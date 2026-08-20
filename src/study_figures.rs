//! Figure generation for the study.
//!
//! Two families, produced at different points in the run:
//!
//!   * per-scene rasters -- heatmaps, difference maps and AABB wireframes -- which
//!     need the per-pixel counters and so are drawn while the scene's results are
//!     still in memory;
//!   * summary charts over the finished table rows, drawn once at the end.
//!
//! The rule that keeps the raster figures honest is that panels which will be
//! compared share one colour scale, computed across every panel in the group
//! before any of them is drawn. A panel that normalised to its own maximum would
//! make the worst heuristic look exactly as cheap as the best.

use std::fmt::Write as _;
use std::io;
use std::path::{Path, PathBuf};

use crate::bvh::AABB;
use crate::camera::Camera;
use crate::charts::{self, AxisScale, BarSeries, BarSpec, LineSeries, LineSpec, ScatterPoint, ScatterSpec};
use crate::diagrams;
use crate::gpu_types::GpuRayCounters;
use crate::study::{Row, SceneSpec, StudyConfig};
use crate::viz::{self, contact_sheet, render_difference_scaled, render_heatmap, HeatmapSpec, Image, Palette, Scale};
use crate::wireframe;

/// Everything a figure needs about one heuristic's run on one scene.
pub struct FigureInput {
    pub heuristic: String,
    /// Display name; carries the seed for the random baseline.
    pub label: String,
    pub records: Vec<GpuRayCounters>,
    pub colors: Vec<[f32; 3]>,
    pub boxes: Vec<(AABB, bool)>,
    pub depth_profile: Vec<usize>,
    pub node_visits_per_ray: f64,
    pub prim_tests_per_ray: f64,
    pub render_mean_ms: Option<f64>,
    pub max_depth: usize,
}

fn per_ray(value: u32, rays: u32) -> f32 {
    if rays == 0 { 0.0 } else { value as f32 / rays as f32 }
}

fn prim_tests_field(input: &FigureInput) -> Vec<f32> {
    input.records.iter().map(|r| per_ray(r.prim_tests, r.ray_count)).collect()
}

fn node_visits_field(input: &FigureInput) -> Vec<f32> {
    input.records.iter().map(|r| per_ray(r.node_visits, r.ray_count)).collect()
}

// ---- Per-scene rasters -----------------------------------------------------

pub fn render_scene_figures(
    dir: &Path,
    spec: &SceneSpec,
    config: &StudyConfig,
    camera: &Camera,
    inputs: &[FigureInput],
) -> io::Result<Vec<PathBuf>> {
    let mut written = Vec::new();
    if inputs.is_empty() {
        return Ok(written);
    }

    let width = config.width;
    let height = config.height;

    // ---- Per-pixel cost, 2x2, one shared absolute scale --------------------
    for (key, legend, extract, mean_of) in [
        (
            "prim_tests",
            "ray-triangle tests per ray",
            prim_tests_field as fn(&FigureInput) -> Vec<f32>,
            (|input| input.prim_tests_per_ray) as fn(&FigureInput) -> f64,
        ),
        (
            "node_visits",
            "node visits per ray",
            node_visits_field as fn(&FigureInput) -> Vec<f32>,
            (|input| input.node_visits_per_ray) as fn(&FigureInput) -> f64,
        ),
    ] {
        let fields: Vec<Vec<f32>> = inputs.iter().map(extract).collect();
        if fields.iter().all(|field| field.is_empty()) {
            continue;
        }

        // Absolute, not percentile-clipped: the caption claims every panel is on
        // the same numbered scale, so nothing may be silently clamped.
        let scale = Scale::absolute_over(fields.iter().map(|f| f.as_slice()));

        let panels: Vec<Image> = inputs
            .iter()
            .zip(&fields)
            .map(|(input, field)| {
                render_heatmap(&HeatmapSpec {
                    values: field,
                    width,
                    height,
                    scale,
                    palette: Palette::Inferno,
                    title: &input.label,
                    subtitle: &format!("{:.2} per ray (scene mean)", mean_of(input)),
                    legend,
                })
            })
            .collect();

        let sheet = contact_sheet(
            &panels,
            2,
            &format!("{} - {}", spec.label, legend),
            &format!(
                "shared absolute {} colour scale, 0 to {}   |   darker = more work",
                scale.kind_label(),
                viz::format_number(scale.max)
            ),
        );
        let path = dir.join(format!("fig_heatmap_{}_{key}.png", spec.key));
        sheet.save(&path)?;
        written.push(path);
    }

    // ---- Difference against SAH -------------------------------------------
    if let Some(reference) = inputs.iter().find(|input| input.heuristic == "Sah") {
        let baseline = prim_tests_field(reference);
        let others: Vec<(&FigureInput, Vec<f32>)> = inputs
            .iter()
            .filter(|input| input.heuristic != "Sah")
            .map(|input| (input, prim_tests_field(input)))
            .collect();

        if !baseline.is_empty() && !others.is_empty() {
            // One symmetric range across all three panels, so "more red" means
            // "more excess cost" between panels and not just within one.
            let peak = viz::difference_peak(
                others.iter().map(|(_, field)| (field.as_slice(), baseline.as_slice())),
            );

            let panels: Vec<Image> = others
                .iter()
                .map(|(input, field)| {
                    render_difference_scaled(
                        field,
                        &baseline,
                        peak,
                        width,
                        height,
                        &format!("{} - SAH", input.label),
                        "red = this heuristic tests more triangles here",
                    )
                })
                .collect();

            let sheet = contact_sheet(
                &panels,
                3,
                &format!("{} - excess triangle tests over SAH", spec.label),
                &format!(
                    "diverging scale centred on zero, shared across panels, +/-{} tests per ray",
                    viz::format_number(peak)
                ),
            );
            let path = dir.join(format!("fig_difference_{}_prim_tests.png", spec.key));
            sheet.save(&path)?;
            written.push(path);
        }
    }

    // ---- AABB wireframes at one tree level ---------------------------------
    if inputs.iter().any(|input| !input.boxes.is_empty()) {
        let panels: Vec<Image> = inputs
            .iter()
            .map(|input| {
                let silhouette = prim_tests_field(input);
                wireframe::render_boxes(
                    &input.boxes,
                    (!silhouette.is_empty()).then_some(silhouette.as_slice()),
                    camera,
                    width,
                    height,
                    &input.label,
                    &format!(
                        "{} boxes at depth {} of {}{}",
                        input.boxes.len(),
                        config.structure_depth,
                        input.max_depth,
                        match input.render_mean_ms {
                            Some(ms) => format!("   {ms:.2} ms/frame"),
                            None => String::new(),
                        }
                    ),
                )
            })
            .collect();

        let sheet = contact_sheet(
            &panels,
            2,
            &format!("{} - node boxes at depth {}", spec.label, config.structure_depth),
            "same camera, same level   |   overlap and enclosed empty space are what the cost model responds to",
        );
        let path = dir.join(format!("fig_structure_{}_depth{:02}.png", spec.key, config.structure_depth));
        sheet.save(&path)?;
        written.push(path);
    }

    // ---- Rendered output, as an integrity check ----------------------------
    //
    // Four different trees over the same geometry must produce the same image.
    // The numeric version of this check is in `provenance.md`; this is the one
    // you can see.
    if inputs.iter().any(|input| !input.colors.is_empty()) {
        let panels: Vec<Image> = inputs
            .iter()
            .map(|input| {
                viz::render_beauty(
                    &input.colors,
                    width,
                    height,
                    &input.label,
                    &match input.render_mean_ms {
                        Some(ms) => format!("{ms:.2} ms per dispatch"),
                        None => "gpu timing unavailable".to_string(),
                    },
                )
            })
            .collect();

        let sheet = contact_sheet(
            &panels,
            2,
            &format!("{} - rendered output", spec.label),
            "all four trees must produce the same image; a visible difference is a bug, not a result",
        );
        let path = dir.join(format!("fig_render_{}.png", spec.key));
        sheet.save(&path)?;
        written.push(path);
    }

    // ---- Nodes per depth level ---------------------------------------------
    let profiles: Vec<(String, Vec<usize>)> = inputs
        .iter()
        .map(|input| (input.heuristic.clone(), input.depth_profile.clone()))
        .collect();
    if profiles.iter().any(|(_, counts)| !counts.is_empty()) {
        let svg = diagrams::depth_histogram_svg(&spec.label, &profiles);
        let path = dir.join(format!("fig_depth_profile_{}.svg", spec.key));
        write_text(&path, &svg)?;
        written.push(path);
    }

    Ok(written)
}

// ---- Summary charts --------------------------------------------------------

/// Heuristics in the order they appear in every legend.
const ORDER: [&str; 4] = ["LongestAxisCentroid", "Median", "Sah", "Random"];

fn display_name(heuristic: &str) -> String {
    match heuristic {
        "LongestAxisCentroid" => "Longest-axis centroid".to_string(),
        "Median" => "Median split".to_string(),
        "Sah" => "SAH".to_string(),
        "Random" => "Random".to_string(),
        other => other.to_string(),
    }
}

pub fn render_summary_figures(dir: &Path, rows: &[Row]) -> io::Result<Vec<PathBuf>> {
    let mut written = Vec::new();

    // Scene order as measured, deduplicated, split by set.
    let mut set_a: Vec<String> = Vec::new();
    let mut set_b: Vec<String> = Vec::new();
    for row in rows {
        let target = if row.set == "A" { &mut set_a } else { &mut set_b };
        if !target.contains(&row.scene_key) {
            target.push(row.scene_key.clone());
        }
    }

    let find = |scene_key: &str, heuristic: &str| -> Option<&Row> {
        rows.iter().find(|row| row.scene_key == scene_key && row.heuristic == heuristic)
    };
    let scene_label = |scene_key: &str| -> String {
        rows.iter()
            .find(|row| row.scene_key == scene_key)
            .map(Row::scene_display)
            .unwrap_or_else(|| scene_key.to_string())
    };

    // ---- Figure 1 and 2: Set A, categorical -> grouped bars ----------------
    if !set_a.is_empty() {
        let groups: Vec<String> = set_a.iter().map(|key| scene_label(key)).collect();

        let visits = BarSpec {
            title: "Figure 1. Node visits per ray, Set A",
            caption: "Mean node visits per traversal query, one bar per heuristic, grouped by scene. \
                      Primitive count is held at ~150k; only the distribution changes.",
            y_label: "node visits per ray",
            groups: groups.clone(),
            series: ORDER
                .iter()
                .map(|heuristic| BarSeries {
                    name: heuristic.to_string(),
                    values: set_a
                        .iter()
                        .map(|key| find(key, heuristic).map(|row| row.node_visits_per_ray))
                        .collect(),
                    errors: set_a.iter().map(|_| None).collect(),
                })
                .collect(),
        };
        written.push(write_text_at(dir, "fig01_setA_node_visits.svg", &charts::grouped_bars(&visits))?);

        let render = BarSpec {
            title: "Figure 2. Render time, Set A",
            caption: "GPU time for one dispatch, same grouping as Figure 1. Error bars are +/-1 SD over \
                      5 timed dispatches (random: over 5 seeds). Compare with Figure 1: this is whether \
                      the traversal-count advantage survives into wall-clock.",
            y_label: "render time (ms)",
            groups,
            series: ORDER
                .iter()
                .map(|heuristic| BarSeries {
                    name: heuristic.to_string(),
                    values: set_a
                        .iter()
                        .map(|key| {
                            find(key, heuristic)
                                .filter(|row| row.render_valid)
                                .map(|row| row.render_mean)
                        })
                        .collect(),
                    errors: set_a
                        .iter()
                        .map(|key| {
                            find(key, heuristic)
                                .filter(|row| row.render_valid)
                                .map(|row| row.render_sd)
                        })
                        .collect(),
                })
                .collect(),
        };
        written.push(write_text_at(dir, "fig02_setA_render_time.svg", &charts::grouped_bars(&render))?);
    }

    // ---- Figures 3-5: Set B, continuous -> lines on a log x axis ------------
    if set_b.len() >= 2 {
        let ticks: Vec<f64> = set_b
            .iter()
            .filter_map(|key| rows.iter().find(|row| row.scene_key == *key))
            .map(|row| row.prim_count as f64)
            .collect();

        let series_from = |extract: &dyn Fn(&Row) -> Option<(f64, Option<f64>)>| -> Vec<LineSeries> {
            ORDER
                .iter()
                .map(|heuristic| LineSeries {
                    name: heuristic.to_string(),
                    points: set_b
                        .iter()
                        .filter_map(|key| {
                            let row = find(key, heuristic)?;
                            let (value, error) = extract(row)?;
                            Some((row.prim_count as f64, value, error))
                        })
                        .collect(),
                })
                .collect()
        };

        let visits = LineSpec {
            title: "Figure 3. Node visits per ray vs. primitive count",
            caption: "Set B: the same model at five densities, so distribution is held fixed and only \
                      count varies. Log x axis.",
            x_label: "primitives in the scene (log scale)",
            y_label: "node visits per ray",
            x_scale: AxisScale::Log10,
            y_scale: AxisScale::Linear,
            series: series_from(&|row| Some((row.node_visits_per_ray, None))),
            x_ticks: Some(ticks.clone()),
        };
        written.push(write_text_at(dir, "fig03_setB_node_visits.svg", &charts::line_chart(&visits))?);

        let build = LineSpec {
            title: "Figure 4. Build time vs. primitive count",
            caption: "CPU construction time, log-log. A steeper line is worse asymptotic scaling; the \
                      full-sweep SAH sorts every axis at every node and should separate visibly. Error \
                      bars are +/-1 SD over 5 timed builds.",
            x_label: "primitives in the scene (log scale)",
            y_label: "build time, ms (log scale)",
            x_scale: AxisScale::Log10,
            y_scale: AxisScale::Log10,
            series: series_from(&|row| Some((row.build_mean, Some(row.build_sd)))),
            x_ticks: Some(ticks.clone()),
        };
        written.push(write_text_at(dir, "fig04_setB_build_time.svg", &charts::line_chart(&build))?);

        let render = LineSpec {
            title: "Figure 5. Render time vs. primitive count",
            caption: "GPU time for one dispatch, log x. Read against Figure 4: where the traversal \
                      saving outgrows the extra construction cost. Error bars are +/-1 SD.",
            x_label: "primitives in the scene (log scale)",
            y_label: "render time (ms)",
            x_scale: AxisScale::Log10,
            y_scale: AxisScale::Linear,
            series: series_from(&|row| {
                row.render_valid.then_some((row.render_mean, Some(row.render_sd)))
            }),
            x_ticks: Some(ticks),
        };
        written.push(write_text_at(dir, "fig05_setB_render_time.svg", &charts::line_chart(&render))?);
    }

    // ---- Figure 6: does the cost model predict measured work? --------------
    let cost_points: Vec<ScatterPoint> = rows
        .iter()
        .map(|row| ScatterPoint {
            x: row.total_sah_cost,
            y: row.node_visits_per_ray,
            group: row.heuristic.to_string(),
            label: format!(
                "{} / {} - SAH cost {:.1}, {:.2} visits/ray",
                row.scene_display(),
                display_name(row.heuristic),
                row.total_sah_cost,
                row.node_visits_per_ray
            ),
        })
        .collect();
    if !cost_points.is_empty() {
        let spec = ScatterSpec {
            title: "Figure 6. Predicted cost vs. measured traversal work",
            caption: "One point per condition. x is the SAH cost model evaluated on the finished tree; \
                      y is what the GPU actually counted. A tight line means the model predicts \
                      traversal work; scatter means it does not.",
            x_label: "total SAH cost (model)",
            y_label: "node visits per ray (measured)",
            points: cost_points,
            unit_diagonal: false,
            fit_line: true,
        };
        written.push(write_text_at(dir, "fig06_sah_cost_vs_visits.svg", &charts::scatter(&spec))?);
    }

    // ---- Figure 7: structural work vs wall-clock, both normalised ----------
    let mut normalised = Vec::new();
    for row in rows {
        let Some(base) = rows
            .iter()
            .find(|other| other.scene_key == row.scene_key && other.heuristic == "Random")
        else {
            continue;
        };
        if !(row.render_valid && base.render_valid) || base.node_visits_per_ray <= 0.0 || base.render_mean <= 0.0 {
            continue;
        }
        normalised.push(ScatterPoint {
            x: row.node_visits_per_ray / base.node_visits_per_ray,
            y: row.render_mean / base.render_mean,
            group: row.heuristic.to_string(),
            label: format!(
                "{} / {} - {:.3}x visits, {:.3}x render",
                row.scene_display(),
                display_name(row.heuristic),
                row.node_visits_per_ray / base.node_visits_per_ray,
                row.render_mean / base.render_mean
            ),
        });
    }
    if !normalised.is_empty() {
        let spec = ScatterSpec {
            title: "Figure 7. Normalised traversal work vs. normalised render time",
            caption: "Both axes are multiples of the random baseline for the same scene. On the diagonal, \
                      a saving in node visits buys a proportional saving in wall-clock; off it, the two \
                      disagree.",
            x_label: "node visits per ray (x random)",
            y_label: "render time (x random)",
            points: normalised,
            unit_diagonal: true,
            fit_line: false,
        };
        written.push(write_text_at(dir, "fig07_normalised_visits_vs_render.svg", &charts::scatter(&spec))?);
    }

    written.push(write_text_at(dir, "README.md", &figure_index(dir, rows))?);
    Ok(written)
}

/// A caption sheet, so every figure travels with the sentence that explains it.
fn figure_index(dir: &Path, rows: &[Row]) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# Figures\n");
    let _ = writeln!(
        out,
        "Each figure answers one question. Captions carry the descriptive load; \
         interpretation belongs in the analysis.\n"
    );

    let _ = writeln!(out, "## Set A -- distribution\n");
    let _ = writeln!(
        out,
        "- **Figure 1** (`fig01_setA_node_visits.svg`) -- Node visits per ray, grouped by scene, one bar \
         per heuristic. Does heuristic choice matter more as the distribution becomes uneven?"
    );
    let _ = writeln!(
        out,
        "- **Figure 2** (`fig02_setA_render_time.svg`) -- Render time, same grouping, +/-1 SD. Does the \
         count advantage survive into wall-clock?\n"
    );

    let _ = writeln!(out, "## Set B -- primitive count\n");
    let _ = writeln!(
        out,
        "- **Figure 3** (`fig03_setB_node_visits.svg`) -- Node visits per ray vs. primitive count, one \
         line per heuristic, log x."
    );
    let _ = writeln!(
        out,
        "- **Figure 4** (`fig04_setB_build_time.svg`) -- Build time vs. primitive count, log-log, so the \
         full-sweep SAH's steeper scaling is visible as a steeper slope."
    );
    let _ = writeln!(
        out,
        "- **Figure 5** (`fig05_setB_render_time.svg`) -- Render time vs. primitive count, log x. Where \
         does the traversal saving outgrow the build cost?\n"
    );

    let _ = writeln!(out, "## Cross-cutting\n");
    let _ = writeln!(
        out,
        "- **Figure 6** (`fig06_sah_cost_vs_visits.svg`) -- Total SAH cost against measured node visits \
         per ray, every condition as one point, with a least-squares fit and r. Does the cost model \
         predict measured traversal work?"
    );
    let _ = writeln!(
        out,
        "- **Figure 7** (`fig07_normalised_visits_vs_render.svg`) -- Normalised node visits against \
         normalised render time. Points off the diagonal are conditions where structural work and \
         wall-clock disagree.\n"
    );

    // Only list the per-scene rasters that were actually produced.
    let mut scene_keys: Vec<(&str, String)> = Vec::new();
    for row in rows {
        if !scene_keys.iter().any(|(key, _)| *key == row.scene_key) {
            scene_keys.push((&row.scene_key, row.scene_display()));
        }
    }

    let mut heading_written = false;
    for (key, label) in &scene_keys {
        let heatmap = format!("fig_heatmap_{key}_prim_tests.png");
        if !dir.join(&heatmap).exists() {
            continue;
        }
        if !heading_written {
            let _ = writeln!(out, "## Per-pixel and structure figures\n");
            heading_written = true;
        }
        let _ = writeln!(out, "### {label}\n");
        let _ = writeln!(
            out,
            "- `{heatmap}` -- Primitive tests per ray, 2x2 across the four heuristics. All four panels \
             share one absolute, perceptually uniform colour scale; the colourbar caption states \
             whether that scale is linear or log."
        );
        let _ = writeln!(
            out,
            "- `fig_heatmap_{key}_node_visits.png` -- The same, for node visits per ray."
        );
        let _ = writeln!(
            out,
            "- `fig_difference_{key}_prim_tests.png` -- Difference image, (heuristic - SAH), diverging \
             colormap centred on zero and shared across the three panels. Isolates where the excess \
             cost lives."
        );
        let structure = std::fs::read_dir(dir)
            .ok()
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .find(|name| name.starts_with(&format!("fig_structure_{key}_depth")));
        if let Some(name) = structure {
            let depth = name
                .rsplit("depth")
                .next()
                .and_then(|tail| tail.strip_suffix(".png"))
                .unwrap_or("?");
            let _ = writeln!(
                out,
                "- `{name}` -- AABB wireframes at tree depth {depth}, same camera, 2x2 across heuristics."
            );
        }
        let _ = writeln!(
            out,
            "- `fig_depth_profile_{key}.svg` -- Nodes per depth level for all four heuristics on one axis."
        );
        let _ = writeln!(
            out,
            "- `fig_render_{key}.png` -- The rendered image from each of the four trees. An integrity \
             check rather than a result: they must be identical.\n"
        );
    }

    out
}

fn write_text_at(dir: &Path, name: &str, contents: &str) -> io::Result<PathBuf> {
    let path = dir.join(name);
    write_text(&path, contents)?;
    Ok(path)
}

fn write_text(path: &Path, contents: &str) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, contents)
}
