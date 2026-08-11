//! Assembles every figure for one scene once all four heuristics have been
//! measured.
//!
//! Deferring until the whole scene is done is what makes the comparison honest:
//! all four heatmaps of a given metric are rendered against one shared colour
//! scale (see `viz::Scale::shared_over`), so brightness differences between
//! panels are real differences in traversal work.

use std::io;
use std::path::{Path, PathBuf};

use crate::bvh::BVHNode;
use crate::diagrams;
use crate::gpu_types::GpuRayCounters;
use crate::viz::{self, contact_sheet, render_beauty, render_difference, render_heatmap, HeatmapSpec, Image, Palette, Scale};

/// Everything the figure generator needs about one heuristic's run.
pub struct FigureInput {
    pub heuristic: String,
    pub records: Vec<GpuRayCounters>,
    pub colors: Vec<[f32; 3]>,
    pub tree: BVHNode,
    pub node_visits_per_ray: f64,
    pub prim_tests_per_ray: f64,
    pub render_time_ms: Option<f64>,
    pub max_depth: usize,
}

/// A per-pixel metric that gets its own heatmap.
struct Metric {
    key: &'static str,
    title: &'static str,
    legend: &'static str,
    palette: Palette,
    /// Extracts the per-pixel value, already normalised per ray where relevant.
    extract: fn(&GpuRayCounters) -> f32,
}

const METRICS: &[Metric] = &[
    Metric {
        key: "node_visits",
        title: "node visits",
        legend: "node visits per ray",
        palette: Palette::Inferno,
        extract: |r| per_ray(r.node_visits, r.ray_count),
    },
    Metric {
        key: "prim_tests",
        title: "primitive tests",
        legend: "ray-triangle tests per ray",
        palette: Palette::Inferno,
        extract: |r| per_ray(r.prim_tests, r.ray_count),
    },
    Metric {
        key: "stack_depth",
        title: "traversal depth",
        legend: "peak stack depth",
        palette: Palette::Turbo,
        // Not divided by ray count: this is a high-water mark, not a total.
        extract: |r| r.max_stack as f32,
    },
];

fn per_ray(value: u32, rays: u32) -> f32 {
    if rays == 0 { 0.0 } else { value as f32 / rays as f32 }
}

/// Writes every figure for one scene. Returns the paths written.
pub fn render_scene_figures(
    output_dir: &Path,
    scene: &str,
    width: u32,
    height: u32,
    inputs: &[FigureInput],
) -> io::Result<Vec<PathBuf>> {
    let mut written = Vec::new();
    if inputs.is_empty() {
        return Ok(written);
    }

    let scene_dir = output_dir.join(scene);

    // ---- Heatmaps, one contact sheet per metric --------------------------
    for metric in METRICS {
        let fields: Vec<Vec<f32>> = inputs
            .iter()
            .map(|input| input.records.iter().map(metric.extract).collect())
            .collect();

        // One scale for all four panels. Without this the panels would each
        // normalise to their own maximum and Random would look as cheap as Sah.
        let scale = Scale::shared_over(fields.iter().map(|f| f.as_slice()));

        let mut panels = Vec::new();
        for (input, field) in inputs.iter().zip(&fields) {
            let subtitle = match metric.key {
                "node_visits" => format!("{:.2} per ray", input.node_visits_per_ray),
                "prim_tests" => format!("{:.2} per ray", input.prim_tests_per_ray),
                _ => format!("tree max depth {}", input.max_depth),
            };

            let panel = render_heatmap(&HeatmapSpec {
                values: field,
                width,
                height,
                scale,
                palette: metric.palette,
                title: &input.heuristic,
                subtitle: &subtitle,
                legend: metric.legend,
            });

            panel.save(&scene_dir.join(format!("{}_{}.png", metric.key, input.heuristic)))?;
            panels.push(panel);
        }

        let sheet = contact_sheet(
            &panels,
            2,
            &format!("{scene} - {}", metric.title),
            &format!(
                "shared colour scale, 0 to {}   |   brighter = more work",
                viz::format_number(scale.max)
            ),
        );
        let path = scene_dir.join(format!("sheet_{}.png", metric.key));
        sheet.save(&path)?;
        written.push(path);
    }

    // ---- Reference renders ------------------------------------------------
    let beauty_panels: Vec<Image> = inputs
        .iter()
        .map(|input| {
            let subtitle = match input.render_time_ms {
                Some(ms) => format!("{ms:.2} ms per frame"),
                None => "gpu timing unavailable".to_string(),
            };
            render_beauty(&input.colors, width, height, &input.heuristic, &subtitle)
        })
        .collect();

    let beauty_sheet = contact_sheet(
        &beauty_panels,
        2,
        &format!("{scene} - rendered output"),
        "all four trees must produce the same image - any visible difference is a bug",
    );
    let path = scene_dir.join("sheet_render.png");
    beauty_sheet.save(&path)?;
    written.push(path);

    // ---- Difference map: best vs worst ------------------------------------
    // Picking the extremes by measured cost rather than hardcoding Sah vs Random
    // keeps the figure meaningful if a heuristic behaves unexpectedly on a scene.
    if inputs.len() >= 2 {
        let cheapest = inputs
            .iter()
            .min_by(|a, b| a.node_visits_per_ray.total_cmp(&b.node_visits_per_ray))
            .unwrap();
        let costliest = inputs
            .iter()
            .max_by(|a, b| a.node_visits_per_ray.total_cmp(&b.node_visits_per_ray))
            .unwrap();

        if !std::ptr::eq(cheapest, costliest) {
            let extract = METRICS[0].extract;
            let a: Vec<f32> = costliest.records.iter().map(extract).collect();
            let b: Vec<f32> = cheapest.records.iter().map(extract).collect();

            let diff = render_difference(
                &a,
                &b,
                width,
                height,
                &format!("{} minus {}", costliest.heuristic, cheapest.heuristic),
                "node visits per ray - red = worst heuristic pays more here",
            );
            let path = scene_dir.join("difference_node_visits.png");
            diff.save(&path)?;
            written.push(path);
        }
    }

    // ---- Structure diagrams ----------------------------------------------
    for input in inputs {
        let svg = diagrams::bvh_icicle_svg(&input.tree, &input.heuristic, scene);
        let path = scene_dir.join(format!("icicle_{}.svg", input.heuristic));
        write_text(&path, &svg)?;
        written.push(path);
    }

    let profiles: Vec<(String, Vec<usize>)> = inputs
        .iter()
        .map(|input| (input.heuristic.clone(), diagrams::depth_profile(&input.tree)))
        .collect();
    let histogram = diagrams::depth_histogram_svg(scene, &profiles);
    let path = scene_dir.join("depth_histogram.svg");
    write_text(&path, &histogram)?;
    written.push(path);

    Ok(written)
}

fn write_text(path: &Path, contents: &str) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, contents)
}
