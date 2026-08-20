//! The full BVH split-heuristic study: one command, every table and figure.
//!
//! `experiment.rs` measures one (scene, heuristic) pair at a time and writes a
//! flat CSV. This module runs the whole protocol -- eight scenes across two sets,
//! four heuristics, repeated timed runs, a seed sweep for the random baseline --
//! and emits the processed tables and figures the write-up refers to.
//!
//! Everything here is arranged around one goal: any difference between two rows
//! should be attributable to the independent variable and nothing else. Three
//! decisions do most of that work.
//!
//! **Repeats are interleaved, not blocked.** Both the CPU build timing and the
//! GPU render timing run round-robin over the variants: every variant is measured
//! once, then every variant again, and so on. Running a variant's five repeats
//! back to back would confound the heuristic with whatever the machine's clocks
//! were doing during that variant's block -- and since the heuristics are always
//! visited in the same order, that confound would be systematic rather than
//! noise. Interleaving spreads any drift evenly across all of them.
//!
//! **Counting and timing never share a dispatch.** Inherited from the harness
//! (Pass A instrumented, Pass B clean), and preserved here.
//!
//! **The random baseline is a mean over seeds, not one lucky tree.** A single
//! random split is a sample from a wide distribution; the quality floor it is
//! meant to establish is the expectation, so it is built and measured once per
//! seed and averaged.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

use glam::Vec3;

use crate::bvh::{self, AABB, BuildStats, BuildParams, SplitHeuristic, DEFAULT_LEAF_PRIM_CUTOFF};
use crate::camera::Camera;
use crate::color::Color;
use crate::gpu_harness::{BvhUpload, CounterSummary, GpuHarness};
use crate::gpu_types::{GpuPlane, GpuRay, GpuRayCounters, GpuTriangle};
use crate::importer::import_obj;
use crate::material::Material;
use crate::objects::{Hittable, Triangle};
use crate::ray::{self, Ray};
use crate::scene::triangle_to_gpu_triangle;
use crate::study_figures;

// ---- Configuration ---------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SceneSpec {
    /// "A" (distribution) or "B" (primitive count).
    pub set: &'static str,
    /// Name as it appears in the tables.
    pub label: String,
    /// Filename-safe identifier for this scene's figures.
    pub key: String,
    /// Primitive count as the study design refers to it, e.g. "150k".
    pub nominal: String,
    /// Mesh files making up the scene. More than one is loaded as a single body
    /// of geometry: the combined bounds are what gets centred and scaled, so the
    /// camera framing stays a controlled variable either way.
    pub paths: Vec<PathBuf>,
}

impl SceneSpec {
    fn new(set: &'static str, label: &str, key: &str, nominal: &str, path: &str) -> Self {
        Self {
            set,
            label: label.to_string(),
            key: key.to_string(),
            nominal: nominal.to_string(),
            paths: vec![PathBuf::from(path)],
        }
    }
}

/// Every controlled variable, in one place.
#[derive(Debug, Clone)]
pub struct StudyConfig {
    pub camera_origin: Vec3,
    pub camera_direction: Vec3,
    pub width: u32,
    pub height: u32,
    pub samples: u32,
    /// Path depth limit. A control: it fixes how much of each dispatch is
    /// secondary-ray work, which is where most of the traversal cost lives.
    pub max_bounces: u32,
    pub rng_seed: u32,
    pub leaf_cutoff: usize,

    /// Builds discarded before the timed builds begin, per variant.
    pub build_warmup: u32,
    /// Timed builds per variant. These are the R1..R5 of Table 2.
    pub build_runs: u32,
    /// Dispatches discarded before the timed dispatches begin, per variant.
    pub render_warmup: u32,
    /// Timed dispatches per variant. These are the R1..R5 of Table 3.
    pub render_runs: u32,

    /// Seeds the random baseline is averaged over.
    pub random_seeds: Vec<u64>,

    pub scenes: Vec<SceneSpec>,
    /// Scene keys that additionally get per-pixel heatmaps and structure
    /// diagrams. Restricted by default because keeping the per-pixel counter
    /// buffers for every variant of a million-triangle scene is a lot of memory
    /// for a figure nobody asked for.
    pub figure_scenes: Vec<String>,
    /// Tree level drawn in the AABB wireframe figure.
    pub structure_depth: usize,

    pub out_dir: PathBuf,
    pub figures: bool,
    /// Run Pass A twice per variant and check the counts match exactly.
    pub verify: bool,
}

/// Extent of the room's longest axis, and therefore of every scene placed in it.
const TARGET_EXTENT: f32 = 3.0;

impl Default for StudyConfig {
    fn default() -> Self {
        Self {
            camera_origin: Vec3::new(0.0, 0.0, 2.3),
            camera_direction: Vec3::new(0.0, 0.0, -1.0),
            // The declared controls for the study. Nothing below is tuned for
            // speed: these are the values every row is measured at, and changing
            // any of them invalidates comparison against previously collected
            // rows.
            width: 800,
            height: 600,
            samples: 150,
            max_bounces: 5,
            rng_seed: 0,
            leaf_cutoff: DEFAULT_LEAF_PRIM_CUTOFF,
            build_warmup: 2,
            build_runs: 5,
            render_warmup: 2,
            render_runs: 5,
            // Five fixed seeds, so the random baseline is reproducible while
            // still being an average over five independent trees.
            random_seeds: vec![
                0x5eed_1234_abcd_0001,
                0x5eed_1234_abcd_0002,
                0x5eed_1234_abcd_0003,
                0x5eed_1234_abcd_0004,
                0x5eed_1234_abcd_0005,
            ],
            scenes: default_scenes(),
            figure_scenes: vec![
                "coral".to_string(),
                "icosphere".to_string(),
                "utah_teapot".to_string(),
            ],
            structure_depth: 6,
            out_dir: PathBuf::from("results/study"),
            figures: true,
            verify: true,
        }
    }
}

/// Set A varies the *distribution* of primitives at a fixed count; Set B varies
/// the count with the distribution held fixed.
pub fn default_scenes() -> Vec<SceneSpec> {
    vec![
        // Icosphere first: a closed, near-uniform shell, so any heuristic that
        // splits sensibly should look much the same on it. It is the even end of
        // the distribution axis and the control the other two are read against.
        SceneSpec::new("A", "Icosphere", "icosphere", "150k", "src/models/uv_sphere_150k.obj"),
        SceneSpec::new("A", "Utah Teapot", "utah_teapot", "150k", "src/models/teapot_150k.obj"),
        // The uneven end: a branching structure whose root box is mostly empty
        // space, with primitives bunched into thin arms. This is where a split
        // plane has somewhere to go badly wrong.
        SceneSpec::new("A", "Coral", "coral", "150k", "src/models/coral_150k.obj"),
        SceneSpec::new("B", "Stanford Dragon", "dragon_10k", "10k", "src/models/standford_dragon_10k.obj"),
        SceneSpec::new("B", "Stanford Dragon", "dragon_50k", "50k", "src/models/standford_dragon_50k.obj"),
        SceneSpec::new("B", "Stanford Dragon", "dragon_150k", "150k", "src/models/standford_dragon_150k.obj"),
        SceneSpec::new("B", "Stanford Dragon", "dragon_400k", "400k", "src/models/standford_dragon_400k.obj"),
        SceneSpec::new("B", "Stanford Dragon", "dragon_1000k", "1000k", "src/models/standford_dragon_1000k.obj"),
    ]
}

// ---- Result types ----------------------------------------------------------

/// One built tree: a heuristic, plus which seed produced it when the heuristic
/// is the random baseline.
#[derive(Debug, Clone, Copy)]
struct Variant {
    heuristic: SplitHeuristic,
    seed: u64,
    /// `None` for the deterministic heuristics.
    seed_index: Option<usize>,
}

impl Variant {
    fn params(&self, leaf_cutoff: usize) -> BuildParams {
        BuildParams { heuristic: self.heuristic, leaf_cutoff, seed: self.seed }
    }

    fn label(&self) -> String {
        match self.seed_index {
            Some(index) => format!("{} (seed {})", self.heuristic.name(), index + 1),
            None => self.heuristic.name().to_string(),
        }
    }
}

/// Everything measured for one built tree.
struct VariantResult {
    variant: Variant,
    build_ms: Vec<f64>,
    render_ms: Vec<f64>,
    counters: CounterSummary,
    stats: BuildStats,
    unmatched_prims: usize,
    determinism_ok: bool,
    depth_profile: Vec<usize>,
    boxes: Vec<(AABB, bool)>,
    records: Option<Vec<GpuRayCounters>>,
    colors: Option<Vec<[f32; 3]>>,
}

/// One row of the tables: a heuristic on a scene, with the random baseline
/// already collapsed over its seeds.
pub struct Row {
    pub set: &'static str,
    pub scene: String,
    pub scene_key: String,
    pub nominal: String,
    pub prim_count: usize,
    pub heuristic: &'static str,
    /// How many trees this row is the average of (>1 only for the random row).
    pub trees: usize,

    pub node_visits_per_ray: f64,
    pub prim_tests_per_ray: f64,
    pub interior_visits_per_ray: f64,

    /// The five figures that go in Table 2. For the deterministic heuristics
    /// these are the five timed builds; for the random baseline each entry is
    /// one seed's own mean.
    pub build_runs: Vec<f64>,
    pub build_mean: f64,
    pub build_sd: f64,
    /// As `build_runs`, for Table 3.
    pub render_runs: Vec<f64>,
    pub render_mean: f64,
    pub render_sd: f64,
    pub render_valid: bool,

    pub node_count: f64,
    pub max_depth: f64,
    pub avg_depth: f64,
    pub avg_leaf_prims: f64,
    pub total_sah_cost: f64,
    pub leaf_count: f64,

    pub incomplete_traversals: u64,
    pub unmatched_prims: usize,
    pub determinism_ok: bool,
}

/// Sample standard deviation (n-1). Returns 0 for fewer than two samples, where
/// a spread is not defined.
fn stddev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

// ---- Driver ----------------------------------------------------------------

pub fn run(config: &StudyConfig) -> io::Result<()> {
    let harness = GpuHarness::new(config.width, config.height).map_err(io::Error::other)?;

    let timestamp_period_ns = harness.timestamp_period_ns();
    let resolution_floor_ms = timestamp_period_ns.map(|ns| ns as f64 / 1.0e6);

    println!("adapter : {} ({})", harness.adapter_name(), harness.backend());
    println!(
        "protocol: {}x{} @ {} spp, {} bounces | builds {}+{} | renders {}+{} | random seeds {} | leaf cutoff {}",
        config.width,
        config.height,
        config.samples,
        config.max_bounces,
        config.build_warmup,
        config.build_runs,
        config.render_warmup,
        config.render_runs,
        config.random_seeds.len(),
        config.leaf_cutoff,
    );
    match resolution_floor_ms {
        Some(floor) => println!(
            "timing  : GPU timestamp period {:.3} ns -> resolution floor {:.6} ms",
            timestamp_period_ns.unwrap_or(0.0),
            floor
        ),
        None => println!("timing  : TIMESTAMP_QUERY unavailable; render columns will be empty"),
    }

    // One ray per pixel, generated once and reused for every scene, heuristic and
    // pass. Regenerating them per scene would risk two passes disagreeing about
    // what "the same ray" means.
    let camera = Camera::new(
        config.width,
        config.height,
        Ray::new(config.camera_origin, config.camera_direction),
    );
    let rays = generate_rays(&camera);
    harness.upload_rays(&rays);

    let planes = room_planes();
    let mut rows: Vec<Row> = Vec::new();
    let mut raw_lines: Vec<String> = Vec::new();
    let mut notes: Vec<String> = Vec::new();

    for spec in &config.scenes {
        let missing: Vec<String> = spec
            .paths
            .iter()
            .filter(|path| !path.exists())
            .map(|path| path.display().to_string())
            .collect();
        if !missing.is_empty() {
            let note = format!("SKIPPED {}: missing {}", spec.label, missing.join(", "));
            eprintln!("{note}");
            notes.push(note);
            continue;
        }

        println!("\n=== {} / {} ({}) ===", spec.set, spec.label, spec.nominal);
        let triangles = load_scene(spec);
        if triangles.is_empty() {
            let note = format!("SKIPPED {}: no triangles", spec.label);
            eprintln!("{note}");
            notes.push(note);
            continue;
        }
        println!("primitives: {}", triangles.len());

        let want_figures = config.figures && config.figure_scenes.contains(&spec.key);
        let results = measure_scene(config, &harness, spec, &triangles, &planes, want_figures);

        for result in &results {
            raw_lines.extend(raw_rows(spec, triangles.len(), result));
        }

        // Every heuristic renders the same geometry with the same rays and the
        // same RNG seed, so the images must agree. A disagreement means the trees
        // are not describing the same scene and no timing comparison is valid.
        if let Some(note) = image_agreement_note(spec, &results) {
            println!("  {note}");
            notes.push(note);
        }

        let scene_rows = collapse(spec, triangles.len(), &results);
        for row in &scene_rows {
            print_row(row);
        }

        if want_figures {
            match study_figures::render_scene_figures(
                &config.out_dir.join("figures"),
                spec,
                config,
                &camera,
                &figure_inputs(&results),
            ) {
                Ok(paths) => println!("  figures: {} file(s)", paths.len()),
                Err(error) => {
                    let note = format!("figure generation failed for {}: {error}", spec.label);
                    eprintln!("  {note}");
                    notes.push(note);
                }
            }
        }

        rows.extend(scene_rows);
    }

    if rows.is_empty() {
        return Err(io::Error::other("no scenes produced any rows"));
    }

    write_outputs(config, &harness, &rows, &raw_lines, &notes, resolution_floor_ms)?;

    if config.figures {
        match study_figures::render_summary_figures(&config.out_dir.join("figures"), &rows) {
            Ok(paths) => println!("summary figures: {} file(s)", paths.len()),
            Err(error) => eprintln!("summary figure generation failed: {error}"),
        }
    }

    println!("\nwrote {} rows to {}", rows.len(), config.out_dir.display());
    Ok(())
}

/// Runs the whole protocol for one scene and returns one result per built tree.
fn measure_scene(
    config: &StudyConfig,
    harness: &GpuHarness,
    spec: &SceneSpec,
    triangles: &[Triangle],
    planes: &[GpuPlane],
    want_figures: bool,
) -> Vec<VariantResult> {
    let variants = variants_for(config);

    // ---- Phase 1: CPU build time, interleaved across variants ---------------
    //
    // Round-robin rather than blocked, so a CPU frequency excursion partway
    // through the scene lands on every heuristic instead of on one of them.
    let mut build_samples: Vec<Vec<f64>> = vec![Vec::new(); variants.len()];
    let total_rounds = config.build_warmup + config.build_runs;
    for round in 0..total_rounds {
        for (index, variant) in rotated(&variants, round) {
            // Cloning the input is loading work, not construction work, so it
            // sits outside the timer -- `construct_bvh_from_tris` consumes it.
            let input = triangles.to_vec();
            let params = variant.params(config.leaf_cutoff);

            let started = Instant::now();
            let tree = bvh::construct_bvh_from_tris(input, params);
            let elapsed = started.elapsed().as_secs_f64() * 1000.0;
            // Keeps the optimiser from concluding the tree is unused and
            // deleting the build we just timed.
            std::hint::black_box(&tree);

            if round >= config.build_warmup {
                build_samples[index].push(elapsed);
            }
            // Dropping a large tree is not free, and it is not construction, so
            // it happens after the clock has stopped.
            drop(tree);
        }
        print!("\r  build round {}/{}    ", round + 1, total_rounds);
        let _ = io_flush();
    }
    println!();

    // ---- Phase 2: materialise each tree once, keep only summaries -----------
    //
    // One extra untimed build per variant buys a bounded memory footprint: the
    // tree is walked, flattened, uploaded, and dropped before the next variant
    // is built, so peak memory is one tree rather than all of them.
    let geometry = {
        let gpu_triangles: Vec<GpuTriangle> =
            triangles.iter().map(triangle_to_gpu_triangle).collect();
        harness.upload_geometry(&gpu_triangles, planes, config.samples, config.rng_seed, config.max_bounces)
    };

    let mut uploads: Vec<BvhUpload> = Vec::with_capacity(variants.len());
    let mut stats: Vec<BuildStats> = Vec::with_capacity(variants.len());
    let mut unmatched: Vec<usize> = Vec::with_capacity(variants.len());
    let mut profiles: Vec<Vec<usize>> = Vec::with_capacity(variants.len());
    let mut boxes: Vec<Vec<(AABB, bool)>> = Vec::with_capacity(variants.len());

    for (index, variant) in variants.iter().enumerate() {
        let params = variant.params(config.leaf_cutoff);
        let tree = bvh::construct_bvh_from_tris(triangles.to_vec(), params);

        let mut variant_stats = bvh::compute_build_stats(&tree);
        variant_stats.build_time_ms = mean(&build_samples[index]);

        if variant_stats.max_depth >= 64 {
            eprintln!(
                "  warning: {} / {} reaches depth {} against a 64-entry GPU traversal stack; \
                 its counts are a lower bound",
                spec.label,
                variant.label(),
                variant_stats.max_depth
            );
        }

        profiles.push(crate::diagrams::depth_profile(&tree));
        boxes.push(if want_figures {
            bvh::aabbs_at_depth(&tree, config.structure_depth)
        } else {
            Vec::new()
        });

        let flattened = bvh::flatten_bvh_for_gpu_checked(&tree, triangles);
        if flattened.unmatched_prims > 0 {
            eprintln!(
                "  warning: {} leaf primitive(s) unmatched in {} / {}",
                flattened.unmatched_prims,
                spec.label,
                variant.label()
            );
        }
        unmatched.push(flattened.unmatched_prims);
        uploads.push(harness.upload_bvh(&geometry, &flattened.nodes, &flattened.indices));
        stats.push(variant_stats);

        drop(tree);
    }

    // ---- Phase 3: Pass A, counts only ---------------------------------------
    let mut counters: Vec<CounterSummary> = Vec::with_capacity(variants.len());
    let mut records: Vec<Option<Vec<GpuRayCounters>>> = Vec::with_capacity(variants.len());
    let mut determinism = vec![true; variants.len()];

    for (index, upload) in uploads.iter().enumerate() {
        let (summary, kept) = harness.collect_counters(upload, want_figures);
        if config.verify {
            let (repeat, _) = harness.collect_counters(upload, false);
            determinism[index] = repeat.total_node_visits == summary.total_node_visits
                && repeat.total_prim_tests == summary.total_prim_tests
                && repeat.total_rays == summary.total_rays;
            if !determinism[index] {
                eprintln!(
                    "  warning: {} / {} produced different counts on a repeat pass",
                    spec.label,
                    variants[index].label()
                );
            }
        }
        counters.push(summary);
        records.push(kept);
    }

    // ---- Phase 4: Pass B, wall-clock only, interleaved ----------------------
    let mut render_samples: Vec<Vec<f64>> = vec![Vec::new(); variants.len()];
    let total_rounds = config.render_warmup + config.render_runs;
    for round in 0..total_rounds {
        for (index, upload) in rotated(&uploads, round) {
            match harness.timed_dispatch(upload) {
                Some(ms) if round >= config.render_warmup => render_samples[index].push(ms),
                // Either the adapter cannot time, the timestamp pair was
                // unusable, or this is a warm-up round. In all three cases the
                // dispatch still ran, which is the point of a warm-up.
                _ => {}
            }
        }
        // Deliberately no per-round progress print here. A dispatch is a few
        // milliseconds; writing to the console is not, and the pause would let
        // the GPU drop its clocks between rounds. `rotated` already keeps that
        // from favouring any one heuristic, but the cheapest fix is not to
        // create the pause at all.
    }
    println!("  render rounds: {total_rounds} ({} discarded)", config.render_warmup);

    // ---- Phase 5: colour capture, strictly after all timing -----------------
    let colors: Vec<Option<Vec<[f32; 3]>>> = uploads
        .iter()
        .map(|upload| Some(harness.capture_color(upload)))
        .collect();

    variants
        .into_iter()
        .enumerate()
        .map(|(index, variant)| VariantResult {
            variant,
            build_ms: std::mem::take(&mut build_samples[index]),
            render_ms: std::mem::take(&mut render_samples[index]),
            counters: counters[index],
            stats: stats[index],
            unmatched_prims: unmatched[index],
            determinism_ok: determinism[index],
            depth_profile: std::mem::take(&mut profiles[index]),
            boxes: std::mem::take(&mut boxes[index]),
            records: records[index].take(),
            colors: colors[index].clone(),
        })
        .collect()
}

/// Yields `(original index, item)` starting at `round % len` and wrapping.
///
/// Interleaving alone still leaves one systematic advantage: the slot at the top
/// of each round follows whatever the driver did between rounds -- a progress
/// print, a stdout flush -- during which the GPU drops its clocks and the CPU
/// loses its caches. Whoever sits in that slot pays for the ramp back up, and
/// with a fixed order that is always the same heuristic. Rotating the start
/// spreads the penalty evenly, which turns a bias into noise the SD already
/// accounts for.
fn rotated<T>(items: &[T], round: u32) -> impl Iterator<Item = (usize, &T)> {
    let length = items.len();
    let offset = if length == 0 { 0 } else { round as usize % length };
    (0..length).map(move |step| {
        let index = (offset + step) % length;
        (index, &items[index])
    })
}

fn variants_for(config: &StudyConfig) -> Vec<Variant> {
    let mut variants = Vec::new();
    for heuristic in SplitHeuristic::ALL {
        if heuristic == SplitHeuristic::Random {
            for (index, seed) in config.random_seeds.iter().enumerate() {
                variants.push(Variant { heuristic, seed: *seed, seed_index: Some(index) });
            }
        } else {
            // The seed is irrelevant to a deterministic heuristic; it is carried
            // only so `BuildParams` has one field fewer to special-case.
            variants.push(Variant { heuristic, seed: 0, seed_index: None });
        }
    }
    variants
}

/// Collapses per-tree results into one row per heuristic, averaging the random
/// baseline over its seeds.
fn collapse(spec: &SceneSpec, prim_count: usize, results: &[VariantResult]) -> Vec<Row> {
    let mut rows = Vec::new();

    for heuristic in SplitHeuristic::ALL {
        let group: Vec<&VariantResult> = results
            .iter()
            .filter(|result| result.variant.heuristic == heuristic)
            .collect();
        if group.is_empty() {
            continue;
        }

        // For a single-tree heuristic the reported runs are its own timed
        // repeats. For the random baseline they are the per-seed means, so the
        // quoted SD carries seed-to-seed structural variance -- which is the
        // uncertainty that actually matters for a control whose whole purpose is
        // to represent "an arbitrary split".
        let (build_runs, render_runs) = if group.len() == 1 {
            (group[0].build_ms.clone(), group[0].render_ms.clone())
        } else {
            (
                group.iter().map(|result| mean(&result.build_ms)).collect(),
                group
                    .iter()
                    .filter(|result| !result.render_ms.is_empty())
                    .map(|result| mean(&result.render_ms))
                    .collect(),
            )
        };

        let over = |f: fn(&VariantResult) -> f64| -> f64 {
            group.iter().map(|result| f(result)).sum::<f64>() / group.len() as f64
        };

        rows.push(Row {
            set: spec.set,
            scene: spec.label.clone(),
            scene_key: spec.key.clone(),
            nominal: spec.nominal.clone(),
            prim_count,
            heuristic: heuristic.name(),
            trees: group.len(),

            node_visits_per_ray: over(|r| r.counters.node_visits_per_ray),
            prim_tests_per_ray: over(|r| r.counters.prim_tests_per_ray),
            interior_visits_per_ray: over(|r| r.counters.interior_visits_per_ray),

            build_mean: mean(&build_runs),
            build_sd: stddev(&build_runs),
            build_runs,
            render_mean: mean(&render_runs),
            render_sd: stddev(&render_runs),
            render_valid: !render_runs.is_empty(),
            render_runs,

            node_count: over(|r| r.stats.node_count as f64),
            max_depth: over(|r| r.stats.max_depth as f64),
            avg_depth: over(|r| r.stats.avg_depth),
            avg_leaf_prims: over(|r| r.stats.avg_leaf_prims),
            total_sah_cost: over(|r| r.stats.total_sah_cost),
            leaf_count: over(|r| r.stats.leaf_count as f64),

            incomplete_traversals: group
                .iter()
                .map(|result| result.counters.incomplete_traversals)
                .sum(),
            unmatched_prims: group.iter().map(|result| result.unmatched_prims).sum(),
            determinism_ok: group.iter().all(|result| result.determinism_ok),
        });
    }

    rows
}

fn print_row(row: &Row) {
    let render = if row.render_valid {
        format!("{:>8.3} +/- {:<6.3}", row.render_mean, row.render_sd)
    } else {
        format!("{:>19}", "n/a")
    };
    println!(
        "  {:<20} visits/ray {:>7.2}  tests/ray {:>7.2}  build {:>9.2} +/- {:<8.2}  render {}  sah {:>8.2}",
        row.heuristic,
        row.node_visits_per_ray,
        row.prim_tests_per_ray,
        row.build_mean,
        row.build_sd,
        render,
        row.total_sah_cost,
    );
    if row.incomplete_traversals > 0 {
        println!(
            "    warning: {} traversal(s) hit the stack guard; counts are a lower bound",
            row.incomplete_traversals
        );
    }
    if !row.determinism_ok {
        println!("    warning: repeat count pass disagreed");
    }
}

/// Reports the worst per-channel disagreement between each heuristic's render
/// and the SAH render, which should be zero.
fn image_agreement_note(spec: &SceneSpec, results: &[VariantResult]) -> Option<String> {
    let reference = results
        .iter()
        .find(|result| result.variant.heuristic == SplitHeuristic::SurfaceAreaHeuristic)?
        .colors
        .as_ref()?;

    let mut worst = 0.0f32;
    for result in results {
        let Some(colors) = result.colors.as_ref() else { continue };
        if colors.len() != reference.len() {
            continue;
        }
        for (a, b) in colors.iter().zip(reference) {
            for channel in 0..3 {
                worst = worst.max((a[channel] - b[channel]).abs());
            }
        }
    }

    Some(format!(
        "image agreement vs SAH on {}: max channel difference {worst:.3e}",
        spec.label
    ))
}

fn figure_inputs(results: &[VariantResult]) -> Vec<study_figures::FigureInput> {
    results
        .iter()
        .filter(|result| result.variant.seed_index.unwrap_or(0) == 0)
        .map(|result| study_figures::FigureInput {
            heuristic: result.variant.heuristic.name().to_string(),
            label: result.variant.label(),
            records: result.records.clone().unwrap_or_default(),
            colors: result.colors.clone().unwrap_or_default(),
            boxes: result.boxes.clone(),
            depth_profile: result.depth_profile.clone(),
            node_visits_per_ray: result.counters.node_visits_per_ray,
            prim_tests_per_ray: result.counters.prim_tests_per_ray,
            render_mean_ms: (!result.render_ms.is_empty()).then(|| mean(&result.render_ms)),
            max_depth: result.stats.max_depth,
        })
        .collect()
}

fn io_flush() -> io::Result<()> {
    use std::io::Write;
    io::stdout().flush()
}

// ---- Scene construction ----------------------------------------------------

/// Loads a scene's geometry and normalises it into the shared room.
///
/// Multiple files are concatenated first and normalised once, so the scale
/// applies to the scene as a whole rather than to each file separately.
fn load_scene(spec: &SceneSpec) -> Vec<Triangle> {
    // Diffuse, non-emissive, identical for every scene: secondary rays scatter
    // back into the room and keep exercising the BVH instead of escaping.
    let material = Material::new(Color::new(0.75, 0.75, 0.75), 1.0, 0.0, 0.0);

    let mut all: Vec<Triangle> = Vec::new();
    for path in &spec.paths {
        let mut mesh = import_obj(&path.to_string_lossy());
        mesh.set_material(material);
        all.extend(mesh.get_triangles());
    }

    if all.is_empty() {
        return all;
    }

    // Centre at the origin and scale the longest extent to the room's target.
    // Without this the camera framing would vary per scene and the "identical
    // camera" control would mean nothing.
    let (min, max) = bounds_of(&all);
    let extent = (max - min).max_element().max(1e-6);
    let scale = TARGET_EXTENT / extent;
    let offset = -(min + max) * 0.5 * scale;

    for triangle in &mut all {
        let [v0, v1, v2] = triangle.get_vertices();
        *triangle = Triangle::new(
            v0 * scale + offset,
            v1 * scale + offset,
            v2 * scale + offset,
            material,
        );
    }

    all
}

fn bounds_of(tris: &[Triangle]) -> (Vec3, Vec3) {
    tris.iter().flat_map(|tri| tri.get_vertices()).fold(
        (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
        |(min, max), v| (min.min(v), max.max(v)),
    )
}

/// A fixed Cornell-style room, identical for every scene. The planes are tested
/// outside the BVH so they never contribute to the traversal counts, but they
/// keep secondary rays alive instead of letting them escape into a black void.
fn room_planes() -> Vec<GpuPlane> {
    fn plane(center: [f32; 3], normal: [f32; 3], size: f32, albedo: [f32; 3], emission: f32) -> GpuPlane {
        GpuPlane {
            center: [center[0], center[1], center[2], 0.0],
            normal: [normal[0], normal[1], normal[2], 0.0],
            width: size,
            length: size,
            _pad2: [0.0, 0.0],
            albedo: [albedo[0], albedo[1], albedo[2], 0.0],
            emission,
            metallic: 0.0,
            roughness: 1.0,
            _pad3: 0.0,
        }
    }

    let white = [0.8, 0.8, 0.8];
    vec![
        plane([0.0, -2.5, 0.0], [0.0, 1.0, 0.0], 5.0, white, 0.0),
        plane([0.0, 2.5, 0.0], [0.0, -1.0, 0.0], 5.0, white, 0.0),
        plane([0.0, 0.0, -2.5], [0.0, 0.0, 1.0], 5.0, white, 0.0),
        plane([-2.5, 0.0, 0.0], [1.0, 0.0, 0.0], 5.0, [0.9, 0.2, 0.2], 0.0),
        plane([2.5, 0.0, 0.0], [-1.0, 0.0, 0.0], 5.0, [0.2, 0.9, 0.2], 0.0),
        plane([0.0, 2.499, 0.0], [0.0, -1.0, 0.0], 3.0, [1.0, 1.0, 1.0], 1.0),
    ]
}

fn generate_rays(camera: &Camera) -> Vec<GpuRay> {
    let mut rays = Vec::with_capacity((camera.width() * camera.height()) as usize);
    camera.for_each_pixel(|x, y| {
        rays.push(ray::get_ray_from_screen(camera, x, y).to_gpu_ray());
    });
    rays
}

// ---- Output ----------------------------------------------------------------

/// Long-format rows: one line per individual timed run, so every number in the
/// processed tables can be recomputed from source.
fn raw_rows(spec: &SceneSpec, prim_count: usize, result: &VariantResult) -> Vec<String> {
    let mut lines = Vec::new();
    let seed = result
        .variant
        .seed_index
        .map(|index| (index + 1).to_string())
        .unwrap_or_else(|| "-".to_string());

    for (index, ms) in result.build_ms.iter().enumerate() {
        lines.push(format!(
            "{},{},{},{},{},{},build,{},{:.6}",
            spec.set,
            csv_field(&spec.label),
            spec.key,
            prim_count,
            result.variant.heuristic.name(),
            seed,
            index + 1,
            ms
        ));
    }
    for (index, ms) in result.render_ms.iter().enumerate() {
        lines.push(format!(
            "{},{},{},{},{},{},render,{},{:.6}",
            spec.set,
            csv_field(&spec.label),
            spec.key,
            prim_count,
            result.variant.heuristic.name(),
            seed,
            index + 1,
            ms
        ));
    }
    lines
}

fn write_outputs(
    config: &StudyConfig,
    harness: &GpuHarness,
    rows: &[Row],
    raw_lines: &[String],
    notes: &[String],
    resolution_floor_ms: Option<f64>,
) -> io::Result<()> {
    fs::create_dir_all(&config.out_dir)?;

    let runs = config.build_runs.max(config.render_runs) as usize;
    let run_headers: Vec<String> = (1..=runs).map(|index| format!("R{index}")).collect();

    // Table 1 -- traversal counts.
    let mut table1 = String::from(
        "set,scene,prim_count,heuristic,node_visits_per_ray,prim_tests_per_ray,interior_visits_per_ray\n",
    );
    for row in rows {
        let _ = writeln!(
            table1,
            "{},{},{},{},{:.4},{:.4},{:.4}",
            row.set,
            csv_field(&row.scene_display()),
            row.prim_count,
            row.heuristic,
            row.node_visits_per_ray,
            row.prim_tests_per_ray,
            // Not part of the table as specified, but it costs nothing to carry
            // and it separates "walked past" from "opened a leaf".
            row.interior_visits_per_ray
        );
    }
    write_file(&config.out_dir.join("table1_traversal_counts.csv"), &table1)?;

    // Tables 2 and 3 -- every timed run, plus mean and SD.
    let table2 = run_table(rows, &run_headers, |row| (&row.build_runs, row.build_mean, row.build_sd, true));
    write_file(&config.out_dir.join("table2_build_time_ms.csv"), &table2)?;

    let table3 = run_table(rows, &run_headers, |row| {
        (&row.render_runs, row.render_mean, row.render_sd, row.render_valid)
    });
    write_file(&config.out_dir.join("table3_render_time_ms.csv"), &table3)?;

    // Table 4 -- tree quality.
    let mut table4 = String::from(
        "set,scene,heuristic,node_count,leaf_count,max_depth,avg_depth,avg_leaf_prims,total_sah_cost\n",
    );
    for row in rows {
        let _ = writeln!(
            table4,
            "{},{},{},{:.1},{:.1},{:.1},{:.4},{:.4},{:.4}",
            row.set,
            csv_field(&row.scene_display()),
            row.heuristic,
            row.node_count,
            row.leaf_count,
            row.max_depth,
            row.avg_depth,
            row.avg_leaf_prims,
            row.total_sah_cost
        );
    }
    write_file(&config.out_dir.join("table4_tree_quality.csv"), &table4)?;

    // Table 5 -- the consolidated summary.
    let mut table5 = String::from(
        "set,scene,prim_count,heuristic,node_visits_per_ray,prim_tests_per_ray,\
         build_mean_ms,build_sd_ms,render_mean_ms,render_sd_ms,total_sah_cost\n",
    );
    for row in rows {
        let _ = writeln!(
            table5,
            "{},{},{},{},{:.4},{:.4},{:.4},{:.4},{},{},{:.4}",
            row.set,
            csv_field(&row.scene_display()),
            row.prim_count,
            row.heuristic,
            row.node_visits_per_ray,
            row.prim_tests_per_ray,
            row.build_mean,
            row.build_sd,
            optional(row.render_valid, row.render_mean),
            optional(row.render_valid, row.render_sd),
            row.total_sah_cost
        );
    }
    write_file(&config.out_dir.join("table5_summary.csv"), &table5)?;

    // Table 6 -- normalised to the random baseline.
    let baselines = random_baselines(rows);
    let mut table6 = String::from(
        "set,scene,prim_count,heuristic,node_visits_x_random,prim_tests_x_random,render_time_x_random\n",
    );
    for row in rows {
        let Some(base) = baselines.get(&row.scene_key) else { continue };
        let _ = writeln!(
            table6,
            "{},{},{},{},{},{},{}",
            row.set,
            csv_field(&row.scene_display()),
            row.prim_count,
            row.heuristic,
            ratio(row.node_visits_per_ray, base.node_visits_per_ray),
            ratio(row.prim_tests_per_ray, base.prim_tests_per_ray),
            if row.render_valid && base.render_valid {
                ratio(row.render_mean, base.render_mean)
            } else {
                String::new()
            }
        );
    }
    write_file(&config.out_dir.join("table6_normalised.csv"), &table6)?;

    // Raw runs.
    let mut raw = String::from("set,scene,scene_key,prim_count,heuristic,seed,measurement,run,value_ms\n");
    for line in raw_lines {
        raw.push_str(line);
        raw.push('\n');
    }
    write_file(&config.out_dir.join("raw_runs.csv"), &raw)?;

    // Markdown rendering of every table, ready to paste into the write-up.
    let markdown = markdown_tables(config, rows, &run_headers, &baselines);
    write_file(&config.out_dir.join("tables.md"), &markdown)?;

    // Provenance and uncertainty.
    let provenance = provenance_text(config, harness, rows, notes, resolution_floor_ms);
    write_file(&config.out_dir.join("provenance.md"), &provenance)?;

    Ok(())
}

impl Row {
    /// Scene name with its size, so the Set B rows are distinguishable from each
    /// other in a table that has no separate size column.
    pub fn scene_display(&self) -> String {
        if self.set == "B" {
            format!("{} {}", self.scene, self.nominal)
        } else {
            self.scene.clone()
        }
    }
}

fn ratio(value: f64, base: f64) -> String {
    if base.abs() < 1e-12 {
        String::new()
    } else {
        format!("{:.4}", value / base)
    }
}

fn optional(valid: bool, value: f64) -> String {
    if valid { format!("{value:.4}") } else { String::new() }
}

/// The random row for each scene, which Table 6 divides by.
fn random_baselines(rows: &[Row]) -> HashMap<String, RandomBaseline> {
    rows.iter()
        .filter(|row| row.heuristic == "Random")
        .map(|row| {
            (
                row.scene_key.clone(),
                RandomBaseline {
                    node_visits_per_ray: row.node_visits_per_ray,
                    prim_tests_per_ray: row.prim_tests_per_ray,
                    render_mean: row.render_mean,
                    render_valid: row.render_valid,
                },
            )
        })
        .collect()
}

struct RandomBaseline {
    node_visits_per_ray: f64,
    prim_tests_per_ray: f64,
    render_mean: f64,
    render_valid: bool,
}

fn run_table(
    rows: &[Row],
    run_headers: &[String],
    extract: impl Fn(&Row) -> (&Vec<f64>, f64, f64, bool),
) -> String {
    let mut out = format!("set,scene,heuristic,{},mean,sd\n", run_headers.join(","));
    for row in rows {
        let (runs, mean_value, sd, valid) = extract(row);
        let mut cells: Vec<String> = (0..run_headers.len())
            .map(|index| match runs.get(index) {
                Some(value) if valid => format!("{value:.4}"),
                _ => String::new(),
            })
            .collect();
        cells.push(optional(valid, mean_value));
        cells.push(optional(valid, sd));

        let _ = writeln!(
            out,
            "{},{},{},{}",
            row.set,
            csv_field(&row.scene_display()),
            row.heuristic,
            cells.join(",")
        );
    }
    out
}

fn markdown_tables(
    config: &StudyConfig,
    rows: &[Row],
    run_headers: &[String],
    baselines: &HashMap<String, RandomBaseline>,
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# BVH split-heuristic study -- tables\n");
    let _ = writeln!(
        out,
        "Generated by `testyo study`. Every figure below is also available as CSV \
         in this directory; `raw_runs.csv` holds each individual timed run.\n"
    );

    let _ = writeln!(out, "## Table 1 -- Traversal counts\n");
    let _ = writeln!(out, "| Set | Scene | Prim count | Heuristic | Node visits / ray | Prim tests / ray |");
    let _ = writeln!(out, "|---|---|---:|---|---:|---:|");
    for row in rows {
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {:.2} | {:.2} |",
            row.set,
            row.scene_display(),
            format_count(row.prim_count),
            heuristic_display(row),
            row.node_visits_per_ray,
            row.prim_tests_per_ray
        );
    }

    let _ = writeln!(out, "\n## Table 2 -- Build time (CPU, ms)\n");
    let _ = writeln!(
        out,
        "All {} timed runs shown; {} warm-up builds were discarded first. \
         For the random baseline each column is one seed's own mean.\n",
        config.build_runs, config.build_warmup
    );
    let _ = writeln!(
        out,
        "| Set | Scene | Heuristic | {} | Mean | SD |",
        run_headers.join(" | ")
    );
    let _ = writeln!(
        out,
        "|---|---|---|{}---:|---:|",
        run_headers.iter().map(|_| "---:|").collect::<String>()
    );
    for row in rows {
        let cells: Vec<String> = (0..run_headers.len())
            .map(|index| {
                row.build_runs
                    .get(index)
                    .map(|value| format!("{value:.2}"))
                    .unwrap_or_default()
            })
            .collect();
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {:.2} | {:.2} |",
            row.set,
            row.scene_display(),
            heuristic_display(row),
            cells.join(" | "),
            row.build_mean,
            row.build_sd
        );
    }

    let _ = writeln!(out, "\n## Table 3 -- Render time (GPU, ms)\n");
    let _ = writeln!(
        out,
        "All {} timed dispatches shown; {} warm-up dispatches were discarded first. \
         For the random baseline each column is one seed's own mean.\n",
        config.render_runs, config.render_warmup
    );
    let _ = writeln!(
        out,
        "| Set | Scene | Heuristic | {} | Mean | SD |",
        run_headers.join(" | ")
    );
    let _ = writeln!(
        out,
        "|---|---|---|{}---:|---:|",
        run_headers.iter().map(|_| "---:|").collect::<String>()
    );
    for row in rows {
        let cells: Vec<String> = (0..run_headers.len())
            .map(|index| {
                row.render_runs
                    .get(index)
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_default()
            })
            .collect();
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} |",
            row.set,
            row.scene_display(),
            heuristic_display(row),
            cells.join(" | "),
            // Three decimals, matching the per-run columns: quoting a mean to
            // more precision than its samples implies precision that is not there.
            if row.render_valid { format!("{:.3}", row.render_mean) } else { "n/a".to_string() },
            if row.render_valid { format!("{:.3}", row.render_sd) } else { "n/a".to_string() }
        );
    }

    let _ = writeln!(out, "\n## Table 4 -- Tree quality statistics\n");
    let _ = writeln!(
        out,
        "Deterministic given the geometry, so one row per scene x heuristic. \
         The random rows are the mean over {} seeds.\n",
        config.random_seeds.len()
    );
    let _ = writeln!(
        out,
        "| Set | Scene | Heuristic | Node count | Max depth | Avg depth | Avg leaf prims | Total SAH cost |"
    );
    let _ = writeln!(out, "|---|---|---|---:|---:|---:|---:|---:|");
    for row in rows {
        let _ = writeln!(
            out,
            "| {} | {} | {} | {:.0} | {:.1} | {:.2} | {:.2} | {:.2} |",
            row.set,
            row.scene_display(),
            heuristic_display(row),
            row.node_count,
            row.max_depth,
            row.avg_depth,
            row.avg_leaf_prims,
            row.total_sah_cost
        );
    }

    let _ = writeln!(out, "\n## Table 5 -- Summary\n");
    let _ = writeln!(
        out,
        "| Scene | Heuristic | Visits/ray | Tests/ray | Build (ms +/- SD) | Render (ms +/- SD) | Total SAH cost |"
    );
    let _ = writeln!(out, "|---|---|---:|---:|---:|---:|---:|");
    for row in rows {
        let render = if row.render_valid {
            format!("{:.3} +/- {:.3}", row.render_mean, row.render_sd)
        } else {
            "n/a".to_string()
        };
        let _ = writeln!(
            out,
            "| {} | {} | {:.2} | {:.2} | {:.2} +/- {:.2} | {} | {:.2} |",
            row.scene_display(),
            heuristic_display(row),
            row.node_visits_per_ray,
            row.prim_tests_per_ray,
            row.build_mean,
            row.build_sd,
            render,
            row.total_sah_cost
        );
    }

    let _ = writeln!(out, "\n## Table 6 -- Normalised to the random baseline\n");
    let _ = writeln!(
        out,
        "Each cell is that heuristic divided by the random split's figure for the \
         same scene. Below 1.00 is cheaper than random.\n"
    );
    let _ = writeln!(
        out,
        "| Scene | Heuristic | Visits/ray (x random) | Tests/ray (x random) | Render time (x random) |"
    );
    let _ = writeln!(out, "|---|---|---:|---:|---:|");
    for row in rows {
        let Some(base) = baselines.get(&row.scene_key) else { continue };
        let render = if row.render_valid && base.render_valid {
            ratio(row.render_mean, base.render_mean)
        } else {
            "n/a".to_string()
        };
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} |",
            row.scene_display(),
            heuristic_display(row),
            ratio(row.node_visits_per_ray, base.node_visits_per_ray),
            ratio(row.prim_tests_per_ray, base.prim_tests_per_ray),
            render
        );
    }

    out
}

fn heuristic_display(row: &Row) -> String {
    match row.heuristic {
        "LongestAxisCentroid" => "Longest-axis centroid".to_string(),
        "Median" => "Median split".to_string(),
        "Sah" => "SAH".to_string(),
        "Random" => format!("Random (mean of {} seeds)", row.trees),
        other => other.to_string(),
    }
}

fn format_count(count: usize) -> String {
    if count >= 1_000_000 {
        format!("{:.2}M", count as f64 / 1.0e6)
    } else if count >= 1_000 {
        format!("{}k", count / 1_000)
    } else {
        count.to_string()
    }
}

fn provenance_text(
    config: &StudyConfig,
    harness: &GpuHarness,
    rows: &[Row],
    notes: &[String],
    resolution_floor_ms: Option<f64>,
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# Provenance and uncertainty\n");

    let _ = writeln!(out, "## Apparatus\n");
    let _ = writeln!(out, "- Adapter: {} ({})", harness.adapter_name(), harness.backend());

    let _ = writeln!(out, "\n### Controlled variables\n");
    let _ = writeln!(out, "| Variable | Value |");
    let _ = writeln!(out, "|---|---|");
    let _ = writeln!(out, "| Resolution | {}x{} |", config.width, config.height);
    let _ = writeln!(out, "| Sample count | {} samples per dispatch |", config.samples);
    let _ = writeln!(out, "| Max bounces | {} |", config.max_bounces);
    let _ = writeln!(out, "| RNG seed | {} |", config.rng_seed);
    let _ = writeln!(
        out,
        "| Scene | Standard Cornell box (white floor/ceiling/back, red left wall, green right wall, \
         ceiling light) with one 3D model |"
    );
    let _ = writeln!(
        out,
        "| Camera | 90 degree vertical FOV, origin {:?} looking {:?}, identical every run |",
        config.camera_origin.to_array(),
        config.camera_direction.to_array()
    );
    let _ = writeln!(out, "| Leaf node primitive cutoff | {} |", config.leaf_cutoff);
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "- Scene normalisation: every mesh centred and scaled so its longest extent is {TARGET_EXTENT}, \
         inside a fixed Cornell-style room whose planes are intersected outside the BVH and so \
         contribute nothing to the traversal counts.\n"
    );

    let _ = writeln!(out, "## How the spread was obtained\n");
    let _ = writeln!(
        out,
        "Build time: {} warm-up builds discarded, then {} timed builds. The clock is a monotonic \
         CPU timer around construction alone -- the triangle list is cloned and the finished tree is \
         dropped outside the timed region, because neither is construction work.",
        config.build_warmup, config.build_runs
    );
    let _ = writeln!(
        out,
        "\nRender time: {} warm-up dispatches discarded, then {} timed dispatches, measured on the \
         GPU timeline with timestamp queries around the compute pass. Counting and timing never \
         share a dispatch: the instrumented shader variant produces the counts, the clean variant \
         produces the times.",
        config.render_warmup, config.render_runs
    );
    let _ = writeln!(
        out,
        "\nRepeats are interleaved rather than blocked. Round r measures every variant once before \
         round r+1 begins, so a thermal or clock excursion partway through a scene is spread across \
         all heuristics instead of landing on whichever one happened to be running. Quoted SD is the \
         sample standard deviation (n-1) over the {} timed runs.",
        config.build_runs
    );
    let _ = writeln!(
        out,
        "\nThe random baseline is different in kind: it is built {} times from {} fixed seeds, and \
         each of its five reported runs is one seed's own mean. Its SD therefore carries \
         seed-to-seed *structural* variance, which is larger than the pure timing noise in the other \
         rows and is the honest uncertainty for a control meant to stand for \"an arbitrary split\".\n",
        config.random_seeds.len(),
        config.random_seeds.len()
    );

    let _ = writeln!(out, "## Timestamp resolution floor\n");
    match resolution_floor_ms {
        Some(floor) => {
            let _ = writeln!(
                out,
                "The adapter reports a timestamp period of {:.4} ns, so the smallest interval it can \
                 distinguish is {floor:.6} ms. Render times are quoted against that floor.\n",
                floor * 1.0e6
            );

            let mut flagged = Vec::new();
            for row in rows {
                if !row.render_valid {
                    continue;
                }
                let relative = floor / row.render_mean.max(1e-12);
                if relative > 0.001 {
                    flagged.push(format!(
                        "- {} / {}: {:.3} ms, which is {:.0}x the resolution floor ({:.3}% quantisation)",
                        row.scene_display(),
                        heuristic_display(row),
                        row.render_mean,
                        row.render_mean / floor,
                        relative * 100.0
                    ));
                }
            }

            if flagged.is_empty() {
                let _ = writeln!(
                    out,
                    "Every measured render time is more than a thousand ticks above the floor, so \
                     quantisation contributes less than 0.1% to any of them.\n"
                );
            } else {
                let _ = writeln!(
                    out,
                    "The following render times sit close enough to the floor that quantisation is \
                     worth more than 0.1% of the value; treat them as the least reliable numbers in \
                     the set.\n"
                );
                for line in &flagged {
                    let _ = writeln!(out, "{line}");
                }
                let _ = writeln!(out);
            }

            // Coefficient of variation is the more useful reliability signal once
            // the floor is comfortably clear.
            let mut noisy: Vec<String> = rows
                .iter()
                .filter(|row| row.render_valid && row.render_mean > 0.0)
                .filter(|row| row.render_sd / row.render_mean > 0.05 && row.heuristic != "Random")
                .map(|row| {
                    format!(
                        "- {} / {}: SD is {:.1}% of the mean",
                        row.scene_display(),
                        heuristic_display(row),
                        100.0 * row.render_sd / row.render_mean
                    )
                })
                .collect();
            if !noisy.is_empty() {
                let _ = writeln!(out, "Run-to-run spread above 5% of the mean:\n");
                noisy.sort();
                for line in noisy {
                    let _ = writeln!(out, "{line}");
                }
                let _ = writeln!(out);
            }
        }
        None => {
            let _ = writeln!(
                out,
                "This adapter does not expose TIMESTAMP_QUERY, so no render times were recorded and \
                 Table 3, Table 6's render column and the render-time figures are empty.\n"
            );
        }
    }

    let _ = writeln!(out, "## Integrity checks\n");
    let unmatched: usize = rows.iter().map(|row| row.unmatched_prims).sum();
    let incomplete: u64 = rows.iter().map(|row| row.incomplete_traversals).sum();
    let nondeterministic = rows.iter().filter(|row| !row.determinism_ok).count();
    let _ = writeln!(
        out,
        "- Primitives lost on the way to the GPU: {unmatched} (must be 0; anything else means the \
         counts describe different geometry than the tree)"
    );
    let _ = writeln!(
        out,
        "- Traversals that hit the 64-entry stack guard: {incomplete} (must be 0; anything else \
         makes the affected counts a lower bound)"
    );
    let _ = writeln!(
        out,
        "- Rows whose repeat counting pass disagreed: {nondeterministic} (must be 0)"
    );
    if !notes.is_empty() {
        let _ = writeln!(out, "\n### Run log\n");
        for note in notes {
            let _ = writeln!(out, "- {note}");
        }
    }

    let _ = writeln!(out, "\n## Scenes\n");
    let _ = writeln!(out, "| Set | Scene | Nominal | Actual prims | Source |");
    let _ = writeln!(out, "|---|---|---|---:|---|");
    let mut seen: Vec<&str> = Vec::new();
    for row in rows {
        if seen.contains(&row.scene_key.as_str()) {
            continue;
        }
        seen.push(&row.scene_key);
        let sources = config
            .scenes
            .iter()
            .find(|spec| spec.key == row.scene_key)
            .map(|spec| {
                spec.paths
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join(" + ")
            })
            .unwrap_or_default();
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | `{}` |",
            row.set,
            row.scene,
            row.nominal,
            row.prim_count,
            sources
        );
    }

    out
}

fn write_file(path: &Path, contents: &str) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, contents)?;
    println!("  wrote {}", path.display());
    Ok(())
}

fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

// ---- Command line ----------------------------------------------------------

pub fn parse_args(args: &[String]) -> Result<StudyConfig, String> {
    let mut config = StudyConfig::default();
    let mut i = 0;

    while i < args.len() {
        let flag = args[i].as_str();
        let value = || -> Result<String, String> {
            args.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{flag} requires a value"))
        };

        match flag {
            "--out" => { config.out_dir = PathBuf::from(value()?); i += 2; }
            "--width" => { config.width = parse_num(flag, &value()?)?; i += 2; }
            "--height" => { config.height = parse_num(flag, &value()?)?; i += 2; }
            "--samples" => { config.samples = parse_num(flag, &value()?)?; i += 2; }
            "--bounces" => { config.max_bounces = parse_num(flag, &value()?)?; i += 2; }
            "--seed" => { config.rng_seed = parse_num(flag, &value()?)?; i += 2; }
            "--leaf-cutoff" => { config.leaf_cutoff = parse_num(flag, &value()?)?; i += 2; }
            "--build-warmup" => { config.build_warmup = parse_num(flag, &value()?)?; i += 2; }
            "--build-runs" => { config.build_runs = parse_num(flag, &value()?)?; i += 2; }
            "--render-warmup" => { config.render_warmup = parse_num(flag, &value()?)?; i += 2; }
            "--render-runs" => { config.render_runs = parse_num(flag, &value()?)?; i += 2; }
            "--structure-depth" => { config.structure_depth = parse_num(flag, &value()?)?; i += 2; }
            "--no-figures" => { config.figures = false; i += 1; }
            "--no-verify" => { config.verify = false; i += 1; }
            "--random-seeds" => {
                let count: usize = parse_num(flag, &value()?)?;
                if count == 0 {
                    return Err("--random-seeds must be at least 1".to_string());
                }
                // Derived from a fixed base so the set is reproducible at any
                // size, and so seed k means the same tree whatever the count.
                config.random_seeds =
                    (0..count).map(|index| 0x5eed_1234_abcd_0000 + index as u64 + 1).collect();
                i += 2;
            }
            "--only" => {
                let wanted: Vec<String> = value()?.split(',').map(|s| s.trim().to_string()).collect();
                config.scenes.retain(|spec| wanted.contains(&spec.key));
                if config.scenes.is_empty() {
                    return Err(format!(
                        "--only matched no scenes; known keys: {}",
                        default_scenes().iter().map(|s| s.key.clone()).collect::<Vec<_>>().join(", ")
                    ));
                }
                i += 2;
            }
            "--figure-scenes" => {
                config.figure_scenes =
                    value()?.split(',').filter(|s| !s.is_empty()).map(|s| s.trim().to_string()).collect();
                i += 2;
            }
            "--scene-path" => {
                // KEY=PATH[+PATH...]. Repeats and multi-file scenes are both
                // allowed, so a scene can be swapped or composed without a
                // recompile.
                let raw = value()?;
                let (key, paths) = raw
                    .split_once('=')
                    .ok_or_else(|| "--scene-path expects KEY=PATH".to_string())?;
                let replacement: Vec<PathBuf> =
                    paths.split('+').filter(|s| !s.is_empty()).map(PathBuf::from).collect();
                if replacement.is_empty() {
                    return Err("--scene-path needs at least one path".to_string());
                }
                match config.scenes.iter_mut().find(|spec| spec.key == key) {
                    Some(spec) => spec.paths = replacement,
                    None => return Err(format!("--scene-path: no scene with key {key:?}")),
                }
                i += 2;
            }
            other => return Err(format!("unknown flag {other}\n\n{}", usage())),
        }
    }

    if config.build_runs == 0 || config.render_runs == 0 {
        return Err("--build-runs and --render-runs must be at least 1".to_string());
    }
    if config.leaf_cutoff == 0 {
        return Err("--leaf-cutoff must be at least 1".to_string());
    }

    Ok(config)
}

fn parse_num<T: std::str::FromStr>(flag: &str, raw: &str) -> Result<T, String> {
    raw.parse::<T>().map_err(|_| format!("{flag}: could not parse {raw:?}"))
}

pub fn usage() -> String {
    "usage: testyo study [options]\n\
     \n\
     runs the whole split-heuristic protocol and writes every table and figure\n\
     into one directory.\n\
     \n\
     options:\n\
     \x20 --out DIR              output directory (default results/study)\n\
     \x20 --width N --height N   render resolution\n\
     \x20 --samples N            samples per dispatch (default 150)\n\
     \x20 --bounces N            path depth limit (default 5)\n\
     \x20 --seed N               path-tracing RNG seed\n\
     \x20 --leaf-cutoff N        max primitives per leaf\n\
     \x20 --build-warmup N       builds discarded before timing (default 2)\n\
     \x20 --build-runs N         timed builds per variant (default 5)\n\
     \x20 --render-warmup N      dispatches discarded before timing (default 2)\n\
     \x20 --render-runs N        timed dispatches per variant (default 5)\n\
     \x20 --random-seeds N       seeds the random baseline averages over (default 5)\n\
     \x20 --only KEY,KEY         restrict to these scene keys\n\
     \x20 --figure-scenes KEYS   scenes that get heatmaps and structure diagrams\n\
     \x20 --structure-depth N    tree level drawn in the AABB figure (default 6)\n\
     \x20 --scene-path KEY=PATH  swap a scene's mesh; PATH+PATH loads several as one\n\
     \x20 --no-figures           tables only\n\
     \x20 --no-verify            skip the repeat counting pass"
        .to_string()
}
