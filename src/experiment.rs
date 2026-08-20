//! BVH data-collection harness.
//!
//! Produces one record per (scene, heuristic) combination and writes them all to
//! a single CSV. Everything except the scene file and the split heuristic is held
//! fixed by `ExperimentConfig` (§8), so any difference between rows is
//! attributable to the independent variables.
//!
//! Per record the driver (§9):
//!   1. loads the scene geometry once and normalises it into a fixed room,
//!   2. builds the BVH under a monotonic timer (construction only),
//!   3. walks the finished tree for its structural statistics,
//!   4. flattens and uploads it,
//!   5. runs Pass A for counts and Pass B for wall-clock, never together.

use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

use glam::Vec3;

use crate::bvh::{
    self, BuildParams, BuildStats, SplitHeuristic, DEFAULT_BUILD_SEED, DEFAULT_LEAF_PRIM_CUTOFF,
};
use crate::camera::Camera;
use crate::color::Color;
use crate::figures;
use crate::gpu_harness::{CounterSummary, GpuHarness, TimingSummary};
use crate::gpu_types::{GpuPlane, GpuRay, GpuTriangle};
use crate::importer::import_obj;
use crate::material::Material;
use crate::model::Mesh;
use crate::objects::{Hittable, Triangle};
use crate::ray::{self, Ray};
use crate::scene::triangle_to_gpu_triangle;

/// Every controlled variable in one place. Identical for all scenes and all
/// heuristics within a run.
#[derive(Debug, Clone)]
pub struct ExperimentConfig {
    pub camera_origin: Vec3,
    pub camera_direction: Vec3,
    pub width: u32,
    pub height: u32,
    /// Samples accumulated inside a single dispatch.
    pub samples: u32,
    /// Seed for the GPU path-tracing RNG.
    pub rng_seed: u32,
    /// Seed for the `Random` split heuristic.
    pub bvh_seed: u64,
    pub leaf_cutoff: usize,
    /// Dispatches discarded before timing begins (shader compile, pipeline warm-up).
    pub warmup_runs: u32,
    /// Timed dispatches averaged into `render_time_ms`.
    pub timed_runs: u32,
    /// Rescale each mesh to a fixed size and centre it, so the camera framing is
    /// the same for every scene.
    pub normalize_scenes: bool,
    pub scenes: Vec<PathBuf>,
    pub output_path: PathBuf,
    /// Run Pass A twice per record and check the counts match exactly.
    pub verify_determinism: bool,
    /// When set, write heatmaps, comparison sheets and structure diagrams here.
    pub figures_dir: Option<PathBuf>,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            // Inside the room, close enough that the mesh fills most of the
            // frame. Framing matters for the figures: rays that miss the model
            // entirely contribute nothing but background to a traversal heatmap.
            camera_origin: Vec3::new(0.0, 0.0, 2.3),
            camera_direction: Vec3::new(0.0, 0.0, -1.0),
            width: 640,
            height: 480,
            samples: 4,
            rng_seed: 0x1234_5678,
            bvh_seed: DEFAULT_BUILD_SEED,
            leaf_cutoff: DEFAULT_LEAF_PRIM_CUTOFF,
            warmup_runs: 2,
            timed_runs: 5,
            normalize_scenes: true,
            // tank.obj is left out by default: an exact full-sweep SAH build over
            // ~500k primitives takes far longer than the rest of the suite
            // combined. Add it explicitly with --scenes when you want it.
            scenes: vec![
                PathBuf::from("src/models/teapot.obj"),
                PathBuf::from("src/models/standford_dragon.obj"),
                PathBuf::from("src/models/train.obj"),
            ],
            output_path: PathBuf::from("results/bvh_metrics.csv"),
            verify_determinism: false,
            figures_dir: None,
        }
    }
}

/// One row of the output CSV.
pub struct MetricRecord {
    pub scene: String,
    pub heuristic: &'static str,
    pub counters: CounterSummary,
    pub timing: TimingSummary,
    pub stats: BuildStats,
    /// Leaf primitives that could not be matched to a GPU triangle index. Must be
    /// zero for the record to be trustworthy.
    pub unmatched_prims: usize,
    pub flattened_index_count: usize,
}

pub fn run(config: &ExperimentConfig) -> io::Result<()> {
    let harness = GpuHarness::new(config.width, config.height)
        .map_err(|e| io::Error::other(e))?;

    println!("adapter: {} ({})", harness.adapter_name(), harness.backend());
    println!(
        "config: {}x{}  samples={}  rng_seed={}  bvh_seed={}  leaf_cutoff={}  warmup={}  runs={}",
        config.width,
        config.height,
        config.samples,
        config.rng_seed,
        config.bvh_seed,
        config.leaf_cutoff,
        config.warmup_runs,
        config.timed_runs,
    );
    if !harness.timestamps_supported() {
        println!("note: TIMESTAMP_QUERY unavailable, render_time columns will be empty");
    }

    // Identical rays for every scene, heuristic and pass. Generated once so there
    // is no chance of the two passes diverging.
    let camera = Camera::new(
        config.width,
        config.height,
        Ray::new(config.camera_origin, config.camera_direction),
    );
    let rays = generate_rays(&camera);
    harness.upload_rays(&rays);

    let planes = room_planes();
    let mut records = Vec::new();

    for scene_path in &config.scenes {
        if !scene_path.exists() {
            eprintln!("skipping missing scene: {}", scene_path.display());
            continue;
        }

        let scene_name = scene_label(scene_path);
        println!("\n=== scene: {scene_name} ===");

        // Loaded once per scene and shared by all four heuristics (§9): loading
        // cost must not land in any heuristic's build time.
        let mesh = load_scene_mesh(scene_path, config.normalize_scenes);
        let triangles = mesh.get_triangles();
        let gpu_triangles: Vec<GpuTriangle> =
            triangles.iter().map(triangle_to_gpu_triangle).collect();
        println!("primitives: {}", triangles.len());

        if triangles.is_empty() {
            eprintln!("skipping {scene_name}: no triangles");
            continue;
        }

        let mut figure_inputs = Vec::new();

        for heuristic in SplitHeuristic::ALL {
            let (record, figure_input) = collect_record(
                config,
                &harness,
                &scene_name,
                heuristic,
                &triangles,
                &gpu_triangles,
                &planes,
            );
            print_record(&record);
            records.push(record);
            if let Some(input) = figure_input {
                figure_inputs.push(input);
            }
        }

        // Figures are generated per scene rather than per record so every panel
        // of a comparison shares one colour scale.
        if let Some(dir) = &config.figures_dir {
            match figures::render_scene_figures(
                dir,
                &scene_name,
                config.width,
                config.height,
                &figure_inputs,
            ) {
                Ok(paths) => println!("figures: {} file(s) in {}", paths.len(), dir.join(&scene_name).display()),
                Err(error) => eprintln!("failed to write figures for {scene_name}: {error}"),
            }
        }
    }

    write_csv(config, &harness, &records)?;
    println!(
        "\nwrote {} record(s) to {}",
        records.len(),
        config.output_path.display()
    );

    Ok(())
}

fn collect_record(
    config: &ExperimentConfig,
    harness: &GpuHarness,
    scene_name: &str,
    heuristic: SplitHeuristic,
    triangles: &[Triangle],
    gpu_triangles: &[GpuTriangle],
    planes: &[GpuPlane],
) -> (MetricRecord, Option<figures::FigureInput>) {
    let params = BuildParams {
        heuristic,
        leaf_cutoff: config.leaf_cutoff,
        seed: config.bvh_seed,
    };

    // The clone is deliberately outside the timer: `construct_bvh_from_tris`
    // consumes its input, and copying the list is not construction work.
    let input = triangles.to_vec();
    let started = Instant::now();
    let tree = bvh::construct_bvh_from_tris(input, params);
    let build_time_ms = started.elapsed().as_secs_f64() * 1000.0;

    let mut stats = bvh::compute_build_stats(&tree);
    stats.build_time_ms = build_time_ms;

    let flattened = bvh::flatten_bvh_for_gpu_checked(&tree, triangles);
    if flattened.unmatched_prims > 0 {
        eprintln!(
            "warning: {} leaf primitive(s) had no matching GPU triangle in {scene_name}/{}",
            flattened.unmatched_prims,
            heuristic.name(),
        );
    }
    // The GPU stack is fixed size; a deeper tree gets truncated traversals, which
    // the `incomplete` counter will also report per ray.
    if stats.max_depth >= 64 {
        eprintln!(
            "warning: {}/{} has max_depth {} which meets the GPU traversal stack limit (64); \
             counts for this row may be truncated",
            scene_name,
            heuristic.name(),
            stats.max_depth,
        );
    }

    let upload = harness.upload_scene(
        gpu_triangles,
        planes,
        &flattened.nodes,
        &flattened.indices,
        config.samples,
        config.rng_seed,
        // The path depth this harness has always used. The study driver makes it
        // a declared control instead; this one keeps its historical value so old
        // CSVs stay comparable.
        10,
    );

    // Pass A: counts only.
    let want_figures = config.figures_dir.is_some();
    let (counters, records) = harness.collect_counters(upload.bvh(), want_figures);

    if config.verify_determinism {
        let (repeat, _) = harness.collect_counters(upload.bvh(), false);
        let identical = repeat.total_node_visits == counters.total_node_visits
            && repeat.total_prim_tests == counters.total_prim_tests
            && repeat.total_rays == counters.total_rays;
        println!(
            "  determinism check: {}",
            if identical { "counts identical" } else { "COUNTS DIFFERED" }
        );
    }

    // Pass B: wall-clock only, same camera/resolution/samples/seed.
    let timing = harness.measure_render_time(upload.bvh(), config.warmup_runs, config.timed_runs);

    // The colour capture is deliberately last: it is an extra dispatch that must
    // not sit between the timed runs.
    let figure_input = records.map(|records| figures::FigureInput {
        heuristic: heuristic.name().to_string(),
        records,
        colors: harness.capture_color(upload.bvh()),
        tree,
        node_visits_per_ray: counters.node_visits_per_ray,
        prim_tests_per_ray: counters.prim_tests_per_ray,
        render_time_ms: timing.valid.then_some(timing.mean_ms),
        max_depth: stats.max_depth,
    });

    let record = MetricRecord {
        scene: scene_name.to_string(),
        heuristic: heuristic.name(),
        counters,
        timing,
        stats,
        unmatched_prims: flattened.unmatched_prims,
        flattened_index_count: flattened.indices.len(),
    };

    (record, figure_input)
}

fn print_record(record: &MetricRecord) {
    let render = if record.timing.valid {
        format!("{:.3} ms", record.timing.mean_ms)
    } else {
        "n/a".to_string()
    };
    println!(
        "{:<20} nodes/ray {:>8.2}  prims/ray {:>8.2}  build {:>9.1} ms  render {:>12}  \
         nodes {:>7}  depth {:>3}  sah {:>8.2}",
        record.heuristic,
        record.counters.node_visits_per_ray,
        record.counters.prim_tests_per_ray,
        record.stats.build_time_ms,
        render,
        record.stats.node_count,
        record.stats.max_depth,
        record.stats.total_sah_cost,
    );
    if record.counters.incomplete_traversals > 0 {
        println!(
            "  warning: {} traversal(s) hit the stack/step guard; counts are a lower bound",
            record.counters.incomplete_traversals
        );
    }
}

// ---- Scene construction ----------------------------------------------------

fn load_scene_mesh(path: &Path, normalize: bool) -> Mesh {
    let mut mesh = import_obj(&path.to_string_lossy());
    // Diffuse and non-emissive: rays scatter off the mesh back into the room and
    // return, so secondary rays keep exercising the BVH instead of escaping.
    mesh.set_material(Material::new(Color::new(0.75, 0.75, 0.75), 1.0, 0.0, 0.0));

    if normalize {
        fit_to_room(&mut mesh);
    }

    mesh
}

/// Centres the mesh at the origin and scales it so its largest extent is
/// `TARGET_EXTENT`. Without this, camera framing would vary per scene and the
/// "identical camera" control would be meaningless.
fn fit_to_room(mesh: &mut Mesh) {
    const TARGET_EXTENT: f32 = 3.0;

    mesh.position = Vec3::ZERO;
    mesh.rotation = Vec3::ZERO;
    mesh.scale = 1.0;

    let tris = mesh.get_triangles();
    if tris.is_empty() {
        return;
    }

    let (min, max) = tris.iter().flat_map(|tri| tri.get_vertices()).fold(
        (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
        |(min, max), v| (min.min(v), max.max(v)),
    );

    let extent = (max - min).max_element();
    let scale = if extent > 0.0 { TARGET_EXTENT / extent } else { 1.0 };
    let center = (min + max) * 0.5;

    mesh.scale = scale;
    mesh.position = -center * scale;
}

/// A fixed Cornell-style room, identical for every scene. The planes are tested
/// outside the BVH so they never contribute to the traversal counts, but they
/// keep secondary rays alive instead of letting them escape to a black void.
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

fn scene_label(path: &Path) -> String {
    path.file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string_lossy().into_owned())
}

// ---- Output ----------------------------------------------------------------

/// Column order of the emitted CSV. The first eleven are the schema from the
/// spec; the rest are diagnostics and the controlled variables, so every row is
/// self-describing when results from several machines are pooled.
const COLUMNS: &[&str] = &[
    "scene",
    "heuristic",
    "node_visits_per_ray",
    "prim_tests_per_ray",
    "build_time_ms",
    "render_time_ms",
    "node_count",
    "max_depth",
    "avg_depth",
    "avg_leaf_prims",
    "total_sah_cost",
    // --- extras ---
    "interior_visits_per_ray",
    "node_visits_max",
    "node_visits_p95",
    "prim_tests_max",
    "prim_tests_p95",
    "leaf_sah_cost",
    "leaf_count",
    "prim_count",
    "total_rays",
    "total_node_visits",
    "total_prim_tests",
    "incomplete_traversals",
    "unmatched_prims",
    "flattened_index_count",
    "render_time_min_ms",
    "render_time_stddev_ms",
    "width",
    "height",
    "samples",
    "rng_seed",
    "bvh_seed",
    "leaf_cutoff",
    "warmup_runs",
    "timed_runs",
    "adapter",
    "backend",
];

fn write_csv(
    config: &ExperimentConfig,
    harness: &GpuHarness,
    records: &[MetricRecord],
) -> io::Result<()> {
    if let Some(parent) = config.output_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let mut out = String::new();
    out.push_str(&COLUMNS.join(","));
    out.push('\n');

    for record in records {
        let mut row: Vec<String> = Vec::with_capacity(COLUMNS.len());

        row.push(csv_field(&record.scene));
        row.push(csv_field(record.heuristic));
        row.push(format!("{:.6}", record.counters.node_visits_per_ray));
        row.push(format!("{:.6}", record.counters.prim_tests_per_ray));
        row.push(format!("{:.4}", record.stats.build_time_ms));
        row.push(optional_ms(&record.timing, record.timing.mean_ms));
        row.push(record.stats.node_count.to_string());
        row.push(record.stats.max_depth.to_string());
        row.push(format!("{:.4}", record.stats.avg_depth));
        row.push(format!("{:.4}", record.stats.avg_leaf_prims));
        row.push(format!("{:.6}", record.stats.total_sah_cost));

        row.push(format!("{:.6}", record.counters.interior_visits_per_ray));
        row.push(format!("{:.6}", record.counters.node_visits_max));
        row.push(format!("{:.6}", record.counters.node_visits_p95));
        row.push(format!("{:.6}", record.counters.prim_tests_max));
        row.push(format!("{:.6}", record.counters.prim_tests_p95));
        row.push(format!("{:.6}", record.stats.leaf_sah_cost));
        row.push(record.stats.leaf_count.to_string());
        row.push(record.stats.prim_count.to_string());
        row.push(record.counters.total_rays.to_string());
        row.push(record.counters.total_node_visits.to_string());
        row.push(record.counters.total_prim_tests.to_string());
        row.push(record.counters.incomplete_traversals.to_string());
        row.push(record.unmatched_prims.to_string());
        // Equals prim_count when every leaf primitive resolved to a GPU index.
        row.push(record.flattened_index_count.to_string());
        row.push(optional_ms(&record.timing, record.timing.min_ms));
        row.push(optional_ms(&record.timing, record.timing.stddev_ms));

        row.push(config.width.to_string());
        row.push(config.height.to_string());
        row.push(config.samples.to_string());
        row.push(config.rng_seed.to_string());
        row.push(config.bvh_seed.to_string());
        row.push(config.leaf_cutoff.to_string());
        row.push(config.warmup_runs.to_string());
        row.push(config.timed_runs.to_string());
        row.push(csv_field(harness.adapter_name()));
        row.push(csv_field(harness.backend()));

        debug_assert_eq!(row.len(), COLUMNS.len(), "row width must match the header");

        let _ = writeln!(out, "{}", row.join(","));
    }

    fs::write(&config.output_path, out)
}

/// Timing cells are left empty rather than filled with a placeholder number when
/// the adapter cannot measure them -- a blank is unambiguous to any analysis tool.
fn optional_ms(timing: &TimingSummary, value: f64) -> String {
    if timing.valid {
        format!("{value:.4}")
    } else {
        String::new()
    }
}

fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

// ---- Command line ----------------------------------------------------------

/// Parses harness flags. Returns `Err` with a usage message on a malformed flag.
pub fn parse_args(args: &[String]) -> Result<ExperimentConfig, String> {
    let mut config = ExperimentConfig::default();
    let mut i = 0;

    while i < args.len() {
        let flag = args[i].as_str();
        let value = || -> Result<String, String> {
            args.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{flag} requires a value"))
        };

        match flag {
            "--scenes" => {
                config.scenes = value()?
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(PathBuf::from)
                    .collect();
                i += 2;
            }
            "--out" => {
                config.output_path = PathBuf::from(value()?);
                i += 2;
            }
            "--width" => {
                config.width = parse_num(flag, &value()?)?;
                i += 2;
            }
            "--height" => {
                config.height = parse_num(flag, &value()?)?;
                i += 2;
            }
            "--samples" => {
                config.samples = parse_num(flag, &value()?)?;
                i += 2;
            }
            "--seed" => {
                config.rng_seed = parse_num(flag, &value()?)?;
                i += 2;
            }
            "--bvh-seed" => {
                config.bvh_seed = parse_num(flag, &value()?)?;
                i += 2;
            }
            "--leaf-cutoff" => {
                config.leaf_cutoff = parse_num(flag, &value()?)?;
                i += 2;
            }
            "--warmup" => {
                config.warmup_runs = parse_num(flag, &value()?)?;
                i += 2;
            }
            "--runs" => {
                config.timed_runs = parse_num(flag, &value()?)?;
                i += 2;
            }
            "--no-normalize" => {
                config.normalize_scenes = false;
                i += 1;
            }
            "--verify" => {
                config.verify_determinism = true;
                i += 1;
            }
            "--figures" => {
                // Bare --figures uses a default directory; a following value that
                // is not itself a flag overrides it.
                let explicit = args.get(i + 1).filter(|next| !next.starts_with("--"));
                match explicit {
                    Some(dir) => {
                        config.figures_dir = Some(PathBuf::from(dir));
                        i += 2;
                    }
                    None => {
                        config.figures_dir = Some(PathBuf::from("results/figures"));
                        i += 1;
                    }
                }
            }
            other => return Err(format!("unknown flag {other}\n\n{}", usage())),
        }
    }

    if config.leaf_cutoff == 0 {
        return Err("--leaf-cutoff must be at least 1".to_string());
    }
    if config.timed_runs == 0 {
        return Err("--runs must be at least 1".to_string());
    }

    Ok(config)
}

fn parse_num<T: std::str::FromStr>(flag: &str, raw: &str) -> Result<T, String> {
    raw.parse::<T>()
        .map_err(|_| format!("{flag}: could not parse {raw:?}"))
}

pub fn usage() -> String {
    "usage: testyo harness [options]\n\
     \n\
     options:\n\
     \x20 --scenes a.obj,b.obj    scene files to measure\n\
     \x20 --out PATH              output CSV (default results/bvh_metrics.csv)\n\
     \x20 --width N --height N    render resolution\n\
     \x20 --samples N             samples per dispatch\n\
     \x20 --seed N                path-tracing RNG seed\n\
     \x20 --bvh-seed N            seed for the Random split heuristic\n\
     \x20 --leaf-cutoff N         max primitives per leaf\n\
     \x20 --warmup N              dispatches discarded before timing\n\
     \x20 --runs N                timed dispatches to average\n\
     \x20 --no-normalize          keep each mesh at its authored scale\n\
     \x20 --verify                run Pass A twice and check counts match\n\
     \x20 --figures [DIR]         write heatmaps and diagrams (default results/figures)"
        .to_string()
}
