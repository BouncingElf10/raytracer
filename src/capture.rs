//! Offscreen capture of the interactive view, overlay included.
//!
//! Accumulates a fixed number of samples and writes the composited framebuffer
//! to a PNG. This is how the BVH wireframe figures get out of the window and into
//! a document -- no screen grabbing, no window management, and the sample count
//! is explicit so two captures are directly comparable.
//!
//! `--depth-sweep` is the interesting mode: it writes one image per level of the
//! tree, which reads as a flip-book of how the heuristic carves up space.

use std::path::PathBuf;

use glam::Vec3;

use crate::bvh::{BuildParams, SplitHeuristic};
use crate::camera::Camera;
use crate::gpu_types::DisplayMode;
use crate::renderer::{BvhDebugView, LiveSettings, Renderer};
use crate::ray::Ray;
use crate::scene::{self, Scene};
use crate::shaders::ShaderVariant;
use crate::viz::Image;
use crate::window::Canvas;
use crate::{bvh, model};

pub struct CaptureConfig {
    pub width: u32,
    pub height: u32,
    /// Accumulated samples before the frame is written.
    pub frames: u32,
    pub output: PathBuf,
    pub show_bvh: bool,
    pub leaves_only: bool,
    /// Draw only this depth level.
    pub depth: Option<usize>,
    /// Write one image per depth level into a directory instead of a single file.
    pub sweep_dir: Option<PathBuf>,
    pub camera_origin: Vec3,
    pub camera_direction: Vec3,
    /// Which BVH the scene is built with before capturing.
    pub heuristic: Option<SplitHeuristic>,
    /// Live shader settings, including the cost view to render.
    pub live: LiveSettings,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            width: 900,
            height: 700,
            frames: 96,
            output: PathBuf::from("results/figures/interactive/bvh_overlay.png"),
            show_bvh: true,
            leaves_only: false,
            depth: None,
            sweep_dir: None,
            // Close enough that the Cornell box fills the frame rather than
            // floating in a black border.
            camera_origin: Vec3::new(0.0, 0.0, 3.6),
            camera_direction: Vec3::new(0.0, 0.0, -1.0),
            heuristic: None,
            live: LiveSettings::default(),
        }
    }
}

pub async fn run(config: &CaptureConfig) -> std::io::Result<()> {
    let mut canvas = Canvas::new(config.width, config.height, "capture").await;
    let camera = Camera::new(
        canvas.width(),
        canvas.height(),
        Ray::new(config.camera_origin, config.camera_direction),
    );
    let mut scene = scene::create_scene();
    let renderer = Renderer::new();

    if let Some(heuristic) = config.heuristic {
        if let Some(mesh) = scene.first_mesh_mut() {
            let params = BuildParams::new(heuristic);
            let tree = bvh::construct_bvh_from_tris(mesh.get_triangles(), params);
            mesh.add_bvh(tree);
            println!("rebuilt BVH with {}", heuristic.name());
        }
    }

    // A cost view needs the counters, so those captures use the instrumented
    // pipeline; a plain render stays on the clean one.
    let variant = if config.live.display_mode == DisplayMode::Beauty {
        ShaderVariant::Clean
    } else {
        ShaderVariant::Instrumented
    };

    let max_depth = max_bvh_depth(&scene);
    let mut view = BvhDebugView::new(max_depth);
    view.enabled = config.show_bvh;
    view.leaves_only = config.leaves_only;
    view.depth_focus = config.depth;

    println!("accumulating {} samples at {}x{}", config.frames, config.width, config.height);
    for _ in 0..config.frames {
        renderer.render_gpu_with(&camera, &scene, &mut canvas, &config.live, variant);
        // The window still has to be pumped or the OS considers it hung.
        canvas.update();
    }

    match &config.sweep_dir {
        Some(dir) => {
            // The path traced image is already converged; only the overlay
            // changes per level, so the expensive part is done once.
            for depth in 0..=max_depth {
                view.enabled = true;
                view.depth_focus = Some(depth);
                let path = dir.join(format!("bvh_depth_{depth:02}.png"));
                compose_and_save(&renderer, &camera, &scene, &mut canvas, &view, &path)?;
                println!("wrote {}", path.display());
            }

            view.depth_focus = None;
            let path = dir.join("bvh_all_depths.png");
            compose_and_save(&renderer, &camera, &scene, &mut canvas, &view, &path)?;
            println!("wrote {}", path.display());
        }
        None => {
            compose_and_save(&renderer, &camera, &scene, &mut canvas, &view, &config.output)?;
            println!("wrote {}", config.output.display());
        }
    }

    // GLFW tears the window down through a context that wgpu has already
    // released, and its error callback aborts the process on the way out. The
    // interactive loop sidesteps the same teardown by exiting outright; do the
    // same here so a successful capture reports success.
    std::process::exit(0);
}

/// Repaints the accumulated image, draws the overlay on top, and saves.
fn compose_and_save(
    renderer: &Renderer,
    camera: &Camera,
    scene: &Scene,
    canvas: &mut Canvas,
    view: &BvhDebugView,
    path: &std::path::Path,
) -> std::io::Result<()> {
    // Repaint from the accumulation buffer so the previous overlay does not stay
    // burned into the framebuffer.
    renderer.repaint_accumulated(camera, canvas);

    if view.enabled {
        renderer.render_bvh_overlay(camera, scene, canvas, view);
    }

    to_image(canvas).save(path)
}

fn to_image(canvas: &Canvas) -> Image {
    let mut image = Image::new(canvas.width(), canvas.height(), [0, 0, 0]);
    let pixels = canvas.pixel_buffer();

    for y in 0..canvas.height() {
        for x in 0..canvas.width() {
            let packed = pixels[(y * canvas.width() + x) as usize];
            image.set(
                x,
                y,
                [
                    ((packed >> 16) & 0xFF) as u8,
                    ((packed >> 8) & 0xFF) as u8,
                    (packed & 0xFF) as u8,
                ],
            );
        }
    }

    image
}

fn max_bvh_depth(scene: &Scene) -> usize {
    scene
        .get_objects()
        .iter()
        .filter_map(|object| object.as_any().downcast_ref::<model::Mesh>())
        .filter_map(|mesh| mesh.bvh.as_ref())
        .map(bvh::tree_max_depth)
        .max()
        .unwrap_or(0)
}

pub fn parse_args(args: &[String]) -> Result<CaptureConfig, String> {
    let mut config = CaptureConfig::default();
    let mut i = 0;

    while i < args.len() {
        let flag = args[i].as_str();
        let value = || -> Result<String, String> {
            args.get(i + 1).cloned().ok_or_else(|| format!("{flag} requires a value"))
        };
        let number = |raw: &str| -> Result<u32, String> {
            raw.parse().map_err(|_| format!("{flag}: could not parse {raw:?}"))
        };

        match flag {
            "--width" => { config.width = number(&value()?)?; i += 2; }
            "--height" => { config.height = number(&value()?)?; i += 2; }
            "--frames" => { config.frames = number(&value()?)?.max(1); i += 2; }
            "--out" => { config.output = PathBuf::from(value()?); i += 2; }
            "--depth" => { config.depth = Some(number(&value()?)? as usize); i += 2; }
            "--leaves" => { config.leaves_only = true; i += 1; }
            "--no-bvh" => { config.show_bvh = false; i += 1; }
            "--mode" => {
                config.live.display_mode = parse_mode(&value()?)?;
                i += 2;
            }
            "--palette" => {
                config.live.palette = parse_palette(&value()?)?;
                i += 2;
            }
            "--heat-scale" => {
                config.live.heat_scale = value()?
                    .parse()
                    .map_err(|_| "--heat-scale expects a number".to_string())?;
                i += 2;
            }
            "--heat-mix" => {
                config.live.heat_mix = value()?
                    .parse::<f32>()
                    .map_err(|_| "--heat-mix expects a number".to_string())?
                    .clamp(0.0, 1.0);
                i += 2;
            }
            "--heuristic" => {
                config.heuristic = Some(parse_heuristic(&value()?)?);
                i += 2;
            }
            "--samples" => {
                config.live.samples = number(&value()?)?.max(1);
                i += 2;
            }
            "--bounces" => {
                config.live.max_bounces = number(&value()?)?.max(1);
                i += 2;
            }
            "--depth-sweep" => {
                let explicit = args.get(i + 1).filter(|next| !next.starts_with("--"));
                match explicit {
                    Some(dir) => { config.sweep_dir = Some(PathBuf::from(dir)); i += 2; }
                    None => {
                        config.sweep_dir = Some(PathBuf::from("results/figures/interactive"));
                        i += 1;
                    }
                }
            }
            other => return Err(format!("unknown flag {other}\n\n{}", usage())),
        }
    }

    if config.depth.is_some() && !config.show_bvh {
        return Err("--depth has no effect with --no-bvh".to_string());
    }

    Ok(config)
}

fn parse_mode(raw: &str) -> Result<DisplayMode, String> {
    Ok(match raw.to_ascii_lowercase().replace('-', "_").as_str() {
        "beauty" | "render" => DisplayMode::Beauty,
        "node_visits" | "nodes" => DisplayMode::NodeVisits,
        "prim_tests" | "prims" | "triangles" => DisplayMode::PrimTests,
        "traversal_depth" | "depth" | "stack" => DisplayMode::TraversalDepth,
        "leaf_visits" | "leaves" => DisplayMode::LeafVisits,
        "interior_visits" | "interior" => DisplayMode::InteriorVisits,
        other => return Err(format!("unknown --mode {other:?}")),
    })
}

fn parse_palette(raw: &str) -> Result<u32, String> {
    Ok(match raw.to_ascii_lowercase().as_str() {
        "inferno" => 0,
        "viridis" => 1,
        "turbo" => 2,
        "gray" | "grey" | "grayscale" => 3,
        other => return Err(format!("unknown --palette {other:?}")),
    })
}

fn parse_heuristic(raw: &str) -> Result<SplitHeuristic, String> {
    let normalised = raw.to_ascii_lowercase().replace('-', "");
    SplitHeuristic::ALL
        .into_iter()
        .find(|h| h.name().to_ascii_lowercase() == normalised)
        .ok_or_else(|| {
            format!(
                "unknown --heuristic {raw:?}; expected one of {}",
                SplitHeuristic::ALL.map(|h| h.name()).join(", ")
            )
        })
}

pub fn usage() -> String {
    "usage: testyo shot [options]\n\
     \n\
     renders the interactive scene offscreen and saves it, with the BVH overlay\n\
     or as a traversal cost view.\n\
     \n\
     options:\n\
     \x20 --width N --height N    capture resolution\n\
     \x20 --frames N              frames to accumulate before capture\n\
     \x20 --samples N             samples per frame\n\
     \x20 --bounces N             path depth limit\n\
     \x20 --out PATH              output PNG\n\
     \x20 --heuristic NAME        rebuild the BVH first (LongestAxisCentroid, Median, Sah, Random)\n\
     \x20 --mode NAME             beauty | nodes | prims | depth | leaves | interior\n\
     \x20 --palette NAME          inferno | viridis | turbo | grayscale\n\
     \x20 --heat-scale N          value mapped to the top of the ramp\n\
     \x20 --heat-mix N            0 = pure cost view, 1 = pure render\n\
     \x20 --depth N               draw only tree level N\n\
     \x20 --leaves                draw only leaf boxes\n\
     \x20 --no-bvh                no wireframe overlay\n\
     \x20 --depth-sweep [DIR]     one image per tree level (default results/figures/interactive)"
        .to_string()
}
