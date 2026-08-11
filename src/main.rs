use crate::camera::Camera;
use crate::ray::Ray;
use crate::window::Canvas;
use glam::Vec3;

mod camera;
mod window;
mod color;
mod objects;
mod scene;
mod ray;
mod movement;
mod renderer;
mod material;
mod model;
mod importer;
mod gpu_types;
mod compute;
mod profiler;
mod bvh;
mod shaders;
mod gpu_harness;
mod experiment;
mod viz;
mod diagrams;
mod figures;
mod capture;
mod ui;

const DEBUG_MODE: bool = true;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("harness") => {
            run_harness(&args[1..]);
            return;
        }
        Some("shot") => {
            run_capture(&args[1..]).await;
            return;
        }
        Some("studio") => {
            run_studio(&args[1..]).await;
            return;
        }
        _ => {}
    }

    profiler::profiler_start("init");
    profiler::profiler_start("window");
    let mut canvas = Canvas::new(80 * 10, 60 * 10, "WINDOW").await;
    profiler::profiler_stop("window");
    let mut camera = Camera::new(canvas.width(), canvas.height(), Ray::new(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0)));
    let mut movement_state = movement::MovementState::new();
    let scene = scene::create_scene();
    let renderer = renderer::Renderer::new();
    let mut delta_time = 0.0;

    let mut bvh_view = renderer::BvhDebugView::new(scene_max_depth(&scene));
    bvh_view.enabled = DEBUG_MODE;
    let mut keys = DebugKeys::default();

    profiler::profiler_stop("init");

    loop {
        profiler::profiler_start("main");

        profiler::profiler_start("render");
        profiler::profiler_start("gpu");

        renderer.render_gpu(&camera, &scene, &mut canvas);

        profiler::profiler_stop("gpu");
        profiler::profiler_start("debug");

        if bvh_view.enabled {
            renderer.render_bvh_overlay(&camera, &scene, &mut canvas, &bvh_view);
        }
        profiler::profiler_stop("debug");
        profiler::profiler_stop("render");

        profiler::profiler_start("text and movement");

        keys.apply(&canvas, &mut bvh_view);

        canvas.set_window_title(&format!("frame in: {:.0}ms   fps: {:.2}   samples: {}   {}   [B]toggle [L]leaves [{{/}}]depth [\\]all",
                                         profiler::get_delta_time() * 1000.0,
                                         1.0 /  profiler::get_delta_time(),
                                         canvas.sample_count,
                                         bvh_view.status()));
        canvas.present().unwrap();
        canvas.update();
        camera.resize(canvas.width(), canvas.height());
        if movement::apply_movements(&mut camera, &canvas, delta_time, &mut movement_state) {
            canvas.reset_accumulation();
        }

        profiler::profiler_stop("text and movement");

        delta_time = profiler::get_delta_time() as f32;

        if !canvas.is_open() {
            profiler::print_profile();
            std::process::exit(0);
        }

        profiler::profiler_stop("main");
        profiler::profiler_reset()
    }
}

/// Deepest level across every mesh BVH in the scene, so the overlay's colour
/// ramp and depth stepping cover the whole tree.
fn scene_max_depth(scene: &scene::Scene) -> usize {
    scene
        .get_objects()
        .iter()
        .filter_map(|object| object.as_any().downcast_ref::<model::Mesh>())
        .filter_map(|mesh| mesh.bvh.as_ref())
        .map(bvh::tree_max_depth)
        .max()
        .unwrap_or(0)
}

/// Edge-triggered debug keys.
///
/// GLFW only reports whether a key is currently down, so a toggle read directly
/// from `is_key_down` would fire on every frame the key is held. Remembering last
/// frame's state turns each press into exactly one action.
#[derive(Default)]
struct DebugKeys {
    toggle: bool,
    leaves: bool,
    previous: bool,
    next: bool,
    reset: bool,
}

impl DebugKeys {
    fn apply(&mut self, canvas: &window::Canvas, view: &mut renderer::BvhDebugView) {
        use glfw::Key;

        fn pressed(down: bool, was_down: &mut bool) -> bool {
            let fired = down && !*was_down;
            *was_down = down;
            fired
        }

        if pressed(canvas.is_key_down(Key::B), &mut self.toggle) {
            view.enabled = !view.enabled;
        }
        if pressed(canvas.is_key_down(Key::L), &mut self.leaves) {
            view.leaves_only = !view.leaves_only;
        }
        if pressed(canvas.is_key_down(Key::LeftBracket), &mut self.previous) {
            view.step_depth(-1);
        }
        if pressed(canvas.is_key_down(Key::RightBracket), &mut self.next) {
            view.step_depth(1);
        }
        if pressed(canvas.is_key_down(Key::Backslash), &mut self.reset) {
            view.show_all_depths();
        }
    }
}

/// Runs the BVH data-collection harness instead of opening the interactive
/// window. Exits non-zero on a bad flag or an I/O failure so it can be driven
/// from a script.
fn run_harness(args: &[String]) {
    let config = match experiment::parse_args(args) {
        Ok(config) => config,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };

    if let Err(error) = experiment::run(&config) {
        eprintln!("harness failed: {error}");
        std::process::exit(1);
    }
}

/// Opens the BVH studio. `--screenshot PATH` renders a fixed number of frames,
/// writes the whole window including the panels, and exits.
async fn run_studio(args: &[String]) {
    let mut width = 1440;
    let mut height = 900;
    let mut screenshot = None;
    let mut frames = 40;

    let mut i = 0;
    while i < args.len() {
        let value = args.get(i + 1);
        match args[i].as_str() {
            "--width" => { width = value.and_then(|v| v.parse().ok()).unwrap_or(width); i += 2; }
            "--height" => { height = value.and_then(|v| v.parse().ok()).unwrap_or(height); i += 2; }
            "--frames" => { frames = value.and_then(|v| v.parse().ok()).unwrap_or(frames); i += 2; }
            "--screenshot" => {
                let Some(path) = value else {
                    eprintln!("--screenshot requires a path");
                    std::process::exit(2);
                };
                screenshot = Some(std::path::PathBuf::from(path));
                i += 2;
            }
            other => {
                eprintln!("unknown flag {other}\n\nusage: testyo studio [--width N] [--height N] [--screenshot PATH [--frames N]]");
                std::process::exit(2);
            }
        }
    }

    match screenshot {
        Some(path) => ui::run_screenshot(width, height, frames, &path).await,
        None => ui::run(width, height).await,
    }
}

/// Renders the interactive scene offscreen and writes it out with the BVH
/// overlay, for figures that need the wireframe rather than a heatmap.
async fn run_capture(args: &[String]) {
    let config = match capture::parse_args(args) {
        Ok(config) => config,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };

    if let Err(error) = capture::run(&config).await {
        eprintln!("capture failed: {error}");
        std::process::exit(1);
    }
}