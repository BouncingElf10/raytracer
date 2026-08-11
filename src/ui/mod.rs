//! The BVH studio: an ImGui front end over everything the renderer and the
//! data-collection harness can do.
//!
//! Three things run at once here. The path tracer accumulates into the canvas,
//! the wireframe overlay draws on top of it, and the panels in `panels` mutate the
//! state both read from. Anything that cannot be applied mid-frame -- rebuilding a
//! tree, exporting a figure set -- is queued as an `Action` and applied after the
//! UI has finished with its borrows.

mod panels;
mod platform;

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};
use std::time::Instant;

use glam::Vec3;
use glfw::{Key, MouseButton, WindowEvent};

use crate::bvh::{self, BuildParams, BuildStats, SplitHeuristic, DEFAULT_LEAF_PRIM_CUTOFF};
use crate::camera::Camera;
use crate::diagrams;
use crate::experiment::ExperimentConfig;
use crate::gpu_types::DisplayMode;
use crate::ray::Ray;
use crate::renderer::{BvhDebugView, LiveSettings, Renderer};
use crate::scene::{self, Scene};
use crate::shaders::ShaderVariant;
use crate::viz::Image;
use crate::window::Canvas;

pub use panels::PaletteChoice;

/// Deferred work requested by a panel.
pub enum Action {
    RebuildBvh,
    ResetAccumulation,
    SaveViewport,
    /// Run the harness over the current scene and write the full figure set.
    ExportFigureSet,
    ExportStructureDiagrams,
    ExportDepthSweep,
    FrameModel,
}

/// Progress reported by the background harness thread.
pub enum HarnessMessage {
    Finished(Result<String, String>),
}

pub struct Studio {
    pub scene: Scene,
    pub camera: Camera,
    pub renderer: Renderer,

    // --- camera rig ---
    pub yaw: f32,
    pub pitch: f32,
    pub move_speed: f32,
    pub look_sensitivity: f32,
    looking: bool,
    last_cursor: (f64, f64),

    // --- shader settings ---
    pub live: LiveSettings,
    pub palette_choice: PaletteChoice,
    pub auto_scale: bool,

    // --- bvh ---
    pub build: BuildParams,
    pub heuristic_index: usize,
    pub stats: BuildStats,
    pub depth_profile: Vec<f32>,
    pub prim_count: usize,

    // --- overlay ---
    pub view: BvhDebugView,

    // --- export ---
    pub output_dir: String,
    pub export_width: u32,
    pub export_height: u32,
    pub export_samples: u32,
    pub export_runs: u32,
    pub export_warmup: u32,
    pub status: String,

    // --- background harness ---
    harness_rx: Option<Receiver<HarnessMessage>>,
    pub harness_running: bool,

    // --- readouts ---
    pub frame_ms: f32,
    pub actions: Vec<Action>,
    pub show_help: bool,
}

impl Studio {
    fn new(scene: Scene, camera: Camera, prim_count: usize) -> Self {
        let direction = camera.ray().direction().normalize();

        Self {
            scene,
            camera,
            renderer: Renderer::new(),
            yaw: direction.z.atan2(direction.x).to_degrees(),
            pitch: direction.y.asin().to_degrees(),
            move_speed: 3.0,
            look_sensitivity: 0.12,
            looking: false,
            last_cursor: (0.0, 0.0),
            live: LiveSettings::default(),
            palette_choice: PaletteChoice::Inferno,
            auto_scale: false,
            build: BuildParams {
                heuristic: SplitHeuristic::SurfaceAreaHeuristic,
                leaf_cutoff: DEFAULT_LEAF_PRIM_CUTOFF,
                // A small seed so the spin box shows a readable number; the
                // harness keeps its own DEFAULT_BUILD_SEED for reproducibility.
                seed: 1,
            },
            heuristic_index: 2,
            stats: empty_stats(),
            depth_profile: Vec::new(),
            prim_count,
            view: BvhDebugView::new(0),
            output_dir: "results/studio".to_string(),
            export_width: 640,
            export_height: 480,
            export_samples: 32,
            export_runs: 5,
            export_warmup: 2,
            status: "ready".to_string(),
            harness_rx: None,
            harness_running: false,
            frame_ms: 0.0,
            actions: Vec::new(),
            show_help: false,
        }
    }

    pub fn palette_index(&self) -> u32 {
        self.palette_choice as u32
    }

    /// Rebuilds the mesh BVH with the current parameters and forces the compute
    /// pipeline to re-upload it on the next frame.
    fn rebuild_bvh(&mut self, canvas: &mut Canvas) {
        let params = self.build;
        let Some(mesh) = self.scene.first_mesh_mut() else {
            self.status = "no mesh in scene".to_string();
            return;
        };

        let triangles = mesh.get_triangles();
        let prim_count = triangles.len();

        let started = Instant::now();
        let tree = bvh::construct_bvh_from_tris(triangles, params);
        let build_ms = started.elapsed().as_secs_f64() * 1000.0;

        let mut stats = bvh::compute_build_stats(&tree);
        stats.build_time_ms = build_ms;

        self.depth_profile = diagrams::depth_profile(&tree)
            .into_iter()
            .map(|count| count as f32)
            .collect();

        mesh.add_bvh(tree);

        self.prim_count = prim_count;

        let first_build = self.view.max_depth == 0;
        self.view.max_depth = stats.max_depth;
        if first_build {
            // Opening on a single mid-tree level rather than every box at once:
            // the whole wireframe on a 19k-triangle mesh is 30k CPU-rasterised
            // lines a frame, and it is an unreadable thicket besides.
            self.view.depth_focus = Some(stats.max_depth / 2);
        } else {
            self.view.depth_focus = self.view.depth_focus.map(|d| d.min(stats.max_depth));
        }

        self.stats = stats;
        self.status = format!(
            "{} rebuilt in {:.1} ms",
            params.heuristic.name(),
            build_ms
        );

        canvas.invalidate_pipeline();
        canvas.reset_accumulation();
    }

    fn experiment_config(&self) -> ExperimentConfig {
        ExperimentConfig {
            camera_origin: self.camera.ray().origin(),
            camera_direction: self.camera.ray().direction(),
            width: self.export_width,
            height: self.export_height,
            samples: self.export_samples,
            rng_seed: self.live.rng_seed,
            bvh_seed: self.build.seed,
            leaf_cutoff: self.build.leaf_cutoff,
            warmup_runs: self.export_warmup,
            timed_runs: self.export_runs.max(1),
            // The studio scene is already positioned; renormalising would move it
            // out from under the camera the user just framed.
            normalize_scenes: false,
            scenes: vec![PathBuf::from(scene::STUDIO_MESH_PATH)],
            output_path: PathBuf::from(&self.output_dir).join("metrics.csv"),
            verify_determinism: false,
            figures_dir: Some(PathBuf::from(&self.output_dir).join("figures")),
        }
    }

    /// Kicks the harness off on its own thread so a multi-minute sweep does not
    /// freeze the window.
    fn start_figure_export(&mut self) {
        if self.harness_running {
            self.status = "export already running".to_string();
            return;
        }

        let config = self.experiment_config();
        let (tx, rx) = mpsc::channel();

        std::thread::spawn(move || {
            let result = crate::experiment::run(&config)
                .map(|_| format!("figures written to {}", config.output_path.display()))
                .map_err(|error| error.to_string());
            let _ = tx.send(HarnessMessage::Finished(result));
        });

        self.harness_rx = Some(rx);
        self.harness_running = true;
        self.status = "running harness in background...".to_string();
    }

    fn poll_harness(&mut self) {
        let Some(rx) = &self.harness_rx else { return };
        if let Ok(HarnessMessage::Finished(result)) = rx.try_recv() {
            self.status = match result {
                Ok(message) => message,
                Err(error) => format!("export failed: {error}"),
            };
            self.harness_running = false;
            self.harness_rx = None;
        }
    }

    fn export_structure_diagrams(&mut self) {
        let dir = PathBuf::from(&self.output_dir).join("diagrams");
        if let Err(error) = std::fs::create_dir_all(&dir) {
            self.status = format!("could not create {}: {error}", dir.display());
            return;
        }

        let name = self.build.heuristic.name().to_string();
        let Some(mesh) = self.scene.first_mesh_mut() else { return };
        let Some(tree) = mesh.bvh.as_ref() else { return };

        let icicle = diagrams::bvh_icicle_svg(tree, &name, "studio");
        let profile = diagrams::depth_profile(tree);
        let histogram = diagrams::depth_histogram_svg("studio", &[(name.clone(), profile)]);

        let icicle_path = dir.join(format!("icicle_{name}.svg"));
        let histogram_path = dir.join(format!("depth_histogram_{name}.svg"));

        let write = std::fs::write(&icicle_path, icicle)
            .and_then(|_| std::fs::write(&histogram_path, histogram));

        self.status = match write {
            Ok(()) => format!("wrote {}", icicle_path.display()),
            Err(error) => format!("diagram export failed: {error}"),
        };
    }

    fn save_viewport(&mut self, canvas: &Canvas) {
        let dir = PathBuf::from(&self.output_dir);
        let mode = match self.live.display_mode {
            DisplayMode::Beauty => "render",
            DisplayMode::NodeVisits => "node_visits",
            DisplayMode::PrimTests => "prim_tests",
            DisplayMode::TraversalDepth => "traversal_depth",
            DisplayMode::LeafVisits => "leaf_visits",
            DisplayMode::InteriorVisits => "interior_visits",
        };
        let path = dir.join(format!(
            "{}_{}_{}spp.png",
            mode,
            self.build.heuristic.name(),
            canvas.sample_count
        ));

        self.status = match canvas_to_image(canvas).save(&path) {
            Ok(()) => format!("wrote {}", path.display()),
            Err(error) => format!("save failed: {error}"),
        };
    }

    /// Writes one overlay image per level of the tree, from the already
    /// accumulated frame.
    fn export_depth_sweep(&mut self, canvas: &mut Canvas) {
        let dir = PathBuf::from(&self.output_dir).join("depth_sweep");
        let max_depth = self.view.max_depth;
        let saved = (self.view.enabled, self.view.depth_focus);

        let mut written = 0usize;
        for depth in 0..=max_depth {
            self.view.enabled = true;
            self.view.depth_focus = Some(depth);

            self.renderer.repaint_accumulated(&self.camera, canvas);
            self.renderer.render_bvh_overlay(&self.camera, &self.scene, canvas, &self.view);

            let path = dir.join(format!("{}_depth_{depth:02}.png", self.build.heuristic.name()));
            if canvas_to_image(canvas).save(&path).is_ok() {
                written += 1;
            }
        }

        (self.view.enabled, self.view.depth_focus) = saved;
        self.status = format!("wrote {written} depth images to {}", dir.display());
    }

    /// Points the camera at the mesh and backs off far enough to frame it.
    fn frame_model(&mut self) {
        let Some(mesh) = self.scene.first_mesh_mut() else { return };
        let triangles = mesh.get_triangles();
        if triangles.is_empty() {
            return;
        }

        let (min, max) = triangles.iter().flat_map(|tri| tri.get_vertices()).fold(
            (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
            |(min, max), v| (min.min(v), max.max(v)),
        );

        let center = (min + max) * 0.5;
        let radius = (max - min).length() * 0.5;
        // 90 degree vertical FOV, so half-extent equals distance; the 1.25 leaves
        // a margin so the model is not flush against the frame edge.
        let distance = (radius * 1.25).max(0.5);

        self.camera.set_ray(Ray::new(
            center + Vec3::new(0.0, 0.0, distance),
            Vec3::new(0.0, 0.0, -1.0),
        ));
        self.yaw = -90.0;
        self.pitch = 0.0;
        self.status = "framed model".to_string();
    }

    /// WASD plus right-drag look, suppressed whenever a panel has focus.
    fn apply_camera(&mut self, canvas: &Canvas, delta_time: f32, ui_wants_mouse: bool, ui_wants_keyboard: bool) -> bool {
        let mut moved = false;
        let mut ray = self.camera.ray();

        let right_down = canvas.is_mouse_button_down(MouseButton::Button2);
        if right_down && !ui_wants_mouse {
            let cursor = canvas.get_mouse_pos();
            if !self.looking {
                self.last_cursor = cursor;
                self.looking = true;
            }

            let dx = (cursor.0 - self.last_cursor.0) as f32 * self.look_sensitivity;
            let dy = (cursor.1 - self.last_cursor.1) as f32 * self.look_sensitivity;
            self.last_cursor = cursor;

            if dx != 0.0 || dy != 0.0 {
                self.yaw += dx;
                self.pitch = (self.pitch - dy).clamp(-89.0, 89.0);
                moved = true;
            }
        } else {
            self.looking = false;
        }

        let direction = Vec3::new(
            self.yaw.to_radians().cos() * self.pitch.to_radians().cos(),
            self.pitch.to_radians().sin(),
            self.yaw.to_radians().sin() * self.pitch.to_radians().cos(),
        )
        .normalize();
        ray = Ray::new(ray.origin(), direction);

        if !ui_wants_keyboard {
            let forward = direction;
            let right = Vec3::Y.cross(forward).normalize();
            let speed = self.move_speed * delta_time;
            let mut movement = Vec3::ZERO;

            if canvas.is_key_down(Key::W) { movement += forward * speed; }
            if canvas.is_key_down(Key::S) { movement -= forward * speed; }
            if canvas.is_key_down(Key::A) { movement -= right * speed; }
            if canvas.is_key_down(Key::D) { movement += right * speed; }
            if canvas.is_key_down(Key::Space) { movement += Vec3::Y * speed; }
            if canvas.is_key_down(Key::LeftShift) { movement -= Vec3::Y * speed; }

            if movement.length_squared() > 0.0 {
                ray = Ray::new(ray.origin() + movement, ray.direction());
                moved = true;
            }
        }

        self.camera.set_ray(ray);
        moved
    }
}

fn empty_stats() -> BuildStats {
    BuildStats {
        node_count: 0,
        leaf_count: 0,
        prim_count: 0,
        max_depth: 0,
        avg_depth: 0.0,
        avg_leaf_prims: 0.0,
        total_sah_cost: 0.0,
        leaf_sah_cost: 0.0,
        build_time_ms: 0.0,
    }
}

fn canvas_to_image(canvas: &Canvas) -> Image {
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

/// Runs the studio for `frames` frames, writes one shot of the whole window
/// including the panels, then exits. Used to document the UI and to check the
/// layout without a screen grab.
pub async fn run_screenshot(width: u32, height: u32, frames: u32, output: &std::path::Path) {
    run_inner(width, height, Some((frames, output.to_path_buf()))).await;
}

pub async fn run(width: u32, height: u32) {
    run_inner(width, height, None).await;
}

async fn run_inner(width: u32, height: u32, screenshot: Option<(u32, PathBuf)>) {
    let mut canvas = Canvas::new(width, height, "BVH Studio").await;
    // The pointer has to reach the panels, so it stays free; look is on
    // right-drag instead.
    canvas.set_cursor_captured(false);

    let camera = Camera::new(
        canvas.width(),
        canvas.height(),
        Ray::new(Vec3::new(0.0, 0.0, 3.6), Vec3::new(0.0, 0.0, -1.0)),
    );

    println!("loading scene...");
    let scene = scene::create_scene();

    let mut imgui = imgui::Context::create();
    imgui.set_ini_filename(None);
    panels::apply_theme(&mut imgui);

    let mut platform = platform::Platform::new(&mut imgui, canvas.width(), canvas.height(), 1.0);
    let mut ui_renderer = imgui_wgpu::Renderer::new(
        &mut imgui,
        canvas.device(),
        canvas.queue(),
        imgui_wgpu::RendererConfig {
            texture_format: canvas.surface_format(),
            ..Default::default()
        },
    );

    let mut studio = Studio::new(scene, camera, 0);
    studio.actions.push(Action::RebuildBvh);

    let mut last_frame = Instant::now();
    let mut frame_index = 0u32;
    let mut active_variant = ShaderVariant::Clean;

    loop {
        frame_index += 1;
        // Section timings are per-frame; without this they would accumulate for
        // the life of the process. The "main" scope is what tells the profiler
        // these sections belong to a frame rather than to start-up.
        crate::profiler::profiler_reset();
        crate::profiler::profiler_start("main");
        let now = Instant::now();
        let elapsed = (now - last_frame).as_secs_f32();
        last_frame = now;
        // Movement integrates against a clamped delta so a stall cannot teleport
        // the camera, but the readout reports what actually happened.
        let delta_time = elapsed.min(0.1);
        studio.frame_ms = elapsed * 1000.0;

        for event in canvas.pump_events() {
            platform.handle_event(&mut imgui, &event);
            if matches!(event, WindowEvent::Size(_, _)) {
                studio.camera.resize(canvas.width(), canvas.height());
            }
        }

        if !canvas.is_open() {
            std::process::exit(0);
        }

        studio.poll_harness();

        let wants_mouse = imgui.io().want_capture_mouse;
        let wants_keyboard = imgui.io().want_capture_keyboard;
        if studio.apply_camera(&canvas, delta_time, wants_mouse, wants_keyboard) {
            canvas.reset_accumulation();
        }

        // --- render the scene -------------------------------------------
        studio.live.palette = studio.palette_index();

        // Only a cost view needs counters. Staying on the clean variant for the
        // rendered image keeps plain viewing at full speed, at the cost of a
        // pipeline rebuild when the mode crosses between the two.
        let wanted_variant = if studio.live.display_mode == DisplayMode::Beauty {
            ShaderVariant::Clean
        } else {
            ShaderVariant::Instrumented
        };
        if wanted_variant != active_variant {
            canvas.invalidate_pipeline();
            canvas.reset_accumulation();
            active_variant = wanted_variant;
        }

        studio.renderer.render_gpu_with(
            &studio.camera,
            &studio.scene,
            &mut canvas,
            &studio.live,
            active_variant,
        );

        if studio.view.enabled {
            crate::profiler::profiler_start("overlay");
            studio.renderer.render_bvh_overlay(&studio.camera, &studio.scene, &mut canvas, &studio.view);
            crate::profiler::profiler_stop("overlay");
        }

        // --- build the UI -----------------------------------------------
        platform.prepare_frame(&mut imgui, canvas.width(), canvas.height());
        let ui = imgui.frame();
        panels::draw(ui, &mut studio);
        let draw_data = imgui.render();

        if let Some((target_frame, path)) = &screenshot {
            if frame_index >= *target_frame {
                let rgb = canvas.capture_with_ui(draw_data, &mut ui_renderer);
                let image = Image::from_rgb(canvas.width(), canvas.height(), rgb);
                match image.save(path) {
                    Ok(()) => println!("wrote {}", path.display()),
                    Err(error) => eprintln!("screenshot failed: {error}"),
                }
                std::process::exit(0);
            }
        }

        canvas.present_with_ui(draw_data, &mut ui_renderer).unwrap();

        // --- apply queued work ------------------------------------------
        let actions: Vec<Action> = studio.actions.drain(..).collect();
        for action in actions {
            match action {
                Action::RebuildBvh => studio.rebuild_bvh(&mut canvas),
                Action::ResetAccumulation => canvas.reset_accumulation(),
                Action::SaveViewport => studio.save_viewport(&canvas),
                Action::ExportFigureSet => studio.start_figure_export(),
                Action::ExportStructureDiagrams => studio.export_structure_diagrams(),
                Action::ExportDepthSweep => studio.export_depth_sweep(&mut canvas),
                Action::FrameModel => {
                    studio.frame_model();
                    canvas.reset_accumulation();
                }
            }
        }

        crate::profiler::profiler_stop("main");
    }
}
