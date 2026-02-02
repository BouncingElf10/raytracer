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

#[tokio::main]
async fn main() {
    profiler::profiler_start("init");
    profiler::profiler_start("window");
    let mut canvas = Canvas::new(80 * 10, 60 * 10, "WINDOW").await;
    profiler::profiler_stop("window");
    let mut camera = Camera::new(canvas.width(), canvas.height(), Ray::new(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0)));
    let mut movement_state = movement::MovementState::new();
    let scene = scene::create_scene();
    let renderer = renderer::Renderer::new();
    let mut delta_time = 0.0;

    profiler::profiler_stop("init");

    loop {
        profiler::profiler_start("main");

        renderer.render_debug(&camera, &scene, &mut canvas);

        profiler::profiler_start("text and movement");

        canvas.set_window_title(&format!("frame in: {:.0}ms   fps: {:.2}   sample count: {}",
                                         profiler::get_delta_time() * 1000.0,
                                         1.0 /  profiler::get_delta_time(),
                                         canvas.sample_count));
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