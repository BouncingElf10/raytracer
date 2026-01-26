use std::time::Instant;
use glam::Vec3;
use crate::camera::Camera;
use crate::ray::Ray;
use crate::window::Canvas;

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

#[tokio::main]
async fn main() {
    let mut canvas = Canvas::new(80 * 10, 60 * 10, "WINDOW").await;
    let mut camera = Camera::new(canvas.width(), canvas.height(), Ray::new(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0)));
    let mut movement_state = movement::MovementState::new();
    let scene = scene::create_scene();
    let renderer = renderer::Renderer::new();
    let mut delta_time = 0.0;

    loop {
        profiler::profiler_start("main");

        renderer.render_gpu(&camera, &scene, &mut canvas);

        profiler::profiler_start("text and movement");

        canvas.set_window_title(&format!("frame in: {:.0}ms   fps: {:.2}   sample count: {}",
                                         profiler::get_delta_time("main") * 1000.0,
                                         1.0 /  profiler::get_delta_time("main"),
                                         canvas.sample_count));
        canvas.present().unwrap();
        canvas.update();
        if movement::apply_movements(&mut camera, &canvas, delta_time, &mut movement_state) {
            canvas.reset_accumulation();
        }

        profiler::profiler_stop("text and movement");

        delta_time = profiler::get_delta_time("main") as f32;
        
        profiler::profiler_stop("main");
        if !canvas.is_open() {
            profiler::print_profile();
            break;
        }
    }
}