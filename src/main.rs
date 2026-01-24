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

#[tokio::main]
async fn main() {
    let mut canvas = Canvas::new(80 * 10, 60 * 10, "WINDOW").await;
    let mut camera = Camera::new(canvas.width(), canvas.height(), Ray::new(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0)));
    let mut movement_state = movement::MovementState::new();
    let scene = scene::create_scene();
    let renderer = renderer::Renderer::new();
    let mut delta_time = 0.0;

    loop {
        let frame_start = Instant::now();

        renderer.render(&camera, &scene, &mut canvas);
        println!("{:?}", camera.ray());
        canvas.set_window_title(&format!("frame in: {}ms   fps: {:.2}   sample count: {}",
                                         frame_start.elapsed().as_millis(),
                                         1.0 / frame_start.elapsed().as_secs_f32(),
                                         canvas.sample_count));
        canvas.present().unwrap();
        canvas.update();
        if movement::apply_movements(&mut camera, &canvas, delta_time, &mut movement_state) {
            canvas.reset_accumulation();
        }

        delta_time = frame_start.elapsed().as_secs_f32();
        if !canvas.is_open() { break; }
    }
}