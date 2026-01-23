use std::time::Instant;
use glam::Vec3;
use crate::camera::Camera;
use crate::color::Color;
use crate::objects::HitInfo;
use crate::ray::Ray;
use crate::window::Canvas;

mod camera;
mod window;
mod color;
mod objects;
mod scene;
mod ray;
mod movement;

#[tokio::main]
async fn main() {
    let mut canvas = Canvas::new(80 * 10, 60 * 10, "WINDOW").await;
    let mut camera = Camera::new(canvas.width(), canvas.height(), Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0)));
    let mut movement_state = movement::MovementState::new();
    let scene = scene::create_scene();
    let mut delta_time = 0.0;
    loop {
        let frame_start = Instant::now();

        camera.for_each_pixel(|x, y| {
            let ray = ray::get_ray_from_screen(&camera, x, y);

            let mut closest_hit: Option<HitInfo> = None;
            let mut closest_t = f64::INFINITY;

            for hittable in scene.get_objects() {
                let info = hittable.hit(&ray);
                if info.has_hit && info.t < closest_t {
                    closest_t = info.t;
                    closest_hit = Some(info);
                }
            }

            if let Some(info) = closest_hit {
                let normal = info.normal_ray.direction();
                canvas.paint_pixel(x, y, Color::newFromNormals(normal.x, normal.y, normal.z).to_u32());
            } else {
                canvas.paint_pixel(x, y, 0);
            }
        });

        canvas.set_window_title(&format!("frame in: {}ms   fps: {:.2}", frame_start.elapsed().as_millis(), 1.0 / frame_start.elapsed().as_secs_f32()));
        canvas.present().unwrap();
        canvas.update();

        movement::apply_movements(&mut camera, &canvas, delta_time, &mut movement_state);

        delta_time = frame_start.elapsed().as_secs_f32();
        if !canvas.is_open() { break; }
    }
}
