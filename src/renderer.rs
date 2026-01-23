use std::ops::Mul;
use crate::camera::Camera;
use crate::color::Color;
use crate::objects::HitInfo;
use crate::ray;
use crate::ray::Ray;
use crate::scene::Scene;
use crate::window::Canvas;

pub struct Renderer {

}

const MAX_RECURSION: u8 = 5;

impl Renderer {
    pub fn new() -> Self {
        Self {}
    }

    pub fn render(&self, camera: &Camera, scene: &Scene, canvas: &mut Canvas) {
        camera.for_each_pixel(|x, y| {
            let ray = ray::get_ray_from_screen(camera, x, y);
            let sample = recursive_bounce(ray, Color::white(), scene, 0);

            let idx = (y * canvas.width() + x) as usize;

            canvas.accum_buffer[idx] = canvas.accum_buffer[idx] + sample;

            let avg = canvas.accum_buffer[idx] / (canvas.sample_count as f32 + 1.0);
            canvas.paint_pixel(x, y, avg.gamma_correct().to_u32());

        });

        canvas.sample_count += 1;
    }


    pub fn clear(&self, camera: &Camera, canvas: &mut Canvas) {
        camera.for_each_pixel(|x, y| {
            canvas.paint_pixel(x, y, Color::black().to_u32());
        });
    }
}

fn recursive_bounce(ray: Ray, color: Color, scene: &Scene, bounce_num: u8) -> Color {
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
        if info.material.emission > 0.0 {
            return color * info.material.albedo * info.material.emission;
        }

        if bounce_num >= MAX_RECURSION {
            return Color::black();
        }

        let normal = info.normal.normalize();
        let bounce_dir = ray::random_cosine_hemisphere(normal);
        let bounce_ray = Ray::new(info.pos + normal * 0.001, bounce_dir);
        let new_color = color * info.material.albedo;

        recursive_bounce(bounce_ray, new_color, scene, bounce_num + 1)
    } else {
        Color::black()
    }
}