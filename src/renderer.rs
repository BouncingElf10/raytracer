use crate::camera::Camera;
use crate::color::Color;
use crate::gpu_types::{GpuColor, GpuRay};
use crate::objects::HitInfo;
use crate::profiler::{profiler_start, profiler_stop};
use crate::ray::Ray;
use crate::scene::Scene;
use crate::window::Canvas;
use crate::{compute, ray};
use wgpu::PollType;

pub struct Renderer {

}

const MAX_RECURSION: u8 = 5;

impl Renderer {
    pub fn new() -> Self {
        Self {}
    }
    #[allow(dead_code)]
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

    pub fn render_debug(&self, camera: &Camera, scene: &Scene, canvas: &mut Canvas) {
        let mut color_index = 0;
        canvas.clear(camera);
        scene.get_objects().iter().for_each(|object| {
            camera.for_each_pixel(|x, y| {
                let ray = ray::get_ray_from_screen(camera, x, y);
                if object.to_aabb().hit(&ray) {
                    canvas.paint_pixel(x, y, Color::random_from_seed(color_index).to_u32())
                }
            });
            color_index += 1;
        });
        canvas.sample_count += 1;
    }

    pub fn render_gpu(&self, camera: &Camera, scene: &Scene, canvas: &mut Canvas) {
        profiler_start("render gpu");

        if canvas.compute_pipeline.is_none() {
            compute::setup_compute_pipeline(canvas, scene);
        }

        let mut rays = Vec::with_capacity((canvas.width() * canvas.height()) as usize);
        camera.for_each_pixel(|x, y| {
            let ray = ray::get_ray_from_screen(camera, x, y);
            rays.push(GpuRay {
                origin: [ray.origin().x, ray.origin().y, ray.origin().z],
                _pad0: 0.0,
                direction: [ray.direction().x, ray.direction().y, ray.direction().z],
                _pad1: 0.0,
            });
        });

        canvas.queue().write_buffer(
            canvas.ray_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&rays)
        );

        canvas.queue().write_buffer(
            canvas.counts_buffer.as_ref().unwrap(),
            20,
            bytemuck::cast_slice(&[canvas.sample_count]),
        );

        let mut encoder = canvas.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Ray Trace Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(canvas.compute_pipeline.as_ref().unwrap());
            compute_pass.set_bind_group(0, canvas.compute_bind_group.as_ref().unwrap(), &[]);

            let workgroups_x = (canvas.width() + 7) / 8;
            let workgroups_y = (canvas.height() + 7) / 8;

            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        encoder.copy_buffer_to_buffer(
            canvas.color_buffer.as_ref().unwrap(),
            0,
            canvas.staging_buffer.as_ref().unwrap(),
            0,
            (canvas.pixel_count() as usize * std::mem::size_of::<GpuColor>()) as u64,
        );

        canvas.queue().submit(std::iter::once(encoder.finish()));
        canvas.device().poll(PollType::Wait { submission_index: None, timeout: None })
            .expect("GPU was NOT polled");

        profiler_stop("render gpu");
        profiler_start("cpu accumulation");

        let buffer_slice = canvas.staging_buffer.as_ref().unwrap().slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        canvas.device().poll(PollType::Wait { submission_index: None, timeout: None })
            .expect("GPU was NOT polled");
        pollster::block_on(rx).unwrap().unwrap();

        {
            let data = buffer_slice.get_mapped_range();
            let colors: &[GpuColor] = bytemuck::cast_slice(&data);

            camera.for_each_pixel(|x, y| {
                let idx = (y * canvas.width() + x) as usize;
                let gpu_color = &colors[idx];
                let color = Color::new(gpu_color.r, gpu_color.g, gpu_color.b);

                canvas.accum_buffer[idx] = canvas.accum_buffer[idx] + color;
                let avg = canvas.accum_buffer[idx] / (canvas.sample_count as f32 + 1.0);
                canvas.paint_pixel(x, y, avg.gamma_correct().to_u32());
            });
        }

        canvas.staging_buffer.as_ref().unwrap().unmap();
        canvas.sample_count += 1;

        profiler_stop("cpu accumulation");
    }
    
    #[allow(dead_code)]
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
            let color = color * info.material.albedo * info.material.emission;
            return color;
        }

        if bounce_num >= MAX_RECURSION {
            return Color::black();
        }

        let normal = info.normal.normalize();
        let diffuse_dir = ray::random_cosine_hemisphere(normal);
        let diffuse_ray = Ray::new(info.pos + normal * 0.001, diffuse_dir);

        let specular_dir = ray.reflect(info.normal);
        let specular_ray = Ray::new(info.pos + normal * 0.001, specular_dir.direction().normalize());
        let final_ray = ray::lerp(&specular_ray, &diffuse_ray, info.material.roughness);

        let final_color = color * info.material.albedo * (info.material.metallic * specular_ray.dot() + (1.0 - info.material.metallic) * diffuse_ray.dot());

        recursive_bounce(final_ray, final_color, scene, bounce_num + 1)
    } else {
        Color::black()
    }
}