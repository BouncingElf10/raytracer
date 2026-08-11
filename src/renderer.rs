use crate::bvh::{traverse_leaf_nodes, traverse_nodes_with_depth, NodeVisit, AABB, DEFAULT_LEAF_PRIM_CUTOFF};
use crate::camera::Camera;
use crate::color::Color;
use crate::gpu_types::{Counts, DisplayMode, GpuColor};
use crate::shaders::ShaderVariant;
use crate::model::Mesh;
use crate::objects::HitInfo;
use crate::profiler::{profiler_start, profiler_stop};
use crate::ray::Ray;
use crate::scene::Scene;
use crate::viz::Palette;
use crate::window::Canvas;
use crate::{compute, ray};
use wgpu::PollType;

pub struct Renderer {

}

const MAX_RECURSION: u8 = 5;

/// Shader settings that can change between frames without rebuilding anything.
#[derive(Debug, Clone, Copy)]
pub struct LiveSettings {
    pub samples: u32,
    pub rng_seed: u32,
    pub display_mode: DisplayMode,
    pub palette: u32,
    pub heat_scale: f32,
    /// 0 = pure cost view, 1 = pure render.
    pub heat_mix: f32,
    pub max_bounces: u32,
}

impl Default for LiveSettings {
    fn default() -> Self {
        Self {
            samples: 1,
            rng_seed: 0,
            display_mode: DisplayMode::Beauty,
            palette: 0,
            heat_scale: 40.0,
            heat_mix: 0.0,
            max_bounces: 10,
        }
    }
}

impl LiveSettings {
    /// The `Counts` fields from `samples` onward, as raw words ready to upload.
    fn tail_words(&self) -> [u32; 7] {
        [
            self.samples.max(1),
            self.rng_seed,
            self.display_mode as u32,
            self.palette,
            self.heat_scale.to_bits(),
            self.heat_mix.to_bits(),
            self.max_bounces.max(1),
        ]
    }
}

/// Interactive controls for the BVH wireframe overlay.
pub struct BvhDebugView {
    pub enabled: bool,
    /// Hide interior nodes and show only the boxes that actually hold geometry.
    pub leaves_only: bool,
    /// When set, draw only this one level of the tree. Stepping through levels is
    /// the clearest way to see how a heuristic partitions space -- drawing every
    /// box at once is an unreadable thicket on anything but a toy mesh.
    pub depth_focus: Option<usize>,
    /// Deepest level present, used to scale the colour ramp.
    pub max_depth: usize,
    /// Ramp used to colour boxes by depth.
    pub palette: Palette,
    /// Brightness of interior boxes relative to leaves.
    pub interior_dim: f32,
    /// Scale leaf brightness by how full the leaf is.
    pub highlight_full_leaves: bool,
}

impl BvhDebugView {
    pub fn new(max_depth: usize) -> Self {
        Self {
            enabled: true,
            leaves_only: false,
            depth_focus: None,
            max_depth,
            palette: Palette::Turbo,
            interior_dim: 0.45,
            highlight_full_leaves: true,
        }
    }

    fn accepts(&self, visit: &NodeVisit) -> bool {
        if self.leaves_only && !visit.is_leaf {
            return false;
        }
        match self.depth_focus {
            Some(depth) => visit.depth == depth,
            None => true,
        }
    }

    fn color_for(&self, visit: &NodeVisit) -> u32 {
        let t = if self.max_depth == 0 {
            0.0
        } else {
            visit.depth as f32 / self.max_depth as f32
        };
        let [r, g, b] = self.palette.sample(t);

        // Interior boxes are dimmed so leaves stay legible when both are drawn.
        // Leaves are brightened by how full they are: a leaf sitting at the
        // primitive cutoff is where the build stopped subdividing, and those are
        // exactly the boxes that dominate intersection cost.
        let brightness = if visit.is_leaf {
            if self.highlight_full_leaves {
                let fill = visit.prim_count as f32 / DEFAULT_LEAF_PRIM_CUTOFF as f32;
                0.5 + 0.5 * fill.clamp(0.0, 1.0)
            } else {
                1.0
            }
        } else {
            self.interior_dim.clamp(0.0, 1.0)
        };

        let scale = |channel: u8| -> u32 { (channel as f32 * brightness) as u32 };
        (scale(r) << 16) | (scale(g) << 8) | scale(b)
    }

    pub fn step_depth(&mut self, delta: i32) {
        let next = match self.depth_focus {
            None => 0,
            Some(depth) => (depth as i32 + delta).max(0) as usize,
        };
        self.depth_focus = Some(next.min(self.max_depth));
    }

    pub fn show_all_depths(&mut self) {
        self.depth_focus = None;
    }

    /// Short status line for the window title.
    pub fn status(&self) -> String {
        if !self.enabled {
            return "bvh off".to_string();
        }
        let level = match self.depth_focus {
            Some(depth) => format!("depth {depth}/{}", self.max_depth),
            None => format!("all depths (0-{})", self.max_depth),
        };
        let scope = if self.leaves_only { "leaves" } else { "all nodes" };
        format!("bvh: {level}, {scope}")
    }
}

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

    /// Repaints the framebuffer from the accumulation buffer without touching the
    /// GPU.
    ///
    /// Needed before drawing a fresh overlay: wireframe lines are written
    /// straight into the framebuffer, so without a repaint every overlay drawn
    /// would stack on top of the last one.
    pub fn repaint_accumulated(&self, camera: &Camera, canvas: &mut Canvas) {
        let samples = canvas.sample_count.max(1) as f32;
        camera.for_each_pixel(|x, y| {
            let idx = (y * canvas.width() + x) as usize;
            let average = canvas.accum_buffer[idx] / samples;
            canvas.paint_pixel(x, y, average.gamma_correct().to_u32());
        });
    }

    /// Draws the BVH wireframe over the rendered image.
    ///
    /// Every box is coloured by its depth in the tree rather than by which object
    /// it belongs to, so the structure of the hierarchy is visible at a glance:
    /// a balanced build shows a smooth colour gradient outward, an unbalanced one
    /// shows deep-end colours concentrated in one region.
    pub fn render_bvh_overlay(&self, camera: &Camera, scene: &Scene, canvas: &mut Canvas, view: &BvhDebugView) {
        for object in scene.get_objects() {
            let Some(mesh) = object.as_any().downcast_ref::<Mesh>() else { continue };
            let Some(bvh) = mesh.bvh.as_ref() else { continue };

            traverse_nodes_with_depth(bvh, &mut |visit| {
                if !view.accepts(&visit) {
                    return;
                }

                let color = view.color_for(&visit);
                for (a, b) in visit.aabb.edges() {
                    if let Some((pa, pb)) = camera.project_segment(a, b) {
                        canvas.draw_line(pa, pb, color);
                    }
                }
            });
        }
    }

    /// Wireframes every non-mesh object's bounds. Useful for sanity-checking the
    /// room geometry against what the shader thinks is there.
    #[allow(dead_code)]
    pub fn render_debug(&self, camera: &Camera, scene: &Scene, canvas: &mut Canvas, should_clear: bool) {
        if should_clear { canvas.clear(camera); }

        for (i, object) in scene.get_objects().iter().enumerate() {
            if let Some(mesh) = object.as_any().downcast_ref::<Mesh>() {
                let bvh = mesh.bvh.as_ref().unwrap();
                traverse_leaf_nodes(&bvh, &mut |aabb: &AABB, _objects| {
                    for (a, b) in aabb.edges() {
                        if let Some((pa, pb)) = camera.project_segment(a, b) {
                            canvas.draw_line(pa, pb, Color::random_from_seed(i as u32).to_u32());
                        }
                    }
                });
                return;
            }

            let aabb = object.to_aabb();
            for (a, b) in aabb.edges() {
                if let Some((pa, pb)) = camera.project_segment(a, b) {
                    canvas.draw_line(pa, pb, Color::random_from_seed(i as u32).to_u32());
                }
            }
        }
    }


    pub fn render_gpu(&self, camera: &Camera, scene: &Scene, canvas: &mut Canvas) {
        self.render_gpu_with(camera, scene, canvas, &LiveSettings::default(), ShaderVariant::Clean);
    }

    /// One accumulation step with explicit shader settings.
    ///
    /// `variant` decides which pipeline gets built on the first frame; the studio
    /// asks for the instrumented one so its cost views have counters to read.
    pub fn render_gpu_with(
        &self,
        camera: &Camera,
        scene: &Scene,
        canvas: &mut Canvas,
        settings: &LiveSettings,
        variant: ShaderVariant,
    ) {
        profiler_start("render gpu");

        if canvas.compute_pipeline.is_none() {
            compute::setup_compute_pipeline_with(canvas, scene, variant);
        }

        let mut rays = Vec::with_capacity((canvas.width() * canvas.height()) as usize);
        camera.for_each_pixel(|x, y| {
            let ray = ray::get_ray_from_screen(camera, x, y);
            rays.push(ray.to_gpu_ray());
        });

        canvas.queue().write_buffer(
            canvas.ray_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&rays)
        );

        canvas.queue().write_buffer(
            canvas.counts_buffer.as_ref().unwrap(),
            Counts::FRAME_NUMBER_OFFSET,
            bytemuck::cast_slice(&[canvas.sample_count]),
        );

        // Everything from `samples` onward, patched as one block. The geometry
        // counts that precede it are set when the pipeline is built and must not
        // be clobbered here.
        canvas.queue().write_buffer(
            canvas.counts_buffer.as_ref().unwrap(),
            32,
            bytemuck::cast_slice(&settings.tail_words()),
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
            (canvas.pixel_count() as usize * size_of::<GpuColor>()) as u64,
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