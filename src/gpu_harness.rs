//! Headless GPU side of the data-collection harness.
//!
//! Owns its own adapter/device so it never touches the interactive window, and
//! builds both shader variants up front (§4). Each record is collected in two
//! separate passes (§6):
//!
//!   * Pass A runs the instrumented variant and keeps only the counters.
//!   * Pass B runs the clean variant and keeps only the GPU timestamp delta.
//!
//! Counts and timings are never taken from the same dispatch, so the extra
//! buffer traffic in Pass A cannot leak into the reported render time.

use crate::gpu_types::{Counts, GpuBVHNode, GpuColor, GpuPlane, GpuRay, GpuRayCounters, GpuSphere, GpuTriangle};
use crate::shaders::{self, ShaderVariant};
use wgpu::util::DeviceExt;
use wgpu::PollType;

/// Geometry uploaded for one (scene, heuristic) combination.
pub struct SceneUpload {
    _sphere_buffer: wgpu::Buffer,
    _triangle_buffer: wgpu::Buffer,
    _plane_buffer: wgpu::Buffer,
    _bvh_node_buffer: wgpu::Buffer,
    _bvh_index_buffer: wgpu::Buffer,
    _counts_buffer: wgpu::Buffer,
    clean_bind_group: wgpu::BindGroup,
    instrumented_bind_group: wgpu::BindGroup,
}

/// Reduction of the per-ray counter buffer (§5 Option B).
///
/// Means divide by the true traversal-query count, so secondary bounces are
/// weighted correctly. Distribution stats are taken over pixels, each pixel
/// contributing its own mean-per-ray.
#[derive(Debug, Clone, Copy, Default)]
pub struct CounterSummary {
    pub total_node_visits: u64,
    pub total_interior_visits: u64,
    pub total_prim_tests: u64,
    pub total_rays: u64,
    /// Traversals that ran out of stack or hit the step guard. Non-zero means the
    /// counts below are a lower bound, not the true cost.
    pub incomplete_traversals: u64,
    pub node_visits_per_ray: f64,
    pub interior_visits_per_ray: f64,
    pub prim_tests_per_ray: f64,
    pub node_visits_max: f64,
    pub node_visits_p95: f64,
    pub prim_tests_max: f64,
    pub prim_tests_p95: f64,
}

/// Wall-clock cost of the traversal dispatch, measured on the GPU timeline.
#[derive(Debug, Clone, Copy, Default)]
pub struct TimingSummary {
    pub mean_ms: f64,
    pub min_ms: f64,
    pub stddev_ms: f64,
    /// False when the adapter lacks TIMESTAMP_QUERY; the millisecond fields are
    /// then meaningless and get written as empty CSV cells.
    pub valid: bool,
}

pub struct GpuHarness {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    backend: String,

    width: u32,
    height: u32,
    pixel_count: usize,

    clean_pipeline: wgpu::ComputePipeline,
    instrumented_pipeline: wgpu::ComputePipeline,
    clean_layout: wgpu::BindGroupLayout,
    instrumented_layout: wgpu::BindGroupLayout,

    ray_buffer: wgpu::Buffer,
    color_buffer: wgpu::Buffer,
    color_staging: wgpu::Buffer,
    counter_buffer: wgpu::Buffer,
    counter_staging: wgpu::Buffer,

    timestamps: Option<TimestampRig>,
}

struct TimestampRig {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    /// Nanoseconds per timestamp tick.
    period_ns: f32,
}

impl GpuHarness {
    pub fn new(width: u32, height: u32) -> Result<Self, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .map_err(|e| format!("no suitable GPU adapter: {e}"))?;

        let info = adapter.get_info();
        let has_timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        if !has_timestamps {
            eprintln!(
                "warning: adapter '{}' does not support TIMESTAMP_QUERY; \
                 render_time_ms will be left empty",
                info.name
            );
        }

        let mut required_features = wgpu::Features::empty();
        if has_timestamps {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("BVH Harness Device"),
            required_features,
            // The instrumented variant binds eight storage buffers, which is
            // exactly the default limit; taking the adapter's own limits keeps
            // headroom and lets large scenes exceed the default 128 MiB binding.
            required_limits: adapter.limits(),
            experimental_features: Default::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: Default::default(),
        }))
        .map_err(|e| format!("failed to create device: {e}"))?;

        let clean_layout = create_bind_group_layout(&device, false);
        let instrumented_layout = create_bind_group_layout(&device, true);

        let clean_pipeline =
            create_pipeline(&device, &clean_layout, ShaderVariant::Clean, "Clean Traversal");
        let instrumented_pipeline = create_pipeline(
            &device,
            &instrumented_layout,
            ShaderVariant::Instrumented,
            "Instrumented Traversal",
        );

        let pixel_count = (width as usize) * (height as usize);

        let ray_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Harness Ray Buffer"),
            size: (pixel_count * size_of::<GpuRay>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let color_bytes = (pixel_count * size_of::<GpuColor>()) as u64;

        let color_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Harness Color Buffer"),
            size: color_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let color_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Harness Color Staging"),
            size: color_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let counter_bytes = (pixel_count * size_of::<GpuRayCounters>()) as u64;

        let counter_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Harness Counter Buffer"),
            size: counter_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let counter_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Harness Counter Staging"),
            size: counter_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let timestamps = has_timestamps.then(|| TimestampRig {
            query_set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("Traversal Timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            }),
            resolve_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Timestamp Resolve"),
                size: 2 * size_of::<u64>() as u64,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            staging_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Timestamp Staging"),
                size: 2 * size_of::<u64>() as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            period_ns: queue.get_timestamp_period(),
        });

        Ok(Self {
            device,
            queue,
            adapter_name: info.name.clone(),
            backend: format!("{:?}", info.backend),
            width,
            height,
            pixel_count,
            clean_pipeline,
            instrumented_pipeline,
            clean_layout,
            instrumented_layout,
            ray_buffer,
            color_buffer,
            color_staging,
            counter_buffer,
            counter_staging,
            timestamps,
        })
    }

    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    pub fn backend(&self) -> &str {
        &self.backend
    }

    pub fn timestamps_supported(&self) -> bool {
        self.timestamps.is_some()
    }

    /// Uploads the camera rays shared by both passes. Called once per experiment:
    /// identical rays for every scene and heuristic is what makes the counts
    /// comparable.
    pub fn upload_rays(&self, rays: &[GpuRay]) {
        assert_eq!(rays.len(), self.pixel_count, "ray count must match resolution");
        self.queue.write_buffer(&self.ray_buffer, 0, bytemuck::cast_slice(rays));
    }

    /// Uploads one flattened BVH plus its geometry and builds both bind groups.
    pub fn upload_scene(
        &self,
        triangles: &[GpuTriangle],
        planes: &[GpuPlane],
        bvh_nodes: &[GpuBVHNode],
        bvh_indices: &[u32],
        samples: u32,
        rng_seed: u32,
    ) -> SceneUpload {
        // Storage buffers may not be zero-sized, so empty categories get one
        // inert element. Spheres are unused by the harness scenes entirely.
        let spheres = [GpuSphere {
            center: [0.0, 0.0, 0.0],
            radius: 0.0,
            albedo: [0.0, 0.0, 0.0],
            emission: 0.0,
            metallic: 0.0,
            roughness: 0.0,
            _padding: [0.0, 0.0],
        }];

        let counts = Counts {
            // Reported as 0 so the shader's sphere loop never runs; the buffer
            // still exists to satisfy the binding.
            sphere_count: 0,
            triangle_count: triangles.len() as u32,
            plane_count: planes.len() as u32,
            width: self.width,
            height: self.height,
            frame_number: 0,
            bvh_node_count: bvh_nodes.len() as u32,
            bvh_index_count: bvh_indices.len() as u32,
            samples,
            rng_seed,
            // The harness always measures the rendered image. Live cost views are
            // a studio display option and must never alter a measurement run.
            display_mode: 0,
            palette: 0,
            heat_scale: 1.0,
            heat_mix: 0.0,
            max_bounces: 10,
            _pad0: 0,
        };

        let sphere_buffer = self.storage_init("Harness Spheres", bytemuck::cast_slice(&spheres));
        let triangle_buffer = self.storage_init("Harness Triangles", bytemuck::cast_slice(triangles));
        let plane_buffer = self.storage_init("Harness Planes", bytemuck::cast_slice(planes));
        let bvh_node_buffer = self.storage_init("Harness BVH Nodes", bytemuck::cast_slice(bvh_nodes));
        let bvh_index_buffer = self.storage_init("Harness BVH Indices", bytemuck::cast_slice(bvh_indices));

        let counts_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Harness Counts"),
            contents: bytemuck::cast_slice(&[counts]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let shared = [
            wgpu::BindGroupEntry { binding: 0, resource: self.ray_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: self.color_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: sphere_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: triangle_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: plane_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: counts_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: bvh_node_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: bvh_index_buffer.as_entire_binding() },
        ];

        let clean_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Harness Clean Bind Group"),
            layout: &self.clean_layout,
            entries: &shared,
        });

        let mut instrumented_entries = shared.to_vec();
        instrumented_entries.push(wgpu::BindGroupEntry {
            binding: 8,
            resource: self.counter_buffer.as_entire_binding(),
        });

        let instrumented_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Harness Instrumented Bind Group"),
            layout: &self.instrumented_layout,
            entries: &instrumented_entries,
        });

        SceneUpload {
            _sphere_buffer: sphere_buffer,
            _triangle_buffer: triangle_buffer,
            _plane_buffer: plane_buffer,
            _bvh_node_buffer: bvh_node_buffer,
            _bvh_index_buffer: bvh_index_buffer,
            _counts_buffer: counts_buffer,
            clean_bind_group,
            instrumented_bind_group,
        }
    }

    /// Pass A (§6): one instrumented dispatch, counters read back, timing ignored.
    ///
    /// When `keep_records` is set the raw per-pixel counters are returned as well,
    /// for the figure generator. They are ~32 bytes per pixel, so the caller asks
    /// for them only when it is actually going to draw something.
    pub fn collect_counters(
        &self,
        scene: &SceneUpload,
        keep_records: bool,
    ) -> (CounterSummary, Option<Vec<GpuRayCounters>>) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Instrumented Pass"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Instrumented Traversal"),
                // Deliberately untimed: instrumentation perturbs the schedule, so
                // any timing taken here would be meaningless.
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.instrumented_pipeline);
            pass.set_bind_group(0, &scene.instrumented_bind_group, &[]);
            let (gx, gy) = self.workgroups();
            pass.dispatch_workgroups(gx, gy, 1);
        }

        encoder.copy_buffer_to_buffer(
            &self.counter_buffer,
            0,
            &self.counter_staging,
            0,
            self.counter_staging.size(),
        );

        self.queue.submit(std::iter::once(encoder.finish()));
        self.wait();

        let bytes = self.read_back(&self.counter_staging);
        let records: &[GpuRayCounters] = bytemuck::cast_slice(&bytes);
        let summary = reduce_counters(records);
        let kept = keep_records.then(|| records.to_vec());
        self.counter_staging.unmap();
        (summary, kept)
    }

    /// One clean dispatch whose colour output is read back, for the reference
    /// image that accompanies the heatmaps. Values are linear HDR; tone mapping
    /// happens in `viz`.
    pub fn capture_color(&self, scene: &SceneUpload) -> Vec<[f32; 3]> {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Colour Capture"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Colour Capture"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.clean_pipeline);
            pass.set_bind_group(0, &scene.clean_bind_group, &[]);
            let (gx, gy) = self.workgroups();
            pass.dispatch_workgroups(gx, gy, 1);
        }

        encoder.copy_buffer_to_buffer(
            &self.color_buffer,
            0,
            &self.color_staging,
            0,
            self.color_staging.size(),
        );

        self.queue.submit(std::iter::once(encoder.finish()));
        self.wait();

        let bytes = self.read_back(&self.color_staging);
        let colors: &[GpuColor] = bytemuck::cast_slice(&bytes);
        let out = colors.iter().map(|c| [c.r, c.g, c.b]).collect();
        self.color_staging.unmap();
        out
    }

    /// Pass B (§6/§7): the clean variant, timed on the GPU timeline.
    ///
    /// The first `warmup_runs` dispatches are discarded -- they include shader
    /// compilation and pipeline warm-up -- and the remaining `timed_runs` are
    /// averaged.
    pub fn measure_render_time(
        &self,
        scene: &SceneUpload,
        warmup_runs: u32,
        timed_runs: u32,
    ) -> TimingSummary {
        for _ in 0..warmup_runs {
            self.dispatch_clean(scene, false);
        }

        let Some(_) = self.timestamps.as_ref() else {
            // No timestamp support: still run the dispatches so the workload is
            // identical, but report nothing rather than a CPU-side guess.
            for _ in 0..timed_runs {
                self.dispatch_clean(scene, false);
            }
            return TimingSummary::default();
        };

        let mut samples = Vec::with_capacity(timed_runs as usize);
        for _ in 0..timed_runs {
            if let Some(ms) = self.dispatch_clean(scene, true) {
                samples.push(ms);
            }
        }

        if samples.is_empty() {
            return TimingSummary::default();
        }

        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let variance = samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n;
        let min = samples.iter().copied().fold(f64::INFINITY, f64::min);

        TimingSummary {
            mean_ms: mean,
            min_ms: min,
            stddev_ms: variance.sqrt(),
            valid: true,
        }
    }

    /// One clean dispatch. Returns the GPU-side duration in ms when timed.
    fn dispatch_clean(&self, scene: &SceneUpload, timed: bool) -> Option<f64> {
        let rig = self.timestamps.as_ref().filter(|_| timed);

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Clean Pass"),
        });

        {
            let timestamp_writes = rig.map(|rig| wgpu::ComputePassTimestampWrites {
                query_set: &rig.query_set,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Clean Traversal"),
                timestamp_writes,
            });
            pass.set_pipeline(&self.clean_pipeline);
            pass.set_bind_group(0, &scene.clean_bind_group, &[]);
            let (gx, gy) = self.workgroups();
            pass.dispatch_workgroups(gx, gy, 1);
        }

        if let Some(rig) = rig {
            encoder.resolve_query_set(&rig.query_set, 0..2, &rig.resolve_buffer, 0);
            encoder.copy_buffer_to_buffer(
                &rig.resolve_buffer,
                0,
                &rig.staging_buffer,
                0,
                rig.staging_buffer.size(),
            );
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.wait();

        let rig = rig?;
        let bytes = self.read_back(&rig.staging_buffer);
        let ticks: &[u64] = bytemuck::cast_slice(&bytes);
        let (start, end) = (ticks[0], ticks[1]);
        rig.staging_buffer.unmap();

        // A wrapped or unwritten pair would produce nonsense; drop it rather than
        // fold a negative duration into the mean.
        if end <= start {
            return None;
        }

        Some((end - start) as f64 * rig.period_ns as f64 / 1.0e6)
    }

    fn workgroups(&self) -> (u32, u32) {
        (self.width.div_ceil(8), self.height.div_ceil(8))
    }

    fn storage_init(&self, label: &str, contents: &[u8]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::STORAGE,
        })
    }

    fn wait(&self) {
        self.device
            .poll(PollType::Wait { submission_index: None, timeout: None })
            .expect("GPU was not polled");
    }

    /// Maps `buffer` and copies its contents out. The caller must `unmap()` once
    /// it is done with the returned bytes.
    fn read_back(&self, buffer: &wgpu::Buffer) -> Vec<u8> {
        let slice = buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.wait();
        pollster::block_on(rx).expect("map callback dropped").expect("buffer map failed");

        let view = slice.get_mapped_range();
        view.to_vec()
    }
}

fn reduce_counters(records: &[GpuRayCounters]) -> CounterSummary {
    let mut summary = CounterSummary::default();

    // Per-pixel mean cost, used for the distribution stats. Pixels that issued no
    // traversal at all (a primary ray that missed the root box) are excluded so
    // they cannot drag the percentiles toward zero.
    let mut node_per_ray: Vec<f64> = Vec::with_capacity(records.len());
    let mut prim_per_ray: Vec<f64> = Vec::with_capacity(records.len());

    for record in records {
        summary.total_node_visits += record.node_visits as u64;
        summary.total_interior_visits += record.interior_visits as u64;
        summary.total_prim_tests += record.prim_tests as u64;
        summary.total_rays += record.ray_count as u64;
        summary.incomplete_traversals += record.incomplete as u64;

        if record.ray_count > 0 {
            let rays = record.ray_count as f64;
            node_per_ray.push(record.node_visits as f64 / rays);
            prim_per_ray.push(record.prim_tests as f64 / rays);
        }
    }

    if summary.total_rays > 0 {
        let rays = summary.total_rays as f64;
        summary.node_visits_per_ray = summary.total_node_visits as f64 / rays;
        summary.interior_visits_per_ray = summary.total_interior_visits as f64 / rays;
        summary.prim_tests_per_ray = summary.total_prim_tests as f64 / rays;
    }

    node_per_ray.sort_by(f64::total_cmp);
    prim_per_ray.sort_by(f64::total_cmp);

    summary.node_visits_max = node_per_ray.last().copied().unwrap_or(0.0);
    summary.prim_tests_max = prim_per_ray.last().copied().unwrap_or(0.0);
    summary.node_visits_p95 = percentile(&node_per_ray, 0.95);
    summary.prim_tests_p95 = percentile(&prim_per_ray, 0.95);

    summary
}

/// Nearest-rank percentile over a pre-sorted slice.
fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (fraction * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

fn create_bind_group_layout(device: &wgpu::Device, instrumented: bool) -> wgpu::BindGroupLayout {
    fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    let mut entries = vec![
        storage(0, true),  // rays
        storage(1, false), // output colors
        storage(2, true),  // spheres
        storage(3, true),  // triangles
        storage(4, true),  // planes
        wgpu::BindGroupLayoutEntry {
            binding: 5,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        storage(6, true), // bvh nodes
        storage(7, true), // bvh indices
    ];

    if instrumented {
        entries.push(storage(8, false)); // per-ray counters
    }

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(if instrumented {
            "Harness Instrumented Layout"
        } else {
            "Harness Clean Layout"
        }),
        entries: &entries,
    })
}

fn create_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    variant: ShaderVariant,
    label: &str,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(shaders::compose(variant).into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[layout],
        immediate_size: 0,
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}
