use crate::bvh::flatten_bvh_for_gpu;
use crate::gpu_types::{Counts, GpuBVHNode, GpuColor, GpuPlane, GpuRay, GpuSphere, GpuTriangle};
use crate::model::Mesh;
use crate::scene::Scene;
use crate::shaders::{self, ShaderVariant};
use crate::window::Canvas;
use wgpu::util::DeviceExt;

/// Builds the compute pipeline for the requested shader variant.
///
/// The studio uses the instrumented variant so its live cost views have counters
/// to read; it costs frame rate, which is the right trade for a tool whose whole
/// job is showing where the cost is.
pub fn setup_compute_pipeline_with(canvas: &mut Canvas, scene: &Scene, variant: ShaderVariant) {
    let instrumented = variant == ShaderVariant::Instrumented;
    let shader_source = shaders::compose(variant);

    let shader = canvas.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Raytrace Compute Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let (gpu_spheres, gpu_triangles, gpu_planes) = extract_scene_data(scene);
    let (bvh_nodes, bvh_indices) = build_scene_bvh(scene, &gpu_triangles);

    let counts = Counts {
        sphere_count: gpu_spheres.len() as u32,
        triangle_count: gpu_triangles.len() as u32,
        plane_count: gpu_planes.len() as u32,
        width: canvas.width(),
        height: canvas.height(),
        frame_number: canvas.sample_count,
        bvh_node_count: bvh_nodes.len() as u32,
        bvh_index_count: bvh_indices.len() as u32,
        // One sample per dispatch; the interactive path accumulates across frames.
        // The studio overwrites these live through `Canvas::write_counts`.
        samples: 1,
        rng_seed: 0,
        display_mode: 0,
        palette: 0,
        heat_scale: 40.0,
        heat_mix: 0.0,
        max_bounces: 10,
        _pad0: 0,
    };

    println!("Creating counts buffer:");
    println!("  spheres: {}", counts.sphere_count);
    println!("  triangles: {}", counts.triangle_count);
    println!("  planes: {}", counts.plane_count);
    println!("  width: {}", counts.width);
    println!("  height: {}", counts.height);
    println!("  BVH nodes: {}", counts.bvh_node_count);
    println!("  BVH indices: {}", counts.bvh_index_count);

    let sphere_buffer = canvas.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Sphere Buffer"),
        contents: bytemuck::cast_slice(&gpu_spheres),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let triangle_buffer = canvas.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Triangle Buffer"),
        contents: bytemuck::cast_slice(&gpu_triangles),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let plane_buffer = canvas.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Plane Buffer"),
        contents: bytemuck::cast_slice(&gpu_planes),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let bvh_node_buffer = canvas.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("BVH Node Buffer"),
        contents: bytemuck::cast_slice(&bvh_nodes),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let bvh_index_buffer = canvas.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("BVH Index Buffer"),
        contents: bytemuck::cast_slice(&bvh_indices),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let pixel_count = canvas.pixel_count() as usize;

    let ray_buffer = canvas.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("Ray Buffer"),
        size: (pixel_count * std::mem::size_of::<GpuRay>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let color_buffer = canvas.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("Color Output Buffer"),
        size: (pixel_count * std::mem::size_of::<GpuColor>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = canvas.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: (pixel_count * std::mem::size_of::<GpuColor>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let counts_buffer = canvas.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Counts Buffer"),
        contents: bytemuck::cast_slice(&[counts]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Only the instrumented variant declares binding 8, so the clean pipeline
    // never allocates this.
    let counter_buffer = instrumented.then(|| {
        canvas.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Live Counter Buffer"),
            size: (pixel_count * std::mem::size_of::<crate::gpu_types::GpuRayCounters>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    });

    let mut layout_entries = vec![
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
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
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
    ];

    if instrumented {
        layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 8,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }

    let bind_group_layout = canvas.device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Compute Bind Group Layout"),
        entries: &layout_entries,
    });

    let mut bind_entries = vec![
        wgpu::BindGroupEntry { binding: 0, resource: ray_buffer.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: color_buffer.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: sphere_buffer.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: triangle_buffer.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: plane_buffer.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 5, resource: counts_buffer.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 6, resource: bvh_node_buffer.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 7, resource: bvh_index_buffer.as_entire_binding() },
    ];

    if let Some(buffer) = &counter_buffer {
        bind_entries.push(wgpu::BindGroupEntry { binding: 8, resource: buffer.as_entire_binding() });
    }

    let bind_group = canvas.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Compute Bind Group"),
        layout: &bind_group_layout,
        entries: &bind_entries,
    });

    let pipeline_layout = canvas.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        immediate_size: 0,
    });

    let pipeline = canvas.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Raytrace Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    canvas.compute_pipeline = Some(pipeline);
    canvas.compute_bind_group = Some(bind_group);
    canvas.sphere_buffer = Some(sphere_buffer);
    canvas.triangle_buffer = Some(triangle_buffer);
    canvas.plane_buffer = Some(plane_buffer);
    canvas.ray_buffer = Some(ray_buffer);
    canvas.color_buffer = Some(color_buffer);
    canvas.staging_buffer = Some(staging_buffer);
    canvas.counts_buffer = Some(counts_buffer);
    canvas.counter_buffer = counter_buffer;
}

fn extract_scene_data(scene: &Scene) -> (Vec<GpuSphere>, Vec<GpuTriangle>, Vec<GpuPlane>) {
    let primitives = scene.export_gpu_data();

    let mut spheres = primitives.0;
    let mut triangles = primitives.1;
    let mut planes = primitives.2;

    if spheres.is_empty() {
        spheres.push(GpuSphere {
            center: [0.0, 0.0, 0.0],
            radius: 0.0,
            albedo: [0.0, 0.0, 0.0],
            emission: 0.0,
            metallic: 0.0,
            roughness: 0.0,
            _padding: [0.0, 0.0],
        });
    }

    if triangles.is_empty() {
        triangles.push(GpuTriangle {
            v0: [0.0, 0.0, 0.0],
            _pad0: 0.0,
            v1: [0.0, 0.0, 0.0],
            _pad1: 0.0,
            v2: [0.0, 0.0, 0.0],
            _pad2: 0.0,
            albedo: [0.0, 0.0, 0.0],
            emission: 0.0,
            metallic: 0.0,
            roughness: 0.0,
            _padding: [0.0, 0.0],
        });
    }

    if planes.is_empty() {
        planes.push(GpuPlane {
            center: [0.0, 0.0, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0, 0.0],
            width: 0.0,
            length: 0.0,
            _pad2: [0.0, 0.0],
            albedo: [0.0, 0.0, 0.0, 0.0],
            emission: 0.0,
            metallic: 0.0,
            roughness: 0.0,
            _pad3: 0.0,
        });
    }

    (spheres, triangles, planes)
}

fn build_scene_bvh(scene: &Scene, triangles: &[GpuTriangle]) -> (Vec<GpuBVHNode>, Vec<u32>) {
    use crate::objects::Triangle;
    use crate::material::Material;
    use crate::color::Color;
    use glam::Vec3;

    let cpu_triangles: Vec<Triangle> = triangles.iter().map(|t| {
        Triangle::new(
            Vec3::new(t.v0[0], t.v0[1], t.v0[2]),
            Vec3::new(t.v1[0], t.v1[1], t.v1[2]),
            Vec3::new(t.v2[0], t.v2[1], t.v2[2]),
            Material::new(
                Color::new(t.albedo[0], t.albedo[1], t.albedo[2]),
                t.emission,
                t.metallic,
                t.roughness,
            )
        )
    }).collect();

    let mut all_nodes = Vec::new();
    let mut all_indices = Vec::new();

    for object in scene.get_objects() {
        if let Some(mesh) = object.as_any().downcast_ref::<Mesh>() {
            let bvh = mesh.bvh.as_ref().unwrap();
            let (nodes, indices) = flatten_bvh_for_gpu(&bvh, &cpu_triangles);
            all_nodes.extend(nodes);
            all_indices.extend(indices);
        }
    }

    if all_nodes.is_empty() {
        all_nodes.push(GpuBVHNode {
            min: [0.0; 3],
            _pad0: 0.0,
            max: [0.0; 3],
            _pad1: 0.0,
            left_first: 0,
            right_count: 0,
            is_leaf: 1,
            _pad2: 0,
        });
        all_indices.push(0);
    }

    (all_nodes, all_indices)
}