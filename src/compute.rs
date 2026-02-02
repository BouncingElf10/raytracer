use crate::gpu_types::{GpuColor, GpuPlane, GpuRay, GpuSphere, GpuTriangle, GpuBVHNode};
use crate::scene::Scene;
use crate::window::Canvas;
use crate::bvh::{construct_bvh, flatten_bvh_for_gpu};
use crate::model::Mesh;
use crate::objects::Hittable;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Counts {
    sphere_count: u32,
    triangle_count: u32,
    plane_count: u32,
    bvh_node_count: u32,
    bvh_index_count: u32,
    width: u32,
    height: u32,
    frame_number: u32,
}

pub fn setup_compute_pipeline(canvas: &mut Canvas, scene: &Scene) {
    let shader_source = format!(
        "{}\n{}\n{}\n{}",
        include_str!("shaders/types.wgsl"),
        include_str!("shaders/hit.wgsl"),
        include_str!("shaders/random.wgsl"),
        include_str!("shaders/raytracer.wgsl"),
    );

    let shader = canvas.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Raytrace Compute Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let (gpu_spheres, mut gpu_triangles, gpu_planes) = extract_scene_data(scene);
    let (bvh_nodes, bvh_indices) = build_scene_bvh(scene, &gpu_triangles);

    let counts = Counts {
        sphere_count: gpu_spheres.len() as u32,
        triangle_count: gpu_triangles.len() as u32,
        plane_count: gpu_planes.len() as u32,
        bvh_node_count: bvh_nodes.len() as u32,
        bvh_index_count: bvh_indices.len() as u32,
        width: canvas.width(),
        height: canvas.height(),
        frame_number: canvas.sample_count,
    };

    println!("Creating buffers:");
    println!("  spheres: {}", counts.sphere_count);
    println!("  triangles: {}", counts.triangle_count);
    println!("  planes: {}", counts.plane_count);
    println!("  BVH nodes: {}", counts.bvh_node_count);
    println!("  BVH indices: {}", counts.bvh_index_count);
    println!("  width: {}", counts.width);
    println!("  height: {}", counts.height);

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

    let bind_group_layout = canvas.device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Compute Bind Group Layout"),
        entries: &[
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
        ],
    });

    let bind_group = canvas.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Compute Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: ray_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: color_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: sphere_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: triangle_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: plane_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: counts_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: bvh_node_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: bvh_index_buffer.as_entire_binding() },
        ],
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
    let mut all_nodes = Vec::new();
    let mut all_indices = Vec::new();

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

    for object in scene.get_objects() {
        if let Some(mesh) = object.as_any().downcast_ref::<Mesh>() {
            let bvh = construct_bvh(mesh);
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