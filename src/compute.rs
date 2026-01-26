use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::util::DeviceExt;
use crate::gpu_types::{vec4_to_vec3, GpuHitInfo, GpuPlane, GpuRay, GpuSphere, GpuTriangle};
use crate::material::Material;
use crate::model::Mesh;
use crate::objects::{Hittable, HitInfo, Sphere, Plane, Triangle};
use crate::ray::Ray;
use crate::scene::Scene;
use crate::window::Canvas;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Counts {
    sphere_count: u32,
    triangle_count: u32,
    plane_count: u32,
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub fn setup_compute_pipeline(canvas: &mut Canvas, scene: &Scene) {
    let shader = canvas.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Raytrace Compute Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/raytracer.wgsl").into()),
    });

    let (gpu_spheres, gpu_triangles, gpu_planes) = extract_scene_data(scene);

    let counts = Counts {
        sphere_count: gpu_spheres.len() as u32,
        triangle_count: gpu_triangles.len() as u32,
        plane_count: gpu_planes.len() as u32,
        width: canvas.width(),
        height: canvas.height(),
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };

    println!("Creating counts buffer:");
    println!("  spheres: {}", counts.sphere_count);
    println!("  triangles: {}", counts.triangle_count);
    println!("  planes: {}", counts.plane_count);
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

    let pixel_count = canvas.pixel_count() as usize;

    let ray_buffer = canvas.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("Ray Buffer"),
        size: (pixel_count * std::mem::size_of::<GpuRay>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let hit_buffer = canvas.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("Hit Buffer"),
        size: (pixel_count * std::mem::size_of::<GpuHitInfo>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = canvas.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: (pixel_count * std::mem::size_of::<GpuHitInfo>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let counts_buffer = canvas.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Counts Buffer"),
        contents: bytemuck::cast_slice(&[counts]),
        usage: wgpu::BufferUsages::UNIFORM,
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
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
        ],
    });

    let bind_group = canvas.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Compute Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: sphere_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: triangle_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: plane_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: ray_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: hit_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: counts_buffer.as_entire_binding() },
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
    canvas.hit_buffer = Some(hit_buffer);
    canvas.staging_buffer = Some(staging_buffer);
    canvas.counts_buffer = Some(counts_buffer);
}

fn extract_scene_data(scene: &Scene) -> (Vec<GpuSphere>, Vec<GpuTriangle>, Vec<GpuPlane>) {
    let mut spheres = Vec::new();
    let mut triangles = Vec::new();
    let mut planes = Vec::new();

    for obj in scene.get_objects() {
        if let Some(sphere) = obj.as_any().downcast_ref::<Sphere>() {
            let center = sphere.center();
            let mat = sphere.material();
            let albedo = mat.albedo();

            spheres.push(GpuSphere {
                center: [center.x, center.y, center.z],
                radius: sphere.radius(),
                albedo: [albedo.r, albedo.g, albedo.b],
                emission: mat.emission(),
                metallic: mat.metallic(),
                roughness: mat.roughness(),
                _padding: [0.0, 0.0],
            });
        }
        else if let Some(plane) = obj.as_any().downcast_ref::<Plane>() {
            let center = plane.center();
            let normal = plane.normal();
            let mat = plane.material();
            let albedo = mat.albedo();

            println!("Extracting plane: center=({}, {}, {}), normal=({}, {}, {}), width={}, length={}",
                     center.x, center.y, center.z,
                     normal.x, normal.y, normal.z,
                     plane.width(), plane.length());

            planes.push(GpuPlane {
                center: [center.x, center.y, center.z, 0.0],
                normal: [normal.x, normal.y, normal.z, 0.0],
                width: plane.width(),
                length: plane.length(),
                _pad2: [0.0, 0.0],
                albedo: [albedo.r, albedo.g, albedo.b, 0.0],
                emission: mat.emission(),
                metallic: mat.metallic(),
                roughness: mat.roughness(),
                _pad3: 0.0,
            });
        }
        else if let Some(triangle) = obj.as_any().downcast_ref::<Triangle>() {
            let v0 = triangle.v0();
            let v1 = triangle.v1();
            let v2 = triangle.v2();
            let mat = triangle.material();
            let albedo = mat.albedo();

            triangles.push(GpuTriangle {
                v0: [v0.x, v0.y, v0.z],
                _pad0: 0.0,
                v1: [v1.x, v1.y, v1.z],
                _pad1: 0.0,
                v2: [v2.x, v2.y, v2.z],
                _pad2: 0.0,
                albedo: [albedo.r, albedo.g, albedo.b],
                emission: mat.emission(),
                metallic: mat.metallic(),
                roughness: mat.roughness(),
                _padding: [0.0, 0.0],
            });
        }
        else if let Some(mesh) = obj.as_any().downcast_ref::<Mesh>() {
            for tri in mesh.get_triangles() {
                let v0 = tri.v0();
                let v1 = tri.v1();
                let v2 = tri.v2();
                let mat = tri.material();
                let albedo = mat.albedo();

                triangles.push(GpuTriangle {
                    v0: [v0.x, v0.y, v0.z],
                    _pad0: 0.0,
                    v1: [v1.x, v1.y, v1.z],
                    _pad1: 0.0,
                    v2: [v2.x, v2.y, v2.z],
                    _pad2: 0.0,
                    albedo: [albedo.r, albedo.g, albedo.b],
                    emission: mat.emission(),
                    metallic: mat.metallic(),
                    roughness: mat.roughness(),
                    _padding: [0.0, 0.0],
                });
            }
        }
    }

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

    (spheres, triangles, planes)
}

pub fn convert_hit_info(gpu_hit: &GpuHitInfo, ray: &Ray) -> HitInfo {
    HitInfo {
        has_hit: gpu_hit.has_hit != 0,
        t: gpu_hit.t as f64,
        pos: vec4_to_vec3(gpu_hit.pos),
        sent_ray: ray.clone(),
        normal: vec4_to_vec3(gpu_hit.normal),
        material: Material {
            albedo: crate::color::Color::new(
                gpu_hit.albedo[0],
                gpu_hit.albedo[1],
                gpu_hit.albedo[2],
            ),
            emission: gpu_hit.emission,
            metallic: gpu_hit.metallic,
            roughness: gpu_hit.roughness,
        },
    }
}