use crate::color::Color;
use crate::importer::import_obj;
use crate::material::Material;
use crate::objects::{Hittable, Plane, Sphere, Triangle};
use glam::Vec3;
use crate::gpu_types::{GpuPlane, GpuSphere, GpuTriangle};
use crate::model::Mesh;
use crate::profiler::{profiler_start, profiler_stop};

pub struct Scene {
    objects: Vec<Box<dyn Hittable>>
}

impl Scene {
    pub fn new() -> Self {
        Self { objects: Vec::new() }
    }
    
    pub fn add_object(&mut self, object: Box<dyn Hittable>) -> &mut Self {
        self.objects.push(object);
        self
    }

    pub fn export_gpu_data(&self) -> (Vec<GpuSphere>, Vec<GpuTriangle>, Vec<GpuPlane>) {
        profiler_start("export gpu data");
        let mut spheres = Vec::new();
        let mut triangles = Vec::new();
        let mut planes = Vec::new();

        for obj in self.get_objects() {
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
                triangles.push(triangle_to_gpu_triangle(triangle));
            }
            else if let Some(mesh) = obj.as_any().downcast_ref::<Mesh>() {
                for tri in mesh.get_triangles() {
                    triangles.push(triangle_to_gpu_triangle(&tri));
                }
            }
        }
        profiler_stop("export gpu data");
        (spheres, triangles, planes)
    }

    pub fn get_objects(&self) -> &[Box<dyn Hittable>] {
        &self.objects
    }
}

pub fn triangle_to_gpu_triangle(tri: &Triangle) -> GpuTriangle {
    let v0 = tri.v0();
    let v1 = tri.v1();
    let v2 = tri.v2();
    let mat = tri.material();
    let albedo = mat.albedo();

    GpuTriangle {
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
    }
}

pub fn create_scene() -> Scene {
    profiler_start("create scene");
    profiler_start("load other models");
    let mut scene = Scene::new();

    // Floor
    scene.add_object(Box::new(Plane::new(
        Vec3::new(0.0, -2.5, 0.0), Vec3::new(0.0, 1.0, 0.0),
        5.0, 5.0,
        Material { albedo: Color::new(0.8, 0.8, 0.8), roughness: 1.0, metallic: 0.0, emission: 0.0 },
    )));

    // Ceiling
    scene.add_object(Box::new(Plane::new(
        Vec3::new(0.0, 2.5, 0.0), Vec3::new(0.0, -1.0, 0.0),
        5.0, 5.0,
        Material { albedo: Color::new(0.8, 0.8, 0.8), roughness: 1.0, metallic: 0.0, emission: 0.0 },
    )));

    // Back wall
    scene.add_object(Box::new(Plane::new(
        Vec3::new(0.0, 0.0, -2.5), Vec3::new(0.0, 0.0, 1.0),
        5.0, 5.0,
        Material { albedo: Color::new(0.8, 0.8, 0.8), roughness: 1.0, metallic: 0.0, emission: 0.0 },
    )));

    // Left wall
    scene.add_object(Box::new(Plane::new(
        Vec3::new(-2.5, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0),
        5.0, 5.0,
        Material { albedo: Color::new(0.9, 0.2, 0.2), roughness: 1.0, metallic: 0.0, emission: 0.0 },
    )));

    // Right wall
    scene.add_object(Box::new(Plane::new(
        Vec3::new(2.5, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0),
        5.0, 5.0,
        Material { albedo: Color::new(0.2, 0.9, 0.2), roughness: 1.0, metallic: 0.0, emission: 0.0 },
    )));

    // Ceiling light
    scene.add_object(Box::new(Plane::new(
        Vec3::new(0.0, 2.499, 0.0), Vec3::new(0.0, -1.0, 0.0),
        3.0, 3.0,
        Material { albedo: Color::new(1.0, 1.0, 1.0), roughness: 0.0, metallic: 0.0, emission: 1.0 },
    )));

    // Spheres
    // scene.add_object(Box::new(Sphere::new(
    //     Vec3::new(-1.5, -1.5, 0.0), 1.0,
    //     Material { albedo: Color::new(0.7, 0.7, 0.7), roughness: 1.0, metallic: 0.0, emission: 0.0 },
    // )));
    //
    // scene.add_object(Box::new(Sphere::new(
    //     Vec3::new(0.0, -1.5, -1.5), 1.0,
    //     Material { albedo: Color::new(0.8, 0.8, 0.9), roughness: 0.0, metallic: 1.0, emission: 0.0 },
    // )));
    //
    // scene.add_object(Box::new(Sphere::new(
    //     Vec3::new(1.5, -1.5, 0.0), 1.0,
    //     Material { albedo: Color::new(0.9, 0.6, 0.2), roughness: 0.6, metallic: 0.0, emission: 0.0 },
    // )));

    profiler_stop("load other models");
    profiler_start("load mesh");

    let mut mesh = import_obj("src/models/standford_dragon.obj");
    mesh.set_material(Material::new(Color::new(0.9, 0.9, 0.9), 1.0, 0.0, 0.0));
    mesh.position = Vec3::new(0.0, -2.5, 0.0);
    mesh.scale = 2.0;
    mesh.rotation = Vec3::new(0.0, 20.0f32.to_radians(), 0.0);
    scene.add_object(Box::new(mesh));

    profiler_stop("load mesh");
    profiler_stop("create scene");
    scene
}