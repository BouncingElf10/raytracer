use glam::Vec3;
use crate::color::Color;
use crate::material::Material;
use crate::objects::{Hittable, Plane, Sphere};

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
    
    pub fn get_objects(&self) -> &Vec<Box<dyn Hittable>> {
        &self.objects
    }
}

pub fn create_scene() -> Scene {
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
        Vec3::new(0.0, 2.4, 0.0), Vec3::new(0.0, -1.0, 0.0),
        3.0, 3.0,
        Material { albedo: Color::new(1.0, 1.0, 1.0), roughness: 0.0, metallic: 0.0, emission: 1.0 },
    )));

    // Spheres
    scene.add_object(Box::new(Sphere::new(
        Vec3::new(-1.5, -1.5, 0.0), 1.0,
        Material { albedo: Color::new(0.7, 0.7, 0.7), roughness: 1.0, metallic: 0.0, emission: 0.0 },
    )));

    scene.add_object(Box::new(Sphere::new(
        Vec3::new(0.0, -1.5, -1.5), 1.0,
        Material { albedo: Color::new(0.8, 0.8, 0.9), roughness: 0.05, metallic: 1.0, emission: 0.0 },
    )));

    scene.add_object(Box::new(Sphere::new(
        Vec3::new(1.5, -1.5, 0.0), 1.0,
        Material { albedo: Color::new(0.9, 0.6, 0.2), roughness: 0.6, metallic: 0.0, emission: 0.0 },
    )));

    scene
}