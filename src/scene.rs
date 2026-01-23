use std::collections::LinkedList;
use glam::Vec3;
use crate::color::Color;
use crate::material::Material;
use crate::objects::{Hittable, Sphere};

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
    let sphere1 = Sphere::new(Vec3::new(0.0, 0.0, 2.0), 0.5, Material::new(Color::new(1.0, 1.0, 1.0), 0.5, 0.5, 1.0));
    let sphere2 = Sphere::new(Vec3::new(0.0, 2.0, 2.0), 1.0, Material::new(Color::new(1.0, 0.5, 0.5), 0.5, 0.5, 0.0));
    let sphere3 = Sphere::new(Vec3::new(0.0, -6.0, 2.0), 5.0, Material::new(Color::new(0.5, 0.5, 1.0), 0.5, 0.5, 0.0));

    scene
        .add_object(Box::new(sphere1))
        .add_object(Box::new(sphere2))
        .add_object(Box::new(sphere3));
    scene
}