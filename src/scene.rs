use std::collections::LinkedList;
use glam::Vec3;
use crate::objects::{Hittable, Sphere};

pub struct Scene {
    objects: Vec<Box<dyn Hittable>>
}

impl Scene {
    pub fn new() -> Self {
        Self { objects: Vec::new() }
    }
    
    pub fn add_object(&mut self, object: Box<dyn Hittable>) {
        self.objects.push(object);
    }
    
    pub fn get_objects(&self) -> &Vec<Box<dyn Hittable>> {
        &self.objects
    }
}

pub fn create_scene() -> Scene {
    let mut scene = Scene::new();
    let sphere = Sphere::new(Vec3::new(0.0, 0.0, -2.0), 0.5);
    
    scene.add_object(Box::new(sphere));
    scene
}