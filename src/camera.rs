use crate::ray::Ray;

pub struct Camera {
    width: u32,
    height: u32,
    ray: Ray
}

impl Camera {
    pub fn new(width: u32, height: u32, ray: Ray) -> Self {
        Self { width, height, ray }
    }

    pub fn for_each_pixel<F>(&self, mut f: F) where F: FnMut(u32, u32) {
        for y in 0..self.height {
            for x in 0..self.width {
                f(x, y);
            }
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }
    
    pub fn width(&self) -> u32 { self.width }
    pub fn height(&self) -> u32 { self.height }
    pub fn ray(&self) -> Ray { self.ray.clone() }
    pub fn set_ray(&mut self, ray: Ray) { self.ray = ray; }
}