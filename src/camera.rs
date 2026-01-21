use crate::ray::Ray;

pub struct Camera {
    width: usize,
    height: usize,
    ray: Ray
}

impl Camera {
    pub fn new(width: usize, height: usize, ray: Ray) -> Self {
        Self { width, height, ray }
    }

    pub fn for_each_pixel<F>(&self, mut f: F) where F: FnMut(usize, usize) {
        for y in 0..self.height {
            for x in 0..self.width {
                f(x, y);
            }
        }
    }
    
    pub fn width(&self) -> usize { self.width }
    pub fn height(&self) -> usize { self.height }
    pub fn ray(&self) -> Ray { self.ray.clone() }
    pub fn set_ray(&mut self, ray: Ray) { self.ray = ray; }
}