use glam::Vec3;
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

    pub fn world_to_screen(&self, p: Vec3) -> Option<(i32, i32)> {
        let forward = self.ray.direction().normalize();

        let world_up = if forward.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
        let right = forward.cross(world_up).normalize();
        let up = right.cross(forward).normalize();

        let rel = p - self.ray.origin();

        let x = rel.dot(right);
        let y = rel.dot(up);
        let z = rel.dot(forward);

        if z <= 0.0 {
            return None;
        }

        let fov_y = 90.0_f32.to_radians();
        let f = 1.0 / (fov_y * 0.5).tan();
        let aspect = self.width as f32 / self.height as f32;

        let ndc_x = (x * f / aspect) / z;
        let ndc_y = (y * f) / z;

        // if ndc_x.abs() > 10.0 || ndc_y.abs() > 10.0 {
        //     return None;
        // }

        let sx = ((ndc_x + 1.0) * 0.5 * self.width as f32) as i32;
        let sy = ((1.0 - ndc_y) * 0.5 * self.height as f32) as i32;

        Some((sx, sy))
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