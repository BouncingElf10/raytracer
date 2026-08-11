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
        let right = world_up.cross(forward).normalize();
        let up = forward.cross(right).normalize();

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

    /// Projects a world-space segment, clipping it against the near plane first.
    ///
    /// `world_to_screen` rejects any point behind the camera, so an edge with one
    /// endpoint behind it would be dropped entirely and the wireframe box would
    /// visibly fall apart as you walk into it. Clipping the segment to the
    /// visible half-space keeps the surviving portion drawable.
    pub fn project_segment(&self, a: Vec3, b: Vec3) -> Option<((i32, i32), (i32, i32))> {
        const NEAR: f32 = 0.01;

        let forward = self.ray.direction().normalize();
        let origin = self.ray.origin();

        // Signed distance in front of the camera.
        let depth_a = (a - origin).dot(forward);
        let depth_b = (b - origin).dot(forward);

        if depth_a < NEAR && depth_b < NEAR {
            return None;
        }

        let (a, b) = if depth_a < NEAR {
            // Slide `a` forward to where the segment crosses the near plane.
            let t = (NEAR - depth_a) / (depth_b - depth_a);
            (a + (b - a) * t, b)
        } else if depth_b < NEAR {
            let t = (NEAR - depth_b) / (depth_a - depth_b);
            (a, b + (a - b) * t)
        } else {
            (a, b)
        };

        Some((self.world_to_screen(a)?, self.world_to_screen(b)?))
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