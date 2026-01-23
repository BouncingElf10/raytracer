use crate::camera::Camera;
use glam::Vec3;

#[derive(Clone)]
pub struct Ray {
    origin: Vec3,
    direction: Vec3
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction }
    }
    pub fn origin(&self) -> Vec3 { self.origin }
    pub fn direction(&self) -> Vec3 { self.direction }

    pub fn rotate_x(&mut self, angle: f32) {
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let y = self.direction[1];
        let z = self.direction[2];
        self.direction[1] = y * cos_a - z * sin_a;
        self.direction[2] = y * sin_a + z * cos_a;
    }

    pub fn rotate_y(&mut self, angle: f32) {
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let x = self.direction[0];
        let z = self.direction[2];
        self.direction[0] = x * cos_a + z * sin_a;
        self.direction[2] = -x * sin_a + z * cos_a;
    }

    pub fn move_origin(&mut self, distance: f32) { self.origin += self.direction * distance; }
    pub fn move_origin_along_direction(&mut self, distance: f32) { self.origin += self.direction * distance; }
    pub fn at(&self, t: f32) -> Vec3 { self.origin + self.direction * t }
}

pub fn get_ray_from_screen(camera: &Camera, x: u32, y: u32) -> Ray {
    let fov_y: f32 = 90.0f32.to_radians();
    let aspect = camera.width() as f32 / camera.height() as f32;

    let ndc_x = (x as f32 + 0.5) / camera.width() as f32;
    let ndc_y = (y as f32 + 0.5) / camera.height() as f32;

    let screen_x = (2.0 * ndc_x - 1.0) * aspect * (fov_y * 0.5).tan();
    let screen_y = (1.0 - 2.0 * ndc_y) * (fov_y * 0.5).tan();

    let forward = camera.ray().direction().normalize();
    let right = Vec3::new(0.0, 1.0, 0.0).cross(forward).normalize();
    let up = forward.cross(right);

    let direction = (forward + right * screen_x + up * screen_y).normalize();

    Ray::new(camera.ray().origin(), direction)
}
