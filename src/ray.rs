use crate::camera::Camera;
use glam::Vec3;
use rand::random;

#[derive(Clone, Debug, Copy)]
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
    pub fn reflect(&self, normal: Vec3) -> Self {
        Self::new(self.origin(), self.direction() - 2.0 * normal.dot(self.direction()) * normal)
    }

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

    pub fn rotate_z(&mut self, angle: f32) {
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let x = self.direction[0];
        let y = self.direction[1];
        self.direction[0] = x * cos_a - y * sin_a;
        self.direction[1] = x * sin_a + y * cos_a;
    }

    pub fn dot(&self) -> f32 { self.direction.dot(self.direction) }
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

pub fn random_cosine_hemisphere(normal: Vec3) -> Vec3 {
    let r1 = random::<f32>();
    let r2 = random::<f32>();

    let phi = 2.0 * std::f32::consts::PI * r1;
    let x = phi.cos() * r2.sqrt();
    let y = phi.sin() * r2.sqrt();
    let z = (1.0 - r2).sqrt();

    let local = Vec3::new(x, y, z);

    let up = if normal.z.abs() < 0.999 {
        Vec3::Z
    } else {
        Vec3::X
    };

    let tangent = normal.cross(up).normalize();
    let bitangent = normal.cross(tangent);

    (tangent * local.x +
        bitangent * local.y +
        normal * local.z).normalize()
}

pub fn lerp(ray1: &Ray, ray2: &Ray, t: f32) -> Ray {
    Ray::new(ray1.origin() + (ray2.origin() - ray1.origin()) * t, ray1.direction() + (ray2.direction() - ray1.direction()) * t)
}

