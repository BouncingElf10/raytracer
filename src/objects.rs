use glam::Vec3;
use crate::material::Material;
use crate::ray::Ray;

pub trait Hittable {
    fn hit(&self, ray: &Ray) -> HitInfo;
    fn set_material(&mut self, material: Material);
}

pub struct HitInfo {
    pub has_hit: bool,
    pub t: f64,
    pub pos: Vec3,
    pub sent_ray: Ray,
    pub normal: Vec3,
    pub material: Material,
}

pub struct Sphere {
    pos: Vec3,
    radius: f32,
    material: Material,
}

impl Sphere {
    pub fn new(pos: Vec3, radius: f32, material: Material) -> Self {
        Self { pos, radius, material }
    }
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray) -> HitInfo {
        let oc = ray.origin() - self.pos;
        let a = ray.direction().dot(ray.direction());
        let b = 2.0 * oc.dot(ray.direction());
        let c = oc.dot(oc) - self.radius * self.radius;
        let discriminant = b * b - 4.0 * a * c;

        if discriminant > 0.0 {
            let sqrt_d = discriminant.sqrt();
            let mut t = (-b - sqrt_d) / (2.0 * a);

            if t < 0.001 {
                t = (-b + sqrt_d) / (2.0 * a);
            }

            if t > 0.001 {
                let hit_pos = ray.at(t);
                let normal = (hit_pos - self.pos).normalize();

                return HitInfo {
                    has_hit: true,
                    t: t as f64,
                    pos: hit_pos,
                    sent_ray: ray.clone(),
                    normal,
                    material: self.material
                };
            }
        }

        HitInfo {
            has_hit: false,
            t: f64::INFINITY,
            pos: Vec3::ZERO,
            sent_ray: ray.clone(),
            normal: Vec3::ZERO,
            material: self.material
        }
    }

    fn set_material(&mut self, material: Material) {
        self.material = material;
    }
}