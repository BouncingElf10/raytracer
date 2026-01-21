use glam::Vec3;
use crate::ray::Ray;

pub trait Hittable {
    fn hit(&self, ray: &Ray) -> HitInfo;
}

pub struct HitInfo {
    pub has_hit: bool,
    pub pos: Vec3,
    pub sent_ray: Ray,
    pub normal_ray: Ray
}

pub struct Sphere {
    pos: Vec3,
    radius: f32,
}

impl Sphere {
    pub fn new(pos: Vec3, radius: f32) -> Self {
        Self { pos, radius }
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
            let t = (-b - discriminant.sqrt()) / (2.0 * a);
            let hit_pos = ray.at(t);
            let normal = (hit_pos - self.pos).normalize();

            HitInfo {
                has_hit: true,
                pos: hit_pos,
                sent_ray: ray.clone(),
                normal_ray: Ray::new(hit_pos, normal),
            }
        } else {
            HitInfo {
                has_hit: false,
                pos: Vec3::ZERO,
                sent_ray: ray.clone(),
                normal_ray: Ray::new(Vec3::ZERO, Vec3::ZERO),
            }
        }
    }
}