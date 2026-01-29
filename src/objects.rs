use std::any::Any;
use glam::Vec3;
use crate::material::Material;
use crate::model::Mesh;
use crate::ray::Ray;

pub trait Hittable {
    fn hit(&self, ray: &Ray) -> HitInfo;
    fn set_material(&mut self, material: Material);
    fn as_any(&self) -> &dyn Any;
}
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
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
    #[allow(dead_code)]
    pub fn new(pos: Vec3, radius: f32, material: Material) -> Self {
        Self { pos, radius, material }
    }

    pub fn center(&self) -> Vec3 {
        self.pos
    }

    pub fn radius(&self) -> f32 {
        self.radius
    }

    pub fn material(&self) -> &Material {
        &self.material
    }
}

pub struct Plane {
    center: Vec3,
    normal: Vec3,
    width: f32,
    length: f32,
    material: Material,
}

impl Plane {
    pub fn new(center: Vec3, normal: Vec3, width: f32, length: f32, material: Material) -> Self {
        Self { center, normal, width, length, material }
    }

    pub fn center(&self) -> Vec3 {
        self.center
    }

    pub fn normal(&self) -> Vec3 {
        self.normal
    }

    pub fn width(&self) -> f32 {
        self.width
    }

    pub fn length(&self) -> f32 {
        self.length
    }

    pub fn material(&self) -> &Material {
        &self.material
    }
}

pub struct Triangle {
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    material: Material,
}

impl Triangle {
    pub fn new(v0: Vec3, v1: Vec3, v2: Vec3, material: Material) -> Self {
        Self { v0, v1, v2, material }
    }

    pub fn v0(&self) -> Vec3 {
        self.v0
    }

    pub fn v1(&self) -> Vec3 {
        self.v1
    }

    pub fn v2(&self) -> Vec3 {
        self.v2
    }

    pub fn get_vertices(&self) -> [Vec3; 3] {
        [self.v0, self.v1, self.v2]
    }

    pub fn material(&self) -> &Material {
        &self.material
    }
}

impl Hittable for Triangle {
    fn hit(&self, ray: &Ray) -> HitInfo {
        let eps = 1e-6;

        let edge1 = self.v1 - self.v0;
        let edge2 = self.v2 - self.v0;

        let h = ray.direction().cross(edge2);
        let a = edge1.dot(h);

        if a.abs() < eps {
            return no_hit(ray, self.material);
        }

        let f = 1.0 / a;
        let s = ray.origin() - self.v0;
        let u = f * s.dot(h);

        if !(0.0..=1.0).contains(&u) {
            return no_hit(ray, self.material);
        }

        let q = s.cross(edge1);
        let v = f * ray.direction().dot(q);

        if v < 0.0 || u + v > 1.0 {
            return no_hit(ray, self.material);
        }

        let t = f * edge2.dot(q);

        if t > 0.001 {
            let hit_pos = ray.at(t);
            let normal = edge1.cross(edge2).normalize();

            return HitInfo {
                has_hit: true,
                t: t as f64,
                pos: hit_pos,
                sent_ray: ray.clone(),
                normal,
                material: self.material,
            };
        }

        no_hit(ray, self.material)
    }

    fn set_material(&mut self, material: Material) {
        self.material = material;
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Hittable for Plane {
    fn hit(&self, ray: &Ray) -> HitInfo {
        let eps = 1e-6;

        let denom = self.normal.dot(ray.direction());
        if denom.abs() < eps {
            return no_hit(ray, self.material);
        }

        let t = (self.center - ray.origin()).dot(self.normal) / denom;
        if t <= 0.001 {
            return no_hit(ray, self.material);
        }

        let hit_pos = ray.at(t);
        let n = self.normal.normalize();

        let tangent = if n.x.abs() > 0.9 {
            Vec3::Y
        } else {
            Vec3::X
        };

        let u = n.cross(tangent).normalize();
        let v = n.cross(u);

        let local = hit_pos - self.center;
        let u_dist = local.dot(u);
        let v_dist = local.dot(v);

        if u_dist.abs() > self.width * 0.5 || v_dist.abs() > self.length * 0.5 {
            return no_hit(ray, self.material);
        }

        HitInfo {
            has_hit: true,
            t: t as f64,
            pos: hit_pos,
            sent_ray: ray.clone(),
            normal: n,
            material: self.material,
        }
    }

    fn set_material(&mut self, material: Material) {
        self.material = material;
    }
    fn as_any(&self) -> &dyn Any {
        self
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

        no_hit(ray, self.material)
    }

    fn set_material(&mut self, material: Material) {
        self.material = material;
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Hittable for Mesh {
    fn hit(&self, ray: &Ray) -> HitInfo {
        *self.faces
            .iter()
            .flat_map(|face| face.to_tris())
            .map(|tri| tri.hit(ray))
            .filter(|hit| hit.has_hit)
            .min_by(|a, b| a.t.partial_cmp(&b.t).unwrap())
            .get_or_insert(no_hit(ray, Material::default()))
    }

    fn set_material(&mut self, material: Material) {
        self.faces.iter_mut().for_each(|face| {
            face.set_material(material);
        });
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn no_hit(ray: &Ray, material: Material) -> HitInfo {
    HitInfo {
        has_hit: false,
        t: f64::INFINITY,
        pos: Vec3::ZERO,
        sent_ray: ray.clone(),
        normal: Vec3::ZERO,
        material,
    }
}