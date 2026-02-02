use glam::Vec3;
use crate::objects::Hittable;
use crate::scene::Scene;

pub struct AABB {
    min: Vec3,
    max: Vec3,
}

impl AABB {
    pub(crate) fn new(min: Vec3, max: Vec3) -> Self {
        AABB { min, max }
    }
    pub fn hit(&self, ray: &crate::ray::Ray) -> bool {
        let inv_dir = 1.0 / ray.direction();

        let t0 = (self.min - ray.origin()) * inv_dir;
        let t1 = (self.max - ray.origin()) * inv_dir;

        let tmin = t0.min(t1).max_element();
        let tmax = t0.max(t1).min_element();

        tmax >= tmin.max(0.0)
    }
}

enum BVHNode {
    BVHNode {
        left: Box<BVHNode>,
        right: Box<BVHNode>,
    },
    LeafNode {
        object: Box<dyn Hittable>,
    },
}

// pub fn construct_bvh(scene: Scene) -> BVHNode {
//
// }