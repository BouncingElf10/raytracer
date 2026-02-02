use glam::Vec3;
use crate::objects::Hittable;
use crate::scene::Scene;

pub struct AABB {
    pub(crate) min: Vec3,
    pub(crate) max: Vec3,
}

impl AABB {
    pub(crate) fn new(min: Vec3, max: Vec3) -> Self {
        AABB { min, max }
    }
    pub fn edges(&self) -> [(Vec3, Vec3); 12] {
        let min = self.min;
        let max = self.max;

        let v000 = Vec3::new(min.x, min.y, min.z);
        let v001 = Vec3::new(min.x, min.y, max.z);
        let v010 = Vec3::new(min.x, max.y, min.z);
        let v011 = Vec3::new(min.x, max.y, max.z);
        let v100 = Vec3::new(max.x, min.y, min.z);
        let v101 = Vec3::new(max.x, min.y, max.z);
        let v110 = Vec3::new(max.x, max.y, min.z);
        let v111 = Vec3::new(max.x, max.y, max.z);

        [
            (v000, v001), (v001, v011), (v011, v010), (v010, v000),
            (v100, v101), (v101, v111), (v111, v110), (v110, v100),
            (v000, v100), (v001, v101), (v010, v110), (v011, v111),
        ]
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