use crate::model::Mesh;
use crate::objects::{Hittable, Triangle};
use glam::Vec3;
use std::sync::Arc;
use crate::gpu_types::GpuBVHNode;
use crate::profiler::profiler_start;

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
    pub(crate) fn get_biggest_axis(&self) -> Axis {
        let x_length = self.max.x - self.min.x;
        let y_length = self.max.y - self.min.y;
        let z_length = self.max.z - self.min.z;
        if x_length > y_length && x_length > z_length { Axis::X } else if y_length > z_length { Axis::Y } else { Axis::Z }
    }
}

pub enum Axis {
    X, Y, Z
}

pub enum BVHNode {
    BVHNode {
        aabb: AABB,
        left: Box<BVHNode>,
        right: Box<BVHNode>,
    },
    LeafNode {
        aabb: AABB,
        objects: Arc<Mesh>,
    },
}

pub fn construct_bvh(mesh: &Mesh) -> BVHNode {
    let prims = mesh.get_triangles();
    let aabb = mesh.to_aabb();

    if prims.len() <= 20 {
        return BVHNode::LeafNode {
            aabb,
            objects: Arc::new(mesh.clone()),
        };
    }

    let (left, right) = split_prims(prims, aabb);
    BVHNode::BVHNode {
        aabb: mesh.to_aabb(),
        left: Box::new(construct_bvh(&to_mesh(left))),
        right: Box::new(construct_bvh(&to_mesh(right)))
    }
}

fn split_prims(prims: Vec<Triangle>, aabb: AABB) -> (Vec<Triangle>, Vec<Triangle>) {
    let mut left = Vec::new();
    let mut right = Vec::new();
    let mut center = Vec3::ZERO;

    for prim in &prims {
        center += prim.center()
    }
    center /= prims.len() as f32;

    let axis = aabb.get_biggest_axis();
    for prim in prims {
        match axis {
            Axis::X => { if prim.center().x < center.x { left.push(prim); } else { right.push(prim); }}
            Axis::Y => { if prim.center().y < center.y { left.push(prim); } else { right.push(prim); }}
            Axis::Z => { if prim.center().z < center.z { left.push(prim); } else { right.push(prim); }}
        }
    }

    (left, right)
}

fn to_mesh(prims: Vec<Triangle>) -> Mesh {
    let mut mesh = Mesh::new();
    for tri in prims {
        mesh.append_tri(tri);
    }
    mesh
}

pub fn traverse_leaf_nodes<F>(node: &BVHNode, f: &mut F) where F: FnMut(&AABB, &Arc<Mesh>) {
    match node {
        BVHNode::LeafNode { aabb, objects } => {
            f(aabb, objects);
        }
        BVHNode::BVHNode { aabb: _aabb, left, right } => {
            traverse_leaf_nodes(left, f);
            traverse_leaf_nodes(right, f);
        }
    }
}

fn vec3_to_hashable(v: Vec3) -> (u32, u32, u32) {
    (v.x.to_bits(), v.y.to_bits(), v.z.to_bits())
}

pub fn flatten_bvh_for_gpu(bvh: &BVHNode, triangles: &[Triangle]) -> (Vec<GpuBVHNode>, Vec<u32>) {
    let mut nodes = Vec::new();
    let mut triangle_indices = Vec::new();

    let mut tri_to_idx = std::collections::HashMap::new();
    for (idx, tri) in triangles.iter().enumerate() {
        let key = (
            vec3_to_hashable(tri.v0()),
            vec3_to_hashable(tri.v1()),
            vec3_to_hashable(tri.v2())
        );
        tri_to_idx.insert(key, idx);
    }

    flatten_node(bvh, &mut nodes, &mut triangle_indices, &tri_to_idx);

    (nodes, triangle_indices)
}
fn flatten_node(node: &BVHNode, nodes: &mut Vec<GpuBVHNode>, triangle_indices: &mut Vec<u32>,
                tri_to_idx: &std::collections::HashMap<((u32, u32, u32), (u32, u32, u32), (u32, u32, u32)), usize>) -> u32 {
    let node_index = nodes.len() as u32;
    nodes.push(GpuBVHNode {
        min: [0.0; 3],
        _pad0: 0.0,
        max: [0.0; 3],
        _pad1: 0.0,
        left_first: 0,
        right_count: 0,
        is_leaf: 0,
        _pad2: 0,
    });

    match node {
        BVHNode::LeafNode { aabb, objects } => {
            let first_tri = triangle_indices.len() as u32;
            let triangles = objects.get_triangles();

            for tri in &triangles {
                let key = (
                    vec3_to_hashable(tri.v0()),
                    vec3_to_hashable(tri.v1()),
                    vec3_to_hashable(tri.v2())
                );
                if let Some(&idx) = tri_to_idx.get(&key) {
                    triangle_indices.push(idx as u32);
                }
            }

            let tri_count = (triangle_indices.len() as u32) - first_tri;

            nodes[node_index as usize] = GpuBVHNode {
                min: [aabb.min.x, aabb.min.y, aabb.min.z],
                _pad0: 0.0,
                max: [aabb.max.x, aabb.max.y, aabb.max.z],
                _pad1: 0.0,
                left_first: first_tri,
                right_count: tri_count,
                is_leaf: 1,
                _pad2: 0,
            };
        }
        BVHNode::BVHNode { aabb, left, right } => {
            let left_index = flatten_node(left, nodes, triangle_indices, tri_to_idx);
            let right_index = flatten_node(right, nodes, triangle_indices, tri_to_idx);

            nodes[node_index as usize] = GpuBVHNode {
                min: [aabb.min.x, aabb.min.y, aabb.min.z],
                _pad0: 0.0,
                max: [aabb.max.x, aabb.max.y, aabb.max.z],
                _pad1: 0.0,
                left_first: left_index,
                right_count: right_index,
                is_leaf: 0,
                _pad2: 0,
            };
        }
    }

    node_index
}
