use std::cmp::Ordering;
use crate::gpu_types::GpuBVHNode;
use crate::model::Mesh;
use crate::objects::{Hittable, Triangle};
use glam::Vec3;
use std::sync::Arc;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Debug, Clone, Copy)]
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

    /// Half the total surface area is the SAH-relevant quantity, but the constant
    /// factor cancels in the argmin, so we return the full surface area.
    fn surface_area(&self) -> f32 {
        let d = self.max - self.min;
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    /// An "inverted" box that acts as the identity for `union`.
    fn empty() -> Self {
        AABB {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
        }
    }

    fn union(&self, other: &AABB) -> AABB {
        AABB {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    X, Y, Z,
}

fn axis_component(v: Vec3, axis: Axis) -> f32 {
    match axis {
        Axis::X => v.x,
        Axis::Y => v.y,
        Axis::Z => v.z,
    }
}

fn tri_bounds(tri: &Triangle) -> AABB {
    let a = tri.v0();
    let b = tri.v1();
    let c = tri.v2();
    AABB::new(a.min(b).min(c), a.max(b).max(c))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitHeuristic {
    /// Split along the node's longest axis at the mean of the triangle centroids.
    LongestAxisCentroid,
    /// Sort along the longest axis, split into two equal-count halves (balanced tree).
    Median,
    /// Binned SAH: pick the axis/plane that minimises SA(L)*N_L + SA(R)*N_R.
    SurfaceAreaHeuristic,
    /// Random axis, random plane position. Baseline / control.
    Random,
}

#[derive(Debug, Clone)]
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

pub fn construct_bvh(mesh: &Mesh, heuristic: SplitHeuristic) -> BVHNode {
    let prims = mesh.get_triangles();
    let aabb = mesh.to_aabb();

    if prims.len() <= 20 {
        return BVHNode::LeafNode {
            aabb,
            objects: Arc::new(mesh.clone()),
        };
    }

    let (left, right) = split_prims(prims, &aabb, heuristic);
    BVHNode::BVHNode {
        aabb,
        left: Box::new(construct_bvh(&to_mesh(left), heuristic)),
        right: Box::new(construct_bvh(&to_mesh(right), heuristic)),
    }
}

fn split_prims(prims: Vec<Triangle>, aabb: &AABB, heuristic: SplitHeuristic)
               -> (Vec<Triangle>, Vec<Triangle>)
{
    let (left, right) = match heuristic {
        SplitHeuristic::LongestAxisCentroid => split_longest_axis_centroid(prims, aabb),
        SplitHeuristic::Median => split_median(prims, aabb),
        SplitHeuristic::SurfaceAreaHeuristic => split_sah(prims, aabb),
        SplitHeuristic::Random => split_random(prims, aabb),
    };

    // Universal degeneracy guard: if a heuristic put everything on one side
    // (coincident centroids, a random plane outside the geometry, etc.) fall
    // back to an index-median split so recursion always makes progress.
    if left.is_empty() || right.is_empty() {
        let mut all = left;
        all.extend(right);
        median_by_index(all)
    } else {
        (left, right)
    }
}

/// Splits [0, mid) | [mid, len) by position in the vec. Guarantees two non-empty
/// halves for len >= 2. Only used as the fallback above.
fn median_by_index(mut prims: Vec<Triangle>) -> (Vec<Triangle>, Vec<Triangle>) {
    let mid = prims.len() / 2;
    let right = prims.split_off(mid);
    (prims, right)
}

fn partition_by_axis(prims: Vec<Triangle>, axis: Axis, split: f32)
                     -> (Vec<Triangle>, Vec<Triangle>)
{
    let mut left = Vec::new();
    let mut right = Vec::new();
    for prim in prims {
        if axis_component(prim.center(), axis) < split {
            left.push(prim);
        } else {
            right.push(prim);
        }
    }
    (left, right)
}

// ---- 1. Longest-axis centroid (the original behaviour) ---------------------

fn split_longest_axis_centroid(prims: Vec<Triangle>, aabb: &AABB) -> (Vec<Triangle>, Vec<Triangle>) {
    let axis = aabb.get_biggest_axis();
    let mut center = Vec3::ZERO;
    for prim in &prims {
        center += prim.center();
    }
    center /= prims.len() as f32;

    partition_by_axis(prims, axis, axis_component(center, axis))
}

// ---- 2. Median split -------------------------------------------------------

fn split_median(mut prims: Vec<Triangle>, aabb: &AABB) -> (Vec<Triangle>, Vec<Triangle>)
{
    let axis = aabb.get_biggest_axis();
    prims.sort_by(|a, b| {
        axis_component(a.center(), axis)
            .partial_cmp(&axis_component(b.center(), axis))
            .unwrap_or(Ordering::Equal)
    });
    let mid = prims.len() / 2;
    let right = prims.split_off(mid);
    (prims, right)
}

// ---- 3. Surface Area Heuristic (binned) ------------------------------------

fn split_sah(prims: Vec<Triangle>, _aabb: &AABB) -> (Vec<Triangle>, Vec<Triangle>) {
    // Precompute each triangle's bounds + centroid once.
    let mut tris: Vec<(Triangle, AABB, Vec3)> = prims
        .into_iter()
        .map(|t| {
            let b = tri_bounds(&t);
            let c = t.center();
            (t, b, c)
        })
        .collect();

    let n = tris.len();
    let mut best_cost = f32::INFINITY;
    let mut best_axis = Axis::X;
    let mut best_k = n / 2;

    for axis in [Axis::X, Axis::Y, Axis::Z] {
        tris.sort_by(|a, b| {
            axis_component(a.2, axis)
                .partial_cmp(&axis_component(b.2, axis))
                .unwrap_or(Ordering::Equal)
        });

        // left_area[k]  = SA of union of the first k triangles      (k = 1..=n)
        // right_area[k] = SA of union of triangles k..n             (k = 0..n)
        let mut left_area = vec![0f32; n + 1];
        let mut acc = AABB::empty();
        for k in 1..=n {
            acc = acc.union(&tris[k - 1].1);
            left_area[k] = acc.surface_area();
        }

        let mut right_area = vec![0f32; n + 1];
        acc = AABB::empty();
        for k in (0..n).rev() {
            acc = acc.union(&tris[k].1);
            right_area[k] = acc.surface_area();
        }

        // Candidate split k: left = [0, k), right = [k, n), for k in 1..n.
        for k in 1..n {
            let cost = left_area[k] * k as f32 + right_area[k] * (n - k) as f32;
            if cost < best_cost {
                best_cost = cost;
                best_axis = axis;
                best_k = k;
            }
        }
    }

    // Re-sort by the winning axis, then split at the winning index.
    tris.sort_by(|a, b| {
        axis_component(a.2, best_axis)
            .partial_cmp(&axis_component(b.2, best_axis))
            .unwrap_or(Ordering::Equal)
    });

    let right: Vec<Triangle> = tris.split_off(best_k).into_iter().map(|(t, _, _)| t).collect();
    let left: Vec<Triangle> = tris.into_iter().map(|(t, _, _)| t).collect();
    (left, right)
}

// ---- 4. Random split -------------------------------------------------------

fn split_random(prims: Vec<Triangle>, aabb: &AABB) -> (Vec<Triangle>, Vec<Triangle>) {
    // Seed from node contents: random across nodes, reproducible across runs.
    let mut seed = prims.len() as u64;
    if let Some(first) = prims.first() {
        let c = first.center();
        seed ^= (c.x.to_bits() as u64) << 1;
        seed ^= (c.y.to_bits() as u64) << 11;
        seed ^= (c.z.to_bits() as u64) << 21;
    }
    let mut rng = StdRng::seed_from_u64(seed);

    let axis = match rng.random_range(0..3) {
        0 => Axis::X,
        1 => Axis::Y,
        _ => Axis::Z,
    };

    let lo = axis_component(aabb.min, axis);
    let hi = axis_component(aabb.max, axis);
    if hi <= lo {
        return (prims, Vec::new()); // guard median-splits it
    }
    let split = rng.random_range(lo..hi);
    partition_by_axis(prims, axis, split)
}

// ---------------------------------------------------------------------------

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