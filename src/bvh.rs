use std::cmp::Ordering;
use crate::gpu_types::GpuBVHNode;
use crate::model::Mesh;
use crate::objects::Triangle;
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
    pub(crate) fn surface_area(&self) -> f32 {
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

/// Bounds of a triangle set. Matches `Mesh::to_aabb`: folds from an inverted box
/// so an empty set would yield `min > max` (we never recurse into empty sets --
/// `split_prims` guarantees both children are non-empty).
fn bounds_of(prims: &[Triangle]) -> AABB {
    let (min, max) = prims
        .iter()
        .flat_map(|tri| tri.get_vertices())
        .fold(
            (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
            |(min, max), p| (min.min(p), max.max(p)),
        );
    AABB::new(min, max)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitHeuristic {
    /// Split along the node's longest axis at the mean of the triangle centroids.
    LongestAxisCentroid,
    /// Sort along the longest axis, split into two equal-count halves (balanced tree).
    Median,
    /// Exact full-sweep SAH: for every axis, sweep every candidate split index and
    /// pick the one minimising SA(L)*N_L + SA(R)*N_R.
    SurfaceAreaHeuristic,
    /// Random axis, random plane position. Baseline / control.
    Random,
}

impl SplitHeuristic {
    /// Every heuristic under comparison, in a fixed order so experiment output is
    /// deterministic row-for-row.
    pub const ALL: [SplitHeuristic; 4] = [
        SplitHeuristic::LongestAxisCentroid,
        SplitHeuristic::Median,
        SplitHeuristic::SurfaceAreaHeuristic,
        SplitHeuristic::Random,
    ];

    /// Stable short name used as the `heuristic` column in the metrics CSV.
    pub fn name(&self) -> &'static str {
        match self {
            SplitHeuristic::LongestAxisCentroid => "LongestAxisCentroid",
            SplitHeuristic::Median => "Median",
            SplitHeuristic::SurfaceAreaHeuristic => "Sah",
            SplitHeuristic::Random => "Random",
        }
    }
}

/// Maximum primitives allowed in a leaf. Identical for every heuristic so the leaf
/// cutoff never becomes a confounding variable; override per-experiment through
/// `BuildParams::leaf_cutoff`.
pub const DEFAULT_LEAF_PRIM_CUTOFF: usize = 20;

/// Default seed for the `Random` heuristic. Any fixed value works; what matters is
/// that it is fixed, so two runs over the same geometry produce the same tree.
pub const DEFAULT_BUILD_SEED: u64 = 0x5eed_1234_abcd_0001;

/// Everything that can vary a build other than the geometry itself. Held constant
/// across heuristics within one experiment.
#[derive(Debug, Clone, Copy)]
pub struct BuildParams {
    pub heuristic: SplitHeuristic,
    pub leaf_cutoff: usize,
    pub seed: u64,
}

impl BuildParams {
    pub fn new(heuristic: SplitHeuristic) -> Self {
        Self {
            heuristic,
            leaf_cutoff: DEFAULT_LEAF_PRIM_CUTOFF,
            seed: DEFAULT_BUILD_SEED,
        }
    }
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
    construct_bvh_with(mesh, BuildParams::new(heuristic))
}

pub fn construct_bvh_with(mesh: &Mesh, params: BuildParams) -> BVHNode {
    construct_bvh_from_tris(mesh.get_triangles(), params)
}

/// Top-down construction over a flat triangle list.
///
/// The experiment harness calls this directly with a pre-extracted list so that
/// `build_time_ms` measures splitting alone -- pulling the transformed triangles
/// out of the `Mesh` is loading work, not construction work.
pub fn construct_bvh_from_tris(prims: Vec<Triangle>, params: BuildParams) -> BVHNode {
    let aabb = bounds_of(&prims);
    build_node(prims, aabb, params)
}

fn build_node(prims: Vec<Triangle>, aabb: AABB, params: BuildParams) -> BVHNode {
    if prims.len() <= params.leaf_cutoff {
        return BVHNode::LeafNode {
            aabb,
            objects: Arc::new(to_mesh(prims)),
        };
    }

    let (left, right) = split_prims(prims, &aabb, params);
    let left_aabb = bounds_of(&left);
    let right_aabb = bounds_of(&right);

    BVHNode::BVHNode {
        aabb,
        left: Box::new(build_node(left, left_aabb, params)),
        right: Box::new(build_node(right, right_aabb, params)),
    }
}

fn split_prims(prims: Vec<Triangle>, aabb: &AABB, params: BuildParams)
               -> (Vec<Triangle>, Vec<Triangle>)
{
    let (left, right) = match params.heuristic {
        SplitHeuristic::LongestAxisCentroid => split_longest_axis_centroid(prims, aabb),
        SplitHeuristic::Median => split_median(prims, aabb),
        SplitHeuristic::SurfaceAreaHeuristic => split_sah(prims, aabb),
        SplitHeuristic::Random => split_random(prims, aabb, params.seed),
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

// ---- 3. Surface Area Heuristic (exact full sweep) --------------------------

struct TriInfo {
    tri: Triangle,
    bounds: AABB,
    centroid: Vec3,
}

struct SplitCandidate {
    axis: Axis,
    index: usize,
    cost: f32,
}

fn prepare_tris(prims: Vec<Triangle>) -> Vec<TriInfo> {
    prims
        .into_iter()
        .map(|tri| {
            let bounds = tri_bounds(&tri);
            let centroid = tri.center();
            TriInfo { tri, bounds, centroid }
        })
        .collect()
}

fn sort_by_axis(tris: &mut [TriInfo], axis: Axis) {
    tris.sort_by(|a, b| {
        axis_component(a.centroid, axis)
            .partial_cmp(&axis_component(b.centroid, axis))
            .unwrap_or(Ordering::Equal)
    });
}

fn swept_surface_areas(tris: &[TriInfo]) -> (Vec<f32>, Vec<f32>) {
    let n = tris.len();

    let mut left_area = vec![0f32; n + 1];
    let mut acc = AABB::empty();
    for k in 1..=n {
        acc = acc.union(&tris[k - 1].bounds);
        left_area[k] = acc.surface_area();
    }

    let mut right_area = vec![0f32; n + 1];
    acc = AABB::empty();
    for k in (0..n).rev() {
        acc = acc.union(&tris[k].bounds);
        right_area[k] = acc.surface_area();
    }

    (left_area, right_area)
}

/// For one node, C_trav, C_isect and SA(B) are identical across every candidate
/// split, so they never change which split wins. Dropping those constants
/// leaves the quantity actually minimised below:
///
///     SA(B_L) * N_L + SA(B_R) * N_R
fn sah_cost(left_area: f32, left_count: usize, right_area: f32, right_count: usize) -> f32 {
    left_area * left_count as f32 + right_area * right_count as f32
}

fn split_sah(prims: Vec<Triangle>, _aabb: &AABB) -> (Vec<Triangle>, Vec<Triangle>) {
    let mut tris = prepare_tris(prims);
    let n = tris.len();

    let mut best = SplitCandidate {
        axis: Axis::X,
        index: n / 2,
        cost: f32::INFINITY,
    };

    for axis in [Axis::X, Axis::Y, Axis::Z] {
        sort_by_axis(&mut tris, axis);
        let (left_areas, right_areas) = swept_surface_areas(&tris);

        for index in 1..n {
            let cost = sah_cost(left_areas[index], index, right_areas[index], n - index);
            if cost < best.cost {
                best = SplitCandidate { axis, index, cost };
            }
        }
    }

    sort_by_axis(&mut tris, best.axis);
    let right = tris.split_off(best.index);
    (unwrap_tris(tris), unwrap_tris(right))
}

fn unwrap_tris(tris: Vec<TriInfo>) -> Vec<Triangle> {
    tris.into_iter().map(|info| info.tri).collect()
}

// ---- 4. Random split -------------------------------------------------------

fn seeded_rng(prims: &[Triangle], seed: u64) -> StdRng {
    // Seed from node contents mixed with the experiment seed: the draw differs
    // between nodes, but is identical run-to-run for the same seed and geometry.
    let mut state = prims.len() as u64;
    if let Some(first) = prims.first() {
        let c = first.center();
        state ^= (c.x.to_bits() as u64) << 1;
        state ^= (c.y.to_bits() as u64) << 11;
        state ^= (c.z.to_bits() as u64) << 21;
    }
    // splitmix64 finaliser, so neighbouring node states don't produce correlated
    // streams the way raw `seed_from_u64(small_int)` would.
    state = state.wrapping_add(seed).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    state ^= state >> 30;
    state = state.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    state ^= state >> 27;
    state = state.wrapping_mul(0x94D0_49BB_1331_11EB);
    state ^= state >> 31;

    StdRng::seed_from_u64(state)
}

fn random_axis(rng: &mut StdRng) -> Axis {
    match rng.random_range(0..3) {
        0 => Axis::X,
        1 => Axis::Y,
        _ => Axis::Z,
    }
}

fn split_random(prims: Vec<Triangle>, aabb: &AABB, seed: u64) -> (Vec<Triangle>, Vec<Triangle>) {
    let mut rng = seeded_rng(&prims, seed);
    let axis = random_axis(&mut rng);

    let lo = axis_component(aabb.min, axis);
    let hi = axis_component(aabb.max, axis);
    if hi <= lo {
        // Degenerate on this axis; hand back an empty child and let the
        // empty-partition fallback in `split_prims` force progress.
        return (prims, Vec::new());
    }

    let split = rng.random_range(lo..hi);
    partition_by_axis(prims, axis, split)
}

// ---- Tree-quality statistics ----------------------------------------------

/// Cost constants for the SAH cost model. Both are 1.0 so `total_sah_cost` is
/// expressed in "expected primitive intersections + node traversals per random
/// ray hitting the root", which keeps the number comparable across scenes.
const C_TRAV: f64 = 1.0;
const C_ISECT: f64 = 1.0;

/// Structural quality of one built tree. Everything here is a pure function of
/// the tree, so it is hardware-independent and reproduces exactly.
#[derive(Debug, Clone, Copy)]
pub struct BuildStats {
    pub node_count: usize,
    pub leaf_count: usize,
    pub prim_count: usize,
    pub max_depth: usize,
    /// Mean depth over leaves (root = depth 0).
    pub avg_depth: f64,
    pub avg_leaf_prims: f64,
    /// Full SAH model:
    ///   C_trav * Σ_interior SA(n)/SA(root) + C_isect * Σ_leaf SA(n)/SA(root) * n_prims
    pub total_sah_cost: f64,
    /// Leaf term only: Σ_leaf SA(n)/SA(root) * n_prims. Reported alongside the
    /// full model so either convention can be used in analysis.
    pub leaf_sah_cost: f64,
    /// Filled in by the caller; construction is timed outside this walk.
    pub build_time_ms: f64,
}

/// Single post-order walk collecting every tree-quality statistic at once.
pub fn compute_build_stats(root: &BVHNode) -> BuildStats {
    let root_area = match root {
        BVHNode::BVHNode { aabb, .. } | BVHNode::LeafNode { aabb, .. } => aabb.surface_area() as f64,
    };
    // A single-triangle or perfectly flat root has zero surface area; normalising
    // by it would produce inf/NaN, so fall back to an unnormalised cost.
    let norm = if root_area > 0.0 { root_area } else { 1.0 };

    let mut acc = StatsAcc {
        norm,
        node_count: 0,
        leaf_count: 0,
        prim_count: 0,
        max_depth: 0,
        depth_sum: 0,
        interior_area_sum: 0.0,
        leaf_cost_sum: 0.0,
    };
    walk_stats(root, 0, &mut acc);

    let leaf_count = acc.leaf_count.max(1) as f64;
    BuildStats {
        node_count: acc.node_count,
        leaf_count: acc.leaf_count,
        prim_count: acc.prim_count,
        max_depth: acc.max_depth,
        avg_depth: acc.depth_sum as f64 / leaf_count,
        avg_leaf_prims: acc.prim_count as f64 / leaf_count,
        total_sah_cost: C_TRAV * acc.interior_area_sum + C_ISECT * acc.leaf_cost_sum,
        leaf_sah_cost: acc.leaf_cost_sum,
        build_time_ms: 0.0,
    }
}

struct StatsAcc {
    norm: f64,
    node_count: usize,
    leaf_count: usize,
    prim_count: usize,
    max_depth: usize,
    depth_sum: usize,
    interior_area_sum: f64,
    leaf_cost_sum: f64,
}

fn walk_stats(node: &BVHNode, depth: usize, acc: &mut StatsAcc) {
    acc.node_count += 1;

    match node {
        BVHNode::LeafNode { aabb, objects } => {
            let prims = objects.get_triangles().len();
            acc.leaf_count += 1;
            acc.prim_count += prims;
            acc.depth_sum += depth;
            acc.max_depth = acc.max_depth.max(depth);
            acc.leaf_cost_sum += (aabb.surface_area() as f64 / acc.norm) * prims as f64;
        }
        BVHNode::BVHNode { aabb, left, right } => {
            acc.interior_area_sum += aabb.surface_area() as f64 / acc.norm;
            walk_stats(left, depth + 1, acc);
            walk_stats(right, depth + 1, acc);
        }
    }
}

// ---------------------------------------------------------------------------

fn to_mesh(prims: Vec<Triangle>) -> Mesh {
    let mut mesh = Mesh::new();
    for tri in prims {
        mesh.append_tri(tri);
    }
    mesh
}

/// One node as seen by the debug overlay.
pub struct NodeVisit<'a> {
    pub aabb: &'a AABB,
    pub depth: usize,
    pub is_leaf: bool,
    /// Primitives in this leaf, or 0 for an interior node.
    pub prim_count: usize,
}

/// Walks every node -- interior and leaf -- carrying its depth.
///
/// The leaf-only walker below cannot show *why* a tree is bad: an unbalanced
/// build looks the same at the leaves as a balanced one until you can see which
/// level each box belongs to.
pub fn traverse_nodes_with_depth<F>(node: &BVHNode, f: &mut F) where F: FnMut(NodeVisit) {
    fn walk<F: FnMut(NodeVisit)>(node: &BVHNode, depth: usize, f: &mut F) {
        match node {
            BVHNode::LeafNode { aabb, objects } => {
                f(NodeVisit { aabb, depth, is_leaf: true, prim_count: objects.get_triangles().len() });
            }
            BVHNode::BVHNode { aabb, left, right } => {
                f(NodeVisit { aabb, depth, is_leaf: false, prim_count: 0 });
                walk(left, depth + 1, f);
                walk(right, depth + 1, f);
            }
        }
    }
    walk(node, 0, f);
}

/// Deepest level in the tree, for scaling the overlay's depth colour ramp.
pub fn tree_max_depth(root: &BVHNode) -> usize {
    let mut max = 0;
    traverse_nodes_with_depth(root, &mut |visit| max = max.max(visit.depth));
    max
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

type TriKey = ((u32, u32, u32), (u32, u32, u32), (u32, u32, u32));

fn tri_key(tri: &Triangle) -> TriKey {
    (
        vec3_to_hashable(tri.v0()),
        vec3_to_hashable(tri.v1()),
        vec3_to_hashable(tri.v2()),
    )
}

/// Result of flattening, including the bookkeeping the harness needs to assert
/// that no primitive was silently lost on the way to the GPU.
pub struct FlattenResult {
    pub nodes: Vec<GpuBVHNode>,
    pub indices: Vec<u32>,
    /// Leaf triangles that had no match in the GPU triangle array. Must be 0;
    /// anything else means the counts below are measuring the wrong geometry.
    pub unmatched_prims: usize,
}

pub fn flatten_bvh_for_gpu(bvh: &BVHNode, triangles: &[Triangle]) -> (Vec<GpuBVHNode>, Vec<u32>) {
    let result = flatten_bvh_for_gpu_checked(bvh, triangles);
    (result.nodes, result.indices)
}

pub fn flatten_bvh_for_gpu_checked(bvh: &BVHNode, triangles: &[Triangle]) -> FlattenResult {
    let mut nodes = Vec::new();
    let mut triangle_indices = Vec::new();

    // Geometrically identical triangles share a key, so the map holds every index
    // that matches and leaves hand them out one at a time. A plain
    // `HashMap<key, usize>` would collapse duplicates onto a single index and
    // silently drop primitives from the flattened tree.
    let mut tri_to_idx: std::collections::HashMap<TriKey, Vec<u32>> =
        std::collections::HashMap::with_capacity(triangles.len());
    for (idx, tri) in triangles.iter().enumerate() {
        tri_to_idx.entry(tri_key(tri)).or_default().push(idx as u32);
    }

    let mut unmatched_prims = 0usize;
    flatten_node(bvh, &mut nodes, &mut triangle_indices, &mut tri_to_idx, &mut unmatched_prims);

    FlattenResult { nodes, indices: triangle_indices, unmatched_prims }
}

fn flatten_node(
    node: &BVHNode,
    nodes: &mut Vec<GpuBVHNode>,
    triangle_indices: &mut Vec<u32>,
    tri_to_idx: &mut std::collections::HashMap<TriKey, Vec<u32>>,
    unmatched_prims: &mut usize,
) -> u32 {
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
                match tri_to_idx.get_mut(&tri_key(tri)).and_then(|slots| slots.pop()) {
                    Some(idx) => triangle_indices.push(idx),
                    None => *unmatched_prims += 1,
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
            let left_index = flatten_node(left, nodes, triangle_indices, tri_to_idx, unmatched_prims);
            let right_index = flatten_node(right, nodes, triangle_indices, tri_to_idx, unmatched_prims);

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

/// Every node's box at exactly `depth`, plus whether that node is a leaf.
///
/// Collected before the tree is dropped so the structure figures can be drawn
/// without keeping a whole scene's worth of trees alive at once.
pub fn aabbs_at_depth(root: &BVHNode, depth: usize) -> Vec<(AABB, bool)> {
    let mut boxes = Vec::new();
    traverse_nodes_with_depth(root, &mut |visit| {
        if visit.depth == depth {
            boxes.push((*visit.aabb, visit.is_leaf));
        }
    });
    boxes
}
