use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuSphere {
    pub(crate) center: [f32; 3],
    pub(crate) radius: f32,
    pub(crate) albedo: [f32; 3],
    pub(crate) emission: f32,
    pub(crate) metallic: f32,
    pub(crate) roughness: f32,
    pub(crate) _padding: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuTriangle {
    pub(crate) v0: [f32; 3],
    pub(crate) _pad0: f32,
    pub(crate) v1: [f32; 3],
    pub(crate) _pad1: f32,
    pub(crate) v2: [f32; 3],
    pub(crate) _pad2: f32,
    pub(crate) albedo: [f32; 3],
    pub(crate) emission: f32,
    pub(crate) metallic: f32,
    pub(crate) roughness: f32,
    pub(crate) _padding: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuPlane {
    pub(crate) center: [f32; 4],
    pub(crate) normal: [f32; 4],
    pub(crate) width: f32,
    pub(crate) length: f32,
    pub(crate) _pad2: [f32; 2],
    pub(crate) albedo: [f32; 4],
    pub(crate) emission: f32,
    pub(crate) metallic: f32,
    pub(crate) roughness: f32,
    pub(crate) _pad3: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuRay {
    pub(crate) origin: [f32; 3],
    pub(crate) _pad0: f32,
    pub(crate) direction: [f32; 3],
    pub(crate) _pad1: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuHitInfo {
    pub has_hit: u32,
    pub t: f32,
    _pad0: [f32; 2],
    pub pos: [f32; 4],
    pub normal: [f32; 4],
    pub albedo: [f32; 4],
    pub emission: f32,
    pub metallic: f32,
    pub roughness: f32,
    _pad1: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub _pad: f32,
}

/// Mirrors `Counts` in `shaders/types.wgsl`. Uploaded as a uniform buffer, so the
/// size must stay a multiple of 16 bytes -- hence the trailing padding.
///
/// `renderer.rs` patches `frame_number` in place at byte offset 20; keep the
/// field order stable if you add anything.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Counts {
    pub sphere_count: u32,
    pub triangle_count: u32,
    pub plane_count: u32,
    pub width: u32,
    pub height: u32,
    pub frame_number: u32,
    pub bvh_node_count: u32,
    pub bvh_index_count: u32,
    pub samples: u32,
    pub rng_seed: u32,
    pub display_mode: u32,
    pub palette: u32,
    pub heat_scale: f32,
    pub heat_mix: f32,
    pub max_bounces: u32,
    pub _pad0: u32,
}

impl Counts {
    /// Byte offset of `frame_number`, patched in place every frame by the
    /// renderer. Kept next to the struct so the two cannot drift apart.
    pub const FRAME_NUMBER_OFFSET: u64 = 20;
}

/// Live cost views, mirroring the `DISPLAY_*` constants in `shaders/types.wgsl`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayMode {
    Beauty = 0,
    NodeVisits = 1,
    PrimTests = 2,
    TraversalDepth = 3,
    LeafVisits = 4,
    InteriorVisits = 5,
}

impl DisplayMode {
    pub const ALL: [DisplayMode; 6] = [
        DisplayMode::Beauty,
        DisplayMode::NodeVisits,
        DisplayMode::PrimTests,
        DisplayMode::TraversalDepth,
        DisplayMode::LeafVisits,
        DisplayMode::InteriorVisits,
    ];

    pub fn label(self) -> &'static str {
        match self {
            DisplayMode::Beauty => "Rendered image",
            DisplayMode::NodeVisits => "Node visits / ray",
            DisplayMode::PrimTests => "Triangle tests / ray",
            DisplayMode::TraversalDepth => "Peak traversal depth",
            DisplayMode::LeafVisits => "Leaf visits / ray",
            DisplayMode::InteriorVisits => "Interior visits / ray",
        }
    }

    /// Sensible ramp ceiling when the view is first selected.
    pub fn default_scale(self) -> f32 {
        match self {
            DisplayMode::Beauty => 1.0,
            DisplayMode::NodeVisits => 40.0,
            DisplayMode::PrimTests => 40.0,
            DisplayMode::TraversalDepth => 24.0,
            DisplayMode::LeafVisits => 16.0,
            DisplayMode::InteriorVisits => 24.0,
        }
    }
}

/// Mirrors `RayCounters` in `shaders/types.wgsl`: one record per pixel, written
/// only by the instrumented shader variant.
#[repr(C)]
#[derive(Copy, Clone, Default, Pod, Zeroable)]
pub struct GpuRayCounters {
    pub node_visits: u32,
    pub prim_tests: u32,
    pub ray_count: u32,
    pub interior_visits: u32,
    pub incomplete: u32,
    pub max_stack: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuBVHNode {
    pub min: [f32; 3],
    pub _pad0: f32,
    pub max: [f32; 3],
    pub _pad1: f32,
    pub left_first: u32,
    pub right_count: u32,
    pub is_leaf: u32,
    pub _pad2: u32,
}