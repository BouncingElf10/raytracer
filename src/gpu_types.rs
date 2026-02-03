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