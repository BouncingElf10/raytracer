struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct Sphere {
    center: vec3<f32>,
    radius: f32,
    albedo: vec3<f32>,
    emission: f32,
    metallic: f32,
    roughness: f32,
}

struct Triangle {
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>,
    albedo: vec3<f32>,
    emission: f32,
    metallic: f32,
    roughness: f32,
}

struct Plane {
    center: vec4<f32>,
    normal: vec4<f32>,
    width: f32,
    length: f32,
    _pad2: vec2<f32>,
    albedo: vec4<f32>,
    emission: f32,
    metallic: f32,
    roughness: f32,
    _pad3: f32,
}

struct HitInfo {
    has_hit: u32,
    t: f32,
    _pad0: vec2<f32>,
    pos: vec4<f32>,
    normal: vec4<f32>,
    albedo: vec4<f32>,
    emission: f32,
    metallic: f32,
    roughness: f32,
    _pad1: f32,
}

struct Counts {
    sphere_count: u32,
    triangle_count: u32,
    plane_count: u32,
    width: u32,
    height: u32,
    frame_number: u32,
    bvh_node_count: u32,
    bvh_index_count: u32,
}

struct BVHNode {
    min: vec3<f32>,
    _pad0: f32,
    max: vec3<f32>,
    _pad1: f32,
    left_first: u32,
    right_count: u32,
    is_leaf: u32,
    _pad2: u32,
}

struct AABB {
    min: vec3<f32>,
    max: vec3<f32>,
}
