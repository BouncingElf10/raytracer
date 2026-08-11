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
    // Samples accumulated inside a single dispatch. The interactive renderer
    // leaves this at 1 and accumulates across frames instead; the data-collection
    // harness raises it so one dispatch is one complete measurement.
    samples: u32,
    // Experiment RNG seed. Folded into the per-pixel seed so a run reproduces
    // exactly given the same config.
    rng_seed: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Per-ray instrumentation counters (§5 Option B: one record per pixel, so the
/// harness can report distribution stats and not just means).
///
/// Written only by the instrumented shader variant; the clean variant does not
/// declare the buffer at all.
struct RayCounters {
    // AABB tests performed, i.e. nodes popped off the traversal stack. This is
    // the standard hardware-independent "node visits" traversal-cost metric.
    node_visits: u32,
    // Calls to hit_triangle from inside a leaf.
    prim_tests: u32,
    // Traversal queries issued by this pixel (primary + every secondary bounce).
    ray_count: u32,
    // Subset of node_visits that were interior (non-leaf) nodes.
    interior_visits: u32,
    // Traversals that hit the stack-depth or step guard and returned early. Any
    // non-zero total means the counts for this scene are truncated.
    incomplete: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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
