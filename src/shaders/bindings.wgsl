// All resource bindings for the path-tracing compute pipeline.
//
// Bindings 0..7 are shared by both shader variants. Binding 8 exists only in the
// instrumented variant: `//#if INSTRUMENTED` blocks are stripped by the composer
// in `src/shaders.rs`, so the clean variant contains no counter buffer, no
// counter writes, and no atomics at all.

@group(0) @binding(0) var<storage, read> rays: array<Ray>;
@group(0) @binding(1) var<storage, read_write> output_colors: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> spheres: array<Sphere>;
@group(0) @binding(3) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(4) var<storage, read> planes: array<Plane>;
@group(0) @binding(5) var<uniform> counts: Counts;
@group(0) @binding(6) var<storage, read> bvh_nodes: array<BVHNode>;
@group(0) @binding(7) var<storage, read> bvh_indices: array<u32>;
//#if INSTRUMENTED
@group(0) @binding(8) var<storage, read_write> ray_counters: array<RayCounters>;
//#endif
