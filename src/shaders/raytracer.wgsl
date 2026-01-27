@group(0) @binding(0) var<storage, read> spheres: array<Sphere>;
@group(0) @binding(1) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(2) var<storage, read> planes: array<Plane>;
@group(0) @binding(3) var<storage, read> rays: array<Ray>;
@group(0) @binding(4) var<storage, read_write> hits: array<HitInfo>;
@group(0) @binding(5) var<uniform> counts: Counts;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ray_idx = global_id.y * 800u + global_id.x;

    if (ray_idx >= arrayLength(&rays)) {
        return;
    }

    let ray = rays[ray_idx];

    var closest_hit: HitInfo;
    closest_hit.has_hit = 0u;
    closest_hit.t = 1e10;
    closest_hit._pad0 = vec2<f32>(0.0, 0.0);
    closest_hit._pad1 = 0.0;

    for (var i = 0u; i < counts.sphere_count; i++) {
        let hit = hit_sphere(spheres[i], ray);
        if (hit.has_hit == 1u && hit.t < closest_hit.t) {
            closest_hit = hit;
        }
    }

    for (var i = 0u; i < counts.triangle_count; i++) {
        let hit = hit_triangle(triangles[i], ray);
        if (hit.has_hit == 1u && hit.t < closest_hit.t) {
            closest_hit = hit;
        }
    }

    for (var i = 0u; i < counts.plane_count; i++) {
        let hit = hit_plane(planes[i], ray);
        if (hit.has_hit == 1u && hit.t < closest_hit.t) {
            closest_hit = hit;
        }
    }

    hits[ray_idx] = closest_hit;
}