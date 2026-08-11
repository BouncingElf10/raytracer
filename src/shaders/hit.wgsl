
fn hit_sphere(sphere: Sphere, ray: Ray) -> HitInfo {
    var hit: HitInfo;
    hit.has_hit = 0u;
    hit.t = 1e10;
    hit._pad0 = vec2<f32>(0.0, 0.0);
    hit._pad1 = 0.0;

    let oc = ray.origin - sphere.center;
    let a = dot(ray.direction, ray.direction);
    let b = 2.0 * dot(oc, ray.direction);
    let c = dot(oc, oc) - sphere.radius * sphere.radius;
    let discriminant = b * b - 4.0 * a * c;

    if (discriminant > 0.0) {
        let sqrt_d = sqrt(discriminant);
        var t = (-b - sqrt_d) / (2.0 * a);

        if (t < 0.001) {
            t = (-b + sqrt_d) / (2.0 * a);
        }

        if (t > 0.001) {
            hit.has_hit = 1u;
            hit.t = t;
            let pos = ray.origin + ray.direction * t;
            hit.pos = vec4<f32>(pos, 0.0);
            let normal = normalize(pos - sphere.center);
            hit.normal = vec4<f32>(normal, 0.0);
            hit.albedo = vec4<f32>(sphere.albedo, 0.0);
            hit.emission = sphere.emission;
            hit.metallic = sphere.metallic;
            hit.roughness = sphere.roughness;
        }
    }

    return hit;
}

fn hit_triangle(tri: Triangle, ray: Ray) -> HitInfo {
    var hit: HitInfo;
    hit.has_hit = 0u;
    hit.t = 1e10;
    hit._pad0 = vec2<f32>(0.0, 0.0);
    hit._pad1 = 0.0;

    let eps = 1e-6;
    let edge1 = tri.v1 - tri.v0;
    let edge2 = tri.v2 - tri.v0;
    let h = cross(ray.direction, edge2);
    let a = dot(edge1, h);

    if (abs(a) < eps) {
        return hit;
    }

    let f = 1.0 / a;
    let s = ray.origin - tri.v0;
    let u = f * dot(s, h);

    if (u < 0.0 || u > 1.0) {
        return hit;
    }

    let q = cross(s, edge1);
    let v = f * dot(ray.direction, q);

    if (v < 0.0 || u + v > 1.0) {
        return hit;
    }

    let t = f * dot(edge2, q);

    if (t > 0.001) {
        hit.has_hit = 1u;
        hit.t = t;
        let pos = ray.origin + ray.direction * t;
        hit.pos = vec4<f32>(pos, 0.0);
        let normal = normalize(cross(edge1, edge2));
        hit.normal = vec4<f32>(normal, 0.0);
        hit.albedo = vec4<f32>(tri.albedo, 0.0);
        hit.emission = tri.emission;
        hit.metallic = tri.metallic;
        hit.roughness = tri.roughness;
    }

    return hit;
}

fn hit_plane(plane: Plane, ray: Ray) -> HitInfo {
    var hit: HitInfo;
    hit.has_hit = 0u;
    hit.t = 1e10;
    hit._pad0 = vec2<f32>(0.0, 0.0);
    hit._pad1 = 0.0;

    let eps = 1e-6;

    let n = normalize(plane.normal.xyz);

    let denom = dot(n, ray.direction);

    if (abs(denom) < eps) {
        return hit;
    }

    let t = dot(plane.center.xyz - ray.origin, n) / denom;

    if (t <= 0.001) {
        return hit;
    }

    let hit_pos = ray.origin + ray.direction * t;

    var tangent: vec3<f32>;
    if (abs(n.x) > 0.9) {
        tangent = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        tangent = vec3<f32>(1.0, 0.0, 0.0);
    }

    let u_vec = normalize(cross(n, tangent));
    let v_vec = cross(n, u_vec);

    let local = hit_pos - plane.center.xyz;
    let u_dist = dot(local, u_vec);
    let v_dist = dot(local, v_vec);

    if (abs(u_dist) > plane.width * 0.5 || abs(v_dist) > plane.length * 0.5) {
        return hit;
    }


    hit.has_hit = 1u;
    hit.t = t;
    hit.pos = vec4<f32>(hit_pos, 0.0);
    hit.normal = vec4<f32>(n, 0.0);
    hit.albedo = vec4<f32>(plane.albedo);
    hit.emission = plane.emission;
    hit.metallic = plane.metallic;
    hit.roughness = plane.roughness;

    return hit;
}


// ---- BVH traversal ---------------------------------------------------------

// Traversal stack depth. Must exceed the deepest tree under test or counts get
// truncated -- the harness compares this against the CPU-side max_depth and
// reports any ray that bailed out via the `incomplete` counter.
const BVH_STACK_SIZE: u32 = 64u;
// Hard step guard so a malformed tree cannot hang the GPU. Set far above any
// legitimate traversal so it never biases the measured counts.
const BVH_MAX_STEPS: u32 = 65536u;

//#if INSTRUMENTED
// Per-invocation counters, accumulated across every bounce of every sample and
// flushed to `ray_counters` once at the end of the entry point.
var<private> ctr_node_visits: u32 = 0u;
var<private> ctr_interior_visits: u32 = 0u;
var<private> ctr_prim_tests: u32 = 0u;
var<private> ctr_ray_count: u32 = 0u;
var<private> ctr_incomplete: u32 = 0u;
var<private> ctr_max_stack: u32 = 0u;
//#endif

/// Reciprocal of a direction with no infinities.
///
/// The naive `1.0 / direction` produces +/-inf on a zero component, and the slab
/// test then evaluates `0 * inf = NaN` whenever the ray origin lies exactly on a
/// slab plane -- exactly what axis-parallel rays through an axis-aligned uniform
/// grid generate. NaN propagates through min/max unpredictably and would corrupt
/// both the hit result and the traversal counts.
///
/// Clamping the magnitude to a tiny non-zero value keeps every product finite
/// while preserving the sign, so a ray parallel to a slab still resolves
/// correctly: outside the slab it yields two same-signed huge values (a miss on
/// that axis), inside it yields -huge/+huge (no constraint).
fn safe_inv_dir(direction: vec3<f32>) -> vec3<f32> {
    let tiny = vec3<f32>(1e-20);
    let magnitude = max(abs(direction), tiny);
    let signs = select(vec3<f32>(-1.0), vec3<f32>(1.0), direction >= vec3<f32>(0.0));
    return signs / magnitude;
}

/// Returns the entry distance t if hit, or -1.0 on a miss.
fn intersect_aabb(ray: Ray, aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> f32 {
    let inv_dir = safe_inv_dir(ray.direction);

    // Distance along the ray to each slab's two planes, per axis.
    let t0 = (aabb_min - ray.origin) * inv_dir;
    let t1 = (aabb_max - ray.origin) * inv_dir;

    // We cannot assume t0 < t1: when a direction component is negative the ray
    // reaches the max plane before the min plane, so t0 > t1 on that axis.
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);

    // Box entry = latest of the three slab entries.
    // Box exit  = earliest of the three slab exits.
    let tmin_max = max(max(tmin.x, tmin.y), tmin.z);
    let tmax_min = min(min(tmax.x, tmax.y), tmax.z);

    // Hit if the box interval is non-empty (entry <= exit) AND lies in front of
    // the ray. The 0.001 is a near-clip epsilon: it rejects hits extremely
    // close to the origin so a ray leaving a surface doesn't immediately
    // re-hit the box it just came from.
    if (tmax_min >= max(0.001, tmin_max)) {
        return tmin_max; // entry distance
    }
    return -1.0; // miss
}

fn traverse_bvh(ray: Ray) -> HitInfo {
    var closest_hit: HitInfo;
    closest_hit.has_hit = 0u;
    var closest_t = 3.402823466e+38;

    if (counts.bvh_node_count == 0u) {
        return closest_hit;
    }

//#if INSTRUMENTED
    ctr_ray_count += 1u;
//#endif

    var stack: array<u32, BVH_STACK_SIZE>;
    var stack_ptr = 0u;

    stack[0] = 0u;
    stack_ptr = 1u;

    var steps = 0u;
    var truncated = false;

    while (stack_ptr > 0u && steps < BVH_MAX_STEPS) {
        steps += 1u;
        stack_ptr -= 1u;

        let node_idx = stack[stack_ptr];

        if (node_idx >= counts.bvh_node_count) {
            continue;
        }

        let node = bvh_nodes[node_idx];

//#if INSTRUMENTED
        // One AABB test per node popped. Counted before the hit/miss branch so
        // the metric reflects work done, not work that paid off.
        ctr_node_visits += 1u;
        if (node.is_leaf != 1u) {
            ctr_interior_visits += 1u;
        }
//#endif

        let aabb_t = intersect_aabb(ray, node.min, node.max);
        if (aabb_t < 0.0 || aabb_t > closest_t) {
            continue;
        }

        if (node.is_leaf == 1u) {
            let first_tri = node.left_first;
            let tri_count = node.right_count;

            for (var i = 0u; i < tri_count; i++) {
                let idx_offset = first_tri + i;

                if (idx_offset >= counts.bvh_index_count) {
                    break;
                }

                let tri_idx = bvh_indices[idx_offset];
                if (tri_idx >= counts.triangle_count) {
                    continue;
                }

//#if INSTRUMENTED
                ctr_prim_tests += 1u;
//#endif

                let hit = hit_triangle(triangles[tri_idx], ray);
                if (hit.has_hit != 0u && hit.t < closest_t) {
                    closest_t = hit.t;
                    closest_hit = hit;
                }
            }
        } else {
            let left_child = node.left_first;
            let right_child = node.right_count;

            // Push right first so left is popped first (rough front-to-back for
            // a left-leaning build). Two slots are needed; if only one is free
            // the traversal would silently drop a subtree, so flag it instead.
            if (stack_ptr + 2u > BVH_STACK_SIZE) {
                truncated = true;
            } else {
                if (right_child < counts.bvh_node_count) {
                    stack[stack_ptr] = right_child;
                    stack_ptr += 1u;
                }
                if (left_child < counts.bvh_node_count) {
                    stack[stack_ptr] = left_child;
                    stack_ptr += 1u;
                }
//#if INSTRUMENTED
                ctr_max_stack = max(ctr_max_stack, stack_ptr);
//#endif
            }
        }
    }

//#if INSTRUMENTED
    if (truncated || steps >= BVH_MAX_STEPS) {
        ctr_incomplete += 1u;
    }
//#endif

    return closest_hit;
}

fn trace_scene(ray: Ray) -> HitInfo {
    var closest_hit: HitInfo;
    closest_hit.has_hit = 0u;
    var closest_t = 3.402823466e+38;

    for (var i = 0u; i < counts.sphere_count; i++) {
        let hit = hit_sphere(spheres[i], ray);
        if (hit.has_hit != 0u && hit.t < closest_t) {
            closest_t = hit.t;
            closest_hit = hit;
        }
    }
    if (counts.bvh_node_count > 1u) {
        let bvh_hit = traverse_bvh(ray);
        if (bvh_hit.has_hit != 0u && bvh_hit.t < closest_t) {
            closest_t = bvh_hit.t;
            closest_hit = bvh_hit;
        }
    } else {
        for (var i = 0u; i < counts.triangle_count; i++) {
            let hit = hit_triangle(triangles[i], ray);
            if (hit.has_hit != 0u && hit.t < closest_t) {
                closest_t = hit.t;
                closest_hit = hit;
            }
        }
    }

    for (var i = 0u; i < counts.plane_count; i++) {
        let hit = hit_plane(planes[i], ray);
        if (hit.has_hit != 0u && hit.t < closest_t) {
            closest_t = hit.t;
            closest_hit = hit;
        }
    }

    return closest_hit;
}
