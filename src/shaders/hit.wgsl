
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

// AABB intersection test
fn intersect_aabb(ray: Ray, aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    let inv_dir = 1.0 / ray.direction;
    let t0 = (aabb_min - ray.origin) * inv_dir;
    let t1 = (aabb_max - ray.origin) * inv_dir;

    let tmin = min(t0, t1);
    let tmax = max(t0, t1);

    let tmin_max = max(max(tmin.x, tmin.y), tmin.z);
    let tmax_min = min(min(tmax.x, tmax.y), tmax.z);

    return tmax_min >= max(0.0, tmin_max);
}

fn traverse_bvh(ray: Ray) -> HitInfo {
    var closest_hit: HitInfo;
    closest_hit.has_hit = 0u;
    var closest_t = 3.402823466e+38;

    if (counts.bvh_node_count == 0u) {
        return closest_hit;
    }

    var stack: array<u32, 32>;
    var stack_ptr = 0u;
    stack[0] = 0u;
    stack_ptr = 1u;

    while (stack_ptr > 0u) {
        stack_ptr -= 1u;
        let node_idx = stack[stack_ptr];

        if (node_idx >= counts.bvh_node_count) {
            continue;
        }

        let node = bvh_nodes[node_idx];
        if (!intersect_aabb(ray, node.min, node.max)) {
            continue;
        }

        if (node.is_leaf == 1u) {
            let first_tri = node.left_first;
            let tri_count = node.right_count;

            for (var i = 0u; i < tri_count; i++) {
                let tri_idx = bvh_indices[first_tri + i];
                if (tri_idx >= counts.triangle_count) {
                    continue;
                }

                let hit = hit_triangle(triangles[tri_idx], ray);
                if (hit.has_hit != 0u && hit.t < closest_t) {
                    closest_t = hit.t;
                    closest_hit = hit;
                }
            }
        } else {
            // Internal node - add children to stack
            let left_child = node.left_first;
            let right_child = node.right_count;

            if (stack_ptr < 31u) {
                stack[stack_ptr] = left_child;
                stack_ptr += 1u;
            }
            if (stack_ptr < 31u) {
                stack[stack_ptr] = right_child;
                stack_ptr += 1u;
            }
        }
    }

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

    if (counts.bvh_node_count > 0u) {
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