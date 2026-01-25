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
    center: vec3<f32>,
    normal: vec3<f32>,
    width: f32,
    length: f32,
    albedo: vec3<f32>,
    emission: f32,
    metallic: f32,
    roughness: f32,
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
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> spheres: array<Sphere>;
@group(0) @binding(1) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(2) var<storage, read> planes: array<Plane>;
@group(0) @binding(3) var<storage, read> rays: array<Ray>;
@group(0) @binding(4) var<storage, read_write> hits: array<HitInfo>;
@group(0) @binding(5) var<uniform> counts: Counts;

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
    let denom = dot(plane.normal, ray.direction);

    if (abs(denom) < eps) {
        return hit;
    }

    let t = dot(plane.center - ray.origin, plane.normal) / denom;

    if (t <= 0.001) {
        return hit;
    }

    let hit_pos = ray.origin + ray.direction * t;
    let n = normalize(plane.normal);

    var tangent: vec3<f32>;
    if (abs(n.x) > 0.9) {
        tangent = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        tangent = vec3<f32>(1.0, 0.0, 0.0);
    }

    let u = normalize(cross(n, tangent));
    let v = cross(n, u);

    let local = hit_pos - plane.center;
    let u_dist = dot(local, u);
    let v_dist = dot(local, v);

    if (abs(u_dist) > plane.width * 0.5 || abs(v_dist) > plane.length * 0.5) {
        return hit;
    }

    hit.has_hit = 1u;
    hit.t = t;
    hit.pos = vec4<f32>(hit_pos, 0.0);
    hit.normal = vec4<f32>(n, 0.0);
    hit.albedo = vec4<f32>(plane.albedo, 0.0);
    hit.emission = plane.emission;
    hit.metallic = plane.metallic;
    hit.roughness = plane.roughness;

    return hit;
}

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