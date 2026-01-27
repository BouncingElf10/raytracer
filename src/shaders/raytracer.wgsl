@group(0) @binding(0) var<storage, read> rays: array<Ray>;
@group(0) @binding(1) var<storage, read_write> output_colors: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> spheres: array<Sphere>;
@group(0) @binding(3) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(4) var<storage, read> planes: array<Plane>;
@group(0) @binding(5) var<uniform> counts: Counts;

const MAX_BOUNCES: u32 = 10u;
const PI: f32 = 3.14159265359;

fn trace_path(initial_ray: Ray, seed: ptr<function, u32>) -> vec3<f32> {
    var ray = initial_ray;
    var throughput = vec3<f32>(1.0, 1.0, 1.0);
    var accumulated_light = vec3<f32>(0.0, 0.0, 0.0);

    for (var bounce = 0u; bounce < MAX_BOUNCES; bounce++) {
        let hit = trace_scene(ray);

        if (hit.has_hit == 0u) {
            break;
        }

        accumulated_light += throughput * hit.albedo.xyz * hit.emission;

        if (hit.emission > 0.0) {
            break;
        }

        let normal = normalize(hit.normal.xyz);
        let diffuse_dir = random_cosine_hemisphere(normal, seed);
        let specular_dir = reflect(ray.direction, normal);
        let final_dir = mix(specular_dir, diffuse_dir, hit.roughness);

        let diffuse_dot = max(dot(diffuse_dir, normal), 0.0);
        let specular_dot = max(dot(normalize(specular_dir), normal), 0.0);
        let brdf = hit.metallic * specular_dot + (1.0 - hit.metallic) * diffuse_dot;
        throughput *= hit.albedo.xyz * brdf;

        ray.origin = hit.pos.xyz + normal * 0.001;
        ray.direction = normalize(final_dir);
    }

    return accumulated_light;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = vec2<u32>(counts.width, counts.height);
        let pixel_coords = global_id.xy;
        let idx = pixel_coords.y * dims.x + pixel_coords.x;

        if (pixel_coords.x >= dims.x || pixel_coords.y >= dims.y) {
            return;
        }

        var rng_state = (pixel_coords.y * 1973u + pixel_coords.x) * 9277u + counts.frame_number * 26699u;

        let ray = Ray(rays[idx].origin, rays[idx].direction);
        let color = trace_path(ray, &rng_state);

        output_colors[idx] = vec4<f32>(color, 1.0);
}