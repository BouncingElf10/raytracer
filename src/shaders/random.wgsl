fn pcg_hash(seed: ptr<function, u32>) -> u32 {
    let state = *seed;
    *seed = state * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn random_float(seed: ptr<function, u32>) -> f32 {
    return f32(pcg_hash(seed)) / 4294967296.0;
}

fn random_cosine_hemisphere(normal: vec3<f32>, seed: ptr<function, u32>) -> vec3<f32> {
    let r1 = random_float(seed);
    let r2 = random_float(seed);

    let phi = 2.0 * 3.14159265359 * r1;
    let cos_theta = sqrt(r2);
    let sin_theta = sqrt(1.0 - r2);

    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal.y) > 0.99);
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);

    return normalize(
        tangent * (cos(phi) * sin_theta) +
        bitangent * (sin(phi) * sin_theta) +
        normal * cos_theta
    );
}