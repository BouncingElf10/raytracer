// Compute entry point. Shared verbatim by the interactive renderer and by both
// passes of the data-collection harness -- the only difference between the
// instrumented and clean builds is what the `//#if INSTRUMENTED` blocks add.

/// Per-pixel RNG seed.
///
/// Depends only on the pixel, the sample index and the configured seed, so the
/// same config replays the same random sequence on any machine. The interactive
/// renderer varies `frame_number` per frame to accumulate; the harness holds it
/// fixed and varies `sample` inside the dispatch instead.
fn pixel_seed(pixel: vec2<u32>, sample_: u32) -> u32 {
    return (pixel.y * 1973u + pixel.x) * 9277u
         + (counts.frame_number + sample_) * 26699u
         + counts.rng_seed * 7919u;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = vec2<u32>(counts.width, counts.height);
    let pixel_coords = global_id.xy;

    if (pixel_coords.x >= dims.x || pixel_coords.y >= dims.y) {
        return;
    }

    let idx = pixel_coords.y * dims.x + pixel_coords.x;

    // Primary rays are generated CPU-side and uploaded, so the instrumented and
    // clean passes are guaranteed to trace exactly the same rays.
    let ray = Ray(rays[idx].origin, rays[idx].direction);

    let samples = max(counts.samples, 1u);
    var color = vec3<f32>(0.0);

    for (var sample = 0u; sample < samples; sample++) {
        var rng_state = pixel_seed(pixel_coords, sample);
        color += trace_path(ray, &rng_state);
    }

    output_colors[idx] = vec4<f32>(color / f32(samples), 1.0);

//#if INSTRUMENTED
    ray_counters[idx] = RayCounters(
        ctr_node_visits,
        ctr_prim_tests,
        ctr_ray_count,
        ctr_interior_visits,
        ctr_incomplete,
        0u, 0u, 0u,
    );
//#endif
}
