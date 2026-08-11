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

    var out_color = color / f32(samples);

//#if INSTRUMENTED
    ray_counters[idx] = RayCounters(
        ctr_node_visits,
        ctr_prim_tests,
        ctr_ray_count,
        ctr_interior_visits,
        ctr_incomplete,
        ctr_max_stack,
        0u, 0u,
    );

    if (counts.display_mode != DISPLAY_BEAUTY) {
        out_color = cost_view(out_color);
    }
//#endif

    output_colors[idx] = vec4<f32>(out_color, 1.0);
}

//#if INSTRUMENTED
/// Maps this invocation's counters onto the selected ramp.
///
/// Written into the same colour buffer the renderer accumulates, so a cost view
/// converges over frames exactly like the render does -- the picture settles
/// instead of flickering with per-frame sampling noise.
fn cost_view(rendered: vec3<f32>) -> vec3<f32> {
    let rays = max(ctr_ray_count, 1u);

    var value = 0.0;
    switch counts.display_mode {
        case DISPLAY_NODE_VISITS: { value = f32(ctr_node_visits) / f32(rays); }
        case DISPLAY_PRIM_TESTS: { value = f32(ctr_prim_tests) / f32(rays); }
        // A high-water mark, not a total, so it is not divided by ray count.
        case DISPLAY_TRAVERSAL_DEPTH: { value = f32(ctr_max_stack); }
        case DISPLAY_LEAF_VISITS: { value = f32(ctr_node_visits - ctr_interior_visits) / f32(rays); }
        case DISPLAY_INTERIOR_VISITS: { value = f32(ctr_interior_visits) / f32(rays); }
        default: { value = 0.0; }
    }

    // Pixels that never entered the hierarchy stay black rather than taking the
    // ramp's darkest colour, so empty space reads as absent, not as cheap.
    if (ctr_ray_count == 0u) {
        return vec3<f32>(0.0);
    }

    let heat = palette_sample(counts.palette, value / max(counts.heat_scale, 0.0001));

    // The colour buffer is gamma-corrected downstream, so undo that here to keep
    // ramp colours exactly as authored.
    let linear_heat = heat * heat;

    return mix(linear_heat, rendered, clamp(counts.heat_mix, 0.0, 1.0));
}
//#endif
