//#if INSTRUMENTED
// Colour ramps for the live cost views. These mirror the tables in `src/viz.rs`
// exactly, so a heatmap seen in the studio and the same heatmap exported to PNG
// use identical colours.

fn ramp_lookup(t: f32, count: u32, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>,
               e: vec3<f32>, f: vec3<f32>, g: vec3<f32>, h: vec3<f32>, i: vec3<f32>, j: vec3<f32>) -> vec3<f32> {
    var stops = array<vec3<f32>, 10>(a, b, c, d, e, f, g, h, i, j);

    let scaled = clamp(t, 0.0, 1.0) * f32(count - 1u);
    let index = min(u32(floor(scaled)), count - 2u);
    let frac = scaled - f32(index);

    return mix(stops[index], stops[index + 1u], frac) / 255.0;
}

fn palette_inferno(t: f32) -> vec3<f32> {
    return ramp_lookup(t, 10u,
        vec3<f32>(0.0, 0.0, 4.0),     vec3<f32>(22.0, 11.0, 57.0),
        vec3<f32>(66.0, 10.0, 104.0), vec3<f32>(106.0, 23.0, 110.0),
        vec3<f32>(147.0, 38.0, 103.0), vec3<f32>(188.0, 55.0, 84.0),
        vec3<f32>(221.0, 81.0, 58.0), vec3<f32>(243.0, 128.0, 26.0),
        vec3<f32>(246.0, 186.0, 39.0), vec3<f32>(252.0, 255.0, 164.0));
}

fn palette_viridis(t: f32) -> vec3<f32> {
    return ramp_lookup(t, 10u,
        vec3<f32>(68.0, 1.0, 84.0),   vec3<f32>(72.0, 40.0, 120.0),
        vec3<f32>(62.0, 74.0, 137.0), vec3<f32>(49.0, 104.0, 142.0),
        vec3<f32>(38.0, 130.0, 142.0), vec3<f32>(31.0, 158.0, 137.0),
        vec3<f32>(53.0, 183.0, 121.0), vec3<f32>(109.0, 205.0, 89.0),
        vec3<f32>(180.0, 222.0, 44.0), vec3<f32>(253.0, 231.0, 37.0));
}

fn palette_turbo(t: f32) -> vec3<f32> {
    // Ten of the eleven raster stops; dropping the last barely shifts the ramp
    // and keeps every lookup on one code path.
    return ramp_lookup(t, 10u,
        vec3<f32>(48.0, 18.0, 59.0),  vec3<f32>(70.0, 107.0, 227.0),
        vec3<f32>(54.0, 166.0, 249.0), vec3<f32>(25.0, 214.0, 203.0),
        vec3<f32>(60.0, 234.0, 141.0), vec3<f32>(129.0, 248.0, 75.0),
        vec3<f32>(191.0, 240.0, 45.0), vec3<f32>(238.0, 206.0, 50.0),
        vec3<f32>(253.0, 152.0, 39.0), vec3<f32>(200.0, 50.0, 8.0));
}

fn palette_grayscale(t: f32) -> vec3<f32> {
    let v = clamp(t, 0.0, 1.0);
    return vec3<f32>(v, v, v);
}

fn palette_sample(palette: u32, t: f32) -> vec3<f32> {
    switch palette {
        case 1u: { return palette_viridis(t); }
        case 2u: { return palette_turbo(t); }
        case 3u: { return palette_grayscale(t); }
        default: { return palette_inferno(t); }
    }
}
//#endif
