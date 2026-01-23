@group(0) @binding(0)
var screen_tex: texture_2d<f32>;

@group(0) @binding(1)
var screen_sampler: sampler;

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VSOut {
    var positions = array<vec2<f32>, 3>(
        vec2(-1.0, -3.0),
        vec2( 3.0,  1.0),
        vec2(-1.0,  1.0),
    );

    var uvs = array<vec2<f32>, 3>(
        vec2(0.0, 2.0),
        vec2(2.0, 0.0),
        vec2(0.0, 0.0),
    );

    var out: VSOut;
    out.pos = vec4(positions[idx], 0.0, 1.0);
    out.uv = uvs[idx];
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let color = textureSample(screen_tex, screen_sampler, in.uv);
    return color;
}
