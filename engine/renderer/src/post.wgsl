// Post-process pass (modular, pack-agnostic).
//
// This pass is deliberately engine-generic: it reads whatever configuration
// the active render settings supply (uniform `params`), so external shader
// packs can drive the same pipeline through feathered-packs' configuration —
// no single shader pack is hardcoded into the engine. (The effect set is a
// re-implementation of translatable shader-pack concepts; see
// docs/SHADER_COMPATIBILITY.md for what is implemented vs not, and which
// upstream designs the parameterizations cite.)
//
// Effects (bit flags in params.flags.x):
//   1 = ACES filmic tonemap, 2 = cheap highlight bloom, 4 = vignette.
// With all flags off the pass is a plain (linear-filtered) upscale, which is
// how the Low quality preset renders at half resolution.

struct Params {
    flags: vec4<u32>,
    // p0: bloom threshold, p1: bloom intensity, p2: vignette strength.
    p0: f32,
    p1: f32,
    p2: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var scene_tex: texture_2d<f32>;
@group(0) @binding(2) var scene_samp: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle from vertex_index — no vertex buffer needed.
@vertex
fn vs_post(@builtin(vertex_index) vi: u32) -> VsOut {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: VsOut;
    let p = pos[vi];
    out.clip = vec4<f32>(p, 0.0, 1.0);
    // WebGPU NDC y points UP: clip y=+1 rasterizes to the TOP framebuffer
    // row (verified empirically — a uv.y gradient painted red lands bright
    // at row 0). Texture v=0 is the top row, so flip y here. Getting this
    // wrong mirrors the pass output; two stacked fullscreen passes used to
    // cancel each other's flip and hide the bug in the staged path.
    out.uv = vec2<f32>((p.x + 1.0) * 0.5, 1.0 - (p.y + 1.0) * 0.5);
    return out;
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// ACES filmic approximation (Hill/Narkowicz fit).
fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_post(in: VsOut) -> @location(0) vec4<f32> {
    let flags = params.flags.x;
    var color = textureSampleLevel(scene_tex, scene_samp, in.uv, 0.0).rgb;

    // Single-pass highlight bloom: 8 taps around the pixel, thresholded.
    if ((flags & 2u) != 0u) {
        let tw = vec2<f32>(textureDimensions(scene_tex));
        let off = 1.5 / tw;
        var blur = vec3<f32>(0.0);
        for (var i = 0; i < 8; i = i + 1) {
            let ang = f32(i) * 0.7853982; // pi/4
            let dir = vec2<f32>(cos(ang), sin(ang)) * off;
            let s = textureSampleLevel(scene_tex, scene_samp, in.uv + dir, 0.0).rgb;
            blur = blur + max(vec3<f32>(0.0), s - vec3<f32>(params.p0));
        }
        color = color + blur * (params.p1 / 8.0) * 4.0;
    }

    if ((flags & 1u) != 0u) {
        color = aces(color * 1.05);
    }

    if ((flags & 4u) != 0u) {
        let d = distance(in.uv, vec2<f32>(0.5, 0.5));
        color = color * (1.0 - params.p2 * smoothstep(0.35, 0.85, d));
    }

    return vec4<f32>(color, 1.0);
}
