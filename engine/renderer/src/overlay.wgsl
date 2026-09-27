// In-game overlays: screen-space HUD + world-space outlines.
//
// One vertex format for both passes (see client/src/overlay.rs):
//   pos   — UI pixels (screen pass) or world blocks (outline pass)
//   color — straight RGBA8 (NOT premultiplied); per-vertex tint
//   px    — screen pass: UV into the bound texture (or x < 0 = flat, untex-
//           tured); outline pass: unused
//   uv    — reserved (padding so both passes share the vertex layout)
//
// The screen pass maps UI pixels → NDC with a uniform viewport size and can
// sample the menu/HUD texture (logo art); flat quads keep the sentinel UV.
// The outline pass uses the scene view-projection and never samples. Both
// are alpha blended over the frame with no depth writes (outlines test
// depth, the HUD does not).

struct ScreenParams {
    size: vec4<f32>, // xy = surface size in px, zw unused
};
struct SceneGlobals {
    view_proj: mat4x4<f32>,
    // Memory-compatible with the CPU's [f32; 64]; uniform storage requires
    // 16-byte array stride, so it is declared as vec4 lanes.
    anim_offsets: array<vec4<f32>, 16>,
};

@group(0) @binding(0) var<uniform> params: ScreenParams;
@group(0) @binding(0) var<uniform> globals: SceneGlobals;
// Group 1 (screen + outline layouts both carry it; the outline pass never
// samples — outline vertices pass the flat-UV sentinel): the overlay art
// textures (logo + full-screen menu background) + one linear sampler.
// UV sentinels: x < 0 = flat quad; 0..1 = logo; x > 1 = background at
// (x - 1, y).
@group(1) @binding(1) var screen_tex: texture_2d<f32>;
@group(1) @binding(2) var screen_samp: sampler;
@group(1) @binding(3) var bg_tex: texture_2d<f32>;
@group(1) @binding(4) var bg_samp: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
};

struct HudVertex {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec4<f32>, // Unorm8x4 arrives pre-normalized 0..1
    @location(2) px: vec2<f32>,    // UV (x < 0 sentinel = flat quad)
    @location(3) uv: vec2<f32>,    // reserved
};

// --- screen-space pass -----------------------------------------------------

@vertex
fn vs_screen(v: HudVertex) -> VsOut {
    var out: VsOut;
    // UI pixels → NDC.
    let ndc = 2.0 * v.pos.xy / params.size.xy - vec2<f32>(1.0);
    out.clip = vec4<f32>(ndc.x, -ndc.y, 0.0, 1.0);
    out.color = v.color;
    out.uv = v.px;
    return out;
}

// --- world-space outline pass ----------------------------------------------

@vertex
fn vs_outline(v: HudVertex) -> VsOut {
    var out: VsOut;
    out.clip = globals.view_proj * vec4<f32>(v.pos, 1.0);
    out.color = v.color;
    out.uv = vec2<f32>(-1.0, 0.0); // outlines never sample
    return out;
}

// --- shared fragment --------------------------------------------------------

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    // Flat quads carry the UV sentinel; textured quads tint the sampled
    // texel (straight-alpha PNG art × per-vertex straight-alpha tint).
    if (in.uv.x < 0.0) {
        return in.color;
    }
    if (in.uv.x > 1.0) {
        let bg = textureSample(bg_tex, bg_samp, vec2<f32>(in.uv.x - 1.0, in.uv.y));
        return bg * in.color;
    }
    let tex = textureSample(screen_tex, screen_samp, in.uv);
    return tex * in.color;
}
