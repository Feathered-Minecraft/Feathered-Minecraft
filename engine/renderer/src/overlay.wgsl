// In-game overlays: screen-space HUD + world-space outlines.
//
// One vertex format for both passes (see client/src/overlay.rs):
//   pos   — UI pixels (screen pass) or world blocks (outline pass)
//   color — straight RGBA8 (NOT premultiplied)
//   px    — unused; padding so both passes share the vertex layout
//
// The screen pass maps UI pixels → NDC with a uniform viewport size; the
// outline pass uses the scene view-projection. Both are alpha blended over
// the frame with no depth writes (outlines test depth, the HUD does not).

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

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

struct HudVertex {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec4<f32>, // Unorm8x4 arrives pre-normalized 0..1
    @location(2) px: vec2<f32>,
};

// --- screen-space pass -----------------------------------------------------

@vertex
fn vs_screen(v: HudVertex) -> VsOut {
    var out: VsOut;
    // UI pixels → NDC.
    let ndc = 2.0 * v.pos.xy / params.size.xy - vec2<f32>(1.0);
    out.clip = vec4<f32>(ndc.x, -ndc.y, 0.0, 1.0);
    out.color = v.color;
    return out;
}

// --- world-space outline pass ----------------------------------------------

@vertex
fn vs_outline(v: HudVertex) -> VsOut {
    var out: VsOut;
    out.clip = globals.view_proj * vec4<f32>(v.pos, 1.0);
    out.color = v.color;
    return out;
}

// --- shared fragment --------------------------------------------------------

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    // Straight-alpha over the already-tonemapped frame.
    return in.color;
}
