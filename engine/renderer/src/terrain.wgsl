// Terrain shader (Phase 1).
// Vertex layout matches feathered_chunk::Vertex (24 bytes):
//   loc0: Float32x3 position (world blocks, chunk-local)
//   loc1: Unorm16x2 uv (atlas texels / 65535)
//   loc2: Unorm8x4 tint (RGBA)
//   loc3: Uint8x2 (shade, anim_id)
//
// Animation: anim_id > 0 indexes globals.anim_offsets; the offset shifts v
// down the sprite's frame strip (each frame is one guttered atlas row).

struct Globals {
    view_proj: mat4x4<f32>,
    // Memory-compatible with the CPU's [f32; 64]; uniform storage requires
    // 16-byte array stride, so it is declared as vec4 lanes.
    anim_offsets: array<vec4<f32>, 16>,
};
@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var atlas_tex: texture_2d<f32>;
@group(1) @binding(1) var atlas_samp: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) tint: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) shade: f32,
    @location(3) @interpolate(flat) anim_id: u32,
};

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    // Unorm16x2 / Unorm8x4 arrive already normalized to 0..1 floats.
    @location(1) uv: vec2<f32>,
    @location(2) tint: vec4<f32>,
    @location(3) flags: vec2<u32>,
) -> VsOut {
    var out: VsOut;
    out.clip = globals.view_proj * vec4<f32>(pos, 1.0);
    out.tint = tint;
    var v = uv.y;
    let slot = flags.y;
    if (slot > 0u) {
        v = v + globals.anim_offsets[slot >> 2u][slot & 3u];
    }
    out.uv = vec2<f32>(uv.x, v);
    out.shade = f32(flags.x) / 255.0;
    out.anim_id = slot;
    return out;
}

// Opaque/cutout path: binary alpha via discard.
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let tex = textureSample(atlas_tex, atlas_samp, in.uv);
    let shaded = tex.rgb * in.tint.rgb * in.shade;
    if (tex.a * in.tint.a < 0.5) {
        discard;
    }
    return vec4<f32>(shaded, 1.0);
}

// Translucent path (water/glass): real alpha blending.
@fragment
fn fs_main_blend(in: VsOut) -> @location(0) vec4<f32> {
    let tex = textureSample(atlas_tex, atlas_samp, in.uv);
    let shaded = tex.rgb * in.tint.rgb * in.shade;
    let a = tex.a * in.tint.a;
    if (a < 0.02) {
        discard;
    }
    return vec4<f32>(shaded, a);
}
