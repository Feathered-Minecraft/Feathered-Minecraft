// Terrain shader (Phase 1 + Phase 2 gbuffer path).
//
// Vertex layout matches feathered_chunk::Vertex (24 bytes):
//   loc0: Float32x3 position (world blocks, chunk-local)
//   loc1: Unorm16x2 uv (atlas texels / 65535)
//   loc2: Unorm8x4 tint (RGBA)
//   loc3: Uint8x2 (shade, anim_id)
//   loc4: Unorm8x2 light (sky, block) — mesher-baked voxel light levels
//
// Animation: anim_id > 0 indexes globals.anim_offsets; the offset shifts v
// down the sprite's frame strip (each frame is one guttered atlas row).
// The animation flag also marks water surfaces for the lighting stage via
// ANIM_WATER_FLAG (bit 7) — Noble's gbuffers_water equivalent.
//
// Phase-1 compatibility: the classic entry points (fs_main / fs_main_blend,
// outputting texel-shaded color directly) still compile and are used by the
// direct quality path. The staged pipeline uses fs_gbuffer[_blend], which
// writes the gbuffer layout the deferred pass consumes.

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
    @location(4) light: vec2<f32>,
};

// Bit 7 of anim_id marks animated water (fluid meshes).
const ANIM_WATER_FLAG: u32 = 128u;

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    // Unorm16x2 / Unorm8x4 arrive already normalized to 0..1 floats.
    @location(1) uv: vec2<f32>,
    @location(2) tint: vec4<f32>,
    @location(3) flags: vec2<u32>,
    // Unorm8x2 light arrives as f32 0..1 = level/255 (mesher stores 0..15).
    @location(4) light: vec2<f32>,
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
    out.light = light;
    return out;
}

// Water carries anim_id slot | 128; the lighting stage reads the flat flag.
fn is_water(a: u32) -> bool {
    return (a & ANIM_WATER_FLAG) != 0u;
}

// Shadow-map entry: depth-only, positions transformed by the sun's ortho
// matrix (globals.view_proj is rebound to the sun matrix for this pass).
@vertex
fn vs_shadow(
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) tint: vec4<f32>,
    @location(3) flags: vec2<u32>,
    @location(4) light: vec2<f32>,
) -> VsOut {
    var out: VsOut;
    out.clip = globals.view_proj * vec4<f32>(pos, 1.0);
    out.tint = tint;
    out.uv = uv;
    out.shade = f32(flags.x) / 255.0;
    out.anim_id = flags.y;
    out.light = light;
    return out;
}

// Shadow casters write binary-alpha coverage (torches, leaves, rails cast
// dithered shadows); translucent water does not cast.
@fragment
fn fs_shadow(in: VsOut) -> @location(0) vec4<f32> {
    if (is_water(in.anim_id)) {
        discard;
    }
    let tex = textureSample(atlas_tex, atlas_samp, in.uv);
    if (tex.a * in.tint.a < 0.5) {
        discard;
    }
    return vec4<f32>(1.0);
}

// --- Gbuffer path (staged pipeline) ---------------------------------------

struct GbufferOut {
    @location(0) albedo_shade: vec4<f32>,
    @location(1) lightmap: vec4<f32>,
};

@fragment
fn fs_gbuffer(in: VsOut) -> GbufferOut {
    let tex = textureSample(atlas_tex, atlas_samp, in.uv);
    if (tex.a * in.tint.a < 0.5) {
        discard;
    }
    var out: GbufferOut;
    // Linear albedo — the atlas is sRGB-encoded, textureSample decodes it;
    // the deferred stage shades in linear space. Face shade rides in alpha.
    out.albedo_shade = vec4<f32>(tex.rgb * in.tint.rgb, in.shade);
    out.lightmap = vec4<f32>(in.light, 0.0, 1.0);
    return out;
}

// Translucent gbuffer: blended write (water alpha to albedo alpha channel is
// consumed as the water marker by the lighting stage via the anim flag).
struct GbufferOutBlend {
    @location(0) albedo_shade: vec4<f32>,
    @location(1) lightmap: vec4<f32>,
};

@fragment
fn fs_gbuffer_blend(in: VsOut) -> GbufferOutBlend {
    let tex = textureSample(atlas_tex, atlas_samp, in.uv);
    let a = tex.a * in.tint.a;
    if (a < 0.02) {
        discard;
    }
    var out: GbufferOutBlend;
    out.albedo_shade = vec4<f32>(tex.rgb * in.tint.rgb, in.shade);
    out.lightmap = vec4<f32>(in.light, 0.0, a);
    return out;
}

// --- Phase-1 direct path (quality presets Low/Medium) ----------------------

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
