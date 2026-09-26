// Feathered deferred lighting pass — generic stage, configured per quality
// preset or by any shader pack through feathered-packs (see shader_config.rs).
//
// Algorithm provenance (see docs/SHADER_COMPATIBILITY.md for the full table):
// * Atmospheric single-scattering (Rayleigh + Mie) follows the structure of
//   Noble Shaders' `include/atmospherics/atmosphere.glsl` (GPL-3.0, © Belmu,
//   github.com/BelmuTM/Noble): ray-marched optical depth with per-step sun
//   transmittance and phase functions. This implementation uses a fixed
//   8-step march with an exponential airmass approximation for the sun ray —
//   Noble's default quality uses 12–24 steps plus a full transmittance march.
// * Water waves are classical Gerstner sums with the parameterization of
//   Noble's `include/fragment/water.glsl` (pow(sin(x)*0.5+0.5, steepness)
//   with octave-rotated directions); Fresnel is Schlick, as in Noble's
//   `include/material/fresnel.glsl` with F0 = 0.02 for water.
// * SSAO here is a depth-only occlusion approximation (see the fn comment).
//   Noble's GTAO/RTAO require a normal buffer, which Feathered's gbuffer
//   does not carry yet — that difference is documented, not hidden.
// * Directional face shading comes from the vanilla shade table baked by the
//   mesher; Noble's per-pixel BRDF needs material normals (unsupported).
//
// The pass reads the terrain gbuffer (color + lightmap target) and depth,
// composites sky/atmosphere, sun + block lighting, water, and fog, then
// tonemaps. It writes display-referred sRGB, which the existing post pass
// (bloom/vignette) consumes.
//
// Debug views (params.flags.w, set via env in lib.rs record_staged):
//   777 = raw atmosphere() output (FEATHERED_MARK_ATMO)
//   778 = uniform echo: rayleigh coeffs + pixel ray (FEATHERED_MARK_UNIFORM)
//   779 = raw pcf_visibility grayscale (FEATHERED_MARK_SHADOW)
//   780 = shadow-projection footprint + map-occupancy (FEATHERED_MARK_SHADOWMAP)
// These render the named intermediate into the lit target directly; they
// exist for diagnostics and are safe to leave set in a dev build.

struct Params {
    // x: effect flags (see EffectFlags in lib.rs), y: pcf sample count,
    // z: tonemap (0 none, 1 aces, 2 reinhard), w: unused.
    flags: vec4<u32>,
    // xyz camera world pos (blocks), w time (seconds).
    camera: vec4<f32>,
    // xyz sun direction (normalized, world), w sun elevation (sin).
    sun_dir: vec4<f32>,
    // x mie g, y sun disk intensity, z ambient strength, w vignette (unused)
    atmosphere: vec4<f32>,
    // rgb rayleigh β, w mie β
    rayleigh: vec4<f32>,
    // x water octaves, y amplitude, z speed, w reflection strength
    water: vec4<f32>,
    // rgb water absorption, w fog density
    absorption: vec4<f32>,
    // rgb fog tint, w exposure compensation
    fog_tint: vec4<f32>,
    // x ao strength, y ao radius, z near, w far
    ao: vec4<f32>,
    // x altitude, y thickness, z coverage, w steps (0 = clouds off)
    clouds: vec4<f32>,
    // x density, y wind speed, z/w unused
    clouds_b: vec4<f32>,
    // Normalized world-space ray directions to the frustum corners
    // (top-left, top-right, bottom-left, bottom-right) for sky-pixel ray
    // reconstruction.
    ray_tl: vec3<f32>,
    ray_tr: vec3<f32>,
    ray_bl: vec3<f32>,
    ray_br: vec3<f32>,
    inv_view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var albedo_tex: texture_2d<f32>;
@group(0) @binding(2) var depth_tex: texture_2d<f32>;
@group(0) @binding(3) var lin_samp: sampler;
@group(0) @binding(4) var lightmap_tex: texture_2d<f32>;
// --- Shadow uniform block (sun ortho matrix + always-bound shadow map; the
// shadows-enabled flag lives in params.flags, so one compiled module serves
// every configuration).
@group(0) @binding(5) var<uniform> shadow_params: ShadowParams;
@group(0) @binding(6) var shadow_tex: texture_depth_2d;
@group(0) @binding(7) var shadow_cmp: sampler_comparison;

struct ShadowParams {
    // Sun ortho projection (view_proj convention: op = P·V with row vectors
    // stored column-major — the CPU builds it exactly like Camera::view_proj).
    sun_view_proj: mat4x4<f32>,
    // x: map extent (blocks, ±), y: bias, z: shadow strength, w: unused.
    config: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle — identical mapping to post.wgsl (NDC y-up, uv y
// flipped; see post.wgsl vs_post).
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
    out.uv = vec2<f32>((p.x + 1.0) * 0.5, 1.0 - (p.y + 1.0) * 0.5);
    return out;
}

const PI: f32 = 3.14159265;

fn saturate(x: vec3<f32>) -> vec3<f32> {
    return clamp(x, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn hash12(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Interleaved gradient noise (Jimenez 2014) — the same jitter Noble uses for
// its AO/shadow dithering.
fn ign(coord: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(0.06711056 * coord.x + 0.00583715 * coord.y));
}

// Depth (0..1) → view distance (blocks). Perspective is the standard
// [0,1]-depth wgpu projection; see Camera::view_proj.
fn linear_depth(z: f32) -> f32 {
    let near = params.ao.z;
    let far = params.ao.w;
    return near * far / (far - z * (far - near));
}

// NDC/uv/depth → world position (blocks).
fn world_pos(uv: vec2<f32>, z: f32) -> vec3<f32> {
    // uv here is the same identity-space coordinate the gbuffer was written
    // at: convert back to NDC with y up (uv.y flipped, see vs_post).
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, z, 1.0);
    let p = params.inv_view_proj * ndc;
    return p.xyz / p.w;
}

// ---------------------------------------------------------------------------
// Atmosphere: ray-marched Rayleigh + Mie single scattering.
// ---------------------------------------------------------------------------

// Optical depth toward the sun along a chord — exponential airmass
// approximation (Kasten-style relative airmass, clamped for grazing rays).
fn sun_airmass(cos_elev: f32) -> f32 {
    return 1.0 / max(cos_elev, 0.035);
}

fn rayleigh_phase(mu: f32) -> f32 {
    return 3.0 / (16.0 * PI) * (1.0 + mu * mu);
}

// Henyey–Greenstein (Noble uses the Klein–Nishina phase with a matching
// shape; HG is the visually equivalent closed form at these step counts).
fn mie_phase(mu: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * mu;
    return 3.0 / (8.0 * PI) * (1.0 - g2) * (1.0 + mu * mu) / (denom * max(denom, 1e-4) * (2.0 + g2) * 0.5);
}

// Sky radiance along a view ray, originating `alt` meters above sea level.
// Returns linear RGB (HDR; tonemap happens at the end of the pass).
fn atmosphere(dir: vec3<f32>, sun_dir: vec3<f32>, alt: f32) -> vec3<f32> {
    let beta_r = params.rayleigh.rgb;
    let beta_m = vec3<f32>(params.rayleigh.w);
    let hr = 8500.0;
    let hm = 1200.0;
    let top = 6471e3;
    let ground = 6371e3;

    // Ray/planet-shell intersection ([R, top]); below horizon → dusk floor.
    let o = vec3<f32>(0.0, ground + max(alt, 2.0), 0.0);
    let b = dot(o, dir);
    let c = dot(o, o) - top * top;
    let disc = max(b * b - c, 0.0);
    let t_far = -b + sqrt(disc);
    // Ground hit: stop at the planet (rays into the ground get horizon color).
    let c2 = dot(o, o) - ground * ground;
    let disc2 = b * b - c2;
    var t_near = 0.0;
    if (disc2 > 0.0) {
        t_near = max(0.0, -b - sqrt(disc2));
    }
    let len = max(t_far - t_near, 0.0);

    let steps = 8.0;
    let step_size = len / steps;
    let mu = dot(dir, sun_dir);
    let ph_r = rayleigh_phase(mu);
    // Mie forward peak tempered for horizon-grazing rays: at dir.y→0 every
    // column samples mu≈1 and the raw phase (≈20×) turns the horizon into
    // per-column streaks. Blend toward isotropic as the ray flattens.
    let horizon_temper = mix(0.12, 1.0, smoothstep(0.0, 0.3, dir.y));
    let ph_m = mie_phase(mu, params.atmosphere.x) * horizon_temper;

    var total_r = vec3<f32>(0.0);
    var total_m = vec3<f32>(0.0);
    var od_r = 0.0;
    var od_m = 0.0;
    var t = t_near + step_size * 0.5;
    for (var i = 0; i < 8; i = i + 1) {
        let h = length(o + dir * t) - ground;
        let d_r = exp(-h / hr) * step_size;
        let d_m = exp(-h / hm) * step_size;
        od_r += d_r;
        od_m += d_m;
        // View→sample optical depth (accumulated) + sample→sun vertical
        // integral above h scaled by the relative airmass (Chapman-style
        // approximation of the transmittance march Noble runs explicitly).
        let view_od = beta_r * od_r + beta_m * od_m;
        let sun_od = sun_airmass(max(sun_dir.y, 0.035))
            * (beta_r * hr * exp(-h / hr) + beta_m * hm * exp(-h / hm));
        total_r += d_r * ph_r * exp(-(view_od + sun_od));
        total_m += d_m * ph_m * exp(-(view_od + sun_od));
        t += step_size;
    }
    // Sun illuminance → sky radiance scale. Calibrated so the Rayleigh sum
    // over the full atmospheric column lands the zenith near ~1.5 HDR
    // (display ~0.65 after ACES) — the 8-step march over-accumulates
    // relative to a full N-layer model, so this constant absorbs that.
    let isun = params.atmosphere.y * 2.5;
    return (total_r * beta_r + total_m * beta_m) * isun;
}

// Transmittance of direct sunlight through the whole atmosphere (vertical
// column from sea level × relative airmass).
fn atmosphere_transmittance(sun_dir: vec3<f32>, alt: f32) -> vec3<f32> {
    let beta_r = params.rayleigh.rgb;
    let beta_m = vec3<f32>(params.rayleigh.w);
    let airmass = sun_airmass(max(sun_dir.y, 0.035));
    let vertical_r = 8500.0 * exp(-alt / 8500.0);
    let vertical_m = 1200.0 * exp(-alt / 1200.0);
    return exp(-(beta_r * vertical_r + beta_m * vertical_m) * airmass);
}

// Sun tint at ground level: white at noon, deep red at the horizon.
fn sun_tint(elev: f32) -> vec3<f32> {
    return mix(vec3<f32>(1.0, 0.42, 0.18), vec3<f32>(1.0, 0.97, 0.93), smoothstep(0.02, 0.38, elev));
}

// ---------------------------------------------------------------------------
// Water: Gerstner wave derivative sum (Noble water.glsl parameterization).
// ---------------------------------------------------------------------------

fn gerstner_derivative(p: vec2<f32>, time: f32, octaves: u32, amplitude: f32, speed: f32) -> vec2<f32> {
    var derivative = vec2<f32>(0.0);
    var steepness = 2.2;
    var amp = 0.06 * amplitude;
    var lambda = 2.1;
    var t = time * speed;
    var dir = vec2<f32>(0.3, 0.5);
    let rot = mat2x2<f32>(0.866, -0.5, 0.5, 0.866); // ~30° per octave (Noble mixes 15°/155°)
    for (var i = 0u; i < octaves; i = i + 1) {
        let k = 2.0 * PI / lambda;
        let x = sqrt(9.81 * k) * t - k * dot(dir, p);
        let dudx = -0.5 * cos(x) * k;
        let sharp = steepness * pow(max(sin(x) * 0.5 + 0.5, 1e-4), steepness - 1.0) * dudx;
        derivative += amp * sharp * -dir;
        steepness = max(1.0, steepness * 0.82);
        amp *= 0.82;
        lambda *= 0.76;
        t += 0.35;
        dir = rot * dir;
    }
    return derivative;
}

fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}

// ---------------------------------------------------------------------------
// SSAO: depth-only occlusion approximation.
//
// Noble's GTAO searches horizon angles around the *normal*; with no normal
// buffer Feathered weights each screen-space tap by how much closer its
// world position is than the sphere of `radius` around the receiver — a
// horizon-free occlusion estimate with matched radius/strength semantics.
// ---------------------------------------------------------------------------
fn ssao(uv: vec2<f32>, frag: vec3<f32>, size: vec2<f32>) -> f32 {
    let strength = params.ao.x;
    let radius = params.ao.y;
    if (strength <= 0.001) {
        return 1.0;
    }
    let jitter = ign(uv * size) * 2.0 * PI;
    var occlusion = 0.0;
    var occluders = 0.0;
    let taps = 8;
    for (var i = 0; i < taps; i = i + 1) {
        let ang = (f32(i) / 8.0) * 2.0 * PI + jitter;
        let spiral = (0.25 + 0.75 * f32(i) / 8.0) * radius;
        let off = vec2<f32>(cos(ang), sin(ang)) * spiral;
        let suv = uv + off / size * vec2<f32>(1.0, -1.0);
        if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0) {
            continue;
        }
        let z = textureLoad(depth_tex, vec2<i32>(floor(suv * size)), 0).x;
        if (z >= 1.0) {
            continue;
        }
        let sp = world_pos(suv, z);
        let d = distance(sp, frag);
        // Occluders live ABOVE the receiver (smaller world y): walls next
        // to open ground don't count, real geometry overhead does.
        let h = frag.y - sp.y;
        if (h > 0.02) {
            occluders += 1.0;
            occlusion += (1.0 - d / radius) * saturate_f(h / radius * 2.0);
        }
    }
    // Normalize by samples that found geometry, not the raw tap count —
    // keeps open-field pixels at AO 1 while pits and seams darken.
    let ao = 1.0 - saturate_f(occlusion / max(occluders, 1.0) * strength);
    return mix(1.0, ao, step(0.5, occluders));
}

fn saturate_f(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Shadows: PCF over the sun ortho map (Noble SHADOWS=1 + SHADOW_SAMPLES).
// ---------------------------------------------------------------------------

// PCF sample grid over the sun-space footprint. `strength` from config.z.
fn pcf_visibility(frag: vec3<f32>, screen_uv: vec2<f32>) -> f32 {
    let clip4 = shadow_params.sun_view_proj * vec4<f32>(frag, 1.0);
    if (clip4.w <= 0.0) {
        return 1.0; // outside the light frustum: no shadow term
    }
    let clip = clip4.xyz / clip4.w;
    if (any(abs(clip.xy) > vec2<f32>(1.0))) {
        return 1.0;
    }
    // wgpu depth is [0,1]; shadow map stores sun-space depth.
    let depth = clamp(clip.z, 0.0, 1.0);
    // Map sun-space NDC to shadow-map texel coords. Rasterization puts
    // clip.y=+1 at the map's TOP row (row 0), while texture v=0 is row 0:
    // flip y (same convention as vs_post; see the comment there).
    let dims = vec2<f32>(textureDimensions(shadow_tex));
    let base = vec2<f32>(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5) * dims;
    let bias = shadow_params.config.y;
    let strength = shadow_params.config.z;
    let samples = params.flags.y;
    let n = max(samples, 1u);
    // Rotate the grid per-pixel (IGN) to decorrelate PCF banding.
    let rot = ign(screen_uv * dims);
    var acc = 0.0;
    for (var i = 0u; i < 16u; i = i + 1) {
        if (i >= n) { break; }
        let ang = f32(i) / f32(n) * 2.0 * PI + rot * 6.28;
        let r = sqrt(f32(i) / f32(n));
        let off = vec2<f32>(cos(ang), sin(ang)) * r * 1.5;
        let d = textureSampleCompareLevel(
            shadow_tex, shadow_cmp,
            (base + off) / dims,
            depth - bias
        );
        acc += d;
    }
    let vis = acc / f32(n);
    return mix(1.0, vis, strength);
}

// ACES filmic approximation (Hill/Narkowicz fit) — matches post.wgsl.
fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn reinhard(x: vec3<f32>) -> vec3<f32> {
    return x / (1.0 + x);
}

// Vanilla-style lightmap curve: 0..15 level → perceived brightness.
fn light_curve(level: f32) -> f32 {
    return pow(saturate_f(level / 15.0), 1.5);
}

// ---------------------------------------------------------------------------
// Clouds: raymarched slab with value-noise fBm (Noble CLOUDS_LAYER0_*:
// altitude, thickness, coverage, density, wind). Feathered uses hash-based
// value noise instead of Noble's curl/shape/detail .dat textures — the noise
// fields are large binary assets, so the character differs and that is
// documented rather than hidden.
// ---------------------------------------------------------------------------

fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash12(i);
    let b = hash12(i + vec2<f32>(1.0, 0.0));
    let c = hash12(i + vec2<f32>(0.0, 1.0));
    let d = hash12(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p: vec2<f32>) -> f32 {
    var v = 0.0;
    var amp = 0.5;
    var q = p;
    for (var i = 0; i < 4; i = i + 1) {
        v += amp * value_noise(q);
        q = q * 2.03 + vec2<f32>(17.3, 9.1);
        amp *= 0.5;
    }
    return v;
}

// Density at a world point; returns 0..1.
fn cloud_density(world: vec3<f32>, cfg: vec4<f32>, cfgb: vec4<f32>, time: f32) -> f32 {
    let base = world.y - cfg.x;                     // 0 at slab floor
    if (base < 0.0 || base > cfg.y) { return 0.0; }
    let frac = base / max(cfg.y, 1.0);              // 0..1 through the slab
    // Rounded vertical profile (denser mid-slab).
    let profile = 1.0 - pow(abs(frac - 0.5) * 2.0, 2.0) * 0.85;
    let scale = 0.012;                              // ~80-block noise cells
    let drift = vec2<f32>(time * cfgb.y, time * cfgb.y * 0.6);
    let n = fbm(world.xz * scale + drift);
    let d = saturate_f((n - (1.0 - cfg.z)) / max(cfg.z, 0.05));
    return saturate_f(d * profile * cfgb.x);
}

// March the [floor, ceiling] slab along `dir` from the camera; returns
// (rgb, alpha) of the cloud layer.
fn raymarch_clouds(dir: vec3<f32>, sun_dir: vec3<f32>, elev: f32, time: f32) -> vec4<f32> {
    let cfg = params.clouds;
    let cfgb = params.clouds_b;
    let steps = u32(max(cfg.w, 1.0));
    let floor_y = cfg.x;
    let ceil_y = cfg.x + cfg.y;
    let o = vec3<f32>(0.0, params.camera.y, 0.0);
    var t0 = (floor_y - o.y) / dir.y;
    var t1 = (ceil_y - o.y) / dir.y;
    // Below the slab looking up: t0 > 0, t1 > t0. Above the slab looking
    // down: both negative — order them.
    if (t0 > t1) { let tmp = t0; t0 = t1; t1 = tmp; }
    t0 = max(t0, 0.0);
    if (t1 <= t0) { return vec4<f32>(0.0); }
    let step_len = min((t1 - t0) / f32(steps), 24.0);

    let sun_tint_c = sun_tint(max(elev, 0.05));
    let ambient_c = mix(vec3<f32>(0.25, 0.3, 0.45), vec3<f32>(0.7, 0.75, 0.9), saturate_f(elev * 2.0 + 0.2));
    var transmittance = 1.0;
    var scatter = vec3<f32>(0.0);
    var t = t0 + step_len * 0.5;
    for (var i = 0u; i < 64u; i = i + 1) {
        if (i >= steps || transmittance < 0.02) { break; }
        let p = o + dir * t;
        let d = cloud_density(p, cfg, cfgb, time);
        if (d > 0.001) {
            // One-sample sun shadow: density toward the sun.
            let ds = cloud_density(p + sun_dir * 10.0, cfg, cfgb, time);
            let lit = exp(-ds * 6.0);
            let col = sun_tint_c * (0.65 + 0.85 * lit) + ambient_c * 0.35;
            let a = 1.0 - exp(-d * step_len * 0.25);
            scatter += col * a * transmittance;
            transmittance *= 1.0 - a;
        }
        t += step_len;
    }
    return vec4<f32>(scatter, 1.0 - transmittance);
}

@fragment
fn fs_lighting(in: VsOut) -> @location(0) vec4<f32> {
    let flags = params.flags.x;
    let size = vec2<f32>(textureDimensions(albedo_tex));
    let uvi = vec2<i32>(floor(in.uv * size));
    let z = textureLoad(depth_tex, uvi, 0).x;
    let albedo = textureLoad(albedo_tex, uvi, 0);
    let lm = textureLoad(lightmap_tex, uvi, 0);
    // Alpha channel of the lightmap target doubles as the water marker
    // (blend gbuffer writes 1.0 there; terrain writes 0.0).

    let sun_dir = normalize(params.sun_dir.xyz);
    let elev = params.sun_dir.w;

    // --- Sky pixels ---------------------------------------------------------
    if (z >= 1.0) {
        var sky: vec3<f32>;
        if ((flags & 4u) != 0u) {
            // uv.y=0 is the TOP row (v convention, see vs_post). ray_tl/tr
            // carry the ndc +y (up) rays, i.e. what the TOP row shows, so the
            // outer mix weight (0 = top rays) is plain uv.y.
            let v_weight = in.uv.y;
            if (params.flags.w == 777u) {
                // Debug: raw atmosphere() output, clamped scale (0 = black).
                let dir = normalize(mix(
                    mix(params.ray_tl, params.ray_tr, in.uv.x),
                    mix(params.ray_bl, params.ray_br, in.uv.x),
                    v_weight,
                ));
                let raw = atmosphere(dir, sun_dir, params.camera.y);
                return vec4<f32>(saturate(raw * 0.05), 1.0);
            }
            if (params.flags.w == 778u) {
                // Debug: uniform echo — rayleigh coefficients (should read
                // blue-dominant) mixed with the pixel ray direction.
                let dir = normalize(mix(
                    mix(params.ray_tl, params.ray_tr, in.uv.x),
                    mix(params.ray_bl, params.ray_br, in.uv.x),
                    v_weight,
                ));
                let beta_vis = saturate(params.rayleigh.rgb * 3.0e4) * 0.5;
                return vec4<f32>(saturate(beta_vis + dir * 0.5), 1.0);
            }
            // Ray-marched atmosphere (H/Ultra). Ray direction comes from the
            // CPU: inverse-projection of the pixel through the far plane is
            // numerically degenerate (w→0), so the renderer writes the four
            // frustum corner rays and we bilinearly interpolate (exact for
            // any perspective frustum). Below-horizon rays (past the world
            // edge) evaluate the horizon ray instead of the planet-hit void.
            let raw_dir = normalize(mix(
                mix(params.ray_tl, params.ray_tr, in.uv.x),
                mix(params.ray_bl, params.ray_br, in.uv.x),
                v_weight,
            ));
            let dir = normalize(vec3<f32>(raw_dir.x, max(raw_dir.y, 0.02), raw_dir.z));
            sky = atmosphere(dir, sun_dir, params.camera.y);
            // Raymarched cloud layer (behind the sun disk is fine: the disk
            // is added after and glints through). Grazing rays fade out:
            // their slab crossing is enormous and the march aliases into
            // per-column streaks at the horizon.
            if (params.clouds.w >= 1.0 && dir.y > 0.015) {
                let cl = raymarch_clouds(dir, sun_dir, params.sun_dir.w, params.camera.w);
                let cloud_fade = smoothstep(0.02, 0.12, dir.y);
                sky = mix(sky, cl.rgb, cl.a * cloud_fade);
            }
            // Sun disk with soft limb (evaluated on the raw ray so it sits
            // at its true elevation even when the sky ray is horizon-clamped).
            let cosr = dot(raw_dir, sun_dir);
            let disk = smoothstep(0.99955, 0.99985, cosr);
            sky = sky + vec3<f32>(1.0, 0.92, 0.78) * disk * params.atmosphere.y * 2.4 * atmosphere_transmittance(sun_dir, params.camera.y);
            // Below-horizon rays were clamped to the horizon ray: fade them
            // toward the ground fog color with the dip angle so the world's
            // edge reads as distance haze, not a glowing wall.
            let dip = saturate_f(-raw_dir.y * 12.0);
            sky = mix(sky, vec3<f32>(0.75, 0.78, 0.82), dip * 0.85);
        } else {
            // Clear-color sky already in the target; pass through.
            return vec4<f32>(albedo.rgb, 1.0);
        }
        sky = tonemap_chain(sky, flags, in.uv);
        return vec4<f32>(sky, 1.0);
    }

    // --- Terrain reconstruction --------------------------------------------
    let frag = world_pos(in.uv, z);
    if (params.flags.w == 779u) {
        // Debug: raw sun visibility (1 = lit, 0 = shadowed) as grayscale.
        let v = pcf_visibility(frag, in.uv);
        return vec4<f32>(vec3<f32>(v), 1.0);
    }
    if (params.flags.w == 780u) {
        // Debug: shadow-projection footprint. rgb = (uv.x, uv.y, written?)
        // blue 1 = map texel still cleared (empty), 0 = geometry present.
        let clip4 = shadow_params.sun_view_proj * vec4<f32>(frag, 1.0);
        let clip = clip4.xyz / max(clip4.w, 1e-5);
        let dims = vec2<f32>(textureDimensions(shadow_tex));
        let uv = vec2<f32>(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5);
        let probe = textureSampleCompareLevel(shadow_tex, shadow_cmp, uv, 0.999);
        return vec4<f32>(uv, 1.0 - probe, 1.0);
    }
    let cam_to = frag - params.camera.xyz;
    let dist = length(cam_to);
    let view_dir = cam_to / max(dist, 1e-4);

    let shade = clamp(albedo.a, 0.0, 1.0); // face shade baked into alpha
    var color = albedo.rgb;

    // Lightmap: gbuffer target 1 carries (sky, block) levels 0..15 / 255.
    let sky_level = lm.r * 255.0;
    let block_level = lm.g * 255.0;

    // Sun visibility: shadow map (PCF) when enabled, else unshadowed.
    var sun_vis = 1.0;
    if ((flags & 2u) != 0u) {
        sun_vis = pcf_visibility(frag, in.uv);
    }

    // Direct sun: Lambert term (N·L ≈ face shade × sun elevation) with the
    // 1/π diffuse normalization so illuminance stays in a sane HDR range
    // (sun ~14 at the surface → ground luminance ~2 for albedo 0.6).
    var direct = vec3<f32>(0.0);
    if (elev > 0.0) {
        // 0.18 folds the HDR sun illuminance (40) down to a Lambert-lit
        // ground luminance that lands mid-scale after tonemap.
        let sun_col = sun_tint(elev) * params.atmosphere.y * 0.18;
        let ndotl = shade * max(elev, 0.0) / PI;
        direct = sun_col * light_curve(sky_level) * sun_vis * ndotl * smoothstep(-0.05, 0.08, elev);
    } else {
        // Moon: dim cool ambient term (Noble celestial handling is richer).
        direct = vec3<f32>(0.16, 0.21, 0.35) * 0.28 * light_curve(sky_level) * shade / PI;
    }

    // Block light: warm torches (Noble BLOCKLIGHT default).
    let torch_col = vec3<f32>(1.0, 0.62, 0.35);
    var block_l = torch_col * light_curve(block_level) * 1.35 * mix(0.75, 1.0, shade);

    // Ambient sky light (cool from the sky dome; ~35% of direct strength).
    let amb_sky = mix(vec3<f32>(0.22, 0.26, 0.38), vec3<f32>(0.5, 0.62, 0.82), saturate_f(elev * 1.6 + 0.25));
    var ambient = amb_sky * light_curve(sky_level) * params.atmosphere.z * 0.35;

    // SSAO applies to ambient + block light, not the direct sun term.
    if ((flags & 1u) != 0u) {
        let ao = ssao(in.uv, frag, size);
        ambient *= ao;
        block_l *= mix(1.0, ao, 0.6);
    }

    color = color * (direct + block_l + ambient);

    // --- Water surface ------------------------------------------------------
    if ((flags & 8u) != 0u && lm.a > 0.5) {
        let wcfg = params.water;
        let n2d = gerstner_derivative(frag.xz, params.camera.w, u32(wcfg.x), wcfg.y, wcfg.z);
        let n = normalize(vec3<f32>(-n2d.x, 1.0, -n2d.y));
        // Reflect the view ray; Fresnel (Schlick, F0 = 0.02 water).
        let r = reflect(view_dir, n);
        let cos_t = saturate_f(dot(-view_dir, n));
        let f = fresnel_schlick(cos_t, 0.02);
        var refl: vec3<f32>;
        if ((flags & 4u) != 0u && r.y > -0.02) {
            refl = atmosphere(normalize(r), sun_dir, params.camera.y) * 0.12 + vec3<f32>(0.25, 0.4, 0.6) * 0.35;
        } else {
            refl = mix(vec3<f32>(0.35, 0.5, 0.7), vec3<f32>(0.62, 0.78, 0.95), saturate_f(elev + 0.3)) * 0.5;
        }
        // Sun glint.
        if (elev > 0.0) {
            let g = pow(saturate_f(dot(normalize(r), sun_dir)), 220.0);
            refl += sun_tint(elev) * g * 2.0;
        }
        color = mix(color, refl, saturate_f(f * wcfg.w * 2.2));
    }

    // --- Aerial fog ---------------------------------------------------------
    if ((flags & 16u) != 0u) {
        let density = params.absorption.w;
        let amount = 1.0 - exp(-dist * density);
        var fog_col = params.fog_tint.rgb;
        // Sun-phase tinting: fog brightens toward the sun.
        let mu = saturate_f(dot(view_dir, sun_dir));
        fog_col *= 0.7 + 0.6 * pow(mu, 6.0) * saturate_f(elev * 2.0);
        fog_col *= mix(vec3<f32>(0.35, 0.4, 0.55), vec3<f32>(1.05), saturate_f(elev * 2.0));
        color = mix(color, fog_col, saturate_f(amount));
    }

    color = tonemap_chain(color, flags, in.uv);
    return vec4<f32>(color, 1.0);
}

// Exposure compensation → tonemap → vignette-free sRGB encode happens via
// the Rgba8UnormSrgb target.
fn tonemap_chain(c: vec3<f32>, flags: u32, uv: vec2<f32>) -> vec3<f32> {
    // Manual exposure (Noble EXPOSURE=0: EV100 from f/ISO/shutter).
    var color = c * params.fog_tint.w;
    let tm = params.flags.z;
    if (tm == 1u) {
        color = aces(color);
    } else if (tm == 2u) {
        color = reinhard(color);
    } else {
        color = saturate(color);
    }
    // Ordered dither removes banding in the atmosphere gradient.
    color = color + (ign(uv * vec2<f32>(textureDimensions(albedo_tex))) - 0.5) / 255.0;
    return color;
}

