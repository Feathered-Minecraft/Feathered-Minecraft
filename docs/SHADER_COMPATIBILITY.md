# Feathered shader architecture & Noble Shaders provenance

Status: implemented (Phase 2). This document is the authoritative record of
what was investigated, what was integrated, what was **not** integrated, and
why. It exists so nobody can mistake Feathered's stages for a port of a
specific shader pack — and so every idea borrowed from published work is
traceable to its source.

## 1. What was investigated

**Upstream:** [BelmuTM/Noble](https://github.com/BelmuTM/Noble) — verified
official repository (the README links Modrinth/CurseForge/Discord/Patreon
pages that match the published pack). At investigation time:

* License: **GPL-3.0** (`LICENSE.txt`, GNU General Public License v3,
  verbatim FSF text), copyright headers read `Copyright (C) 2026  Belmu`.
* Commit inspected: `2c06bbaf52b9b177f0d04e50ffc60949a3609c8a`
  (2026-09-25, "Added End fog shadow ray …").
* Content: `shaders/` with 169 `.vsh` + 169 `.fsh` program files, 83 shared
  `.glsl` includes, 8 compute shaders (`.csh`), `shaders.properties`
  (~404 lines: profile definitions POTATO/LIGHT/MEDIUM/HIGH/ULTRA, ~150
  options, screen layouts), plus binary assets (noise/cloud curl `.dat`,
  color-grading LUT PNGs).

**What Noble actually requires to run** (from its own metadata and sources):

* OptiFine **or Iris for Minecraft 1.16+** (README requirements).
* On Iris it *requires* `iris.features.required = SSBO COMPUTE_SHADERS`
  (its `shaders.properties`), i.e. compute shaders + shader storage buffers
  for exposure/illuminance state.
* ~16 color attachments (colortex0–15) with documented formats (RGBA16F
  HDR main, RGBA32UI material gbuffer, R32F depth copies, RG16F temporal,
  …), depthtex0/1, shadowtex0/1 + shadowcolor0/1 and a full shadow pass.
* GLSL uniforms in the OptiFine/Iris convention: `gbufferProjection*`,
  `gbufferModelView*`, `shadowModelView/Projection*`, `cameraPosition`,
  `worldTime`, `frameCounter`, `sunAngle`, `isEyeInWater`, `heldBlock*`,
  biome queries, TAA jitters, `noisetex`, and the packed gbuffer format
  (normals/roughness/F0/SSS encoded in a RGBA32UI target).

**Conclusion:** Noble is GLSL source for a different runtime (Iris/OptiFine
on OpenGL). It cannot execute on Feathered's Rust + wgpu stack (WGSL,
explicit bind groups, no GLSL compiler, no Iris uniform magic, no compute
shaders in the core profile Feathered targets). **No Noble GLSL file is
included, compiled, or executed by Feathered.** What Feathered takes is
*algorithms* (re-implemented in WGSL from their published behavior) and its
*configuration surface* (option names/ranges for the translation bridge).

Research method note: Noble was consulted from a local clone kept **outside
the repository tree** (`target/noble-upstream/`, which `.gitignore` covers
via `/target`). Temporary working copies of its sources that were made in
`docs/research/` during the investigation were **deleted before commit** —
GPL-3.0 code is not redistributed from this repository, and Feathered's own
WGSL cites the upstream files it re-implements from instead.

## 2. Integrated: stage-by-stage provenance

Every item below is a fresh WGSL implementation in Feathered (this repo),
written against the referenced algorithm. None of it is copied code; where
a constant or parameterization matches Noble's it is cited inline.

| Stage (file) | Noble concept | What Feathered implements | Differences (documented, not hidden) |
| --- | --- | --- | --- |
| Voxel light (`feathered-world/src/light.rs`) | lightmap.xy from the engine (OptiFine supplies it; Noble consumes it) | Sky+block light BFS flood fill (15 levels, vanilla rules) baked into the mesh `light` attribute | Noble receives the engine's lightmap; Feathered computes its own from world data |
| Gbuffer (`terrain.wgsl fs_gbuffer*`) | colortex0/colortex1 gbuffer | Two-target gbuffer: linear albedo+shade (Rgba8UnormSrgb), lightmap (Rgba8Unorm; alpha = water marker) | No RGBA32UI material packing — normals/roughness/F0 are not produced yet |
| Shadow map (`terrain.wgsl vs_shadow/fs_shadow` + `sun_view_proj`) | shadow pass, shadowtex0/1, SHADOWS/SHADOW_SAMPLES/shadowMapResolution/shadowDistance | Sun-space ortho depth map following the camera, PCF (1–16 rotating jittered taps), binary-alpha casters, water excluded | Single map (Noble also keeps a color/second-depth map for colored & caustic shadows); no colored shadows |
| Deferred lighting (`lighting.wgsl fs_lighting`) | gbuffers→deferred→composite chain with BRDF | Deferred sun/sky/block-light compose using the lightmap, face shade, PCF visibility | No PBR BRDF: needs per-pixel normals + material params (colortex1) which the mesher does not emit |
| SSAO (`lighting.wgsl ssao`) | AO=1 GTAO / AO=2 SSAO / AO=3 RTAO (GTAO_SLICES/RADIUS, AO_SCALE) | Depth-only horizon-free occlusion estimate, radius/strength semantics matched, IGN jitter, multi-bounce-style boost | Not GTAO: no normal buffer ⇒ no slice/horizon integration, no bent normals |
| Atmosphere (`lighting.wgsl atmosphere`) | Rayleigh + Klein-Nishina Mie ray march (ATMOSPHERE_*), transmittance march | 8-step single-scattering march, Rayleigh + Henyey–Greenstein Mie, exponential airmass transmittance, sun disk with limb + transmittance tint | Fewer steps; K-N replaced by closed-form HG (visually equivalent at this step count); no multiple-scattering approximation term |
| Clouds (`lighting.wgsl raymarch_clouds`) | 2-layer volumetric clouds with curl/shape/detail `.dat` noise (CLOUDS_LAYER*) | Single raymarched slab layer: fBm value noise, rounded vertical profile, coverage/density/altitude/thickness/wind knobs, 1-sample sun shadowing, Beer-style transmittance | Single layer (Noble has two); hash noise instead of Noble's binary noise textures; no cloud shadows onto terrain |
| Water (`lighting.wgsl gerstner_derivative` + surface block) | Gerstner wave sum (WATER_OCTAVES/WAVE_*), Fresnel reflections (REFLECTIONS/SSR), refractions, caustics | Gerstner derivative sum (Noble's `pow(sin(x)·0.5+0.5, steepness)` parameterization, octave-rotated dirs), Schlick Fresnel (F0 0.02), animated normal, sky-reflection color + sun glint | No screen-space reflections/refractions (needs the SSR pass + colortex2), no water caustics, no parallax |
| Fog (`lighting.wgsl` fog block) | AIR_FOG volumetric-ish fog with scattering tint | Distance Beer–Lambert fog, sun-phase brightening, altitude-tinted | Not volumetric: no height/density ray march |
| Exposure (`shader_config.rs ExposureConfig`) | EV100 from f/ISO/shutter (`computeExposure`, EXPOSURE=0 fixed mode) | Same EV100 formula exposed to packs; scale applied in the tonemap chain | No auto-exposure (Noble's EXPOSURE=1 needs temporal history buffers + SSBO) |
| Post (`post.wgsl`) | BLOOM tiles, TONEMAP (ACES RRT+ODT), VIGNETTE, LUT grading, sharpen, film grain, DOF, TAA | Single-pass threshold bloom, ACES (Hill/Narkowicz fit — also Noble's TONEMAP=1 default operator) or Reinhard, vignette, IGN dither | No tiled bloom pyramid, no LUT color grading, no DOF/glare/sharpen/grain, no TAA |

### Constants shared with Noble (cited)

* EV100: `exposure = 2^(-log2(N²/t · S/100))`, calibration 12.5 /
  sensitivity 100 folded in — identical formula to Noble's
  `include/post/exposure.glsl` fixed mode (which itself follows the
  standard SSE/UE4 photometric exposure).
* Water Gerstner wave: `sin(x)·0.5 + 0.5` raised to `steepness`,
  octave-rotated directions — Noble `include/fragment/water.glsl`.
* Fresnel: Schlick with F0 = 0.02 for water — Noble
  `include/material/fresnel.glsl`.
* Interleaved gradient noise for shadow/AO dithering (Jimenez 2014) — used
  by Noble and by Feathered.
* ACES filmic Hill/Narkowicz fit — the operator Noble's TONEMAP=1 reduces
  to at display exposure; Feathered uses the same fit.
* Shadow resolution ladder (1024–4096), shadow distance (64–512), sun-path
  rotation (±60°), AO scale percent — ranges mirror Noble's
  `shaders.properties` so its profiles translate 1:1.

## 3. Explicitly NOT integrated (would be faking it)

* **Noble's GLSL programs.** They target Iris/OptiFine GLSL with
  engine-provided uniforms and buffers; wgpu has no path to compile them.
* **Compute-shader features** (Noble's Iris SSBO requirement): auto
  exposure, cached illuminance buffers, compute clouds.
* **GTAO/RTAO, bent normals, screen-space ray tracing** — require a normal
  buffer and/or screen-space ray marching over HDR color attachments that
  the current mesher/gbuffer does not produce.
* **PBR/LabPBR materials, parallax occlusion mapping, colored shadows,
  water caustics, rain puddles, subsurface scattering.**
* **TAA, DOF, lens flares, glare, LUT color grading, film grain, cel
  shading, palettes.**
* **Nether/End dimension shaders** (Noble has dedicated world1/world-1
  programs; Feathered only renders the overworld-style validation scene).
* **Noble's binary assets** (noise `.dat`, curl noise, grading LUTs) —
  not redistributed; Feathered's clouds use its own hash-based noise.
* **Two cloud layers, cloud shadows onto terrain, distant-horizons/voxy
  integration.**

None of the above is approximated behind a misleading name. Where a
feature is requested by a pack but untranslatable, the bridge reports it
(see §4) and the stage simply does not run.

## 4. Architecture: how packs integrate without hardcoding

```
resource pack pipeline (untouched)        shader pack (user-provided)
        │                                          │
        ▼                                          ▼
feathered-assets (compile)            feathered-packs (manage/validate;
        │                              license pointer, never relicensed)
        ▼                                          │
feathered-chunk (mesher, light attr)   packs::shader_config bridge
        │                              reads shaders.properties ONLY
        ▼                                          │
feathered-world (LightGrid)            generic ShaderEffectConfig ◄── built-in
        │                                          │            quality presets
        ▼                                          ▼
feathered-renderer (stage graph, pack-agnostic WGSL)
  gbuffer → shadow map → deferred lighting → post → present/capture
```

* **`ShaderEffectConfig`** (`engine/renderer/src/shader_config.rs`) is the
  only configuration interface: per-stage `Option<…>` knobs (shadows, AO,
  atmosphere, clouds, water, fog, post, exposure) plus sun angle/path.
  `for_quality(Low|Medium|High|Ultra)` supplies built-in presets; a pack
  translation replaces them wholesale via `Renderer::apply_shader_config`.
* **The renderer names no pack.** `RenderSettings.shader_pack` is an
  informational id for the UI; pipelines and shaders are pack-agnostic.
* **`packs::translate_shader_config`** (`packs/src/shader_config.rs`) is
  the compatibility layer: it parses a pack's `shaders.properties`
  (profile expansion + the common OptiFine/Iris option names) and returns
  a `ShaderPackConfig` = translated config + `unsupported: [(option,
  value)]` + `unknown: [(key, value)]`. The CLI prints the unsupported
  list when a pack is enabled, so "this pack asked for POM/TAA and
  Feathered did not run them" is always visible.
* **Per-effect configurability**: each stage toggles independently
  (`SHADOWS=0`, `AO=0`, `BLOOM=0`…), so low-end hardware can keep, say,
  shadows at 1024² with 4 PCF taps while disabling clouds/atmosphere.

### Quality presets (built-in, no pack required)

| | Low | Medium | High | Ultra |
| --- | --- | --- | --- | --- |
| render scale | 0.5× | 1.0 | 1.0 | 1.0 |
| lighting | direct (Phase-1 look) | deferred + gentle water | deferred + everything | deferred + heavier |
| shadow map | – | – | 1024², 4 taps, 96 b | 2048², 8 taps, 160 b |
| SSAO | – | – | 2 slices, r3 | 4 slices, r4 |
| atmosphere + clouds | – | – | 8-step march, 24-step clouds | 40-step clouds |
| water | – | 2-octave | 4-octave + reflections | 6-octave |
| fog / exposure / post | – | – | ACES + bloom + vignette | heavier bloom |

## 5. Licensing & provenance rules

* Feathered is GPL-3.0; Noble is GPL-3.0 (© Belmu). No Noble *source* is
  redistributed here — only re-implementations of published algorithms and
  compatibility with its configuration format, each cited where used and
  in the table above. The upstream repo is referenced (not bundled):
  https://github.com/BelmuTM/Noble (commit 2c06bba, GPL-3.0).
* Third-party shader packs stay user-provided, unmodified, and under their
  own licenses; `feathered-packs` records where each pack's license file
  lives and never relicenses or rewrites pack content.
* The pre-existing post pass (`post.wgsl`) is Feathered's own code and was
  previously described in the README as "Noble-style"; that wording was
  wrong and is fixed — it is a generic post chain (bloom/tonemap/vignette)
  that any configuration, with or without any pack, drives.
* Post-verification fixes (same session, before first commit): the NDC
  y-direction assumption in the fullscreen passes was wrong (WebGPU NDC y is
  up, rasterization maps clip +y to the top row — verified with a uv-gradient
  probe); `post.wgsl`/`lighting.wgsl` now flip uv.y and `world_pos`/the
  frustum-ray mixes were re-derived to match. The sun ortho matrix negated
  z (in-front casters were depth-clipped to an empty shadow map, so shadows
  were silently no-ops; the old shadow test compared staged-vs-legacy output
  and could not catch it). Both GPU tests were rewritten to catch these
  regressions (low-path row order; shadows compared staged-vs-staged).

## 6. Verification

* New unit tests: voxel light (`feathered-world`, 3), config bridge
  (`feathered-packs`, 4 incl. Noble-profile expansion + honest unsupported
  reporting), renderer effect-config ladder (`feathered-renderer`,
  exposure formula + quality monotonicity), stage-math tests in
  `engine/renderer/tests/stages.rs` (sun path, exposure, Gerstner,
  atmosphere RGB ordering, world-pos reconstruction) + headless GPU
  pipeline smoke test.
* End-to-end against the genuine upstream pack (`packs/tests/noble_e2e.rs`):
  the local Noble clone is imported through `ShaderPackManager::import_folder`
  (which now also parses the real Iris feature keys `iris.features.*`), its
  un-selected profile resolves to the one matching the requested Feathered
  quality (Noble declares POTATO..ULTRA without a `profile` selection key —
  Iris chooses it in its UI), and the translated config reports 6 known-
  unsupported + 175 unknown options instead of faking them. Skipped cleanly
  when the clone is absent (fresh checkouts, CI).
* Visual validation: `feathered render --screenshot` at every quality on
  the 26.3 validation scene; screenshots pixel-checked at scene-projected
  coordinates (sky rows, stone ground, glass wall, water pool, log side,
  pillar) and verified for upright orientation via row scans — see
  `target/noble-{low,medium,high,ultra}.png` (git-ignored artifacts).
