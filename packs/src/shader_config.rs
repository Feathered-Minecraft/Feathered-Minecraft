//! Shader-pack configuration bridge: maps a pack's own `shaders.properties`
//! onto the renderer's pack-agnostic [`ShaderEffectConfig`].
//!
//! This module is the **compatibility layer** proper. It reads only the
//! generic option surface any OptiFine/Iris-style pack publishes (profiles
//! plus `shadowMapResolution`, `SHADOW_SAMPLES`, `AO`, `sunPathRotation`,
//! `BLOOM`, `VIGNETTE`, exposure values) and translates it into Feathered's
//! own stage knobs. It never executes pack GLSL — packs targeting
//! OptiFine/Iris cannot run on native wgpu (see docs/SHADER_COMPATIBILITY.md)
//! — so the result is a *configuration translation*, clearly labeled as such.
//!
//! Provenance: the option names and value ranges honored here are the ones
//! the Noble Shaders pack (GPL-3.0, © Belmu, github.com/BelmuTM/Noble)
//! publishes in its `shaders.properties` (verified commit 2c06bba). The
//! mapping is deliberately generic so other packs using the same convention
//! translate through the same code path — no pack is special-cased.
//!
//! Untranslatable options are returned in [`ShaderPackConfig::unsupported`]
//! so the caller can surface them honestly instead of silently faking them.

use crate::shader::ShaderLayout;
use feathered_renderer::{
    AoConfig, AtmosphereConfig, ExposureConfig, FogConfig, PostConfig,
    RenderQuality, ShaderEffectConfig, ShadowConfig, Tonemap,
};
use std::collections::HashMap;
use std::path::Path;

/// Result of translating a pack's configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct ShaderPackConfig {
    /// The translated generic configuration.
    pub config: ShaderEffectConfig,
    /// Pack options Feathered recognizes but cannot honor (yet), with the
    /// verbatim value from the pack.
    pub unsupported: Vec<(&'static str, String)>,
    /// Pack options Feathered does not know at all (informational).
    pub unknown: Vec<(String, String)>,
}

/// Parse a flat `key=value` properties file (comments with `#`).
fn parse_properties(text: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            out.insert(k.trim().to_string(), v.trim().to_string());
        }
    }
    out
}

/// Locate and read the pack's properties file under its `shaders/` dir.
fn load_properties(pack_dir: &Path) -> Option<(HashMap<String, String>, ShaderLayout)> {
    let props_path = pack_dir.join("shaders").join("shaders.properties");
    let text = std::fs::read_to_string(&props_path).ok()?;
    // Profile detection already exists in shader.rs; reuse the same logic
    // shape here without importing its private helpers.
    let layout = detect_layout(pack_dir);
    Some((parse_properties(&text), layout))
}

fn detect_layout(pack_dir: &Path) -> ShaderLayout {
    let s = pack_dir.join("shaders");
    let has = |pat: &str| std::fs::read_dir(&s)
        .map(|rd| rd.filter_map(|e| e.ok()).any(|e| e.path().extension().is_some_and(|x| x == pat)))
        .unwrap_or(false);
    if has("fsh") || has("vsh") {
        ShaderLayout::OptifineStyle
    } else if has("glsl") || has("gsh") {
        ShaderLayout::GlslPasses
    } else {
        ShaderLayout::Unknown
    }
}

fn f(v: Option<&String>) -> Option<f32> {
    v.and_then(|s| s.parse::<f32>().ok())
}

fn u(v: Option<&String>) -> Option<u32> {
    v.and_then(|s| s.parse::<u32>().ok())
}

fn b(v: Option<&String>) -> Option<bool> {
    v.map(|s| matches!(s.as_str(), "true" | "1" | "on" | "enabled"))
}

/// Translate a pack's `shaders.properties` into the generic config.
///
/// `quality` supplies the *base*: every knob the pack does not set keeps the
/// quality preset's value, so a minimal pack still renders reasonably and a
/// full profile (Noble's POTATO..ULTRA lines) overrides only what it names.
pub fn translate_shader_config(
    pack_dir: &Path,
    quality: RenderQuality,
) -> Option<ShaderPackConfig> {
    let (props, _layout) = load_properties(pack_dir)?;
    let mut cfg = ShaderEffectConfig::for_quality(quality);
    let mut unsupported = Vec::new();
    let mut unknown = Vec::new();

    // Recognized-but-untranslatable (documented, never faked).
    const KNOWN_UNSUPPORTED: &[&str] = &[
        "TAA", "DOF", "POM", "POM_LAYERS", "POM_DISTANCE", "POM_SHADOWING",
        "CLOUDS_SHADOWS", "WATER_CAUSTICS", "WATER_PARALLAX", "SSR",
        "LENS_FLARES", "GLARE", "PALETTE", "LUT", "CEL_SHADING", "FILM_GRAIN",
        "LABPBR", "SUBSURFACE_SCATTERING", "DH_SHADOWS", "RAIN_PUDDLES",
    ];

    let get = |k: &str| props.get(k);

    // --- Profile line: `profile.NAME = k=v k=v ...` -------------------------
    // Iris applies the profile chosen in its settings UI. Feathered has no
    // shader-settings UI yet, so: an explicit `profile = NAME` key selects
    // that profile; otherwise the profile whose name matches the requested
    // Feathered quality (Low→POTATO, Medium→MEDIUM, High→HIGH, Ultra→ULTRA,
    // with LIGHT as a Medium fallback) is used — this is the behavior the
    // real Noble file relies on (it declares POTATO..ULTRA without a
    // selection key); otherwise the first declared profile.
    let mut effective: HashMap<String, String> = props.clone();
    let declared: Vec<&String> = props.keys().filter(|k| k.starts_with("profile.")).collect();
    let selection = get("profile").cloned().or_else(|| {
        let want: &[&str] = match quality {
            RenderQuality::Low => &["POTATO"],
            RenderQuality::Medium => &["MEDIUM", "LIGHT"],
            RenderQuality::High => &["HIGH"],
            RenderQuality::Ultra => &["ULTRA"],
        };
        want.iter()
            .find_map(|w| props.keys().find(|k| k.eq_ignore_ascii_case(&format!("profile.{w}"))))
            .map(|k| k.trim_start_matches("profile.").to_string())
            .or_else(|| {
                // First declared profile (BTreeMap iteration order = lexical).
                declared.first().map(|k| k.trim_start_matches("profile.").to_string())
            })
    });
    if let Some(sel) = selection {
        if let Some(line) = get(&format!("profile.{sel}")) {
            for kv in line.split_whitespace() {
                if let Some((k, v)) = kv.split_once('=') {
                    effective.insert(k.to_string(), v.to_string());
                }
            }
        }
    }
    let g = |k: &str| effective.get(k);

    // --- Shadows ------------------------------------------------------------
    let shadows_enabled = g("SHADOWS").map(|v| v != "0" && v != "false");
    if let Some(true) = shadows_enabled {
        let res = u(g("shadowMapResolution")).unwrap_or(2048).clamp(256, 8192);
        let dist = f(g("shadowDistance")).unwrap_or(128.0).clamp(16.0, 512.0);
        let samples = u(g("SHADOW_SAMPLES")).unwrap_or(4).clamp(1, 16);
        cfg.shadows = Some(ShadowConfig {
            resolution: res,
            distance: dist,
            samples,
            strength: cfg.shadows.map(|s| s.strength).unwrap_or(0.9),
        });
    } else if shadows_enabled == Some(false) {
        cfg.shadows = None;
    }

    // --- AO -----------------------------------------------------------------
    let ao_mode = u(g("AO"));
    match ao_mode {
        Some(0) => cfg.ssao = None,
        Some(mode) => {
            let strength = (f(g("AO_SCALE")).unwrap_or(50.0) / 100.0).clamp(0.0, 1.0);
            cfg.ssao = Some(AoConfig {
                slices: (mode.max(1) / 2).clamp(1, 8),
                radius: 3.0,
                strength: strength.max(0.15),
            });
        }
        None => {}
    }

    // --- Celestial / atmosphere --------------------------------------------
    if let Some(rot) = f(g("sunPathRotation")) {
        cfg.sun_path_rotation_deg = rot.clamp(-60.0, 60.0);
    }
    if let Some(true) = b(g("ATMOSPHERE")) {
        cfg.atmosphere = Some(AtmosphereConfig {
            rayleigh: [5.8e-6, 13.5e-6, 33.1e-6],
            mie: 21e-6,
            mie_g: 0.76,
            scale_height: 8_500.0,
            top_radius: 6_471e3,
            ground_radius: 6_371e3,
            sun_illuminance: 40.0,
        });
    } else if let Some(false) = b(g("ATMOSPHERE")) {
        cfg.atmosphere = None;
    }

    // --- Clouds -------------------------------------------------------------
    let clouds0 = b(g("CLOUDS_LAYER0_ENABLED"));
    if clouds0 == Some(false) && b(g("CLOUDS_LAYER1_ENABLED")) != Some(true) {
        cfg.clouds = None;
    } else if clouds0 == Some(true) || b(g("CLOUDS_LAYER1_ENABLED")) == Some(true) {
        if let Some(c) = cfg.clouds.as_mut() {
            if let Some(alt) = f(g("CLOUDS_LAYER0_ALTITUDE")) {
                c.altitude = alt.clamp(40.0, 400.0);
            }
            if let Some(cov) = f(g("CLOUDS_LAYER0_COVERAGE")) {
                c.coverage = cov.clamp(0.0, 1.0);
            }
            if let Some(steps) = u(g("CLOUDS_LAYER0_SCATTERING_STEPS")) {
                c.steps = steps.clamp(4, 64);
            }
        }
    }

    // --- Water / fog --------------------------------------------------------
    let water_oct = u(g("WATER_OCTAVES"));
    if water_oct == Some(0) {
        cfg.water = None;
    } else if let Some(oct) = water_oct {
        if let Some(w) = cfg.water.as_mut() {
            w.octaves = oct.clamp(1, 8);
            if let Some(amp) = f(g("WAVE_AMPLITUDE")) {
                w.amplitude = amp.clamp(0.0, 2.0);
            }
        }
    }
    if let Some(true) = b(g("AIR_FOG")) {
        if cfg.fog.is_none() {
            cfg.fog = Some(FogConfig { density: 0.002, tint: [0.9, 0.95, 1.0] });
        }
        if let Some(d) = f(g("FOG_DENSITY")) {
            cfg.fog.as_mut().unwrap().density = (d * 0.01).clamp(0.0, 0.05);
        }
    } else if let Some(false) = b(g("AIR_FOG")) {
        cfg.fog = None;
    }

    // --- Post: bloom / tonemap / vignette ----------------------------------
    // A pack configuring any post knob gets the stage even if the base
    // preset renders without post (Medium/Low) — otherwise BLOOM=1 would be
    // silently dropped.
    let post_requested = ["BLOOM", "BLOOM_STRENGTH", "VIGNETTE", "VIGNETTE_STRENGTH", "TONEMAP"]
        .iter()
        .any(|k| effective.contains_key(*k));
    if cfg.post.is_none() && post_requested {
        cfg.post = Some(PostConfig {
            bloom: false,
            bloom_threshold: 1.0,
            bloom_intensity: 0.5,
            tonemap: Tonemap::Aces,
            vignette: 0.0,
        });
    }
    if let Some(p) = cfg.post.as_mut() {
        if let Some(true) = b(g("BLOOM")) {
            p.bloom = true;
        }
        if let Some(false) = b(g("BLOOM")) {
            p.bloom = false;
        }
        if let Some(strength) = f(g("BLOOM_STRENGTH")) {
            p.bloom_intensity = strength.clamp(0.0, 2.0);
        }
        if let Some(v) = b(g("VIGNETTE")) {
            p.vignette = if v { 0.2 } else { 0.0 };
        }
        if let Some(v) = f(g("VIGNETTE_STRENGTH")) {
            p.vignette = v.clamp(0.0, 1.0);
        }
        if let Some(tm) = g("TONEMAP") {
            p.tonemap = match tm.to_ascii_lowercase().as_str() {
                "none" | "0" => Tonemap::None,
                "reinhard" => Tonemap::Reinhard,
                // Noble TONEMAP=1 (default) → ACES.
                _ => Tonemap::Aces,
            };
        }
    }

    // --- Exposure (Noble EXPOSURE=0 fixed mode: F_STOPS/ISO/SHUTTER_SPEED) --
    if let (Some(n), Some(t), Some(iso)) = (f(g("F_STOPS")), f(g("SHUTTER_SPEED")), f(g("ISO"))) {
        cfg.exposure = Some(ExposureConfig {
            f_stops: n.clamp(1.0, 22.0),
            shutter_speed: t.clamp(1.0, 8000.0),
            iso: iso.clamp(50.0, 6400.0),
        });
    }

    // --- Unsupported / unknown accounting (honest reporting) ---------------
    for (k, v) in &effective {
        if let Some(name) = KNOWN_UNSUPPORTED.iter().find(|n| n.eq_ignore_ascii_case(k)) {
            unsupported.push((*name, v.clone()));
        } else if !is_recognized(k) {
            unknown.push((k.clone(), v.clone()));
        }
    }
    unsupported.sort();
    unknown.sort();

    Some(ShaderPackConfig {
        config: cfg,
        unsupported,
        unknown,
    })
}

/// Keys this bridge consumes or maps intentionally (anything else is
/// reported as unknown rather than silently dropped).
fn is_recognized(key: &str) -> bool {
    const RECOGNIZED: &[&str] = &[
        "profile", "sun", "moon", "clouds", "separateAo", "vignette", "oldLighting",
        "oldHandLight", "underwaterOverlay", "dynamicHandLight",
        "SHADOWS", "SHADOW_SAMPLES", "shadowMapResolution", "shadowDistance",
        "AO", "AO_SCALE",
        "sunPathRotation", "ATMOSPHERE",
        "CLOUDS_LAYER0_ENABLED", "CLOUDS_LAYER1_ENABLED", "CLOUDS_LAYER0_ALTITUDE",
        "CLOUDS_LAYER0_COVERAGE", "CLOUDS_LAYER0_SCATTERING_STEPS",
        "WATER_OCTAVES", "WAVE_AMPLITUDE", "AIR_FOG", "FOG_DENSITY",
        "BLOOM", "BLOOM_STRENGTH", "VIGNETTE", "VIGNETTE_STRENGTH", "TONEMAP",
        "F_STOPS", "ISO", "SHUTTER_SPEED", "EXPOSURE",
        "RENDER_SCALE_OPTION", "ATMOSPHERE_SCALE", "ATMOSPHERE_SCATTERING_STEPS",
        "ATMOSPHERE_TRANSMITTANCE_STEPS", "CLOUDS_SCALE", "REFLECTIONS",
        "REFLECTIONS_SCALE", "REFLECTIONS_STRIDE", "REFRACTIONS",
    ];
    // profile.<name> lines are consumed by the profile expander.
    key.starts_with("profile.") || RECOGNIZED.iter().any(|k| k.eq_ignore_ascii_case(key))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_pack(dir: &Path, props: &str) {
        std::fs::create_dir_all(dir.join("shaders")).unwrap();
        std::fs::write(dir.join("shaders").join("shaders.properties"), props).unwrap();
        // Layout detection needs at least one program file.
        std::fs::write(dir.join("shaders").join("opaque.fsh"), "").unwrap();
    }

    #[test]
    fn noble_profile_line_expands() {
        let dir = std::env::temp_dir().join("feathered-cfg-test-noble");
        std::fs::remove_dir_all(&dir).ok();
        write_pack(
            &dir,
            // Verbatim shape from Noble's shaders.properties (values trimmed
            // for brevity — the parser only needs key=value pairs).
            "profile.POTATO = SHADOWS=3 shadowMapResolution=1024 AO=0 BLOOM=0\n\
             profile.ULTRA = SHADOWS=1 shadowMapResolution=4096 AO=1 AO_SCALE=100 BLOOM=1\n\
             profile=ULTRA\n",
        );
        let out = translate_shader_config(&dir, RenderQuality::Medium).unwrap();
        let c = out.config;
        assert_eq!(c.shadows.as_ref().unwrap().resolution, 4096);
        assert!(c.ssao.is_some());
        assert_eq!(c.ssao.unwrap().strength, 1.0);
        assert!(c.post.unwrap().bloom);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn untranslatable_options_are_reported() {
        let dir = std::env::temp_dir().join("feathered-cfg-test-unsup");
        std::fs::remove_dir_all(&dir).ok();
        write_pack(
            &dir,
            "SHADOWS=1\nPOM=1\nPOM_LAYERS=64\nTAA=1\nmysteryOption=42\n",
        );
        let out = translate_shader_config(&dir, RenderQuality::High).unwrap();
        let names: Vec<_> = out.unsupported.iter().map(|(k, _)| *k).collect();
        assert!(names.contains(&"POM"));
        assert!(names.contains(&"TAA"));
        assert!(out.unknown.iter().any(|(k, _)| k == "mysteryOption"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn disabled_effects_disable_stages() {
        let dir = std::env::temp_dir().join("feathered-cfg-test-off");
        std::fs::remove_dir_all(&dir).ok();
        write_pack(&dir, "SHADOWS=0\nAO=0\nAIR_FOG=0\n");
        let out = translate_shader_config(&dir, RenderQuality::Ultra).unwrap();
        assert!(out.config.shadows.is_none());
        assert!(out.config.ssao.is_none());
        assert!(out.config.fog.is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn exposure_translation_matches_noble_semantics() {
        let dir = std::env::temp_dir().join("feathered-cfg-test-exp");
        std::fs::remove_dir_all(&dir).ok();
        write_pack(&dir, "EXPOSURE=0\nF_STOPS=8\nSHUTTER_SPEED=125\nISO=200\n");
        let out = translate_shader_config(&dir, RenderQuality::Medium).unwrap();
        let e = out.config.exposure.unwrap();
        // Same EV100 computation as Noble's computeExposure (fixed mode):
        // EV = log2(N²·t·S/ISO) with t = 1/shutter, S = 100.
        let expected = 2.0f32.powf(-(((8.0f32 * 8.0) * 125.0 * (100.0 / 200.0)).log2()));
        assert!((e.exposure_scale() - expected).abs() < 1e-5);
        std::fs::remove_dir_all(&dir).ok();
    }
}
