//! Modular shader-effect configuration — Feathered's shader-pack abstraction.
//!
//! This module is deliberately **pack-agnostic**: it describes WHAT the
//! renderer's stages do (shadows, AO, atmosphere, clouds, water, bloom,
//! exposure, tonemap) and HOW MUCH of each to run. It never names a specific
//! shader pack. `feathered-packs` maps a pack's own configuration file onto
//! these knobs (see `packs/src/shader_config.rs`); the built-in quality
//! presets use `ShaderEffectConfig::for_quality`, so no pack at all is a
//! fully supported configuration.
//!
//! Provenance note: the *tuning ranges* (e.g. shadow map 1024–4096, sun-path
//! rotation ±60°, exposure calibration constants) mirror the option surface
//! of the Noble Shaders pack (GPL-3.0 © Belmu, github.com/BelmuTM/Noble) so
//! that translating its settings is a 1:1 mapping. All shader code in this
//! crate is Feathered's own WGSL; see docs/SHADER_COMPATIBILITY.md.

use crate::RenderQuality;

/// Which pipeline runs for a frame. `None` = the phase-1 direct path
/// (quality presets only, cheapest possible). `Some` = the full modular
/// chain: gbuffer pass → lighting stages → post chain.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ShaderEffectConfig {
    /// Orbiting sun angle (0..1 of a full day). Drives sun direction,
    /// atmosphere phase and shadow projection.
    pub sun_angle: f32,
    /// Tilt of the sun's east-west path in degrees (Noble: sunPathRotation,
    /// default −35°).
    pub sun_path_rotation_deg: f32,

    /// Directional shadow mapping (the `SHADOWS` knob family).
    pub shadows: Option<ShadowConfig>,
    /// Screen-space ambient occlusion (GTAO-family; Noble `AO`).
    pub ssao: Option<AoConfig>,
    /// Physically based sky + aerial perspective (Noble `ATMOSPHERICS`).
    pub atmosphere: Option<AtmosphereConfig>,
    /// Raymarched cloud layer (Noble `CLOUDS_LAYER0/1`).
    pub clouds: Option<CloudsConfig>,
    /// Water surface + Fresnel sky reflection (Noble `WATER`).
    pub water: Option<WaterConfig>,
    /// Volumetric-ish air fog with sun tint (Noble `AIR_FOG`).
    pub fog: Option<FogConfig>,
    /// Post chain: bloom, tonemap, vignette (Noble `POST_PROCESSING` screen).
    pub post: Option<PostConfig>,
    /// Manual EV100 exposure (auto-exposure requires history buffers; Noble
    /// `EXPOSURE=0` fixed mode uses these).
    pub exposure: Option<ExposureConfig>,
}

/// Directional shadow-map settings (Noble: shadowMapResolution,
/// shadowDistance, SHADOW_SAMPLES).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShadowConfig {
    /// Shadow map resolution (square). 1024 = Light, 4096 = Ultra (Noble).
    pub resolution: u32,
    /// Render distance covered by the shadow map, in blocks (Noble
    /// shadowDistance 64–512).
    pub distance: f32,
    /// PCF taps per fragment (Noble SHADOW_SAMPLES 4–16).
    pub samples: u32,
    /// Strength of the sun disk cast through the map (0..1).
    pub strength: f32,
}

/// SSAO settings (Noble AO=1 GTAO; AO_SCALE / GTAO_SLICES / GTAO_RADIUS).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AoConfig {
    /// Half-slices per pixel (Noble GTAO_SLICES 1–16 → this is half of it).
    pub slices: u32,
    /// Occlusion radius in blocks (Noble GTAO_RADIUS).
    pub radius: f32,
    /// Occlusion strength (Noble AO_STRENGTH / AO_SCALE).
    pub strength: f32,
}

/// Physically based atmosphere (Noble: ATMOSPHERE_SCALE, SCATTERING_STEPS,
/// TRANSMITTANCE_STEPS, sunPathRotation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtmosphereConfig {
    /// Rayleigh scattering coefficient (per meter, βR at sea level).
    pub rayleigh: [f32; 3],
    /// Mie scattering coefficient (βM).
    pub mie: f32,
    /// Mie anisotropy g (Henyey–Greenstein / Klein–Nishina shape).
    pub mie_g: f32,
    /// Rayleigh scale height (meters) and atmosphere top radius (meters).
    pub scale_height: f32,
    pub top_radius: f32,
    /// Ground radius (meters) — Feathered's world is block-scale, so the
    /// "planet" is centered far below the camera.
    pub ground_radius: f32,
    /// Sun disk illuminance multiplier (kcd/m²-ish, tuned by eye).
    pub sun_illuminance: f32,
}

/// Volumetric cloud settings (Noble CLOUDS_LAYER0_*: altitude, thickness,
/// coverage, density, wind speed).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloudsConfig {
    /// Cloud layer altitude (blocks above camera-plane world y=0).
    pub altitude: f32,
    /// Layer thickness (blocks).
    pub thickness: f32,
    /// Coverage 0..1 (Noble CLOUDS_LAYER0_COVERAGE).
    pub coverage: f32,
    /// Density multiplier 0..1.
    pub density: f32,
    /// Wind speed (blocks/second) for the noise drift.
    pub wind_speed: f32,
    /// March steps through the slab (quality knob).
    pub steps: u32,
}

/// Water settings (Noble WATER_OCTAVES, WAVE_*, WATER_ABSORPTION).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaterConfig {
    /// Gerstner octaves (Noble WATER_OCTAVES 1–8).
    pub octaves: u32,
    /// Wave amplitude multiplier (Noble WAVE_AMPLITUDE).
    pub amplitude: f32,
    /// Wave speed multiplier (Noble WAVE_SPEED).
    pub speed: f32,
    /// Deep-water absorption coefficient (Beer–Lambert, per block).
    pub absorption: [f32; 3],
    /// Sky reflection strength (Fresnel-boosted).
    pub reflection_strength: f32,
}

/// Air fog (Noble AIR_FOG: density + scattering tint).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FogConfig {
    /// Fog density (Beer–Lambert per block of distance).
    pub density: f32,
    /// Scattering color tint (wakes with the sun via `phase`).
    pub tint: [f32; 3],
}

/// Post-processing chain (Noble POST_PROCESSING screen: BLOOM, TONEMAP,
/// VIGNETTE; plus TAA which Feathered does not implement — see docs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PostConfig {
    /// Enable bloom.
    pub bloom: bool,
    /// Bloom threshold in HDR units.
    pub bloom_threshold: f32,
    /// Bloom strength.
    pub bloom_intensity: f32,
    /// Tonemap operator (Noble TONEMAP).
    pub tonemap: Tonemap,
    /// Vignette strength 0..1 (Noble VIGNETTE_STRENGTH).
    pub vignette: f32,
}

/// Tonemap operators (Noble TONEMAP setting; ACES is the Noble default).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Tonemap {
    /// No tonemap — clamped linear passthrough.
    None,
    /// ACES filmic approximation (Hill/Narkowicz fit — the same fit Noble's
    /// aces.odt path reduces to at display exposure).
    #[default]
    Aces,
    /// Simple Reinhard-Jodie curve.
    Reinhard,
}

/// Manual exposure (Noble EXPOSURE=0 mode: F_STOPS/ISO/SHUTTER_SPEED).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExposureConfig {
    /// F-stop number N.
    pub f_stops: f32,
    /// Shutter speed as 1/seconds.
    pub shutter_speed: f32,
    /// ISO sensitivity.
    pub iso: f32,
}

impl ExposureConfig {
    /// EV100 → exposure scale, exactly Noble's fixed-exposure path
    /// (`computeExposure`, EXPOSURE=0):
    /// `EV = log2(N² / (1/t) · S/ISO)` with t = 1/shutter, S = 100
    /// (sensorSensitivity; the 12.5 calibration constant cancels into it),
    /// `exposure = 2^-EV`.
    pub fn exposure_scale(&self) -> f32 {
        let ev100 = ((self.f_stops * self.f_stops)
            * self.shutter_speed
            * (100.0 / self.iso))
            .log2();
        2.0f32.powf(-ev100)
    }
}

impl ShaderEffectConfig {
    /// Built-in quality presets. These are Feathered's own defaults — a
    /// *configuration*, not a shader-pack port — and a pack config fully
    /// replaces them (see feathered-packs).
    pub fn for_quality(q: RenderQuality) -> ShaderEffectConfig {
        let (shadows, ssao, atmosphere, clouds, water, fog, post, exposure) = match q {
            RenderQuality::Low => (None, None, None, None, None, None, None, None),
            RenderQuality::Medium => (
                None,
                None,
                None,
                None,
                Some(WaterConfig {
                    octaves: 2,
                    amplitude: 0.35,
                    speed: 1.0,
                    absorption: [0.45, 0.08, 0.03],
                    reflection_strength: 0.35,
                }),
                None,
                None,
                None,
            ),
            RenderQuality::High => (
                Some(ShadowConfig {
                    resolution: 1024,
                    distance: 96.0,
                    samples: 4,
                    strength: 0.9,
                }),
                Some(AoConfig { slices: 2, radius: 3.0, strength: 0.7 }),
                Some(AtmosphereConfig {
                    rayleigh: [5.8e-6, 13.5e-6, 33.1e-6],
                    mie: 21e-6,
                    mie_g: 0.76,
                    scale_height: 8_500.0,
                    top_radius: 6_471e3,
                    ground_radius: 6_371e3,
                    sun_illuminance: 40.0,
                }),
                Some(CloudsConfig {
                    altitude: 160.0,
                    thickness: 60.0,
                    coverage: 0.5,
                    density: 0.7,
                    wind_speed: 3.0,
                    steps: 24,
                }),
                Some(WaterConfig {
                    octaves: 4,
                    amplitude: 0.5,
                    speed: 1.0,
                    absorption: [0.45, 0.08, 0.03],
                    reflection_strength: 0.6,
                }),
                Some(FogConfig { density: 0.0015, tint: [0.9, 0.95, 1.0] }),
                Some(PostConfig {
                    bloom: true,
                    bloom_threshold: 1.0,
                    bloom_intensity: 0.5,
                    tonemap: Tonemap::Aces,
                    vignette: 0.2,
                }),
                Some(ExposureConfig { f_stops: 8.0, shutter_speed: 125.0, iso: 200.0 }),
            ),
            RenderQuality::Ultra => (
                Some(ShadowConfig {
                    resolution: 2048,
                    distance: 160.0,
                    samples: 8,
                    strength: 1.0,
                }),
                Some(AoConfig { slices: 4, radius: 4.0, strength: 0.85 }),
                Some(AtmosphereConfig {
                    rayleigh: [5.8e-6, 13.5e-6, 33.1e-6],
                    mie: 21e-6,
                    mie_g: 0.76,
                    scale_height: 8_500.0,
                    top_radius: 6_471e3,
                    ground_radius: 6_371e3,
                    sun_illuminance: 40.0,
                }),
                Some(CloudsConfig {
                    altitude: 160.0,
                    thickness: 80.0,
                    coverage: 0.5,
                    density: 0.75,
                    wind_speed: 3.0,
                    steps: 40,
                }),
                Some(WaterConfig {
                    octaves: 6,
                    amplitude: 0.6,
                    speed: 1.0,
                    absorption: [0.45, 0.08, 0.03],
                    reflection_strength: 0.75,
                }),
                Some(FogConfig { density: 0.0025, tint: [0.9, 0.95, 1.0] }),
                Some(PostConfig {
                    bloom: true,
                    bloom_threshold: 0.85,
                    bloom_intensity: 0.7,
                    tonemap: Tonemap::Aces,
                    vignette: 0.25,
                }),
                Some(ExposureConfig { f_stops: 8.0, shutter_speed: 125.0, iso: 200.0 }),
            ),
        };
        ShaderEffectConfig {
            sun_angle: 0.18, // morning sun: long shadows, warm sky
            sun_path_rotation_deg: -35.0,
            shadows,
            ssao,
            atmosphere,
            clouds,
            water,
            fog,
            post,
            exposure,
        }
    }

    /// True when the full staged pipeline (gbuffer + deferred + post) runs.
    pub fn uses_staged_pipeline(&self) -> bool {
        self.post.is_some() || self.shadows.is_some() || self.ssao.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposure_matches_noble_fixed_mode() {
        // N=8, t=1/125, ISO 200 → EV100 = log2(64·125·0.5) = log2(4000) ≈ 11.97
        let e = ExposureConfig { f_stops: 8.0, shutter_speed: 125.0, iso: 200.0 };
        let ev = ((8.0f32 * 8.0) * 125.0 * (100.0 / 200.0)).log2();
        let expected = 2.0f32.powf(-ev);
        assert!((e.exposure_scale() - expected).abs() < 1e-6);
        assert!(expected > 0.0 && expected < 0.01, "HDR-scale exposure");
    }

    #[test]
    fn quality_ladder_is_monotonic() {
        // Low has nothing staged; Ultra has everything.
        for q in [
            RenderQuality::Low,
            RenderQuality::Medium,
            RenderQuality::High,
            RenderQuality::Ultra,
        ] {
            let c = ShaderEffectConfig::for_quality(q);
            match q {
                RenderQuality::Low => {
                    assert!(c.shadows.is_none() && c.atmosphere.is_none() && c.post.is_none());
                }
                RenderQuality::Medium => {
                    assert!(c.water.is_some() && c.post.is_none() && c.shadows.is_none());
                }
                RenderQuality::High => {
                    assert!(c.shadows.is_some() && c.atmosphere.is_some() && c.post.is_some());
                    assert_eq!(c.shadows.unwrap().resolution, 1024);
                }
                RenderQuality::Ultra => {
                    assert_eq!(c.shadows.unwrap().resolution, 2048);
                    assert_eq!(c.shadows.unwrap().samples, 8);
                    assert!(c.clouds.unwrap().steps > c.shadows.unwrap().samples);
                }
            }
        }
    }
}
