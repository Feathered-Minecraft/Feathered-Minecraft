//! User settings persisted next to the pack data directory
//! (`<data>/feathered/settings.json`).

use feathered_assets::error::{AssetError, AssetResult};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Render quality preset. Maps to internal render scale + feature flags so
/// Feathered stays playable on low-end hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Quality {
    /// Integrated GPUs / old laptops: half-res render, no extras.
    Low,
    /// Default: full-res render, no post processing.
    #[default]
    Medium,
    /// Full-res + post processing (bloom, vignette, tonemap).
    High,
    /// High + soft shadows (higher shadow-map budget).
    Ultra,
}

impl Quality {
    /// Render-scale factor applied to the 3D scene before upscaling.
    pub fn render_scale(self) -> f32 {
        match self {
            Quality::Low => 0.5,
            Quality::Medium => 1.0,
            Quality::High => 1.0,
            Quality::Ultra => 1.0,
        }
    }

    /// Whether the optional post-process pass runs.
    pub fn post_process(self) -> bool {
        matches!(self, Quality::High | Quality::Ultra)
    }

    /// Next preset in Low → Medium → High → Ultra (settings hotkey).
    pub fn next(self) -> Self {
        match self {
            Quality::Low => Quality::Medium,
            Quality::Medium => Quality::High,
            Quality::High => Quality::Ultra,
            Quality::Ultra => Quality::Low,
        }
    }
}

/// Everything Feathered remembers between launches.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PackSettings {
    /// Id of the currently selected resource pack, if any.
    #[serde(default)]
    pub active_pack: Option<String>,
    /// Cached `pack_format` of the active pack (for UI display).
    #[serde(default)]
    pub pack_format: Option<u32>,
    /// Id of the currently enabled shader pack, if any.
    #[serde(default)]
    pub active_shader: Option<String>,
    /// Render quality preset.
    #[serde(default)]
    pub quality: Quality,
    /// True once the user has completed first-launch onboarding.
    #[serde(default)]
    pub onboarded: bool,
}

/// Load settings, creating a default file when missing/corrupt.
pub fn load_or_create(path: &Path) -> AssetResult<PackSettings> {
    if let Ok(text) = std::fs::read_to_string(path) {
        if let Ok(s) = serde_json::from_str::<PackSettings>(&text) {
            return Ok(s);
        }
        // Corrupt settings are not fatal: fall through and rewrite defaults.
    }
    let s = PackSettings::default();
    save(path, &s)?;
    Ok(s)
}

/// Persist settings atomically (write + rename).
pub fn save(path: &Path, settings: &PackSettings) -> AssetResult<()> {
    let text = serde_json::to_string_pretty(settings).map_err(|e| AssetError {
        path: path.display().to_string(),
        message: format!("settings serialize failed: {e}"),
    })?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| AssetError {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, text).map_err(|e| AssetError {
        path: tmp.display().to_string(),
        message: e.to_string(),
    })?;
    std::fs::rename(&tmp, path).map_err(|e| AssetError {
        path: path.display().to_string(),
        message: e.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_presets_have_sane_scales() {
        assert_eq!(Quality::Low.render_scale(), 0.5);
        assert_eq!(Quality::Medium.render_scale(), 1.0);
        assert!(!Quality::Medium.post_process());
        assert!(Quality::High.post_process());
        assert!(Quality::Ultra.post_process());
        assert_eq!(Quality::Ultra.next(), Quality::Low);
    }

    #[test]
    fn settings_round_trip_and_corruption_recovery() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("settings.json");

        let mut s = PackSettings::default();
        s.active_pack = Some("abc".into());
        s.quality = Quality::High;
        s.onboarded = true;
        save(&p, &s).unwrap();

        let loaded = load_or_create(&p).unwrap();
        assert_eq!(loaded.active_pack.as_deref(), Some("abc"));
        assert_eq!(loaded.quality, Quality::High);

        // Corrupt file → defaults, no panic.
        std::fs::write(&p, "{not json").unwrap();
        let recovered = load_or_create(&p).unwrap();
        assert_eq!(recovered, PackSettings::default());
    }
}
