//! User settings from the menu (persisted, applied live in-game).
//!
//! Stored at `<worlds root>/settings.json`. Values map onto the renderer's
//! quality presets and the client's camera/controls so every knob here has
//! an immediate effect — no restart required.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Camera field of view, degrees (70 = classic).
pub const FOV_MIN: f32 = 50.0;
pub const FOV_MAX: f32 = 110.0;
/// Mouse sensitivity multiplier bounds.
pub const SENS_MIN: f32 = 0.1;
pub const SENS_MAX: f32 = 5.0;
/// Streaming view distance in chunks.
pub const VIEW_MIN: i32 = 2;
pub const VIEW_MAX: i32 = 16;
/// Day length bounds, seconds (20 min = vanilla).
pub const DAY_MIN: f32 = 60.0;
pub const DAY_MAX: f32 = 7200.0;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Settings {
    /// Render quality preset name ("Low"|"Medium"|"High"|"Ultra").
    pub quality: String,
    /// Mouse sensitivity multiplier (1.0 = default).
    pub sensitivity: f32,
    /// Field of view in degrees.
    pub fov: f32,
    /// View distance in chunks.
    pub view_distance: i32,
    /// Day length in seconds.
    pub day_length: f32,
    /// HUD visible by default.
    pub hud: bool,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            quality: "Medium".into(),
            sensitivity: 1.0,
            fov: 70.0,
            view_distance: 6,
            day_length: 1200.0,
            hud: true,
        }
    }
}

impl Settings {
    fn clamp(&mut self) {
        self.sensitivity = self.sensitivity.clamp(SENS_MIN, SENS_MAX);
        self.fov = self.fov.clamp(FOV_MIN, FOV_MAX);
        self.view_distance = self.view_distance.clamp(VIEW_MIN, VIEW_MAX);
        self.day_length = self.day_length.clamp(DAY_MIN, DAY_MAX);
        if !matches!(self.quality.as_str(), "Low" | "Medium" | "High" | "Ultra") {
            self.quality = "Medium".into();
        }
    }

    /// Load from `path` (missing/corrupt → defaults).
    pub fn load(path: &std::path::Path) -> (Settings, bool) {
        match std::fs::read(path) {
            Ok(bytes) => match serde_json::from_slice::<Settings>(&bytes) {
                Ok(mut s) => {
                    s.clamp();
                    (s, true)
                }
                Err(_) => (Settings::default(), false),
            },
            Err(_) => (Settings::default(), false),
        }
    }

    /// Save (best effort).
    pub fn save(&self, path: &std::path::Path) -> bool {
        let mut c = self.clone();
        c.clamp();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match serde_json::to_vec_pretty(&c) {
            Ok(bytes) => std::fs::write(path, bytes).is_ok(),
            Err(_) => false,
        }
    }

    /// Cycle Low → Medium → High → Ultra (matches the in-game F4 order).
    pub fn cycle_quality(&mut self, forward: bool) {
        const ORDER: [&str; 4] = ["Low", "Medium", "High", "Ultra"];
        let idx = ORDER.iter().position(|q| *q == self.quality).unwrap_or(1);
        let next = if forward {
            (idx + 1) % ORDER.len()
        } else {
            (idx + ORDER.len() - 1) % ORDER.len()
        };
        self.quality = ORDER[next].into();
    }

    /// Step an f32 value by `delta`, clamped to [min, max].
    pub fn adjust_f32(field: &mut f32, delta: f32, min: f32, max: f32) {
        *field = (*field + delta).clamp(min, max);
    }

    /// Data dir for a worlds root (settings.json lives beside the worlds).
    pub fn path_for(worlds_dir: &std::path::Path) -> PathBuf {
        worlds_dir.join("settings.json")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sane() {
        let s = Settings::default();
        assert_eq!(s.quality, "Medium");
        assert_eq!(s.view_distance, 6);
        assert_eq!(s.fov, 70.0);
    }

    #[test]
    fn quality_cycles_both_directions() {
        let mut s = Settings::default();
        s.cycle_quality(true);
        assert_eq!(s.quality, "High");
        s.cycle_quality(true);
        assert_eq!(s.quality, "Ultra");
        s.cycle_quality(true);
        assert_eq!(s.quality, "Low", "wraps forward");
        s.cycle_quality(false);
        assert_eq!(s.quality, "Ultra", "wraps backward");
    }

    #[test]
    fn clamp_bounds_everything() {
        let mut s = Settings {
            quality: "Absurd".into(),
            sensitivity: 99.0,
            fov: 10.0,
            view_distance: 999,
            day_length: 0.0,
            hud: true,
        };
        s.clamp();
        assert_eq!(s.quality, "Medium", "unknown preset falls back");
        assert_eq!(s.sensitivity, SENS_MAX);
        assert_eq!(s.fov, FOV_MIN);
        assert_eq!(s.view_distance, VIEW_MAX);
        assert_eq!(s.day_length, DAY_MIN);
    }

    #[test]
    fn save_load_round_trip_and_corruption() {
        let dir = std::env::temp_dir().join(format!("feathered-settings-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = Settings::path_for(&dir);
        let mut s = Settings::default();
        s.fov = 90.0;
        s.quality = "Ultra".into();
        assert!(s.save(&path));
        let (loaded, ok) = Settings::load(&path);
        assert!(ok);
        assert_eq!(loaded.fov, 90.0);
        assert_eq!(loaded.quality, "Ultra");

        std::fs::write(&path, b"{{{").unwrap();
        let (fallback, ok) = Settings::load(&path);
        assert!(!ok);
        assert_eq!(fallback, Settings::default());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
