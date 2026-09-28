//! Configurable keybinds with sensible defaults.
//!
//! Actions map to winit `KeyCode`s. Defaults match the classic FPS scheme
//! (WASD + Space + Shift, E = capture, Esc = release/quit, F3/F4 debug and
//! quality). The mapping is loaded from (or written to)
//! `controls.json` next to the world when the app starts, so players can
//! rebind without touching code; unknown or missing entries fall back to
//! defaults. Pure data + mapping — fully testable without a window.

use serde::{Deserialize, Serialize};
use winit::keyboard::KeyCode;

/// Every bindable action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    Forward,
    Back,
    Left,
    Right,
    Jump,
    Sprint,
    #[serde(rename = "mouseCapture")]
    MouseCapture,
    #[serde(rename = "breakDebug")]
    BreakDebug,
    #[serde(rename = "cycleQuality")]
    CycleQuality,
    #[serde(rename = "toggleHud")]
    ToggleHud,
    #[serde(rename = "hotbar1")]
    Hotbar1,
    #[serde(rename = "hotbar2")]
    Hotbar2,
    #[serde(rename = "hotbar3")]
    Hotbar3,
    #[serde(rename = "hotbar4")]
    Hotbar4,
    #[serde(rename = "hotbar5")]
    Hotbar5,
    #[serde(rename = "hotbar6")]
    Hotbar6,
    #[serde(rename = "hotbar7")]
    Hotbar7,
    #[serde(rename = "hotbar8")]
    Hotbar8,
    #[serde(rename = "hotbar9")]
    Hotbar9,
}

/// The keymap: action → key. Missing/unknown file entries fall back to
/// defaults per action (partial overrides stay valid).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Controls {
    pub forward: KeyCode,
    pub back: KeyCode,
    pub left: KeyCode,
    pub right: KeyCode,
    pub jump: KeyCode,
    pub sprint: KeyCode,
    #[serde(rename = "mouseCapture")]
    pub mouse_capture: KeyCode,
    #[serde(rename = "breakDebug")]
    pub break_debug: KeyCode,
    #[serde(rename = "cycleQuality")]
    pub cycle_quality: KeyCode,
    #[serde(rename = "toggleHud")]
    pub toggle_hud: KeyCode,
    #[serde(rename = "hotbar1")]
    pub hotbar1: KeyCode,
    #[serde(rename = "hotbar2")]
    pub hotbar2: KeyCode,
    #[serde(rename = "hotbar3")]
    pub hotbar3: KeyCode,
    #[serde(rename = "hotbar4")]
    pub hotbar4: KeyCode,
    #[serde(rename = "hotbar5")]
    pub hotbar5: KeyCode,
    #[serde(rename = "hotbar6")]
    pub hotbar6: KeyCode,
    #[serde(rename = "hotbar7")]
    pub hotbar7: KeyCode,
    #[serde(rename = "hotbar8")]
    pub hotbar8: KeyCode,
    #[serde(rename = "hotbar9")]
    pub hotbar9: KeyCode,
}

impl Default for Controls {
    fn default() -> Controls {
        Controls {
            forward: KeyCode::KeyW,
            back: KeyCode::KeyS,
            left: KeyCode::KeyA,
            right: KeyCode::KeyD,
            jump: KeyCode::Space,
            sprint: KeyCode::ShiftLeft,
            mouse_capture: KeyCode::KeyE,
            break_debug: KeyCode::F3,
            cycle_quality: KeyCode::F4,
            toggle_hud: KeyCode::F1,
            hotbar1: KeyCode::Digit1,
            hotbar2: KeyCode::Digit2,
            hotbar3: KeyCode::Digit3,
            hotbar4: KeyCode::Digit4,
            hotbar5: KeyCode::Digit5,
            hotbar6: KeyCode::Digit6,
            hotbar7: KeyCode::Digit7,
            hotbar8: KeyCode::Digit8,
            hotbar9: KeyCode::Digit9,
        }
    }
}

impl Controls {
    /// Which action does `key` trigger (if any)?
    pub fn action_of(&self, key: KeyCode) -> Option<Action> {
        let c = self;
        Some(match key {
            k if k == c.forward => Action::Forward,
            k if k == c.back => Action::Back,
            k if k == c.left => Action::Left,
            k if k == c.right => Action::Right,
            k if k == c.jump => Action::Jump,
            k if k == c.sprint => Action::Sprint,
            k if k == c.mouse_capture => Action::MouseCapture,
            k if k == c.break_debug => Action::BreakDebug,
            k if k == c.cycle_quality => Action::CycleQuality,
            k if k == c.toggle_hud => Action::ToggleHud,
            k if k == c.hotbar1 => Action::Hotbar1,
            k if k == c.hotbar2 => Action::Hotbar2,
            k if k == c.hotbar3 => Action::Hotbar3,
            k if k == c.hotbar4 => Action::Hotbar4,
            k if k == c.hotbar5 => Action::Hotbar5,
            k if k == c.hotbar6 => Action::Hotbar6,
            k if k == c.hotbar7 => Action::Hotbar7,
            k if k == c.hotbar8 => Action::Hotbar8,
            k if k == c.hotbar9 => Action::Hotbar9,
            _ => return None,
        })
    }

    /// Load controls from `path`; missing or malformed files fall back to
    /// defaults (the game must always start). Returns (controls, loaded?).
    pub fn load(path: &std::path::Path) -> (Controls, bool) {
        match std::fs::read(path) {
            Ok(bytes) => match serde_json::from_slice::<Controls>(&bytes) {
                Ok(c) => (c, true),
                Err(_) => (Controls::default(), false),
            },
            Err(_) => (Controls::default(), false),
        }
    }

    /// Save controls (pretty JSON) — best effort, returns success.
    pub fn save(&self, path: &std::path::Path) -> bool {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match serde_json::to_vec_pretty(self) {
            Ok(bytes) => std::fs::write(path, bytes).is_ok(),
            Err(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_cover_the_classic_scheme() {
        let c = Controls::default();
        assert_eq!(c.action_of(KeyCode::KeyW), Some(Action::Forward));
        assert_eq!(c.action_of(KeyCode::Space), Some(Action::Jump));
        assert_eq!(c.action_of(KeyCode::F3), Some(Action::BreakDebug));
        assert_eq!(c.action_of(KeyCode::Digit9), Some(Action::Hotbar9));
        assert_eq!(c.action_of(KeyCode::KeyQ), None, "unbound keys are None");
    }

    #[test]
    fn rebinding_changes_the_mapping() {
        let mut c = Controls::default();
        c.forward = KeyCode::ArrowUp;
        assert_eq!(c.action_of(KeyCode::ArrowUp), Some(Action::Forward));
        assert_eq!(c.action_of(KeyCode::KeyW), None, "old binding released");
    }

    #[test]
    fn save_load_round_trip_and_corruption_fallback() {
        let dir = std::env::temp_dir().join(format!("feathered-controls-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("controls.json");
        let mut c = Controls::default();
        c.jump = KeyCode::KeyC;
        assert!(c.save(&path));
        let (loaded, ok) = Controls::load(&path);
        assert!(ok);
        assert_eq!(loaded.jump, KeyCode::KeyC);
        assert_eq!(
            loaded.forward,
            KeyCode::KeyW,
            "untouched keys keep defaults"
        );

        // Corrupt file → clean defaults, never a crash.
        std::fs::write(&path, b"{ not json ").unwrap();
        let (fallback, ok) = Controls::load(&path);
        assert!(!ok);
        assert_eq!(fallback, Controls::default());

        // Missing file → defaults, reported as not-loaded.
        let (fresh, ok) = Controls::load(&dir.join("missing.json"));
        assert!(!ok);
        assert_eq!(fresh, Controls::default());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
