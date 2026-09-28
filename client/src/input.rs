//! Keyboard + mouse input state for the sandbox.
//!
//! winit events funnel into here each frame; the controller reads a stable
//! snapshot instead of poking at raw event state. Nothing here allocates.

use crate::controls::Action;
use crate::controls::Controls;
use std::collections::HashSet;
use winit::keyboard::KeyCode;

/// Movement intent collected from the current key state.
#[derive(Debug, Clone, Copy, Default)]
pub struct MoveIntent {
    pub forward: f32, // +1 forward, -1 back
    pub right: f32,   // +1 right, -1 left
    pub jump: bool,
    pub sprint: bool,
}

impl MoveIntent {
    /// Is any movement requested (used to gate sprint-FOV feedback)?
    pub fn is_moving(self) -> bool {
        self.forward.abs() > 0.0 || self.right.abs() > 0.0
    }
}

/// Accumulated input for one frame.
#[derive(Debug, Default)]
pub struct InputState {
    keys: HashSet<KeyCode>,
    /// Unconsumed mouse deltas (pixels) since the last frame.
    pub mouse_dx: f32,
    pub mouse_dy: f32,
    /// Unconsumed hotbar wheel steps (positive = scroll up/next).
    pub wheel_steps: i32,
    /// The active keymap (rebindable; defaults until set otherwise).
    pub controls: Controls,
}

impl InputState {
    pub fn key_down(&self, key: KeyCode) -> bool {
        self.keys.contains(&key)
    }

    /// Is the action's key currently held?
    pub fn action_down(&self, action: Action) -> bool {
        let c = &self.controls;
        let key = match action {
            Action::Forward => c.forward,
            Action::Back => c.back,
            Action::Left => c.left,
            Action::Right => c.right,
            Action::Jump => c.jump,
            Action::Sprint => c.sprint,
            _ => return false,
        };
        self.key_down(key)
    }

    pub fn set_key(&mut self, key: KeyCode, down: bool) {
        if down {
            self.keys.insert(key);
        } else {
            self.keys.remove(&key);
        }
    }

    pub fn clear(&mut self) {
        self.keys.clear();
        self.mouse_dx = 0.0;
        self.mouse_dy = 0.0;
        self.wheel_steps = 0;
    }

    /// Take the movement intent (does not clear keys — held keys persist).
    /// Movement, jump and sprint respect the configured keymap.
    pub fn move_intent(&self) -> MoveIntent {
        let mut m = MoveIntent::default();
        if self.action_down(Action::Forward) {
            m.forward += 1.0;
        }
        if self.action_down(Action::Back) {
            m.forward -= 1.0;
        }
        if self.action_down(Action::Right) {
            m.right += 1.0;
        }
        if self.action_down(Action::Left) {
            m.right -= 1.0;
        }
        m.jump = self.action_down(Action::Jump);
        m.sprint = self.action_down(Action::Sprint);
        m
    }

    /// Consume accumulated mouse movement (call once per frame).
    pub fn take_mouse(&mut self) -> (f32, f32) {
        let d = (self.mouse_dx, self.mouse_dy);
        self.mouse_dx = 0.0;
        self.mouse_dy = 0.0;
        d
    }

    /// Consume accumulated wheel steps.
    pub fn take_wheel(&mut self) -> i32 {
        let w = self.wheel_steps;
        self.wheel_steps = 0;
        w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intent_is_normalized_by_the_controller_not_input() {
        let mut input = InputState::default();
        input.set_key(KeyCode::KeyW, true);
        input.set_key(KeyCode::KeyD, true);
        let m = input.move_intent();
        assert_eq!(m.forward, 1.0);
        assert_eq!(m.right, 1.0);
        assert!(!m.sprint);
        input.set_key(KeyCode::ShiftLeft, true);
        assert!(input.move_intent().sprint);
    }

    #[test]
    fn custom_keymap_changes_movement_bindings() {
        let mut input = InputState::default();
        input.controls.forward = KeyCode::ArrowUp;
        input.controls.jump = KeyCode::KeyC;
        input.set_key(KeyCode::ArrowUp, true);
        input.set_key(KeyCode::KeyC, true);
        input.set_key(KeyCode::KeyW, true); // no longer bound to forward
        let m = input.move_intent();
        assert_eq!(m.forward, 1.0, "rebound key drives forward");
        assert!(m.jump);
        assert!(
            !input.action_down(Action::Sprint),
            "unrelated keys untouched"
        );
    }

    #[test]
    fn opposite_keys_cancel() {
        let mut input = InputState::default();
        input.set_key(KeyCode::KeyW, true);
        input.set_key(KeyCode::KeyS, true);
        assert_eq!(input.move_intent().forward, 0.0);
    }

    #[test]
    fn mouse_and_wheel_accumulate_then_consume() {
        let mut input = InputState::default();
        input.mouse_dx += 3.0;
        input.mouse_dx += 4.0;
        input.wheel_steps += 2;
        assert_eq!(input.take_mouse(), (7.0, 0.0));
        assert_eq!(input.take_mouse(), (0.0, 0.0), "consumed once");
        assert_eq!(input.take_wheel(), 2);
        assert_eq!(input.take_wheel(), 0);
    }

    #[test]
    fn clear_releases_all_keys() {
        let mut input = InputState::default();
        input.set_key(KeyCode::KeyW, true);
        input.mouse_dx = 5.0;
        input.clear();
        assert_eq!(input.move_intent().forward, 0.0);
        assert_eq!(input.take_mouse(), (0.0, 0.0));
    }
}
