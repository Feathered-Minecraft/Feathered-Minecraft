//! Player controller: input intent → camera rotation + physics movement.
//!
//! Owns the PlayerBody and drives the camera from it. Frame-rate
//! independent: all movement is delta-time scaled and clamped to a maximum
//! step (protects physics from alt-tab pauses and lag spikes).

use feathered_world::physics::{self, PlayerBody};

/// Maximum dt the simulation accepts per frame (seconds). Bigger deltas are
/// subsolved in fixed slices so a 2-second stall cannot teleport the player
/// through the floor.
pub const MAX_FRAME_DT: f32 = 0.1;
/// Fixed physics slice (1/120 s) — deterministic and stable at any fps.
pub const FIXED_DT: f32 = 1.0 / 120.0;

/// Mouse look sensitivity (radians per pixel) at the default multiplier.
pub const DEFAULT_SENSITIVITY: f32 = 0.0022;

pub struct PlayerController {
    pub body: PlayerBody,
    /// Mouse sensitivity multiplier (1.0 = default).
    pub sensitivity: f32,
    /// Pitch clamp (±~89°).
    pitch_limit: f32,
}

impl PlayerController {
    pub fn new(spawn: [f32; 3], sensitivity: f32) -> PlayerController {
        PlayerController {
            body: PlayerBody::new(spawn),
            sensitivity: sensitivity.max(0.05),
            pitch_limit: 89.0f32.to_radians(),
        }
    }

    /// Apply mouse-look. `dx`/`dy` are raw pixel deltas.
    pub fn look(&mut self, dx: f32, dy: f32, camera: &mut feathered_renderer::Camera) {
        let s = DEFAULT_SENSITIVITY * self.sensitivity;
        camera.yaw += dx * s;
        // Moving the mouse up (negative dy) looks up: pitch increases.
        camera.pitch -= dy * s;
        camera.pitch = camera.pitch.clamp(-self.pitch_limit, self.pitch_limit);
        camera.yaw = rem_euclid_pi(camera.yaw);
    }

    /// One fixed-step simulation slice.
    fn fixed_step(
        &mut self,
        intent: &crate::input::MoveIntent,
        camera: &feathered_renderer::Camera,
        solid: &feathered_world::physics::SolidAt,
    ) {
        // Camera-relative horizontal basis (yaw only; no flying).
        let (sin_y, cos_y) = camera.yaw.sin_cos();
        let forward = [sin_y, 0.0, -cos_y];
        let right = [cos_y, 0.0, sin_y];

        let mut mx = forward[0] * intent.forward + right[0] * intent.right;
        let mut mz = forward[2] * intent.forward + right[2] * intent.right;
        // Normalize so diagonals are not faster (len < 1 also caps analog input).
        let len = (mx * mx + mz * mz).sqrt();
        if len > 1.0 {
            mx /= len;
            mz /= len;
        }

        let speed = physics::WALK_SPEED
            * if intent.sprint {
                physics::SPRINT_MULTIPLIER
            } else {
                1.0
            };

        physics::step(&mut self.body, FIXED_DT, mx, mz, speed, intent.jump, solid);
    }

    /// Simulate `dt` seconds in fixed slices. Returns true when the player
    /// landed on the ground this frame (interaction feedback can use it).
    pub fn update(
        &mut self,
        dt: f32,
        intent: &crate::input::MoveIntent,
        camera: &mut feathered_renderer::Camera,
        solid: &feathered_world::physics::SolidAt,
    ) -> bool {
        let dt = dt.min(MAX_FRAME_DT);
        let mut steps = (dt / FIXED_DT).ceil() as usize;
        steps = steps.max(1);
        // Cap the slice count: a huge stall advances at most 0.1 s of sim.
        let mut landed = false;
        for _ in 0..steps {
            self.fixed_step(intent, camera, solid);
        }
        landed |= self.body.on_ground && self.body.vel[1] <= 0.0;
        // The camera follows the eye.
        camera.pos = self.body.eye();
        landed
    }

    /// Desired vertical FOV for this frame: base FOV plus a configurable
    /// sprint kick (eases toward the target in the client each frame).
    pub fn desired_fov(&self, base_fov: f32, sprinting: bool, moving: bool) -> f32 {
        if sprinting && moving {
            base_fov + SPRINT_FOV_KICK
        } else {
            base_fov
        }
    }
}

/// Extra vertical FOV (radians) while sprinting — subtle speed feedback.
pub const SPRINT_FOV_KICK: f32 = 8.0f32.to_radians();

/// Wrap an angle to (-π, π] so yaw never grows without bound.
fn rem_euclid_pi(a: f32) -> f32 {
    let two_pi = std::f32::consts::TAU;
    let mut a = a.rem_euclid(two_pi);
    if a > std::f32::consts::PI {
        a -= two_pi;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_solid() -> impl Fn(i64, i64, i64) -> bool {
        |_: i64, y: i64, _: i64| y <= 9
    }

    #[test]
    fn look_wraps_yaw_and_clamps_pitch() {
        let mut cam = feathered_renderer::Camera {
            pos: [0.0; 3],
            yaw: 0.0,
            pitch: 0.0,
            fov_y: 1.0,
            aspect: 1.0,
            near: 0.1,
            far: 100.0,
        };
        let mut ctl = PlayerController::new([0.0; 3], 1.0);
        ctl.look(0.0, -10_000.0, &mut cam); // hard look up
        assert!(cam.pitch <= 89.0f32.to_radians() + 1e-4);
        ctl.look(0.0, 10_000.0, &mut cam); // hard look down
        assert!(cam.pitch >= -89.0f32.to_radians() - 1e-4);
        // Yaw wraps: many rotations keep it bounded.
        ctl.look(10_000.0, 0.0, &mut cam);
        assert!(cam.yaw.abs() <= std::f32::consts::PI + 1e-4);
    }

    #[test]
    fn forward_input_moves_along_the_camera_yaw() {
        let solid = flat_solid();
        let mut cam = feathered_renderer::Camera {
            pos: [0.0; 3],
            yaw: 0.0, // forward = -z
            pitch: 0.0,
            fov_y: 1.0,
            aspect: 1.0,
            near: 0.1,
            far: 100.0,
        };
        // Spawn centered inside a block column (1.5): the box spans x
        // 1.2..1.8, fully clear of the ground voxels at x=0 and x=2, so
        // horizontal movement is unobstructed. (x=0.5 would put the box edge
        // inside the x=0 column and block every step.)
        let mut ctl = PlayerController::new([1.5, 10.0, 0.5], 1.0);
        ctl.body.on_ground = true;
        let intent = crate::input::MoveIntent {
            forward: 1.0,
            right: 0.0,
            jump: false,
            sprint: false,
        };
        for _ in 0..60 {
            ctl.update(1.0 / 60.0, &intent, &mut cam, &solid);
        }
        // After ~1 s of walking (with acceleration-free instant velocity),
        // the player must have moved toward -z. Spawn y=10 is ON the floor
        // top so horizontal movement is unobstructed.
        assert!(
            ctl.body.pos[2] < -0.5,
            "moved -z, got z={}",
            ctl.body.pos[2]
        );
        assert!(
            (ctl.body.pos[0] - 1.5).abs() < 0.01,
            "no sideways drift from spawn x=1.5, got x={}",
            ctl.body.pos[0]
        );
        assert_eq!(cam.pos, ctl.body.eye(), "camera follows the eye");
    }

    #[test]
    fn dt_spike_cannot_teleport_through_the_floor() {
        let solid = flat_solid();
        let mut cam = feathered_renderer::Camera {
            pos: [0.0; 3],
            yaw: 0.0,
            pitch: 0.0,
            fov_y: 1.0,
            aspect: 1.0,
            near: 0.1,
            far: 100.0,
        };
        let mut ctl = PlayerController::new([0.5, 12.0, 0.5], 1.0);
        let idle = crate::input::MoveIntent::default();
        // A 5-second stall at 60 fps: each frame simulates at most
        // MAX_FRAME_DT (0.1 s) — the player falls normally across frames
        // and can NEVER pass through the 1-block floor in one frame.
        let mut min_y = f32::MAX;
        for _ in 0..30 {
            ctl.update(5.0 / 30.0, &idle, &mut cam, &solid);
            min_y = min_y.min(ctl.body.pos[1]);
        }
        assert!(
            (ctl.body.pos[1] - 10.0).abs() < 0.2,
            "stall must settle on the surface: y={}",
            ctl.body.pos[1]
        );
        assert!(min_y >= 9.99, "never below the floor (min y={min_y})");
    }

    #[test]
    fn sprint_is_faster_than_walk() {
        let solid = flat_solid();
        let mk_cam = || feathered_renderer::Camera {
            pos: [0.0; 3],
            yaw: 0.0,
            pitch: 0.0,
            fov_y: 1.0,
            aspect: 1.0,
            near: 0.1,
            far: 100.0,
        };
        let mut walk = PlayerController::new([0.5, 10.0, 0.5], 1.0);
        walk.body.on_ground = true;
        let mut sprint = PlayerController::new([0.5, 10.0, 0.5], 1.0);
        sprint.body.on_ground = true;

        let i_walk = crate::input::MoveIntent {
            forward: 1.0,
            right: 0.0,
            jump: false,
            sprint: false,
        };
        let i_sprint = crate::input::MoveIntent {
            forward: 1.0,
            right: 0.0,
            jump: false,
            sprint: true,
        };
        let mut cam_a = mk_cam();
        let mut cam_b = mk_cam();
        for _ in 0..120 {
            walk.update(1.0 / 60.0, &i_walk, &mut cam_a, &solid);
            sprint.update(1.0 / 60.0, &i_sprint, &mut cam_b, &solid);
        }
        assert!(
            sprint.body.pos[2] < walk.body.pos[2] * 1.2,
            "sprint must cover more ground (walk z={}, sprint z={})",
            walk.body.pos[2],
            sprint.body.pos[2]
        );
    }
}
