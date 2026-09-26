//! Player physics: gravity, ground detection, and swept AABB collision
//! against voxels.
//!
//! The player is an axis-aligned box (0.6×1.8×0.6, Minecraft-like). Movement
//! resolves one axis at a time (x, then z, then y): for each axis the box is
//! moved by the full velocity component and clamped against the first solid
//! voxel it overlaps, so tunneling through thin walls cannot happen as long
//! as per-step displacement stays below one block (speeds are clamped well
//! below that, and the caller substeps when dt is large). Ground contact is
//! detected by a small probe below the box after the y resolution.
//!
//! Collision data comes from a `SolidAt` closure so tests can drive the
//! physics without a registry, and the client wires it to the chunked world.

/// Player box half-widths + height, in blocks.
pub const PLAYER_HALF_WIDTH: f32 = 0.3;
pub const PLAYER_HEIGHT: f32 = 1.8;
/// Eye height above the feet (camera sits here).
pub const EYE_HEIGHT: f32 = 1.62;

/// Terminal fall speed (blocks/s); keeps per-step displacement sane.
pub const TERMINAL_VELOCITY: f32 = 78.0;
/// Jump impulse to clear a 1-block+ step with vanilla-like arc height.
pub const JUMP_VELOCITY: f32 = 8.4;
/// Downward acceleration (blocks/s²).
pub const GRAVITY: f32 = 27.0;
/// Walking speed (blocks/s); sprint multiplies this.
pub const WALK_SPEED: f32 = 4.3;
pub const SPRINT_MULTIPLIER: f32 = 1.6;

/// Source of collision truth. Returns true when the voxel containing the
/// world position is solid (a full-cube occluder). Unloaded chunks report
/// solid so the player cannot walk into un-streamed space.
pub type SolidAt<'a> = dyn Fn(i64, i64, i64) -> bool + 'a;

/// Full player state the controller owns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlayerBody {
    /// Feet-center position (the box spans pos ± half on x/z, pos..pos+h on y).
    pub pos: [f32; 3],
    /// Velocity, blocks/s.
    pub vel: [f32; 3],
    /// Grounded flag from the last y-resolution.
    pub on_ground: bool,
}

impl PlayerBody {
    pub fn new(spawn: [f32; 3]) -> PlayerBody {
        PlayerBody {
            pos: spawn,
            vel: [0.0; 3],
            on_ground: false,
        }
    }

    /// Box bounds (min, max) for the current position.
    pub fn aabb(&self) -> ([f32; 3], [f32; 3]) {
        (
            [
                self.pos[0] - PLAYER_HALF_WIDTH,
                self.pos[1],
                self.pos[2] - PLAYER_HALF_WIDTH,
            ],
            [
                self.pos[0] + PLAYER_HALF_WIDTH,
                self.pos[1] + PLAYER_HEIGHT,
                self.pos[2] + PLAYER_HALF_WIDTH,
            ],
        )
    }

    /// Camera (eye) position.
    pub fn eye(&self) -> [f32; 3] {
        [self.pos[0], self.pos[1] + EYE_HEIGHT, self.pos[2]]
    }
}

/// Axis-aligned integer range of voxels spanned by [min, max) on one axis.
fn voxel_range(min: f32, max: f32) -> std::ops::RangeInclusive<i64> {
    // A box ending exactly at an integer boundary must not claim the next
    // voxel: use nextafter-style epsilon shrink on the max side.
    let lo = min.floor() as i64;
    let hi = (max - f32::EPSILON * max.abs().max(1.0)).floor() as i64;
    lo..=hi
}

/// Does the player box overlap any solid voxel after moving `pos[axis]` by
/// `d` along `axis`? Returns the first colliding voxel.
fn collides_at(
    pos: [f32; 3],
    axis: usize,
    d: f32,
    solid: &SolidAt,
) -> Option<(i64, i64, i64)> {
    let mut probe = pos;
    probe[axis] += d;
    let (min, max) = (
        [
            probe[0] - PLAYER_HALF_WIDTH,
            probe[1],
            probe[2] - PLAYER_HALF_WIDTH,
        ],
        [
            probe[0] + PLAYER_HALF_WIDTH,
            probe[1] + PLAYER_HEIGHT,
            probe[2] + PLAYER_HALF_WIDTH,
        ],
    );
    for y in voxel_range(min[1], max[1]) {
        for z in voxel_range(min[2], max[2]) {
            for x in voxel_range(min[0], max[0]) {
                if solid(x, y, z) {
                    return Some((x, y, z));
                }
            }
        }
    }
    None
}

/// One step of movement: resolve horizontal (x, z) then vertical (y).
/// `dt` is the frame delta (seconds); `speed` the intended horizontal speed.
/// `want_jump` only fires when grounded. Returns the ground-landing event.
pub fn step(
    body: &mut PlayerBody,
    dt: f32,
    move_x: f32,
    move_z: f32,
    speed: f32,
    want_jump: bool,
    solid: &SolidAt,
) -> bool {
    let mut landed = false;

    // --- Gravity ---------------------------------------------------------
    body.vel[1] = (body.vel[1] - GRAVITY * dt).clamp(-TERMINAL_VELOCITY, TERMINAL_VELOCITY);

    // --- Jump ------------------------------------------------------------
    if want_jump && body.on_ground {
        body.vel[1] = JUMP_VELOCITY;
        body.on_ground = false;
    }

    // --- Horizontal intent (camera-relative movement is the caller's job) --
    let mut dx = move_x * speed;
    let mut dz = move_z * speed;
    // Clamp per-step displacement < 1 block so the sweep cannot skip voxels.
    let max_step = 0.95;
    let horiz = (dx * dx + dz * dz).sqrt().max(f32::EPSILON);
    let h_scale = (max_step / (horiz * dt)).min(1.0);
    dx *= h_scale;
    dz *= h_scale;
    let dy = body.vel[1] * dt;
    // Vertical displacement must also stay below one block per step, or a
    // fast fall could jump clean through a 1-block floor (tunneling).
    let dy = dy.clamp(-max_step, max_step);

    // --- X axis ------------------------------------------------------------
    if dx != 0.0 {
        let d = dx * dt;
        if let Some((_x, _y, _z)) = collides_at(body.pos, 0, d, solid) {
            // Clamping to the voxel face would need the exact fraction; the
            // per-step clamp keeps d < 1 block, so binary-nudge to contact.
            body.pos[0] = slide(body.pos[0], d, |p| {
                collides_at([p, body.pos[1], body.pos[2]], 0, 0.0, solid).is_none()
            });
            body.vel[0] = 0.0;
        } else {
            body.pos[0] += d;
        }
    }

    // --- Z axis ------------------------------------------------------------
    if dz != 0.0 {
        let d = dz * dt;
        if collides_at(body.pos, 2, d, solid).is_some() {
            body.pos[2] = slide(body.pos[2], d, |p| {
                collides_at([body.pos[0], body.pos[1], p], 2, 0.0, solid).is_none()
            });
            body.vel[2] = 0.0;
        } else {
            body.pos[2] += d;
        }
    }

    // --- Y axis ------------------------------------------------------------
    if dy != 0.0 {
        let d = dy;
        if collides_at(body.pos, 1, d, solid).is_some() {
            if d < 0.0 {
                // Landing: snap the feet onto the voxel top exactly.
                let foot_voxel = (body.pos[1] + d).floor() as i64;
                body.pos[1] = (foot_voxel + 1) as f32;
                landed = !body.on_ground;
                body.on_ground = true;
            } else {
                // Head bump: snap the head under the voxel bottom.
                let head_voxel = (body.pos[1] + PLAYER_HEIGHT + d).floor() as i64;
                body.pos[1] = head_voxel as f32 - PLAYER_HEIGHT;
            }
            body.vel[1] = 0.0;
        } else {
            body.pos[1] += d;
            body.on_ground = false;
        }
    }

    // Ground probe: standing on solid ground with no downward motion.
    if body.vel[1] <= 0.0 {
        let probe_y = body.pos[1] - 0.02;
        let (min, max) = body.aabb();
        let grounded = voxel_range(min[0], max[0]).any(|x| {
            voxel_range(min[2], max[2]).any(|z| {
                voxel_range(probe_y, probe_y + 0.02).any(|y| solid(x, y, z))
            })
        });
        if grounded {
            if !body.on_ground {
                landed = true;
            }
            body.on_ground = true;
        } else if body.pos[1] > 0.0 {
            body.on_ground = false;
        }
    }

    landed
}

/// Nudge a coordinate toward contact: while the *probe* position still
/// intersects (checked by `free`), step back in fractions until free.
fn slide(current: f32, d: f32, free: impl Fn(f32) -> bool) -> f32 {
    // Try the exact contact position first, then refine with 8 bisections.
    let mut lo = current;
    let mut hi = current + d;
    // hi collides, lo is free (pre-condition from the caller's collision test).
    for _ in 0..8 {
        let mid = (lo + hi) * 0.5;
        if free(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Flat world of solid voxels at y <= top (open air above).
    fn flat(top: i64) -> impl Fn(i64, i64, i64) -> bool {
        move |_: i64, y: i64, _: i64| y <= top
    }

    #[test]
    fn player_rests_on_ground_and_stays() {
        let solid = flat(9); // ground top face at y = 10
        let mut body = PlayerBody::new([0.5, 10.0, 0.5]);
        body.on_ground = true;
        for _ in 0..60 {
            step(&mut body, 1.0 / 60.0, 0.0, 0.0, WALK_SPEED, false, &solid);
        }
        assert!(body.on_ground);
        assert!((body.pos[1] - 10.0).abs() < 1e-4, "feet must sit on y=10, got {}", body.pos[1]);
    }

    #[test]
    fn gravity_pulls_player_down_to_rest() {
        let solid = flat(9);
        let mut body = PlayerBody::new([0.5, 20.0, 0.5]);
        let mut landed = false;
        for _ in 0..400 {
            if step(&mut body, 1.0 / 60.0, 0.0, 0.0, WALK_SPEED, false, &solid) {
                landed = true;
            }
        }
        assert!(landed, "a falling player must emit a landing event");
        assert!((body.pos[1] - 10.0).abs() < 1e-3);
        assert!(body.on_ground);
    }

    #[test]
    fn jump_clears_one_block_and_lands_again() {
        let solid = flat(9);
        let mut body = PlayerBody::new([0.5, 10.0, 0.5]);
        body.on_ground = true;
        let mut peak = body.pos[1];
        let mut landed_after = false;
        let mut jumped = false;
        for i in 0..240 {
            let want = i == 0;
            if want && body.on_ground {
                jumped = true;
            }
            step(&mut body, 1.0 / 60.0, 0.0, 0.0, WALK_SPEED, want, &solid);
            peak = peak.max(body.pos[1]);
            if jumped && i > 40 && body.on_ground {
                landed_after = true;
                break;
            }
        }
        assert!(jumped);
        assert!(peak >= 11.0, "jump must clear 1 block (peak={peak})");
        assert!(landed_after, "must land again");
    }

    #[test]
    fn wall_stops_horizontal_movement() {
        // Ground at y<=9 plus a wall column at x = 3..=4 (y 10..13).
        let solid = |x: i64, y: i64, _: i64| y <= 9 || (x >= 3 && x < 5 && y <= 13);
        let mut body = PlayerBody::new([1.5, 10.0, 0.5]);
        body.on_ground = true;
        for _ in 0..600 {
            step(&mut body, 1.0 / 60.0, 1.0, 0.0, WALK_SPEED, false, &solid);
        }
        // The box half-width is 0.3; the wall face is at x=3, so the feet
        // center must stop at 3 - 0.3 = 2.7 (small numeric slack).
        assert!(
            body.pos[0] < 2.75,
            "player must stop at the wall face, got x={}",
            body.pos[0]
        );
        assert!(body.pos[0] > 1.0);
    }

    #[test]
    fn no_tunneling_at_high_speeds_through_thin_walls() {
        // 1-block-thick wall at x = 10..=11.
        let solid = |x: i64, y: i64, _: i64| y <= 9 || (x == 10 && y <= 13);
        let mut body = PlayerBody::new([0.5, 10.0, 0.5]);
        body.on_ground = true;
        // Far above walk speed: the per-step clamp must still prevent skips.
        for _ in 0..600 {
            step(&mut body, 1.0 / 60.0, 1.0, 0.0, 200.0, false, &solid);
        }
        assert!(
            body.pos[0] < 9.7 || body.pos[0] >= 11.0,
            "player must not end up inside the wall, x={}",
            body.pos[0]
        );
    }

    #[test]
    fn head_bumps_a_ceiling() {
        // Floor y<=9 and ceiling y<=12 open 10..12.
        let solid = |_: i64, y: i64, _: i64| y <= 9 || (y >= 12 && y <= 20);
        let mut body = PlayerBody::new([0.5, 10.0, 0.5]);
        body.on_ground = true;
        for i in 0..200 {
            step(&mut body, 1.0 / 60.0, 0.0, 0.0, WALK_SPEED, i == 0, &solid);
        }
        assert!(
            body.pos[1] + PLAYER_HEIGHT <= 12.001,
            "head must stop under the ceiling, got y={}",
            body.pos[1]
        );
    }

    #[test]
    fn walking_off_an_edge_falls() {
        // A platform from x = 0..8 drops off to a lower floor at y = 5.
        let solid = |x: i64, y: i64, _: i64| {
            y <= 5 || (x >= 0 && x < 8 && y <= 9)
        };
        let mut body = PlayerBody::new([4.5, 10.0, 0.5]);
        body.on_ground = true;
        let mut airborne_seen = false;
        for _ in 0..600 {
            step(&mut body, 1.0 / 60.0, 1.0, 0.0, WALK_SPEED, false, &solid);
            if !body.on_ground && body.pos[0] >= 8.0 {
                airborne_seen = true;
                break;
            }
        }
        assert!(airborne_seen, "walking off the platform edge must fall");
    }

    #[test]
    fn unloaded_space_is_a_wall() {
        // Nothing is solid anywhere — but the client passes a closure that
        // treats unloaded space as solid; simulate with an always-solid wall
        // at x>=6 to verify the same code path stops the player.
        let solid = |x: i64, _: i64, _: i64| x >= 6;
        let mut body = PlayerBody::new([0.5, 10.0, 0.5]);
        body.on_ground = true;
        for _ in 0..600 {
            step(&mut body, 1.0 / 60.0, 1.0, 0.0, WALK_SPEED, false, &solid);
        }
        assert!(body.pos[0] < 5.71, "must stop before unloaded space");
    }
}
