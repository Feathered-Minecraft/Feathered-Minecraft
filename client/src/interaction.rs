//! Block interaction: camera raycast targeting, breaking and placing.
//!
//! Rules:
//! * Reach is 5 blocks (vanilla-ish survival reach).
//! * Breaking takes time: each block has a hardness (seconds); the target
//!   accumulates progress while continuously aimed and the button is held;
//!   switching targets or releasing resets it. A small cooldown follows
//!   each completed break.
//! * Placing has a 0.2 s cooldown, must not intersect the player's AABB,
//!   and goes on the hit face's outside (correct at any coordinate,
//!   including chunk borders and negative space).
//! * The target is the first non-air voxel in reach.

use feathered_world::physics::{PLAYER_HALF_WIDTH, PLAYER_HEIGHT};
use feathered_world::raycast::{raycast, RayHit};
use feathered_world::Registry;

/// Survival reach in blocks.
pub const REACH: f32 = 5.0;
/// Cooldown after a completed break (instant break = still rate-limited).
pub const BREAK_COOLDOWN: f32 = 0.1;
pub const PLACE_COOLDOWN: f32 = 0.2;

/// Per-block hardness (seconds to break). Deterministic and pack-agnostic;
/// unknown blocks default to 0.5 s.
pub fn hardness(block: u32) -> f32 {
    match block {
        0 => 0.0,
        // Dirt-like and sand: quick.
        3 => 0.4,
        // Wood: medium.
        5 => 0.9,
        // Stone-like and bricks: slow.
        1 | 4 | 8 | 9 => 1.1,
        // Glass shatters.
        7 => 0.2,
        _ => 0.5,
    }
}

/// What the player is looking at this frame.
#[derive(Debug, Clone, Copy)]
pub struct Target {
    /// Voxel to break.
    pub hit: RayHit,
}

impl Target {
    /// The cell a placement would occupy (outside the hit face).
    pub fn place_cell(&self) -> (i64, i64, i64) {
        self.hit.face.neighbor(self.hit.x, self.hit.y, self.hit.z)
    }
}

/// Compute the current target from the camera (eye) + view direction.
/// `targetable(x,y,z)` decides which voxels stop the ray (non-air).
pub fn current_target(
    eye: [f32; 3],
    dir: [f32; 3],
    targetable: &dyn Fn(i64, i64, i64) -> bool,
) -> Option<Target> {
    raycast(eye, dir, REACH, targetable).map(|hit| Target { hit })
}

/// Validation for a placement at `pos`:
/// * inside world height,
/// * not intersecting the player box (expanded by a small epsilon),
/// * (the caller checks the chunk is loaded and the slot non-empty).
pub fn can_place_at(pos: [i64; 3], player_feet: [f32; 3], world_h: u32) -> bool {
    if pos[1] < 0 || pos[1] >= world_h as i64 {
        return false;
    }
    // Player AABB (feet-centered) vs the block cell [pos, pos+1).
    let (pmin, pmax) = (
        [
            player_feet[0] - PLAYER_HALF_WIDTH,
            player_feet[1],
            player_feet[2] - PLAYER_HALF_WIDTH,
        ],
        [
            player_feet[0] + PLAYER_HALF_WIDTH,
            player_feet[1] + PLAYER_HEIGHT,
            player_feet[2] + PLAYER_HALF_WIDTH,
        ],
    );
    let eps = 1e-4;
    let overlap =
        |pmin: f32, pmax: f32, bmin: f32, bmax: f32| pmin < bmax - eps && pmax > bmin + eps;
    !(overlap(pmin[0], pmax[0], pos[0] as f32, pos[0] as f32 + 1.0)
        && overlap(pmin[1], pmax[1], pos[1] as f32, pos[1] as f32 + 1.0)
        && overlap(pmin[2], pmax[2], pos[2] as f32, pos[2] as f32 + 1.0))
}

/// The result of one interaction tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionEvent {
    /// Broke the block at these coords.
    Broke(i64, i64, i64),
    /// Placed a block at these coords.
    Placed(i64, i64, i64),
}

/// Stateful interaction driver: cooldowns + mining progress.
#[derive(Debug, Default)]
pub struct Interaction {
    break_cd: f32,
    place_cd: f32,
    /// Accumulated break progress on the current target (seconds).
    progress: f32,
    /// The voxel progress belongs to (reset when the target changes).
    progress_at: Option<(i64, i64, i64)>,
}

impl Interaction {
    pub fn tick(&mut self, dt: f32) {
        self.break_cd = (self.break_cd - dt).max(0.0);
        self.place_cd = (self.place_cd - dt).max(0.0);
    }

    /// Break progress on the current target in 0..1 (for the HUD bar).
    pub fn progress(&self) -> f32 {
        let Some((x, y, z)) = self.progress_at else {
            return 0.0;
        };
        // Progress ratio needs the target's hardness; store normalized when
        // accumulating — see `update_break`. Returns the stored value.
        let _ = (x, y, z);
        self.progress.min(1.0)
    }

    /// The voxel the progress belongs to, if any.
    pub fn progress_target(&self) -> Option<(i64, i64, i64)> {
        self.progress_at
    }

    /// Advance breaking on `target`. Call every frame while the break
    /// button is held: progress accumulates only while the same voxel is
    /// targeted; releasing, switching voxels, or completion resets it.
    /// `block_at(x,y,z)` → block id (0/absent = air); `do_break` performs
    /// the removal (return false to veto, which also resets progress).
    pub fn update_break(
        &mut self,
        target: &Target,
        dt: f32,
        block_at: &dyn Fn(i64, i64, i64) -> Option<u32>,
        do_break: &mut dyn FnMut(i64, i64, i64) -> bool,
    ) -> Option<InteractionEvent> {
        let cell = (target.hit.x, target.hit.y, target.hit.z);
        if self.progress_at != Some(cell) {
            // Target changed: start fresh.
            self.progress = 0.0;
            self.progress_at = Some(cell);
        }
        if self.break_cd > 0.0 {
            return None;
        }
        let block = block_at(cell.0, cell.1, cell.2).unwrap_or(0);
        let h = hardness(block).max(0.05);
        self.progress += dt / h;
        if self.progress >= 1.0 {
            if do_break(cell.0, cell.1, cell.2) {
                self.progress = 0.0;
                self.progress_at = None;
                self.break_cd = BREAK_COOLDOWN;
                Some(InteractionEvent::Broke(cell.0, cell.1, cell.2))
            } else {
                // Vetoed (e.g. bedrock/void): progress restarts.
                self.progress = 0.0;
                self.progress_at = None;
                None
            }
        } else {
            None
        }
    }

    /// Cancel any accumulated progress (button released / target lost).
    pub fn reset_progress(&mut self) {
        self.progress = 0.0;
        self.progress_at = None;
    }

    /// Try to break the targeted block instantly (legacy path, used when
    /// hardness is disabled). Kept for tests and fallback.
    pub fn try_break(
        &mut self,
        target: &Target,
        do_break: &mut dyn FnMut(i64, i64, i64) -> bool,
    ) -> Option<InteractionEvent> {
        if self.break_cd > 0.0 {
            return None;
        }
        let (x, y, z) = (target.hit.x, target.hit.y, target.hit.z);
        if do_break(x, y, z) {
            self.break_cd = BREAK_COOLDOWN;
            Some(InteractionEvent::Broke(x, y, z))
        } else {
            None
        }
    }

    /// Try to place on the target's hit face. `do_place` validates and
    /// performs the write (chunk loaded, cell empty, not in the player).
    pub fn try_place(
        &mut self,
        target: &Target,
        block: &str,
        registry: &Registry,
        player_feet: [f32; 3],
        world_h: u32,
        do_place: &mut dyn FnMut(i64, i64, i64, u32, u32) -> bool,
    ) -> Option<InteractionEvent> {
        if self.place_cd > 0.0 || block.is_empty() {
            return None;
        }
        let (nx, ny, nz) = target
            .hit
            .face
            .neighbor(target.hit.x, target.hit.y, target.hit.z);
        if !can_place_at([nx, ny, nz], player_feet, world_h) {
            return None;
        }
        // Default state 0; the block must exist in the pack.
        let def = registry.block(block)?;
        let state = 0;
        let _ = def;
        if do_place(nx, ny, nz, registry.block_id(block).unwrap_or(0), state) {
            self.place_cd = PLACE_COOLDOWN;
            Some(InteractionEvent::Placed(nx, ny, nz))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placement_into_the_player_is_rejected() {
        // Player standing at (0.5, 10, 0.5): box x 0.2..0.8, y 10..11.8.
        let feet = [0.5, 10.0, 0.5];
        assert!(
            !can_place_at([0, 10, 0], feet, 128),
            "inside the player box"
        );
        assert!(
            !can_place_at([0, 11, 0], feet, 128),
            "overlaps the head area"
        );
        assert!(can_place_at([2, 10, 2], feet, 128), "clearly away is fine");
        assert!(
            can_place_at([0, 12, 0], feet, 128),
            "above the head is fine"
        );
        assert!(
            !can_place_at([0, 128, 0], feet, 128),
            "above world height rejected"
        );
        assert!(!can_place_at([0, -1, 0], feet, 128), "below world rejected");
    }

    #[test]
    fn breaking_respects_the_cooldown() {
        let mut inter = Interaction::default();
        let target = Target {
            hit: RayHit {
                x: 1,
                y: 1,
                z: 1,
                face: feathered_world::raycast::Face::Up,
                point: [1.0, 2.0, 1.0],
                distance: 3.0,
            },
        };
        let mut calls = 0;
        let ev = inter.try_break(&target, &mut |_, _, _| {
            calls += 1;
            true
        });
        assert_eq!(ev, Some(InteractionEvent::Broke(1, 1, 1)));
        // Within the cooldown: vetoed without calling.
        let ev2 = inter.try_break(&target, &mut |_, _, _| {
            calls += 1;
            true
        });
        assert_eq!(ev2, None);
        assert_eq!(calls, 1);
        // After the cooldown passes, it fires again.
        inter.tick(BREAK_COOLDOWN + 0.01);
        let ev3 = inter.try_break(&target, &mut |_, _, _| {
            calls += 1;
            true
        });
        assert!(ev3.is_some());
        assert_eq!(calls, 2);
    }

    #[test]
    fn placing_uses_the_hit_face_outside() {
        let mut inter = Interaction::default();
        let registry = Registry::from_names_for_tests(&["stone"]);
        // Target: top face of the floor block (0,9,0) — placement goes to
        // (0,10,0).
        let target = Target {
            hit: RayHit {
                x: 0,
                y: 9,
                z: 0,
                face: feathered_world::raycast::Face::Up,
                point: [0.5, 10.0, 0.5],
                distance: 2.0,
            },
        };
        let feet = [8.5, 10.0, 8.5]; // far from the placement
        let mut placed_at = None;
        let ev = inter.try_place(
            &target,
            "stone",
            &registry,
            feet,
            128,
            &mut |x, y, z, b, s| {
                placed_at = Some((x, y, z, b, s));
                true
            },
        );
        assert_eq!(ev, Some(InteractionEvent::Placed(0, 10, 0)));
        assert_eq!(
            placed_at,
            Some((0, 10, 0, registry.block_id("stone").unwrap(), 0))
        );
        // Unknown block names are rejected cleanly.
        inter.tick(PLACE_COOLDOWN + 0.01);
        let ev2 = inter.try_place(
            &target,
            "not_a_real_block",
            &registry,
            feet,
            128,
            &mut |_, _, _, _, _| true,
        );
        assert_eq!(ev2, None);
    }

    #[test]
    fn breaking_takes_time_and_reports_progress() {
        let mut inter = Interaction::default();
        let target = Target {
            hit: RayHit {
                x: 0,
                y: 9,
                z: 0,
                face: feathered_world::raycast::Face::Up,
                point: [0.5, 10.0, 0.5],
                distance: 2.0,
            },
        };
        let block = Some(1u32); // stone-ish: hardness 1.1 s
        let broke_at = std::cell::Cell::new(None);
        let mut ev = None;
        let mut frames = 0;
        loop {
            frames += 1;
            assert!(frames < 500, "must break within 500 frames");
            ev = inter.update_break(&target, 1.0 / 60.0, &|_, _, _| block, &mut |x, y, z| {
                broke_at.set(Some((x, y, z)));
                true
            });
            if ev.is_some() {
                break;
            }
        }
        assert_eq!(ev, Some(InteractionEvent::Broke(0, 9, 0)));
        assert_eq!(broke_at.get(), Some((0, 9, 0)));
        // ~1.1 s + one frame of slack.
        assert!(
            (frames as f32 / 60.0) < 1.3,
            "stone broke in {frames} frames"
        );
        // Progress resets after the break.
        assert_eq!(inter.progress(), 0.0);
        // Cooldown gates an immediate re-break.
        let ev2 = inter.update_break(&target, 0.0, &|_, _, _| block, &mut |_, _, _| true);
        assert_eq!(ev2, None, "cooldown after completed break");
    }

    #[test]
    fn switching_targets_resets_progress() {
        let mut inter = Interaction::default();
        let mk_target = |x: i64| Target {
            hit: RayHit {
                x,
                y: 9,
                z: 0,
                face: feathered_world::raycast::Face::Up,
                point: [x as f32 + 0.5, 10.0, 0.5],
                distance: 2.0,
            },
        };
        let block = Some(3u32); // dirt: 0.4 s
                                // Break half of block A.
        for _ in 0..10 {
            inter.update_break(
                &mk_target(0),
                1.0 / 60.0,
                &|_, _, _| block,
                &mut |_, _, _| true,
            );
        }
        let half = inter.progress();
        assert!(half > 0.2 && half < 0.6, "mid-break progress, got {half}");
        // Switch to block B: progress restarts from zero there.
        for _ in 0..3 {
            inter.update_break(
                &mk_target(5),
                1.0 / 60.0,
                &|_, _, _| block,
                &mut |_, _, _| true,
            );
        }
        assert!(inter.progress() < half, "switching targets resets progress");
        assert_eq!(inter.progress_target(), Some((5, 9, 0)));
    }

    #[test]
    fn vetoed_breaks_reset_progress() {
        let mut inter = Interaction::default();
        let target = Target {
            hit: RayHit {
                x: 0,
                y: 0,
                z: 0,
                face: feathered_world::raycast::Face::Up,
                point: [0.5, 1.0, 0.5],
                distance: 2.0,
            },
        };
        let block = Some(1u32);
        // do_break always vetoes (e.g. unbreakable bottom layer).
        for _ in 0..200 {
            let ev =
                inter.update_break(&target, 1.0 / 60.0, &|_, _, _| block, &mut |_, _, _| false);
            assert!(ev.is_none());
            if inter.progress_at.is_none() {
                break; // progress was reset after a veto
            }
        }
        assert_eq!(
            inter.progress_at, None,
            "veto resets and stops accumulation"
        );
    }

    #[test]
    fn hardness_table_is_sane() {
        assert_eq!(hardness(0), 0.0);
        assert!(hardness(7) < hardness(3), "glass breaks faster than dirt");
        assert!(hardness(1) > hardness(5), "stone slower than wood");
        assert!(hardness(999) > 0.0, "unknown blocks have a default");
    }

    #[test]
    fn placing_inside_the_player_vetoes_without_consuming_cooldown() {
        let mut inter = Interaction::default();
        let registry = Registry::from_names_for_tests(&["stone"]);
        let target = Target {
            hit: RayHit {
                x: 0,
                y: 9,
                z: 0,
                face: feathered_world::raycast::Face::Up,
                point: [0.5, 10.0, 0.5],
                distance: 2.0,
            },
        };
        let feet = [0.5, 10.0, 0.5]; // player right there
        let mut calls = 0;
        let ev = inter.try_place(
            &target,
            "stone",
            &registry,
            feet,
            128,
            &mut |_, _, _, _, _| {
                calls += 1;
                true
            },
        );
        assert_eq!(ev, None, "placement inside the player is vetoed");
        assert_eq!(calls, 0);
        // Cooldown not consumed: an immediate valid placement works.
        let ev2 = inter.try_place(
            &Target {
                hit: RayHit {
                    x: 5,
                    y: 9,
                    z: 5,
                    face: feathered_world::raycast::Face::Up,
                    point: [5.5, 10.0, 5.5],
                    distance: 2.0,
                },
            },
            "stone",
            &registry,
            feet,
            128,
            &mut |_, _, _, _, _| true,
        );
        assert!(ev2.is_some());
    }
}
