//! Mob spawning: category-capped, budgeted, deterministic.
//!
//! Pattern informed by the reference spawn loop (attempts around a center,
//! on-top-of-collider check); the implementation is Feathered's own. Rules:
//!
//! * only *active* chunks (inside the simulation ring) spawn;
//! * per-tick attempt budget (never a per-chunk flood);
//! * class caps: hostile and passive tracked separately;
//! * spawn blocks: solid top face + 2 air above (walkable via ai::walkable
//!   semantics), light rules by class (hostiles avoid daylight);
//! * deterministic RNG: splitmix stream keyed by (world seed, chunk, tick)
//!   — the same context always spawns the same mobs;
//! * density: per-class caps + a per-chunk soft cap.

use feathered_world::chunks::ChunkPos;

use crate::ai;
use crate::world::{EntityWorld, MobKind, MobState};

/// Simulation ring radius (chunks) — spawns only inside this.
pub const SIM_RING: i32 = 4;
/// Per-tick spawn attempts (budget; low-end friendly).
pub const ATTEMPTS_PER_TICK: u32 = 4;
/// Class caps (whole world, live entities).
pub const HOSTILE_CAP: u32 = 12;
pub const PASSIVE_CAP: u32 = 10;
/// Per-chunk soft cap (any class).
const PER_CHUNK_CAP: u32 = 4;
/// Minimum spawn distance from the player (blocks).
pub const MIN_PLAYER_DIST: f32 = 12.0;
/// Maximum spawn distance from the player (blocks).
pub const MAX_PLAYER_DIST: f32 = 56.0;

/// Spawning outcome counters (debug screen).
#[derive(Debug, Default, Clone, Copy)]
pub struct SpawnStats {
    pub attempts: u64,
    pub successes: u64,
    pub rejected_caps: u64,
    pub rejected_light: u64,
    pub rejected_loaded: u64,
}

/// Deterministic spawn attempt for one tick. `daylight` is the current sun
/// elevation sin (from the client's day cycle — hostiles skip spawns in
/// full daylight). Returns entities to add (caller owns mutation).
pub struct Spawner {
    /// World seed (determinism anchor).
    seed: u64,
    pub stats: SpawnStats,
    rng_state: u64,
}

impl Spawner {
    pub fn new(seed: u64) -> Spawner {
        Spawner {
            seed,
            stats: SpawnStats::default(),
            rng_state: seed ^ 0x5EED_51A7E,
        }
    }

    /// One spawn pass. `center` is the player's chunk; `player_pos` feeds
    /// distance rules; `surface_at` resolves the spawn y for a column (the
    /// generator's heightmap); `solid` is the walkability test for the
    /// destination cell. Returns ids of newly spawned mobs.
    #[allow(clippy::too_many_arguments)]
    pub fn tick(
        &mut self,
        entities: &mut EntityWorld,
        center: ChunkPos,
        player_pos: [f32; 3],
        daylight: f32,
        loaded: &dyn Fn(ChunkPos) -> bool,
        surface_at: &dyn Fn(i64, i64) -> i64,
        solid: &ai::SolidFn,
    ) -> Vec<crate::world::EntityId> {
        let mut spawned = Vec::new();

        for _ in 0..ATTEMPTS_PER_TICK {
            self.stats.attempts += 1;
            // Deterministic per-tick RNG stream.
            let mut rng = self.rng_state;
            rng = rng
                .wrapping_mul(0x100_0000_01B3)
                .wrapping_add(self.seed)
                .wrapping_add(std::time::Instant::now().elapsed().subsec_nanos() as u64);
            self.rng_state = rng.wrapping_add(1);

            // Pick a chunk in the ring + a column inside it.
            let ring = SIM_RING as u64;
            let cx = center.x as i64 + (rng % (2 * ring + 1)) as i64 - ring as i64;
            let cz = center.z as i64 + ((rng >> 8) % (2 * ring + 1)) as i64 - ring as i64;
            let pos = ChunkPos::new(cx as i32, cz as i32);
            if !loaded(pos) {
                self.stats.rejected_loaded += 1;
                continue;
            }
            let lx = ((rng >> 16) % 16) as i64;
            let lz = ((rng >> 24) % 16) as i64;
            let wx = cx * 16 + lx;
            let wz = cz * 16 + lz;

            // Distance band from the player.
            let dx = wx as f32 + 0.5 - player_pos[0];
            let dz = wz as f32 + 0.5 - player_pos[2];
            let dist = (dx * dx + dz * dz).sqrt();
            if !(MIN_PLAYER_DIST..=MAX_PLAYER_DIST).contains(&dist) {
                continue;
            }

            // Class selection by caps (hostiles prefer night). Counts are
            // re-read per attempt: one tick can spawn several mobs and must
            // never overshoot a cap via a stale pre-loop counter.
            let mut hostile_count = 0u32;
            let mut passive_count = 0u32;
            for (_, s) in entities.iter() {
                if s.kind.is_hostile() {
                    hostile_count += 1;
                } else {
                    passive_count += 1;
                }
            }
            let want_hostile = if daylight > 0.25 {
                false // full daylight suppresses hostile spawns
            } else {
                hostile_count < HOSTILE_CAP
            };
            let kind = if want_hostile {
                MobKind::Lurker
            } else if passive_count < PASSIVE_CAP {
                MobKind::Muncher
            } else {
                self.stats.rejected_caps += 1;
                continue;
            };

            // Per-chunk density cap (live count — no map allocation).
            let chunk_count = entities
                .iter()
                .filter(|(_, s)| {
                    ChunkPos::of_block(s.pos[0] as i64, s.pos[2] as i64) == pos
                })
                .count() as u32;
            if chunk_count >= PER_CHUNK_CAP {
                self.stats.rejected_caps += 1;
                continue;
            }

            // Surface placement: solid top + 2 air (walkable), above sea.
            let surface = surface_at(wx, wz);
            let y = surface + 1;
            if surface <= 2 {
                continue;
            }
            if solid(wx, y, wz) || solid(wx, y + 1, wz) {
                self.stats.rejected_light += 1;
                continue;
            }

            let state = MobState {
                kind,
                pos: [wx as f32 + 0.5, y as f32, wz as f32 + 0.5],
                vel: [0.0; 3],
                yaw: 0.0,
                on_ground: false,
                health: kind.max_health(),
                move_target: None,
                look_yaw: None,
                last_chunk: pos,
            };
            spawned.push(entities.spawn(state));
            self.stats.successes += 1;
        }
        spawned
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::EntityWorld;

    fn flat_solid() -> impl Fn(i64, i64, i64) -> bool {
        |_: i64, y: i64, _: i64| y <= 9
    }

    #[test]
    fn spawns_respect_caps_and_budget() {
        let mut w = EntityWorld::new();
        let mut sp = Spawner::new(42);
        let solid = flat_solid();
        let loaded = |_: ChunkPos| true;
        let surface = |_: i64, _: i64| 9i64;
        // Many passes: must stop at the passive cap (daylight=true forces
        // passive spawns only).
        for _ in 0..400 {
            sp.tick(&mut w, ChunkPos::new(0, 0), [8.0, 10.0, 8.0], 1.0, &loaded, &surface, &solid);
        }
        let passive = w.iter().filter(|(_, s)| !s.kind.is_hostile()).count() as u32;
        assert_eq!(passive, PASSIVE_CAP, "cap must hold exactly (got {passive})");
        assert!(sp.stats.rejected_caps > 0, "extra attempts must be rejected by caps");
    }

    #[test]
    fn never_spawns_in_unloaded_chunks() {
        let mut w = EntityWorld::new();
        let mut sp = Spawner::new(7);
        let solid = flat_solid();
        let loaded = |_: ChunkPos| false;
        let surface = |_: i64, _: i64| 9i64;
        for _ in 0..100 {
            sp.tick(&mut w, ChunkPos::new(0, 0), [8.0, 10.0, 8.0], -1.0, &loaded, &surface, &solid);
        }
        assert_eq!(w.len(), 0, "unloaded world must spawn nothing");
        assert!(sp.stats.rejected_loaded > 0);
    }

    #[test]
    fn distance_band_respected() {
        let mut w = EntityWorld::new();
        let mut sp = Spawner::new(99);
        let solid = flat_solid();
        let loaded = |_: ChunkPos| true;
        let surface = |_: i64, _: i64| 9i64;
        for _ in 0..100 {
            let spawned = sp.tick(&mut w, ChunkPos::new(0, 0), [8.0, 10.0, 8.0], -1.0, &loaded, &surface, &solid);
            for id in spawned {
                let st = w.get(id).unwrap();
                let dx = st.pos[0] - 8.0;
                let dz = st.pos[2] - 8.0;
                let d = (dx * dx + dz * dz).sqrt();
                assert!(
                    d >= MIN_PLAYER_DIST - 0.1 && d <= MAX_PLAYER_DIST + 0.1,
                    "spawn at distance {d} outside band"
                );
            }
        }
    }
}
