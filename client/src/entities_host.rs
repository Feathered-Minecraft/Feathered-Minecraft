//! Client-side entity host: connects `feathered-entity` to the sandbox.
//!
//! Responsibilities per sim frame (`EntityHost::update`):
//! 1. **Spawn** — budgeted attempts via `Spawner` (caps enforce no flooding).
//! 2. **Activate** — only entities whose chunk is inside the simulation ring
//!    tick; others freeze (they despawn with their chunk unless persisted —
//!    persistence lands with the save-format work).
//! 3. **AI** — per-mob `GoalSelector` tick with a player snapshot.
//! 4. **Physics** — intent (`move_target`/`look_yaw`) → `physics::step`
//!    through the SAME solid-closure the player uses (identical collision).
//! 5. **Render** — debug boxes into the world-space overlay list (replaced
//!    by resource-pack models when the renderer grows entity support).
//!
//! The player is not an entity: `PlayerController` stays authoritative; the
//! AI only reads a snapshot of the player's position.

use feathered_entity::ai::{self, GoalSelector, MobCtx};
use feathered_entity::world::{EntityId, EntityWorld, MobState};
use feathered_entity::spawn::Spawner;
use feathered_world::chunks::ChunkPos;
use feathered_world::physics;

use crate::overlay::{self, TriList};

/// Colors for the debug boxes (hostile = warm, passive = cool).
fn kind_color(hostile: bool) -> [u8; 4] {
    if hostile {
        [235, 90, 80, 210]
    } else {
        [110, 200, 120, 210]
    }
}

/// Wire box: reuse the block-outline beam builder at a scaled cell.
fn draw_debug_box(list: &mut TriList, pos: [f32; 3], half: f32, height: f32, color: [u8; 4]) {
    let x0 = pos[0] - half;
    let z0 = pos[2] - half;
    let x1 = pos[0] + half;
    let z1 = pos[2] + half;
    let y0 = pos[1];
    let y1 = pos[1] + height;
    // Ground ring (4 horizontal beams at the feet) + top ring + 4 posts.
    let beams: [([f32; 3], [f32; 3]); 12] = [
        ([x0, y0, z0], [x1, y0, z0]),
        ([x1, y0, z0], [x1, y0, z1]),
        ([x1, y0, z1], [x0, y0, z1]),
        ([x0, y0, z1], [x0, y0, z0]),
        ([x0, y1, z0], [x1, y1, z0]),
        ([x1, y1, z0], [x1, y1, z1]),
        ([x1, y1, z1], [x0, y1, z1]),
        ([x0, y1, z1], [x0, y1, z0]),
        ([x0, y0, z0], [x0, y1, z0]),
        ([x1, y0, z0], [x1, y1, z0]),
        ([x1, y0, z1], [x1, y1, z1]),
        ([x0, y0, z1], [x0, y1, z1]),
    ];
    for (a, b) in beams {
        overlay::beam3(list, a, b, 0.03, color);
    }
}

/// Per-frame entity work (called from `sandbox_update` with the world's
/// solid closure already built).
pub struct EntityHost;

impl EntityHost {
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        dt: f32,
        tick: &mut u64,
        entities: &mut EntityWorld,
        brains: &mut std::collections::HashMap<u32, (GoalSelector, [f32; 4])>,
        spawner: &mut Spawner,
        player_pos: [f32; 3],
        _player_chunk: ChunkPos,
        daylight: f32,
        center: ChunkPos,
        loaded: &dyn Fn(ChunkPos) -> bool,
        surface_at: &dyn Fn(i64, i64) -> i64,
        solid: &dyn Fn(i64, i64, i64) -> bool,
        view_distance: i32,
    ) {
        let _ = dt;
        *tick += 1;
        let parity = *tick % 2 == 0;

        // 1. Spawning (internally budgeted by ATTEMPTS_PER_TICK).
        spawner.tick(
            entities,
            center,
            player_pos,
            daylight,
            loaded,
            surface_at,
            solid,
        );

        // 2. AI + 3. physics for active entities only. Collect intent via a
        // first pass (immutable), then apply physics per entity (mutable).
        let sim_ring = view_distance.min(feathered_entity::spawn::SIM_RING);
        struct Intent {
            id: EntityId,
            mx: f32,
            mz: f32,
            speed: f32,
            pos: [f32; 3],
            vel: [f32; 3],
            on_ground: bool,
            look_yaw: Option<f32>,
        }
        let mut intents: Vec<Intent> = Vec::new();
        for (id, state) in entities.iter() {
            if !ai::chunk_active(state.last_chunk, center, sim_ring) {
                continue;
            }
            let (dx, dz) = (player_pos[0] - state.pos[0], player_pos[2] - state.pos[2]);
            let dist = (dx * dx + dz * dz).sqrt();
            // Brain for this slot (created on first sight).
            let entry = brains.entry(id.index).or_insert_with(|| {
                (feathered_entity::mobs::brain(state.kind), [0.0; 4])
            });
            let (sel, scratch) = (&mut entry.0, &mut entry.1);
            // Feed the player position for chase/flee goals.
            scratch[2] = player_pos[0];
            scratch[3] = player_pos[2];
            let rng = 0x1234_5678 ^ (id.index as u64).wrapping_mul(0x9E37) ^ (*tick);
            // The AI works on a clone (EntityWorld borrows stay disjoint);
            // the resulting state is written back after physics below.
            let mut working = state.clone();
            {
                let mut ctx = MobCtx {
                    mob: &mut working,
                    player_dist: Some(dist),
                    rng,
                    scratch,
                    solid,
                };
                sel.tick(&mut ctx, parity);
                if scratch[3] > 0.0 {
                    scratch[3] = 0.0; // melee signal consumed (host applies later)
                }
            }
            // Movement intent from the AI's move_target.
            let (mx, mz) = match working.move_target {
                Some(t) => {
                    let (ddx, ddz) = (t[0] - working.pos[0], t[2] - working.pos[2]);
                    let d = (ddx * ddx + ddz * ddz).sqrt();
                    if d < 0.15 {
                        (0.0, 0.0)
                    } else {
                        (ddx / d, ddz / d)
                    }
                }
                None => (0.0, 0.0),
            };
            intents.push(Intent {
                id,
                mx,
                mz,
                speed: working.kind.speed(),
                pos: working.pos,
                vel: working.vel,
                on_ground: working.on_ground,
                look_yaw: working.look_yaw,
            });
        }

        // Physics: apply intents through the shared voxel collision.
        let mut to_remove: Vec<EntityId> = Vec::new();
        for it in intents {
            let Some(state) = entities.get_mut(it.id) else {
                continue;
            };
            let mut body = physics::PlayerBody {
                pos: it.pos,
                vel: it.vel,
                on_ground: it.on_ground,
            };
            physics::step(&mut body, 0.05, it.mx, it.mz, it.speed, false, solid);
            state.pos = body.pos;
            state.vel = body.vel;
            state.on_ground = body.on_ground;
            // Face movement or the AI's look intent (stashed pre-physics).
            if let Some(t) = state.move_target {
                state.yaw = (t[2] - state.pos[2]).atan2(t[0] - state.pos[0]);
            } else if let Some(look) = it.look_yaw {
                state.yaw = look;
            }
            state.last_chunk = ChunkPos::of_block(state.pos[0] as i64, state.pos[2] as i64);
            // Fell out of the world → despawn.
            if state.pos[1] < -8.0 {
                to_remove.push(it.id);
            }
        }
        for id in to_remove {
            entities.despawn(id);
            brains.remove(&id.index);
        }
    }

    /// Debug-boxes into the world overlay list (Phase 6 renderer).
    pub fn render_debug(entities: &EntityWorld, list: &mut TriList) {
        for (_, st) in entities.iter() {
            let (half, height) = st.kind.dims();
            draw_debug_box(list, st.pos, half, height, kind_color(st.kind.is_hostile()));
        }
    }
}

/// Wire entities' snapshot for save (Phase 8).
pub fn entity_snapshot(entities: &EntityWorld) -> Vec<MobState> {
    entities.snapshot()
}

/// Restore entities from a save snapshot.
pub fn entity_restore(
    entities: &mut EntityWorld,
    brains: &mut std::collections::HashMap<u32, (GoalSelector, [f32; 4])>,
    states: Vec<MobState>,
) {
    entities.restore(states);
    for (id, st) in entities.iter() {
        brains.insert(id.index, (feathered_entity::mobs::brain(st.kind), [0.0; 4]));
    }
}
