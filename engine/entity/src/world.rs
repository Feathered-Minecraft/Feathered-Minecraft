//! Entity storage and lifecycle.
//!
//! `EntityWorld` owns every non-player entity. `EntityId` is (index,
//! generation): the generation field makes stale references to despawned
//! entities detectable (accessing a recycled slot with an old id fails
//! cleanly). Storage is a dense `Vec` of slots reused on despawn; tick
//! order is id order — deterministic for a given world state, which makes
//! simulations reproducible in tests.
//!
//! Physics reuses Feathered's existing voxel `physics::step` via the same
//! `SolidAt` closure contract the player controller uses — entities and
//! players collide with the world identically. The player is NOT an entity
//! here (it lives in `PlayerController`); mobs can *see* it through a
//! `PlayerRef` snapshot fed in per tick.

use feathered_world::chunks::ChunkPos;
use std::collections::HashMap;

/// Stable entity handle: slot index + generation guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct EntityId {
    pub index: u32,
    pub generation: u32,
}

/// Which mob archetype an entity is (open enum; new kinds register here).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum MobKind {
    /// Passive: wanders, flees when hurt.
    Muncher,
    /// Hostile: idles, chases the player when close, melee attacks.
    Lurker,
}

impl MobKind {
    /// Display name (debug screen).
    pub fn name(self) -> &'static str {
        match self {
            MobKind::Muncher => "Muncher",
            MobKind::Lurker => "Lurker",
        }
    }

    /// Hostility class (spawn caps are per class).
    pub fn is_hostile(self) -> bool {
        matches!(self, MobKind::Lurker)
    }

    /// Body dimensions (half-width, height) — collision + rendering.
    pub fn dims(self) -> (f32, f32) {
        match self {
            MobKind::Muncher => (0.35, 0.7),
            MobKind::Lurker => (0.3, 1.6),
        }
    }

    /// Walk speed (blocks/s).
    pub fn speed(self) -> f32 {
        match self {
            MobKind::Muncher => 2.0,
            MobKind::Lurker => 3.2,
        }
    }

    /// Max health.
    pub fn max_health(self) -> f32 {
        match self {
            MobKind::Muncher => 8.0,
            MobKind::Lurker => 14.0,
        }
    }
}

/// One AI goal's control flags (simplified from the reference design).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Controls(pub u8);
impl Controls {
    pub const MOVE: u8 = 1;
    pub const LOOK: u8 = 2;
    pub const CHASE: u8 = 4;
    pub fn any(self, other: Controls) -> bool {
        self.0 & other.0 != 0
    }
}

/// Runtime mob state (the "components" — plain data, serializable).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MobState {
    pub kind: MobKind,
    /// Feet-center position.
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub yaw: f32,
    pub on_ground: bool,
    pub health: f32,
    /// Current path target (AI writes, physics reads).
    pub move_target: Option<[f32; 3]>,
    /// Desired look direction (yaw) — LOOK goals write it.
    pub look_yaw: Option<f32>,
    /// Where the mob was when its chunk was last known active.
    pub last_chunk: ChunkPos,
}

/// One stored entity slot.
#[derive(Debug, Clone)]
struct Slot {
    generation: u32,
    state: Option<MobState>,
}

/// Entity container with deterministic tick order.
#[derive(Default)]
pub struct EntityWorld {
    slots: Vec<Slot>,
    free: Vec<u32>,
    count: u32,
}

impl EntityWorld {
    pub fn new() -> EntityWorld {
        EntityWorld::default()
    }

    /// Spawn a mob; returns its stable id.
    pub fn spawn(&mut self, state: MobState) -> EntityId {
        let (index, generation) = match self.free.pop() {
            Some(i) => {
                let g = self.slots[i as usize].generation;
                self.slots[i as usize].state = Some(state);
                (i, g)
            }
            None => {
                let generation = 0;
                self.slots.push(Slot {
                    generation,
                    state: Some(state),
                });
                (self.slots.len() as u32 - 1, generation)
            }
        };
        self.count += 1;
        EntityId { index, generation }
    }

    /// Despawn (returns false when the id was already stale).
    pub fn despawn(&mut self, id: EntityId) -> bool {
        let Some(slot) = self.slots.get_mut(id.index as usize) else {
            return false;
        };
        if slot.generation != id.generation || slot.state.is_none() {
            return false;
        }
        slot.state = None;
        slot.generation += 1;
        self.free.push(id.index);
        self.count -= 1;
        true
    }

    /// Read access (None when stale).
    pub fn get(&self, id: EntityId) -> Option<&MobState> {
        self.slots.get(id.index as usize).and_then(|s| {
            (s.generation == id.generation)
                .then_some(())
                .and(s.state.as_ref())
        })
    }

    /// Write access (None when stale).
    pub fn get_mut(&mut self, id: EntityId) -> Option<&mut MobState> {
        self.slots.get_mut(id.index as usize).and_then(|s| {
            (s.generation == id.generation)
                .then_some(())
                .and(s.state.as_mut())
        })
    }

    /// Live entity count.
    pub fn len(&self) -> u32 {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Iterate (id, state) pairs in deterministic id order.
    pub fn iter(&self) -> impl Iterator<Item = (EntityId, &MobState)> {
        self.slots.iter().enumerate().filter_map(|(i, s)| {
            s.state.as_ref().map(|st| {
                (
                    EntityId {
                        index: i as u32,
                        generation: s.generation,
                    },
                    st,
                )
            })
        })
    }

    /// Mutable iterate in id order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (EntityId, &mut MobState)> {
        self.slots.iter_mut().enumerate().filter_map(|(i, s)| {
            s.state.as_mut().map(|st| {
                (
                    EntityId {
                        index: i as u32,
                        generation: s.generation,
                    },
                    st,
                )
            })
        })
    }

    /// Nearest entity of a kind within `range` of `pos` (linear scan —
    /// entity counts are small; a spatial index would be premature).
    pub fn nearest_of_kind(
        &self,
        pos: [f32; 3],
        kind: MobKind,
        range: f32,
    ) -> Option<(EntityId, [f32; 3])> {
        let r2 = range * range;
        self.iter()
            .filter(|(_, st)| st.kind == kind)
            .map(|(id, st)| {
                let d = (st.pos[0] - pos[0]).powi(2)
                    + (st.pos[1] - pos[1]).powi(2)
                    + (st.pos[2] - pos[2]).powi(2);
                (id, d, st.pos)
            })
            .filter(|(_, d, _)| *d <= r2)
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(id, _, p)| (id, p))
    }

    /// Entities grouped by chunk (the activation gate: only chunks in the
    /// simulation ring tick).
    pub fn by_chunk(&self) -> HashMap<ChunkPos, Vec<EntityId>> {
        let mut map: HashMap<ChunkPos, Vec<EntityId>> = HashMap::new();
        for (id, st) in self.iter() {
            map.entry(ChunkPos::of_block(st.pos[0] as i64, st.pos[2] as i64))
                .or_default()
                .push(id);
        }
        map
    }

    /// Snapshot for persistence (only live entities serialize).
    pub fn snapshot(&self) -> Vec<MobState> {
        self.iter().map(|(_, st)| st.clone()).collect()
    }

    /// Restore from a persistence snapshot (replaces all state).
    pub fn restore(&mut self, states: Vec<MobState>) {
        self.slots.clear();
        self.free.clear();
        self.count = 0;
        for st in states {
            self.spawn(st);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mob(x: f32, kind: MobKind) -> MobState {
        MobState {
            kind,
            pos: [x, 20.0, 0.0],
            vel: [0.0; 3],
            yaw: 0.0,
            on_ground: false,
            health: 10.0,
            move_target: None,
            look_yaw: None,
            last_chunk: ChunkPos::new(0, 0),
        }
    }

    #[test]
    fn ids_and_generations_guard_stale_access() {
        let mut w = EntityWorld::new();
        let a = w.spawn(mob(0.0, MobKind::Muncher));
        assert!(w.despawn(a));
        // Same slot reused: new entity gets a bumped generation.
        let b = w.spawn(mob(1.0, MobKind::Muncher));
        assert_eq!(a.index, b.index);
        assert_ne!(a.generation, b.generation);
        assert!(w.get(a).is_none(), "stale id must not resolve");
        assert!(w.get(b).is_some());
        assert!(!w.despawn(a), "double-despawn of a stale id fails cleanly");
        assert_eq!(w.len(), 1);
    }

    #[test]
    fn iteration_order_is_deterministic() {
        let mut w = EntityWorld::new();
        let _ = w.spawn(mob(3.0, MobKind::Lurker));
        let _ = w.spawn(mob(1.0, MobKind::Muncher));
        let _ = w.spawn(mob(2.0, MobKind::Lurker));
        let xs: Vec<f32> = w.iter().map(|(_, s)| s.pos[0]).collect();
        assert_eq!(xs, [3.0, 1.0, 2.0], "insertion (slot) order is stable");
        // Repeating the construction gives the same order.
        let mut w2 = EntityWorld::new();
        let _ = w2.spawn(mob(3.0, MobKind::Lurker));
        let _ = w2.spawn(mob(1.0, MobKind::Muncher));
        let _ = w2.spawn(mob(2.0, MobKind::Lurker));
        let xs2: Vec<f32> = w2.iter().map(|(_, s)| s.pos[0]).collect();
        assert_eq!(xs, xs2);
    }

    #[test]
    fn spatial_queries_work() {
        let mut w = EntityWorld::new();
        let _ = w.spawn(mob(0.0, MobKind::Lurker));
        let _ = w.spawn(mob(50.0, MobKind::Lurker));
        let _ = w.spawn(mob(2.0, MobKind::Muncher));
        let (id, p) = w
            .nearest_of_kind([0.0, 20.0, 0.0], MobKind::Lurker, 10.0)
            .unwrap();
        assert_eq!(p[0], 0.0);
        assert!(w.get(id).is_some());
        assert!(w
            .nearest_of_kind([0.0, 20.0, 0.0], MobKind::Muncher, 1.0)
            .is_none());
    }

    #[test]
    fn by_chunk_groups_entities() {
        let mut w = EntityWorld::new();
        let mut m = mob(1.0, MobKind::Muncher);
        m.pos = [1.0, 20.0, 1.0];
        let _ = w.spawn(m);
        let mut m2 = mob(2.0, MobKind::Muncher);
        m2.pos = [-20.0, 20.0, -20.0]; // negative chunk (-2,-2)
        let _ = w.spawn(m2);
        let groups = w.by_chunk();
        assert!(groups.contains_key(&ChunkPos::new(0, 0)));
        assert!(groups.contains_key(&ChunkPos::new(-2, -2)));
    }

    #[test]
    fn snapshot_roundtrip() {
        let mut w = EntityWorld::new();
        let _ = w.spawn(mob(1.0, MobKind::Muncher));
        let _ = w.spawn(mob(2.0, MobKind::Lurker));
        let snap = serde_json::to_string(&w.snapshot()).unwrap();
        let mut w2 = EntityWorld::new();
        w2.restore(serde_json::from_str(&snap).unwrap());
        assert_eq!(w2.len(), 2);
        assert_eq!(
            w.iter().map(|(_, s)| s.pos).collect::<Vec<_>>(),
            w2.iter().map(|(_, s)| s.pos).collect::<Vec<_>>()
        );
    }
}
