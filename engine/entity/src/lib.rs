//! feathered-entity — a Feathered-native entity/mob system.
//!
//! Independent implementation informed by the *architectural patterns* of
//! PumpkinMC's entity/AI code (goal selector with controls, spawn attempt
//! loops); no PumpkinMC source is copied — see
//! `docs/WORLDGEN_ENTITIES_REPORT.md` for the licensing audit. Server-only
//! machinery (ArcSwap state, protocol sync, plugin events, NBT) is
//! deliberately absent: Feathered is a client/engine, not a server.
//!
//! Layers:
//! * [`world`] — `EntityWorld`: storage, `EntityId` lifecycle, ticking,
//!   spatial queries. Deterministic iteration order (id-sorted).
//! * [`ai`] — goal selector + reusable goals + bounded voxel A* navigation.
//! * [`spawn`] — category-capped, budgeted, deterministic mob spawning.
//! * [`mobs`] — the initial mob set (one passive, one hostile).

pub mod ai;
pub mod mobs;
pub mod spawn;
pub mod world;

pub use world::{EntityId, EntityWorld, MobKind, MobState};
