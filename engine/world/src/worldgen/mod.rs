//! Feathered world generation: Seed → Climate/Biomes → Density/Shape →
//! Carvers → Surface rules → Features → Chunk finalization.
//!
//! Independent Feathered implementation informed by the architectural
//! patterns of the locally consulted reference trees (see
//! `docs/WORLDGEN_ENTITIES_REPORT.md` for the licensing audit — no source
//! was copied; SteelMC is AGPL-3.0, so ideas only).

pub mod biomes;
pub mod noise;
pub mod shape;

pub use shape::{GenStats, WorldGenerator, BASE_HEIGHT, SEA_LEVEL};
