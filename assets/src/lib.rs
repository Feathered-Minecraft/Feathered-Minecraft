//! feathered-assets — the only crate in Feathered allowed to touch JSON.
//!
//! Pipeline: DISCOVERY (pack) → VALIDATION/PARSING (sprites, models,
//! blockstates) → COMPILATION (atlas, prototypes) → CACHE (binary blob).

pub mod atlas;
pub mod blockstates;
pub mod cache;
pub mod compiled;
pub mod compiler;
pub mod error;
pub mod meta;
pub mod models;
pub mod models_raw;
pub mod pack;
pub mod sprites;
pub mod texture;
pub mod tint;

pub use models::{ModelId, SpriteId};
