//! Compiled pack structures — the serialized runtime data model.
//!
//! These types are what the binary cache stores and what the runtime crates
//! consume. No strings beyond interning tables; everything else is indices.

use serde::{Deserialize, Serialize};

use crate::models::{Direction, FaceQuad, GuiLight, SpriteId};
use crate::blockstates::{Occlusion, Predicate};

/// A compiled model in the runtime table (index = ModelId).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeModel {
    pub quads: Vec<FaceQuad>,
    pub particle: SpriteId,
    pub ambient_occlusion: bool,
    pub gui_light: GuiLight,
}

/// Per-state data for one block: appearance + occlusion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeState {
    pub appearance: CompiledAppearance,
    pub occlusion: Occlusion,
    /// Full-cube geometry with a see-through sprite (glass, ice): vanilla
    /// culls faces between two adjacent blocks of the same kind.
    #[serde(default)]
    pub same_block_cull: bool,
}

/// Appearance referencing ModelIds (post blockstate compilation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompiledAppearance {
    Static(Vec<crate::blockstates::ModelInstance>),
    Variants(Vec<(Predicate, Vec<crate::blockstates::ModelInstance>)>),
    Multipart(Vec<(Predicate, Vec<crate::blockstates::ModelInstance>)>),
}

/// One block in the runtime registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeBlock {
    /// Registry name, e.g. "oak_log".
    pub name: String,
    /// Property schema (names + values) for state packing.
    pub schema: Vec<(String, Vec<String>)>,
    /// One entry per state combination (state id = packed property index).
    pub states: Vec<RuntimeState>,
}

/// Direction of an animation lookup: sprite -> animation row offsetting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeAnim {
    pub sprite: SpriteId,
    pub frametime: u32,
    /// Playback order; empty means sequential over `strip_frames` rows.
    pub frames: Vec<u16>,
    /// Total rows in the strip (frame count for sequential animations).
    pub strip_frames: u32,
    pub interpolate: bool,
}

/// The complete compiled pack.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompiledPack {
    /// Interned sprite names, in id order.
    pub sprite_names: Vec<(String, String)>,
    /// Runtime model table (index = ModelId).
    pub models: Vec<RuntimeModel>,
    /// Block registry (index = block id assigned by feathered-world).
    pub blocks: Vec<RuntimeBlock>,
    /// Animations for animated sprites.
    pub animations: Vec<RuntimeAnim>,
    /// Vanilla face shade table (renderer input, baked here for the cache).
    pub face_shade: [f32; 6],
}

impl CompiledPack {
    pub fn vanilla_face_shade() -> [f32; 6] {
        // up, down, north, south, west, east — vanilla directional brightness.
        [
            1.0, // up
            0.5, // down
            0.8, // north
            0.8, // south
            0.6, // west
            0.6, // east
        ]
    }

    pub fn sprite_id(&self, ns: &str, name: &str) -> Option<crate::models::SpriteId> {
        self.sprite_names
            .iter()
            .position(|(n, p)| n == ns && p == name)
            .map(|i| crate::models::SpriteId(i as u32))
    }
}

/// Re-export Direction for downstream crates convenience.
pub type Dir = Direction;
