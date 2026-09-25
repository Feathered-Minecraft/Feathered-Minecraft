//! Raw JSON model definitions (serde view of the pack format).

use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize)]
pub struct RawModel {
    #[serde(default)]
    pub parent: Option<String>,
    #[serde(default)]
    pub textures: HashMap<String, RawTextureValue>,
    #[serde(default)]
    pub elements: Option<Vec<RawElement>>,
    #[serde(default)]
    pub ambientocclusion: Option<bool>,
    #[serde(default)]
    pub gui_light: Option<String>,
    #[serde(default)]
    pub display: HashMap<String, RawDisplay>,
}

impl RawModel {
    pub fn parse(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// Texture values are either plain strings (`"#side"`, `"minecraft:block/x"`)
/// or `{ "sprite": ..., "force_translucent": true }` dicts (redstone dust).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum RawTextureValue {
    Str(String),
    Sprite { sprite: String, force_translucent: Option<bool> },
}

impl RawTextureValue {
    pub fn sprite_ref(&self) -> &str {
        match self {
            RawTextureValue::Str(s) => s,
            RawTextureValue::Sprite { sprite, .. } => sprite,
        }
    }

    pub fn force_translucent(&self) -> bool {
        matches!(self, RawTextureValue::Sprite { force_translucent: Some(true), .. })
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct RawDisplay {
    pub rotation: Option<[f32; 3]>,
    pub translation: Option<[f32; 3]>,
    pub scale: Option<[f32; 3]>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RawElement {
    /// `[x0, y0, z0]` in 0..16 block units (fractional allowed).
    pub from: [f32; 3],
    pub to: [f32; 3],
    #[serde(default)]
    pub rotation: Option<RawRotation>,
    #[serde(default)]
    pub shade: Option<bool>,
    /// Per-face light emission 0..15 (lantern-style emissive geometry).
    #[serde(default)]
    pub light_emission: Option<u8>,
    pub faces: HashMap<String, RawFace>,
    /// `shade_direction_override` (26.x): forces the directional shade normal.
    #[serde(default, rename = "shade_direction_override")]
    pub shade_direction_override: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RawRotation {
    pub origin: [f32; 3],
    /// Classic form: single axis + angle. 26.x also allows arbitrary-axis
    /// euler rotations (`x`/`y`/`z` angles directly, no `axis`/`angle`).
    #[serde(default)]
    pub axis: Option<String>,
    #[serde(default)]
    pub angle: Option<f32>,
    #[serde(default)]
    pub x: Option<f32>,
    #[serde(default)]
    pub y: Option<f32>,
    #[serde(default)]
    pub z: Option<f32>,
    #[serde(default)]
    pub rescale: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RawFace {
    /// `[x1, y1, x2, y2]` in 0..16 texture units. Defaults to the full face.
    #[serde(default)]
    pub uv: Option<[f32; 4]>,
    /// Texture variable (`"#side"`) or direct sprite name.
    pub texture: String,
    /// `cullface`: direction whose neighbor hides this face.
    #[serde(default)]
    pub cullface: Option<String>,
    /// Biome-tint index (0 = primary tint).
    #[serde(default)]
    pub tintindex: Option<i32>,
}
