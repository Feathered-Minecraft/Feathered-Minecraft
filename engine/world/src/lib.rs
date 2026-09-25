//! feathered-world — block registry and world block access.
//!
//! Consumes the compiled pack (from `feathered-assets`) but performs no
//! parsing itself. The renderer and mesher talk to this crate only.

pub mod grid;

use feathered_assets::blockstates::{ModelInstance, Occlusion, Predicate};
use feathered_assets::compiled::{CompiledAppearance, CompiledPack};
use feathered_assets::models::ModelId;

/// Appearance as seen by the mesher: borrowed view over compiled data.
#[derive(Debug, Clone, Copy)]
pub enum AppearanceRef<'a> {
    Static(&'a [ModelInstance]),
    Variants(&'a [(Predicate, Vec<ModelInstance>)]),
    Multipart(&'a [(Predicate, Vec<ModelInstance>)]),
}

/// One block state in the registry.
#[derive(Debug, Clone)]
pub struct StateEntry {
    pub appearance: CompiledAppearance,
    pub occlusion: Occlusion,
    /// Same-kind face culling (glass): see `RuntimeState::same_block_cull`.
    pub same_block_cull: bool,
}

/// One block in the registry.
#[derive(Debug, Clone)]
pub struct BlockDef {
    pub name: String,
    /// Property schema: (property name, possible values).
    pub schema: Vec<(String, Vec<String>)>,
    /// One per state; state id = packed property index (schema order).
    pub states: Vec<StateEntry>,
}

impl BlockDef {
    /// Pack a (property -> value) assignment into a state id.
    /// Missing properties default to their first value.
    pub fn pack_state(&self, props: &[(String, String)]) -> u32 {
        let mut id: u32 = 0;
        let mut multiplier: u32 = 1;
        for (prop, values) in &self.schema {
            let chosen = props
                .iter()
                .find(|(k, _)| k == prop)
                .map(|(_, v)| v.clone())
                .unwrap_or_else(|| values.first().cloned().unwrap_or_default());
            let idx = values
                .iter()
                .position(|v| *v == chosen)
                .unwrap_or(0) as u32;
            id += idx * multiplier;
            multiplier *= values.len().max(1) as u32;
        }
        id
    }

    /// Unpack a state id back into property assignments.
    pub fn unpack_state(&self, mut id: u32) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for (_, values) in &self.schema {
            let n = values.len().max(1) as u32;
            let idx = (id % n) as usize;
            id /= n;
            out.push((
                String::new(), // filled below by zipping with schema names
                values.get(idx).cloned().unwrap_or_default(),
            ));
        }
        for ((name, _), slot) in self.schema.iter().zip(out.iter_mut()) {
            slot.0 = name.clone();
        }
        out
    }

    pub fn state(&self, id: u32) -> Option<&StateEntry> {
        self.states.get(id as usize)
    }
}

/// The block registry. Holds the compiled models for direct lookup.
#[derive(Debug, Clone, Default)]
pub struct Registry {
    blocks: Vec<BlockDef>,
    by_name: std::collections::HashMap<String, u32>,
    sprite_names: Vec<(String, String)>,
    face_shade: [f32; 6],
    models: Vec<feathered_assets::compiled::RuntimeModel>,
    anims: Vec<(u32, u32, Vec<u16>, u32, bool)>, // (sprite_id, frametime, frames, strip_frames, interpolate)
}

impl Registry {
    /// Build from the compiled pack (zero parsing — the pack is already data).
    ///
    /// Block ids are **1-based**: id 0 is reserved for air so world cells can
    /// store `0` for empty without colliding with a real block.
    pub fn from_compiled(pack: CompiledPack) -> Registry {
        let mut blocks = Vec::new();
        let mut by_name = std::collections::HashMap::new();
        for (i, b) in pack.blocks.iter().enumerate() {
            by_name.insert(b.name.clone(), (i + 1) as u32);
            blocks.push(BlockDef {
                name: b.name.clone(),
                schema: b.schema.clone(),
                states: b
                    .states
                    .iter()
                    .map(|s| StateEntry {
                        appearance: s.appearance.clone(),
                        occlusion: s.occlusion,
                        same_block_cull: s.same_block_cull,
                    })
                    .collect(),
            });
        }
        let anims = pack
            .animations
            .iter()
            .map(|a| (a.sprite.0, a.frametime, a.frames.clone(), a.strip_frames, a.interpolate))
            .collect();
        Registry {
            blocks,
            by_name,
            sprite_names: pack.sprite_names.clone(),
            face_shade: pack.face_shade,
            models: pack.models,
            anims,
        }
    }

    pub fn block(&self, name: &str) -> Option<&BlockDef> {
        // by_name stores 1-based ids; `blocks` is 0-indexed.
        self.by_name.get(name).map(|&i| &self.blocks[(i - 1) as usize])
    }

    pub fn block_id(&self, name: &str) -> Option<u32> {
        self.by_name.get(name).copied()
    }

    pub fn block_by_id(&self, id: u32) -> Option<&BlockDef> {
        if id == 0 {
            return None; // air
        }
        self.blocks.get((id - 1) as usize)
    }

    /// Model table lookup.
    pub fn model(&self, id: ModelId) -> Option<&feathered_assets::compiled::RuntimeModel> {
        self.models.get(id.0 as usize)
    }

    /// First model of a block state (validation helper). For variant
    /// appearances this inspects the first variant entry (states differ only
    /// in model choice, not in structure).
    pub fn model_of(&self, block: &str, state: u32) -> Option<&feathered_assets::compiled::RuntimeModel> {
        let b = self.block(block)?;
        let s = b.state(state)?;
        let mi = match &s.appearance {
            CompiledAppearance::Static(m) => m.first()?,
            CompiledAppearance::Variants(v) => v.first()?.1.first()?,
            CompiledAppearance::Multipart(v) => v.first()?.1.first()?,
        };
        self.model(mi.model)
    }

    /// Render slot for an animated sprite (index into the animation table + 1;
    /// 0 = static sprite). The renderer maps slot -> current frame offset.
    pub fn anim_slot(&self, sprite: feathered_assets::models::SpriteId) -> Option<u8> {
        self.anims
            .iter()
            .position(|(sid, _, _, _, _)| *sid == sprite.0)
            .filter(|i| *i < 255)
            .map(|i| (i + 1) as u8)
    }

    /// Sprite id by interned name (registry lookup, runtime-safe).
    pub fn sprite_id_by_name(&self, ns: &str, name: &str) -> Option<feathered_assets::models::SpriteId> {
        self.sprite_names
            .iter()
            .position(|(n, p)| n == ns && p == name)
            .map(|i| feathered_assets::models::SpriteId(i as u32))
    }

    /// Animation frame count for a sprite id (None = not animated).
    pub fn anim_frames(&self, sprite: feathered_assets::models::SpriteId) -> Option<u32> {
        self.anims
            .iter()
            .find(|(sid, _, _, _, _)| *sid == sprite.0)
            .map(|(_, _, frames, strip_frames, _)| {
                if frames.is_empty() { *strip_frames } else { frames.len() as u32 }
            })
    }

    pub fn anims(&self) -> &[(u32, u32, Vec<u16>, u32, bool)] {
        &self.anims
    }

    pub fn sprite_names(&self) -> &[(String, String)] {
        &self.sprite_names
    }

    pub fn face_shade(&self) -> &[f32; 6] {
        &self.face_shade
    }
}

/// Resolve which model instances apply to a state, given its properties.
///
/// Random pools: a variant key whose apply list holds several models (or a
/// `Static` appearance with several instances — the compiler collapses
/// single-key pools into `Static`) renders exactly ONE model, chosen
/// deterministically from `rng` (the caller's per-position hash). Multipart
/// entries all render, but each entry's own list is still a pool.
pub fn resolve_appearance<'a>(
    appearance: AppearanceRef<'a>,
    props: &[(String, String)],
    rng: u64,
) -> Vec<ModelInstance> {
    match appearance {
        AppearanceRef::Static(m) => vec![pick_weighted(m, rng)],
        AppearanceRef::Variants(entries) => entries
            .iter()
            .find(|(p, _)| p.matches(props))
            .map(|(_, m)| vec![pick_weighted(m, rng)])
            .unwrap_or_default(),
        AppearanceRef::Multipart(entries) => entries
            .iter()
            .filter(|(p, _)| p.matches(props))
            .map(|(_, m)| pick_weighted(m, rng))
            .collect(),
    }
}

/// Weighted pick from a random pool (weights default to 1; single-entry
/// pools are the common case and skip the hash entirely).
fn pick_weighted(models: &[ModelInstance], rng: u64) -> ModelInstance {
    match models.len() {
        0 => ModelInstance {
            model: feathered_assets::models::ModelId(0),
            rot: Default::default(),
            weight: 1,
        },
        1 => models[0],
        _ => {
            let total: u64 = models.iter().map(|m| m.weight.max(1) as u64).sum();
            let mut pick = rng % total.max(1);
            for m in models {
                let w = m.weight.max(1) as u64;
                if pick < w {
                    return *m;
                }
                pick -= w;
            }
            models[0]
        }
    }
}
