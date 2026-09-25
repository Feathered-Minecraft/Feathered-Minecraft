//! Pack compiler orchestration: runs every stage in order and produces the
//! `CompiledPack` that gets written into the binary cache.
//!
//! Block selection: Phase 1 compiles the whole `minecraft` namespace (it is
//! fast — ~1,288 blockstates / ~4,192 models), but only the validation-scene
//! blocks are *required* to succeed. Blocks with unresolved references are
//! skipped with a warning rather than failing the build.

use crate::atlas::Atlas;
use crate::blockstates::{self, StateAppearance};
use crate::compiled::{CompiledAppearance, CompiledPack, RuntimeAnim, RuntimeBlock, RuntimeModel, RuntimeState};
use crate::error::AssetResult;
use crate::models::{self, ModelId, SpriteId, SpriteInterner};
use crate::pack::PackIndex;
use crate::sprites::SpriteStore;
use std::collections::HashMap;

/// Blocks whose failure aborts compilation (the milestone validation set).
pub const REQUIRED_BLOCKS: &[&str] = &[
    "oak_log",
    "grass_block",
    "stone",
    "glass",
    "torch",
    "rail",
    "short_grass",
    "vine",
    "redstone_wire",
    "water",
];

pub struct CompileStats {
    pub sprites: usize,
    pub models: usize,
    pub blocks: usize,
    pub animations: usize,
    pub skipped_blocks: usize,
    pub atlas_size: (u32, u32),
}

/// Compile the whole pack. `required_only = true` compiles just the ten
/// validation blocks (fast iteration); `false` compiles everything.
pub fn compile_pack(index: &PackIndex, required_only: bool) -> AssetResult<(CompiledPack, Atlas, CompileStats)> {
    // ---- Stage 1: sprites -------------------------------------------------
    let store = SpriteStore::load(index)?;

    // ---- Stage 2: raw models + blockstates --------------------------------
    let raw_models = models::load_raw(index)?;
    let raw_blockstates = blockstates::load_raw(index)?;

    // Decide which blocks to compile.
    let mut block_names: Vec<(String, String)> = raw_blockstates
        .keys()
        .filter(|(ns, _)| ns == "minecraft")
        .cloned()
        .collect();
    block_names.sort();
    if required_only {
        block_names.retain(|(_, name)| REQUIRED_BLOCKS.contains(&name.as_str()));
    }

    // ---- Stage 3: compile models reachable from those blockstates ---------
    let mut interner = SpriteInterner::default();
    let mut model_table: Vec<models::CompiledModel> = Vec::new();
    let mut model_ids: HashMap<(String, String), ModelId> = HashMap::new();
    let compile_model = |name: &(String, String),
                             interner: &mut SpriteInterner,
                             model_table: &mut Vec<models::CompiledModel>,
                             model_ids: &mut HashMap<(String, String), ModelId>|
     -> Option<ModelId> {
        if let Some(id) = model_ids.get(name) {
            return Some(*id);
        }
        match models::compile(name, &raw_models, interner) {
            Ok(cm) => {
                let id = ModelId(model_table.len() as u32);
                model_table.push(cm);
                model_ids.insert(name.clone(), id);
                Some(id)
            }
            Err(e) => {
                eprintln!("asset warning: model {}:{} skipped: {}", name.0, name.1, e);
                None
            }
        }
    };

    let mut blocks: Vec<RuntimeBlock> = Vec::new();
    let mut skipped = 0usize;

    for (ns, block_name) in &block_names {
        let Some(raw_bs) = raw_blockstates.get(&(ns.clone(), block_name.clone())) else {
            continue;
        };
        let schema = blockstates::extract_schema(raw_bs);

        // Model-id resolver closure over the table.
        let mut resolve = |model_ref: &str| -> ModelId {
            let normalized = match model_ref.split_once(':') {
                Some((mns, p)) => (mns.to_string(), p.to_string()),
                None => (ns.clone(), model_ref.to_string()),
            };
            compile_model(&normalized, &mut interner, &mut model_table, &mut model_ids)
                .unwrap_or(ModelId(0))
        };

        // Choose appearance kind.
        let appearance: Option<StateAppearance> = if raw_bs.variants.is_some() {
            Some(blockstates::compile_variants(raw_bs, &mut resolve)?)
        } else if raw_bs.multipart.is_some() {
            Some(blockstates::compile_multipart(raw_bs, &mut resolve)?)
        } else {
            None
        };
        let Some(appearance) = appearance else {
            skipped += 1;
            eprintln!("asset warning: blockstate {ns}:{block_name} has neither variants nor multipart");
            continue;
        };

        // Enumerate all states and classify occlusion per state.
        // Occlusion = full opaque cube geometry AND an alpha-opaque particle
        // sprite (glass is a 6-face cube but must never hide neighbors).
        let particle_is_opaque = |sprite_ref: &(String, String)| -> bool {
            store
                .get(&sprite_ref.0, &sprite_ref.1)
                .map(|s| {
                    // Sample frame 0: fully opaque when no alpha < 255.
                    s.tex.rgba.chunks_exact(4).all(|p| p[3] == 255)
                })
                .unwrap_or(false)
        };
        let combos = enumerate_states(&schema);
        let mut states = Vec::with_capacity(combos.len());
        for combo in &combos {
            let resolved = resolve_state(&appearance, combo);
            // Occlusion: from the first resolved model (all variants of a cube
            // share geometry; rotations don't affect classification).
            let (occlusion, same_block_cull) = resolved
                .first()
                .and_then(|mi| model_table.get(mi.model.0 as usize))
                .map(|m| {
                    let geometry_full = blockstates::classify_occlusion(&m.quads) == blockstates::Occlusion::Full;
                    let sprite_opaque = particle_is_opaque(&m.particle);
                    let occ = if geometry_full && sprite_opaque {
                        blockstates::Occlusion::Full
                    } else {
                        blockstates::Occlusion::None
                    };
                    // Glass-like: full cube geometry but see-through sprite —
                    // vanilla culls faces between two of the same such block.
                    let same = geometry_full && !sprite_opaque;
                    (occ, same)
                })
                .unwrap_or((blockstates::Occlusion::default(), false));
            states.push(RuntimeState {
                appearance: to_compiled(&appearance, resolved),
                occlusion,
                same_block_cull,
            });
        }

        blocks.push(RuntimeBlock {
            name: block_name.clone(),
            schema: schema.properties,
            states,
        });
        let _ = &interner; // particle interning happens below with the table
    }

    // ---- Stage 3b: bake variant rotations into dedicated model entries ----
    // Blockstate variants rotate whole models (`oak_log axis=x`, furnace
    // facing, …). We pre-rotate the quads at compile time into new model-table
    // entries so the runtime mesher never does per-block transform work.
    let mut baked: HashMap<(u32, u32, u32), ModelId> = HashMap::new();
    for block in &mut blocks {
        for state in &mut block.states {
            let instances = match &mut state.appearance {
                CompiledAppearance::Static(m) => m.as_mut_slice(),
                CompiledAppearance::Variants(v) | CompiledAppearance::Multipart(v) => {
                    // Rewrite every instance list inside the appearance.
                    for (_, models) in v.iter_mut() {
                        for inst in models.iter_mut() {
                            inst.model = bake_rotated(
                                inst.model,
                                &inst.rot,
                                &mut model_table,
                                &mut baked,
                            );
                        }
                    }
                    continue;
                }
            };
            for inst in instances.iter_mut() {
                inst.model = bake_rotated(inst.model, &inst.rot, &mut model_table, &mut baked);
            }
        }
    }

    // ---- Stage 3c: intern particle sprites ------------------------------
    // Faces intern their sprites during baking, but particles don't — and
    // fluid/water models are particle-only. Interning here guarantees every
    // referenced particle lands in the runtime sprite table (and the atlas).
    for m in &model_table {
        interner.intern(m.particle.clone());
    }

    // Required blocks must all have compiled.
    for req in REQUIRED_BLOCKS {
        if !blocks.iter().any(|b| &b.name == req) {
            return crate::error::err(
                "blockstates",
                format!("required block `{req}` failed to compile"),
            );
        }
    }

    // ---- Stage 4: atlas over referenced sprites ---------------------------
    // Skip internee names that don't exist as real textures (e.g. the barrier
    // model's literal `all:` ref and other vanilla placeholders). Build the
    // remap table interner-id -> runtime-id first, because face quads and
    // particle refs carry interner ids that shift when entries are dropped.
    let mut id_remap: Vec<Option<u32>> = Vec::with_capacity(interner.len());
    let mut sprite_names: Vec<(String, String)> = Vec::new();
    for name in interner.names() {
        if store.get(&name.0, &name.1).is_some() {
            id_remap.push(Some(sprite_names.len() as u32));
            sprite_names.push(name.clone());
        } else {
            id_remap.push(None);
        }
    }
    let remap = |id: SpriteId| -> SpriteId {
        SpriteId(id_remap.get(id.0 as usize).copied().flatten().unwrap_or(0))
    };
    for m in &mut model_table {
        for q in &mut m.quads {
            q.sprite = remap(q.sprite);
        }
    }
    let atlas = crate::atlas::build(&store, &sprite_names)?;

    // ---- Stage 5: assemble CompiledPack -----------------------------------
    let mut animations = Vec::new();
    for (i, name) in sprite_names.iter().enumerate() {
        if let Some((anim, entry)) = atlas.animations.get(name).map(|a| (a, atlas.get(&name.0, &name.1).unwrap())) {
            animations.push(RuntimeAnim {
                sprite: SpriteId(i as u32),
                frametime: anim.frametime,
                frames: anim.frames.clone(),
                strip_frames: entry.frames,
                interpolate: anim.interpolate,
            });
        }
    }

    let pack = CompiledPack {
        sprite_names: sprite_names.clone(),
        models: model_table
            .into_iter()
            .map(|m| {
                // The particle ref on the pre-runtime model is a SpriteRef;
                // find its runtime id in the filtered list.
                let particle_id = sprite_names
                    .iter()
                    .position(|n| *n == m.particle)
                    .map(|i| SpriteId(i as u32))
                    .unwrap_or(SpriteId(0));
                RuntimeModel {
                    quads: m.quads,
                    particle: particle_id,
                    ambient_occlusion: m.ambient_occlusion,
                    gui_light: m.gui_light,
                }
            })
            .collect(),
        blocks,
        animations,
        face_shade: CompiledPack::vanilla_face_shade(),
    };

    let stats = CompileStats {
        sprites: store.sprites.len(),
        models: pack.models.len(),
        blocks: pack.blocks.len(),
        animations: pack.animations.len(),
        skipped_blocks: skipped,
        atlas_size: (atlas.width, atlas.height),
    };
    Ok((pack, atlas, stats))
}

fn to_compiled(
    appearance: &StateAppearance,
    resolved: Vec<crate::blockstates::ModelInstance>,
) -> CompiledAppearance {
    match appearance {
        // Static / single-matching-variant states collapse to Static.
        StateAppearance::Static(_) | StateAppearance::Variants(_) => {
            CompiledAppearance::Static(resolved)
        }
        StateAppearance::Multipart(entries) => CompiledAppearance::Multipart(
            entries
                .iter()
                .map(|e| (e.when.clone(), e.models.clone()))
                .collect(),
        ),
    }
}

/// ModelId of a pre-rotated copy of `base` by `rot` (shared between all
/// states/blocks that use the same base+rotation combination).
fn bake_rotated(
    base: ModelId,
    rot: &crate::blockstates::VariantRot,
    table: &mut Vec<models::CompiledModel>,
    memo: &mut HashMap<(u32, u32, u32), ModelId>,
) -> ModelId {
    if rot.x == 0.0 && rot.y == 0.0 {
        return base;
    }
    // Quantize to 1° so the memo key is stable.
    let key = (
        base.0,
        rot.x.rem_euclid(360.0).round() as u32,
        rot.y.rem_euclid(360.0).round() as u32,
    );
    if let Some(&id) = memo.get(&key) {
        return id;
    }
    let mut m = table[base.0 as usize].clone();
    for q in &mut m.quads {
        models::apply_variant_rotation(q, rot);
    }
    let id = ModelId(table.len() as u32);
    table.push(m);
    memo.insert(key, id);
    id
}

/// Resolve which model instances apply for a property combo.
pub fn resolve_state(
    appearance: &StateAppearance,
    props: &[(String, String)],
) -> Vec<crate::blockstates::ModelInstance> {
    match appearance {
        StateAppearance::Static(models) => models.clone(),
        StateAppearance::Variants(entries) => entries
            .iter()
            .find(|e| e.when.matches(props))
            .map(|e| e.models.clone())
            .unwrap_or_default(),
        StateAppearance::Multipart(entries) => entries
            .iter()
            .filter(|e| e.when.matches(props))
            .flat_map(|e| e.models.iter().cloned())
            .collect(),
    }
}

/// Enumerate all property combinations as (name, value) pairs.
fn enumerate_states(schema: &blockstates::PropertySchema) -> Vec<Vec<(String, String)>> {
    let mut combos: Vec<Vec<(String, String)>> = vec![vec![]];
    for (name, values) in &schema.properties {
        let mut next = Vec::with_capacity(combos.len() * values.len());
        for combo in &combos {
            for v in values {
                let mut c = combo.clone();
                c.push((name.clone(), v.clone()));
                next.push(c);
            }
        }
        combos = next;
    }
    combos
}
