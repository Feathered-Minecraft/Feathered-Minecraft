//! feathered-chunk — the runtime chunk mesher.
//!
//! Consumes compiled `RuntimeModel` prototypes (already rotation-baked by the
//! asset compiler) and world neighbor data; emits the compact vertex format:
//! `position: f32×3 (world blocks) · uv: u16×2 (atlas texels) ·
//!  tint: u32 · shade: u8 · anim_id: u8 · light: u8×2`.
//!
//! Division of labor (per the architecture plan): the asset compiler bakes
//! quads once; the mesher owns face-existence decisions (neighbor occlusion),
//! per-position variant selection, biome tint and layer classification. The
//! mesher never parses assets and the compiler never sees world data.

use bytemuck::{Pod, Zeroable};
use feathered_assets::models::{face_corners, Direction, FaceQuad, SpriteId};
use feathered_assets::compiled::CompiledAppearance;
use feathered_world::grid::World;
use feathered_world::{resolve_appearance, AppearanceRef, Registry};

/// Vertex: 3×f32 pos + 2×u16 uv + u32 tint + shade + anim_id + 2×u8 light
/// = 24 B. Positions are f32 because rotated elements (45° crosses, 22.5°
/// stairs landings) land off the 1/16 grid; i16 quantization visibly shears
/// them.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],   // 0..12 world blocks (chunk-local)
    pub uv: [u16; 2],    // 12..16 (atlas texels / 65535 normalized)
    pub tint: u32,       // 16..20 RGBA
    pub shade: u8,       // 20
    pub anim_id: u8,     // 21 (row-offset slot; 0 = static)
    pub light: [u8; 2],  // 22..24 (block, sky — Phase 1 fullbright)
}

const _: () = assert!(std::mem::size_of::<Vertex>() == 24);

/// One mesh layer's output.
#[derive(Debug, Default)]
pub struct MeshLayer {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl MeshLayer {
    fn push_quad(&mut self, verts: [Vertex; 4]) {
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&verts);
        self.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

/// Render buckets the mesher fills.
#[derive(Debug, Default)]
pub struct MeshedChunk {
    pub opaque: MeshLayer,
    pub cutout: MeshLayer,
    pub translucent: MeshLayer,
}

/// Tint decisions for the validation scene (plains biome everywhere).
pub struct TintPolicy {
    /// RGBA tint applied to tintindex-0 faces (plains grass from the colormap).
    pub tint0: [u8; 4],
}

/// Face shade table (vanilla directional brightness): up, down, n, s, w, e.
pub const FACE_SHADE: [u8; 6] = [255, 128, 204, 204, 153, 153];

fn dir_index(d: Direction) -> usize {
    match d {
        Direction::Down => 1,
        Direction::Up => 0,
        Direction::North => 2,
        Direction::South => 3,
        Direction::West => 4,
        Direction::East => 5,
    }
}

fn cull_offset(d: Direction) -> (i64, i64, i64) {
    match d {
        Direction::Down => (0, -1, 0),
        Direction::Up => (0, 1, 0),
        Direction::North => (0, 0, -1),
        Direction::South => (0, 0, 1),
        Direction::West => (-1, 0, 0),
        Direction::East => (1, 0, 0),
    }
}

/// UV corners matching the CCW corner order, from the quad's uv rect
/// [u0, v0, u1, v1] in 0..16 sprite space (v grows downward in the atlas).
fn quad_uvs(q: &FaceQuad) -> [[f32; 2]; 4] {
    let [u0, v0, u1, v1] = q.uv;
    [[u0, v1], [u1, v1], [u1, v0], [u0, v0]]
}

/// Deterministic per-position hash for random variant selection (vanilla
/// picks one variant per position via a position hash; stable across rebuilds).
fn pos_hash(x: i64, y: i64, z: i64) -> u64 {
    let mut h = 0x9E37_79B9_7F4A_7C15u64
        ^ (x as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
        ^ (y as u64).wrapping_mul(0x94D0_49BB_1331_11EB)
        ^ (z as u64).wrapping_mul(0x2545_F491_4F6C_DD1D);
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 31;
    h
}

/// Precomputed per-sprite rect/opaque tables (indexed by SpriteId). Built once
/// per mesh job so the hot loop never calls back into the atlas closure.
type SpriteRect = (u32, u32, u32, u32, u32, u32, bool); // x,y,fw,fh,frames,stride,opaque

fn sprite_tables(
    registry: &Registry,
    uv_rects: &dyn Fn(u32) -> Option<SpriteRect>,
) -> Vec<Option<SpriteRect>> {
    (0..registry.sprite_names().len() as u32)
        .map(uv_rects)
        .collect()
}

/// Emit one quad at a world position: UV bake, shading, tint, layer choice.
fn emit_quad(
    out: &mut MeshedChunk,
    rects: &[Option<SpriteRect>],
    atlas_size: (u32, u32),
    tint0: [u8; 4],
    q: &FaceQuad,
    x: i64,
    y: i64,
    z: i64,
    anim_id: u8,
) {
    let Some(&(ax, ay, fw, fh, _frames, _stride, sprite_opaque)) =
        rects.get(q.sprite.0 as usize).and_then(|r| r.as_ref())
    else {
        return;
    };
    let (au, av) = (atlas_size.0 as f32, atlas_size.1 as f32);
    let uv_norm = |u: f32, v: f32| -> [u16; 2] {
        [
            ((ax as f32 + u / 16.0 * fw as f32) / au * 65535.0) as u16,
            ((ay as f32 + v / 16.0 * fh as f32) / av * 65535.0) as u16,
        ]
    };

    let uvs = quad_uvs(q);
    let shade = match q.shade_override {
        Some(d) => FACE_SHADE[dir_index(d)],
        None => FACE_SHADE[dir_index(q.dir)],
    };
    let tint_rgba: [u8; 4] = match q.tint {
        Some(0) => tint0,
        _ => [255, 255, 255, 255],
    };
    let tint_packed = u32::from_le_bytes(tint_rgba);

    // Geometry is baked by the compiler (corners already rotated/scaled);
    // only the world offset is applied here.
    let verts: [Vertex; 4] = std::array::from_fn(|i| {
        let p = q.corners[i];
        Vertex {
            pos: [x as f32 + p[0] / 16.0, y as f32 + p[1] / 16.0, z as f32 + p[2] / 16.0],
            uv: uv_norm(uvs[i][0], uvs[i][1]),
            tint: tint_packed,
            shade,
            anim_id,
            light: [255, 255],
        }
    });

    // Layer selection (vanilla-correct): forced-translucent quads (fluids,
    // stained glass) → translucent; fully opaque sprites → opaque; everything
    // with any transparency (glass, leaves, plants, rails, torches) → cutout.
    let layer = if q.force_translucent {
        &mut out.translucent
    } else if sprite_opaque {
        &mut out.opaque
    } else {
        &mut out.cutout
    };
    layer.push_quad(verts);
}

/// Engine-side still-water rendering (Phase 1). Vanilla renders fluids outside
/// the model system — `blockstates/water.json` has no elements — so fluid
/// geometry is the mesher's job: a 14/16-tall cube from the animated
/// `water_still` sprite on the translucent layer, faces dropped against water
/// neighbors and occluders, frames advanced via `anim_id`.
fn emit_fluid(
    out: &mut MeshedChunk,
    rects: &[Option<SpriteRect>],
    atlas_size: (u32, u32),
    registry: &Registry,
    world: &World,
    x: i64,
    y: i64,
    z: i64,
    sprite: SpriteId,
) {
    let anim_id = registry.anim_slot(sprite).unwrap_or(0);
    let h = 14.0f32; // source-block surface height, 0..16 units

    let at = |dx: i64, dy: i64, dz: i64| world.get(x + dx, y + dy, z + dz);
    let is_water = |dx: i64, dy: i64, dz: i64| {
        at(dx, dy, dz)
            .map(|(nid, _)| {
                registry
                    .block_by_id(nid)
                    .map(|nb| nb.name == "water")
                    .unwrap_or(false)
            })
            .unwrap_or(false)
    };
    let occludes = |dx: i64, dy: i64, dz: i64| match at(dx, dy, dz) {
        Some((0, _)) => false, // air
        Some((nid, nsid)) => registry
            .block_by_id(nid)
            .and_then(|nb| nb.state(nsid))
            .map(|ns| ns.occlusion.hides_neighbor())
            .unwrap_or(false),
        None => true, // world border never exposed
    };

    let mk = |dir: Direction, bounds: [f32; 6]| FaceQuad {
        bounds,
        corners: face_corners(dir, bounds),
        dir,
        sprite,
        uv: [0.0, 0.0, 16.0, 16.0],
        tint: None,
        cull: None,
        force_translucent: true,
        light_emission: 0,
        shade_override: None,
    };

    let mut faces = Vec::with_capacity(6);
    if !is_water(0, 1, 0) && !occludes(0, 1, 0) {
        faces.push(mk(Direction::Up, [0.0, h, 0.0, 16.0, h, 16.0]));
    }
    if !is_water(0, -1, 0) && !occludes(0, -1, 0) {
        faces.push(mk(Direction::Down, [0.0, 0.0, 0.0, 16.0, 0.0, 16.0]));
    }
    for (dir, d, bounds) in [
        (Direction::North, (0, 0, -1), [0.0, 0.0, 0.0, 16.0, h, 0.0]),
        (Direction::South, (0, 0, 1), [0.0, 0.0, 16.0, 16.0, h, 16.0]),
        (Direction::West, (-1, 0, 0), [0.0, 0.0, 0.0, 0.0, h, 16.0]),
        (Direction::East, (1, 0, 0), [16.0, 0.0, 0.0, 16.0, h, 16.0]),
    ] {
        if !is_water(d.0, d.1, d.2) && !occludes(d.0, d.1, d.2) {
            faces.push(mk(dir, bounds));
        }
    }

    for f in &faces {
        emit_quad(out, rects, atlas_size, [255, 255, 255, 255], f, x, y, z, anim_id);
    }
}

/// Mesh one world. `registry` supplies state→appearance and occlusion data;
/// `uv_rects` maps SpriteId → (x, y, frame_w, frame_h, frames, stride,
/// fully-opaque-sprite).
pub fn mesh_world(
    world: &World,
    registry: &Registry,
    uv_rects: &dyn Fn(u32) -> Option<SpriteRect>,
    atlas_size: (u32, u32),
    tint: &TintPolicy,
) -> MeshedChunk {
    let mut out = MeshedChunk::default();
    let [sx, sy, sz] = world.size;
    let rects = sprite_tables(registry, uv_rects);

    for y in 0..sy as i64 {
        for z in 0..sz as i64 {
            for x in 0..sx as i64 {
                let Some((block_id, state_id)) = world.get(x, y, z) else { continue };
                if block_id == 0 {
                    continue;
                }
                let Some(block) = registry.block_by_id(block_id) else { continue };
                let Some(state) = block.state(state_id) else { continue };

                // Resolve appearance for this state's properties. Random
                // pools pick ONE model per position (deterministic hash).
                let props = block.unpack_state(state_id);
                let appearance = match &state.appearance {
                    CompiledAppearance::Static(m) => AppearanceRef::Static(m),
                    CompiledAppearance::Variants(v) => AppearanceRef::Variants(v),
                    CompiledAppearance::Multipart(m) => AppearanceRef::Multipart(m),
                };
                let instances = resolve_appearance(appearance, &props, pos_hash(x, y, z));

                for mi in instances {
                    let Some(model) = registry.model(mi.model) else { continue };
                    for q in &model.quads {
                        // Neighbor culling: only faces with a cull dir are
                        // hidden when the neighbor occludes. Out-of-world
                        // borders count as occluding so the world shell is
                        // never meshed.
                        if let Some(cull) = q.cull {
                            let (dx, dy, dz) = cull_offset(cull);
                            let neighbor = world.get(x + dx, y + dy, z + dz);
                            let neighbor_occludes = match neighbor {
                                Some((0, _)) => false, // air
                                Some((nid, nsid)) => {
                                    let Some(nb) = registry.block_by_id(nid) else { continue };
                                    let Some(ns) = nb.state(nsid) else { continue };
                                    // Same-kind culling (glass vs glass) hides
                                    // the face even though neither occludes.
                                    if ns.same_block_cull && nid == block_id {
                                        continue;
                                    }
                                    ns.occlusion.hides_neighbor()
                                }
                                None => true, // out of world: borders never expose faces
                            };
                            if neighbor_occludes {
                                continue;
                            }
                        }
                        // Variant rotations are baked into the model table at
                        // compile time; anim_id 0 (block models are static
                        // sprites — animated block textures don't exist in the
                        // model path in this pack).
                        emit_quad(&mut out, &rects, atlas_size, tint.tint0, q, x, y, z, 0);
                    }
                }

                // Fluids (Phase 1: still water).
                if block.name == "water" {
                    match registry.sprite_id_by_name("minecraft", "block/water_still") {
                        Some(sid) => {
                            if std::env::var("FEATHERED_DEBUG_FLUID").is_ok() {
                                eprintln!("[fluid] water at {x},{y},{z} sprite {sid:?}");
                            }
                            emit_fluid(&mut out, &rects, atlas_size, registry, world, x, y, z, sid);
                        }
                        None => {
                            if std::env::var("FEATHERED_DEBUG_FLUID").is_ok() {
                                let names = registry.sprite_names();
                                eprintln!("[fluid] water_still NOT FOUND; have {} sprites, sample: {:?}", names.len(), names.iter().find(|(_, p)| p.contains("water")));
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// Unused ModelId import guard (used indirectly via registry API).
#[allow(dead_code)]
fn _witness(_: feathered_assets::models::ModelId) {}
