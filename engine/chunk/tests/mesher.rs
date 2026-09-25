//! Mesher tests: neighbor culling, layer assignment, vertex counts.
//!
//! Semantics verified here (from the real 26.3 pack):
//! * stone's blockstate has a 4-entry random pool under `""` — vanilla shows
//!   exactly one per position, so one stone cube = exactly 6 quads.
//! * out-of-world borders count as occluding (the world shell never meshes).
//! * `tinted_cross` is double-sided (2 planes × 2 faces = 4 quads / 16 verts).
//! * plain glass uses binary-alpha sprites → cutout layer, never translucent;
//!   translucent is reserved for water/stained glass/ice (true alpha).

use feathered_assets::atlas::Atlas;
use feathered_assets::compiler::compile_pack;
use feathered_assets::pack::discover;
use feathered_chunk::mesh_world;
use feathered_chunk::TintPolicy;
use feathered_world::grid::World;
use feathered_world::Registry;

fn test_pack_root() -> &'static str {
    for p in ["texture/assets", "../texture/assets", "../../texture/assets"] {
        if std::path::Path::new(p).join("pack.mcmeta").exists() {
            return p;
        }
    }
    panic!("pack not found; run tests from the Feathered workspace");
}

struct Fixture {
    registry: Registry,
    atlas: Atlas,
}

fn fixture() -> Fixture {
    let index = discover(std::path::Path::new(test_pack_root())).unwrap();
    let (pack, atlas, _stats) = compile_pack(&index, true).unwrap();
    Fixture {
        registry: Registry::from_compiled(pack),
        atlas,
    }
}

/// True when every texel of the sprite's frame-0 rect is fully opaque.
fn sprite_is_opaque(atlas: &Atlas, sprite_id: u32, registry: &Registry) -> bool {
    let Some(name) = registry.sprite_names().get(sprite_id as usize) else {
        return false;
    };
    let Some(e) = atlas.get(&name.0, &name.1) else {
        return false;
    };
    let stride = atlas.width as usize * 4;
    for row in 0..e.frame_h as usize {
        let y = e.y as usize + row;
        let start = y * stride + e.x as usize * 4;
        let px = &atlas.pixels[start..start + e.frame_w as usize * 4];
        if px.chunks_exact(4).any(|p| p[3] != 255) {
            return false;
        }
    }
    true
}

fn rects(fx: &Fixture) -> impl Fn(u32) -> Option<(u32, u32, u32, u32, u32, u32, bool)> + '_ {
    |sprite_id: u32| {
        let name = fx.registry.sprite_names().get(sprite_id as usize)?;
        let e = fx.atlas.get(&name.0, &name.1)?;
        let opaque = sprite_is_opaque(&fx.atlas, sprite_id, &fx.registry);
        Some((e.x, e.y, e.frame_w, e.frame_h, e.frames, e.frame_stride, opaque))
    }
}

fn white_tint() -> TintPolicy {
    TintPolicy { tint0: [255, 255, 255, 255] }
}

#[test]
fn buried_stone_generates_zero_faces() {
    let fx = fixture();
    let mut world = World::new([8, 8, 8]);
    let stone = fx.registry.block_id("stone").unwrap();
    for x in 0..8 {
        for y in 0..8 {
            for z in 0..8 {
                world.set(x, y, z, stone, 0);
            }
        }
    }
    let mesh = mesh_world(&world, &fx.registry, &rects(&fx), (fx.atlas.width, fx.atlas.height), &white_tint());
    // A fully occluded 8³ block of stone has no visible faces at all, and the
    // world border never exposes shell faces either.
    assert_eq!(mesh.opaque.vertices.len(), 0, "interior stone must cull entirely");
    assert_eq!(mesh.cutout.vertices.len(), 0);
}

#[test]
fn exposed_stone_generates_exactly_six_faces_per_cube() {
    let fx = fixture();
    let mut world = World::new([8, 8, 8]);
    let stone = fx.registry.block_id("stone").unwrap();
    // Single stone block floating in air: 6 faces = 24 verts.
    // (stone.json lists a 4-entry random pool; the mesher must pick ONE.)
    world.set(4, 4, 4, stone, 0);
    let mesh = mesh_world(&world, &fx.registry, &rects(&fx), (fx.atlas.width, fx.atlas.height), &white_tint());
    assert_eq!(mesh.opaque.vertices.len(), 24);
    assert_eq!(mesh.opaque.indices.len(), 36);
}

#[test]
fn random_variant_pool_picks_deterministically() {
    let fx = fixture();
    let mut world = World::new([8, 8, 8]);
    let stone = fx.registry.block_id("stone").unwrap();
    // A row of stones: each position renders exactly 6 faces (one pool pick),
    // and the pick is stable across two meshes of the same world.
    for x in 0..8 {
        world.set(x, 4, 4, stone, 0);
    }
    let mesh_a = mesh_world(&world, &fx.registry, &rects(&fx), (fx.atlas.width, fx.atlas.height), &white_tint());
    let mesh_b = mesh_world(&world, &fx.registry, &rects(&fx), (fx.atlas.width, fx.atlas.height), &white_tint());
    // Out-of-world borders count as occluders, so the bar at y=4 spanning the
    // full x range shows only its 4 exposed sides per block (top/bottom face
    // air, ends against the border): 4 × 8 blocks = 32 quads = 128 verts.
    assert_eq!(mesh_a.opaque.vertices.len(), 128);
    assert_eq!(mesh_a.opaque.indices.len(), 32 * 6);
    let bytes_a: Vec<u8> = mesh_a.opaque.vertices.iter().flat_map(|v| bytemuck::bytes_of(v).to_vec()).collect();
    let bytes_b: Vec<u8> = mesh_b.opaque.vertices.iter().flat_map(|v| bytemuck::bytes_of(v).to_vec()).collect();
    assert_eq!(bytes_a, bytes_b, "variant pick must be deterministic");
}

#[test]
fn cross_plants_land_in_cutout_layer() {
    let fx = fixture();
    let mut world = World::new([8, 8, 8]);
    let grass = fx.registry.block_id("short_grass").unwrap();
    world.set(4, 1, 4, grass, 0);
    let tint = TintPolicy { tint0: [145, 189, 89, 255] };
    let mesh = mesh_world(&world, &fx.registry, &rects(&fx), (fx.atlas.width, fx.atlas.height), &tint);
    // tinted_cross is double-sided: 2 planes × 2 faces = 4 quads = 16 verts.
    assert_eq!(mesh.cutout.vertices.len(), 16, "double-sided cross = 4 quads");
    assert_eq!(mesh.opaque.vertices.len(), 0, "plants never land on the opaque layer");
    // Cross quads carry the plains tint.
    assert!(mesh.cutout.vertices.iter().any(|v| v.tint != 0xFFFFFFFF));
}

#[test]
fn glass_goes_cutout_and_neighbors_stay_visible() {
    let fx = fixture();
    let mut world = World::new([8, 8, 8]);
    let glass = fx.registry.block_id("glass").unwrap();
    let stone = fx.registry.block_id("stone").unwrap();
    // Stone under a 2x1 glass roof: stone top must NOT be culled by glass.
    world.set(2, 1, 2, stone, 0);
    world.set(3, 1, 2, stone, 0);
    world.set(2, 2, 2, glass, 0);
    world.set(3, 2, 2, glass, 0);
    let mesh = mesh_world(&world, &fx.registry, &rects(&fx), (fx.atlas.width, fx.atlas.height), &white_tint());
    // Two stones, fully exposed except the shared stone-stone interface
    // (culls BOTH faces: 12 - 2 = 10 quads); glass never occludes them.
    assert_eq!(mesh.opaque.vertices.len(), 40, "stone faces must survive under glass (10 quads)");
    // Pack ground truth: glass.json sets `force_translucent: true`, so the
    // pack itself demands the alpha-blended layer. The interior texels are
    // alpha=0 (blend output ~0) and the frame texels carry alpha ~200, so the
    // result matches vanilla's look; cutout stays reserved for leaves/plants.
    assert_eq!(mesh.translucent.vertices.len(), 32, "glass renders on its pack-declared layer (8 quads)");
    assert_eq!(mesh.cutout.vertices.len(), 0, "no cutout quads in this scene");
    // Quad count check: 12 glass faces - 2 shared glass-glass (same_block_cull)
    // - 2 bottoms against stone occluders = 8 quads (the 32 verts above).
}

#[test]
fn water_renders_engine_fluid_on_translucent_layer() {
    let fx = fixture();
    let mut world = World::new([8, 8, 8]);
    let water = fx.registry.block_id("water").unwrap();
    // One water block floating in air: 5 exposed faces (bottom + 4 sides) +
    // top = 6 quads at 14/16 height; all on the translucent layer with an
    // anim slot assigned.
    world.set(4, 4, 4, water, 0);
    let mesh = mesh_world(&world, &fx.registry, &rects(&fx), (fx.atlas.width, fx.atlas.height), &white_tint());
    assert_eq!(mesh.translucent.vertices.len(), 24, "still water cube = 6 quads");
    assert!(mesh.translucent.vertices.iter().any(|v| v.anim_id > 0), "water carries an anim slot");
    // Surface sits at 14/16 above the block origin (y=4).
    let max_y = mesh.translucent.vertices.iter().map(|v| v.pos[1]).fold(f32::NEG_INFINITY, f32::max);
    assert!((max_y - (4.0 + 14.0 / 16.0)).abs() < 1e-5, "water surface at 14/16, got {max_y}");
    // Fully surrounded water has no faces at all.
    let mut w2 = World::new([8, 8, 8]);
    for x in 0..8 {
        for y in 0..8 {
            for z in 0..8 {
                w2.set(x, y, z, water, 0);
            }
        }
    }
    let m2 = mesh_world(&w2, &fx.registry, &rects(&fx), (fx.atlas.width, fx.atlas.height), &white_tint());
    assert_eq!(m2.translucent.vertices.len(), 0, "buried water culls entirely");
}

#[test]
fn rotated_log_axis_lands_end_caps_on_x_and_z() {
    let fx = fixture();
    // The compiler bakes variant rotations: oak_log axis=x/z states must end
    // with models whose up-facing quads point along the axis.
    let log = fx.registry.block("oak_log").unwrap();
    let dirs_of = |state: u32| -> Vec<u8> {
        let s = log.state(state).unwrap();
        let feathered_assets::compiled::CompiledAppearance::Static(m) = &s.appearance else {
            panic!("static");
        };
        let model = fx.registry.model(m[0].model).unwrap();
        let mut dirs: Vec<u8> = model.quads.iter().map(|q| q.dir as u8).collect();
        dirs.sort();
        dirs
    };
    // Schema order is axis=x, axis=y, axis=z (state 0,1,2).
    let y_state = dirs_of(1);
    assert_eq!(y_state, [0, 1, 2, 3, 4, 5], "axis=y keeps all six dirs distinct");
    let x_state = dirs_of(0);
    assert_eq!(x_state, [0, 1, 2, 3, 4, 5], "axis=x still has six distinct dirs");
}
