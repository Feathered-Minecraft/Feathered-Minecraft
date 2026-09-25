//! Golden tests against the real 26.3 pack. These encode the observed
//! semantics from the asset report: parent chains, rotations, fractional
//! geometry, tint indices, occlusion classes and animation timing.

use feathered_assets::cache;
use feathered_assets::compiler::{compile_pack, REQUIRED_BLOCKS};
use feathered_assets::pack::discover;

fn test_pack_root() -> &'static str {
    for p in ["texture/assets", "../texture/assets", "../../texture/assets"] {
        if std::path::Path::new(p).join("pack.mcmeta").exists() {
            return p;
        }
    }
    panic!("pack not found; run tests from the Feathered workspace");
}

#[test]
fn pack_discovery_finds_26_3() {
    let index = discover(std::path::Path::new(test_pack_root())).unwrap();
    assert_eq!(index.packs.len(), 1);
    let desc = index.packs[0].description.clone().unwrap();
    assert!(desc.contains("26.3"), "unexpected description: {desc}");
    // Sanity: the whole pack indexes (measured 11,577 at analysis time).
    assert!(index.files.len() > 11_000, "index: {}", index.files.len());
}

#[test]
fn water_animation_strip_resolves() {
    let index = discover(std::path::Path::new(test_pack_root())).unwrap();
    let store = feathered_assets::sprites::SpriteStore::load(&index).unwrap();
    let water = store.get("minecraft", "block/water_still").unwrap();
    assert_eq!(water.frame_count(), 32);
    assert_eq!(water.animation.as_ref().unwrap().frametime, 2);
    let lava = store.get("minecraft", "block/lava_still").unwrap();
    // lava_still has an explicit 38-entry palindrome frame list.
    assert_eq!(lava.frame_count(), 38);
}

#[test]
fn oak_log_compiles_to_six_quads_with_two_sprites() {
    let index = discover(std::path::Path::new(test_pack_root())).unwrap();
    let (pack, _atlas, _stats) = compile_pack(&index, true).unwrap();
    let log = pack.blocks.iter().find(|b| b.name == "oak_log").unwrap();
    assert_eq!(log.states.len(), 3); // axis x|y|z
    for state in &log.states {
        let feathered_assets::compiled::CompiledAppearance::Static(instances) = &state.appearance
        else {
            panic!("oak_log state should be static");
        };
        assert_eq!(instances.len(), 1);
        let model = &pack.models[instances[0].model.0 as usize];
        assert_eq!(model.quads.len(), 6, "oak_log must be a 6-quad cube");
        let sprites: std::collections::HashSet<_> = model.quads.iter().map(|q| q.sprite).collect();
        assert_eq!(sprites.len(), 2, "end + side sprites");
    }
}

#[test]
fn grass_block_has_tinted_faces() {
    let index = discover(std::path::Path::new(test_pack_root())).unwrap();
    let (pack, _atlas, _stats) = compile_pack(&index, true).unwrap();
    let grass = pack.blocks.iter().find(|b| b.name == "grass_block").unwrap();
    let snow_states: Vec<bool> = grass
        .states
        .iter()
        .map(|s| match &s.appearance {
            feathered_assets::compiled::CompiledAppearance::Static(m) => {
                let model = &pack.models[m[0].model.0 as usize];
                model.quads.iter().any(|q| q.tint.is_some())
            }
            _ => false,
        })
        .collect();
    // snowy=false has tinted top+overlay; snowy=true (grass_block_snow) does not.
    assert!(snow_states.iter().any(|t| *t), "tinted grass state missing");
    assert!(snow_states.len() == 2, "expected snowy and plain states");
}

#[test]
fn stone_occludes_glass_does_not() {
    let index = discover(std::path::Path::new(test_pack_root())).unwrap();
    let (pack, _atlas, _stats) = compile_pack(&index, true).unwrap();
    let stone = pack.blocks.iter().find(|b| b.name == "stone").unwrap();
    assert!(stone.states.iter().all(|s| s.occlusion.hides_neighbor()));
    let glass = pack.blocks.iter().find(|b| b.name == "glass").unwrap();
    assert!(glass.states.iter().all(|s| !s.occlusion.hides_neighbor()));
}

#[test]
fn rail_is_flat_geometry() {
    let index = discover(std::path::Path::new(test_pack_root())).unwrap();
    let (pack, _atlas, _stats) = compile_pack(&index, true).unwrap();
    let rail = pack.blocks.iter().find(|b| b.name == "rail").unwrap();
    // Every `shape` state resolves; flat shapes (north_south/east_west) are
    // 1/16-thick plates, raised shapes are tilted slabs.
    let flat = rail
        .states
        .iter()
        .filter_map(|s| match &s.appearance {
            feathered_assets::compiled::CompiledAppearance::Static(m) => Some(m[0].model),
            _ => None,
        })
        .any(|model_id| {
            pack.models[model_id.0 as usize]
                .quads
                .iter()
                .any(|q| q.bounds[4] - q.bounds[1] < 2.0)
        });
    assert!(flat, "rail flat states must contain a 1/16-tall quad");
    assert_eq!(rail.states.len(), 10); // 10 shape variants
}

#[test]
fn redstone_wire_multipart_states() {
    let index = discover(std::path::Path::new(test_pack_root())).unwrap();
    let (pack, _atlas, _stats) = compile_pack(&index, true).unwrap();
    let wire = pack.blocks.iter().find(|b| b.name == "redstone_wire").unwrap();
    assert_eq!(wire.states.len(), 81); // north/east/south/west × side|up|none-ish combos
    assert!(wire.states.iter().any(|s| matches!(
        s.appearance,
        feathered_assets::compiled::CompiledAppearance::Multipart(_)
    )));
}

#[test]
fn all_required_blocks_compile() {
    let index = discover(std::path::Path::new(test_pack_root())).unwrap();
    let (pack, _atlas, stats) = compile_pack(&index, true).unwrap();
    for req in REQUIRED_BLOCKS {
        assert!(
            pack.blocks.iter().any(|b| b.name == *req),
            "required block {req} missing"
        );
    }
    assert_eq!(stats.skipped_blocks, 0);
}

#[test]
fn cache_round_trip_preserves_pack() {
    let index = discover(std::path::Path::new(test_pack_root())).unwrap();
    let (pack, atlas, _stats) = compile_pack(&index, true).unwrap();
    let payload = cache::CachePayload {
        atlas: cache::atlas_to_cached(&atlas, &pack.sprite_names),
        pack: pack.clone(),
    };
    let digest = cache::content_digest(&[("test".into(), b"hello".to_vec())]);
    let blob = cache::encode(digest, &payload).unwrap();
    let (digest2, decoded) = cache::decode(&blob).unwrap();
    assert_eq!(digest, digest2);
    assert_eq!(decoded.pack.blocks.len(), pack.blocks.len());
    assert_eq!(decoded.atlas.width, atlas.width);

    // Header validation: corrupt the magic -> rejection.
    let mut bad = blob.clone();
    bad[0] = b'X';
    assert!(cache::decode(&bad).is_err());
    // Truncate -> rejection.
    assert!(cache::decode(&blob[..30]).is_err());
}

#[test]
fn atlas_packs_oak_log_top_correctly() {
    let index = discover(std::path::Path::new(test_pack_root())).unwrap();
    let store = feathered_assets::sprites::SpriteStore::load(&index).unwrap();
    let (pack, atlas, _stats) = compile_pack(&index, true).unwrap();
    let _ = pack;
    let entry = atlas.get("minecraft", "block/oak_log_top").unwrap();
    let sprite = store.get("minecraft", "block/oak_log_top").unwrap();
    assert_eq!(entry.frame_w, sprite.tex.width);
    assert_eq!(entry.frame_h, sprite.tex.height);
    // Atlas is a power of two with a mip chain.
    assert!(atlas.width.is_power_of_two());
    assert!(atlas.mip_levels() > 1);
}
