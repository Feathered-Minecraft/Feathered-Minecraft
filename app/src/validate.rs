//! Milestone validation: the ten-block checklist, checked structurally
//! against the compiled pack (geometry, sprites, animations, occlusion).
//! Lives in `feathered-app` because it observes both the compiled pack and
//! the runtime registry without either depending on the other.

use feathered_assets::atlas::Atlas;
use feathered_assets::compiled::CompiledAppearance;
use feathered_assets::error::AssetResult;
use feathered_assets::sprites::SpriteStore;
use feathered_world::Registry;

pub struct ValidationReport {
    pub lines: Vec<(String, bool, String)>,
}

impl ValidationReport {
    pub fn failures(&self) -> usize {
        self.lines.iter().filter(|l| !l.1).count()
    }

    pub fn print(&self) {
        for (name, ok, detail) in &self.lines {
            println!("  [{}] {:<14} {}", if *ok { "x" } else { " " }, name, detail);
        }
    }
}

pub fn validate(runtime: &Registry, atlas: &Atlas, store: &SpriteStore) -> AssetResult<ValidationReport> {
    let mut lines = Vec::new();

    // --- oak_log: 3 axis variants, cube of 6 quads, distinct end/side -----
    {
        let three_axis = runtime.block("oak_log").map(|b| {
            b.states.len() == 3
                && b.states.iter().all(|s| match &s.appearance {
                    CompiledAppearance::Static(m) => {
                        m.len() == 1
                            && runtime
                                .model(m[0].model)
                                .map(|m| m.quads.len() == 6)
                                .unwrap_or(false)
                    }
                    _ => false,
                })
        }).unwrap_or(false);
        let end_side = runtime
            .model_of("oak_log", 0)
            .map(|m| {
                let sprites: std::collections::HashSet<_> = m.quads.iter().map(|q| q.sprite).collect();
                sprites.len() == 2
            })
            .unwrap_or(false);
        lines.push((
            "oak_log".into(),
            three_axis && end_side,
            if three_axis && end_side {
                "3 axis variants, 6 quads each, end+side sprites distinct".into()
            } else {
                format!("three_axis={three_axis} end_side={end_side}")
            },
        ));
    }

    // --- grass_block: tinted faces -----------------------------------------
    // Snowy and plain states have different variant models; scan them all.
    {
        let has_tint = runtime
            .block("grass_block")
            .map(|b| {
                b.states.iter().any(|s| {
                    let models: Vec<_> = match &s.appearance {
                        CompiledAppearance::Static(m) => m.iter().map(|mi| mi.model).collect(),
                        CompiledAppearance::Variants(v) | CompiledAppearance::Multipart(v) => {
                            v.iter().flat_map(|(_, ms)| ms.iter().map(|mi| mi.model)).collect()
                        }
                    };
                    models.iter().any(|m| {
                        runtime
                            .model(*m)
                            .map(|m| m.quads.iter().any(|q| q.tint.is_some()))
                            .unwrap_or(false)
                    })
                })
            })
            .unwrap_or(false);
        lines.push((
            "grass_block".into(),
            has_tint,
            if has_tint { "tinted faces present (top + side overlay)".into() } else { "no tinted faces".into() },
        ));
    }

    // --- stone: occluder ------------------------------------------------------
    {
        let occ = runtime
            .block("stone")
            .map(|b| b.states.iter().all(|s| s.occlusion.hides_neighbor()))
            .unwrap_or(false);
        lines.push((
            "stone".into(),
            occ,
            if occ { "classified full opaque occluder".into() } else { "not classified as occluder".into() },
        ));
    }

    // --- glass: non-occluder ----------------------------------------------------
    {
        let no_occ = runtime
            .block("glass")
            .map(|b| b.states.iter().all(|s| !s.occlusion.hides_neighbor()))
            .unwrap_or(false);
        lines.push((
            "glass".into(),
            no_occ,
            if no_occ { "non-occluding (neighbor faces preserved)".into() } else { "wrongly occluding".into() },
        ));
    }

    // --- torch: cutout geometry ---------------------------------------------------
    {
        let torch_ok = runtime
            .model_of("torch", 0)
            .map(|m| m.quads.len() >= 2)
            .unwrap_or(false);
        lines.push((
            "torch".into(),
            torch_ok,
            if torch_ok { "non-cube geometry with side faces".into() } else { "geometry missing".into() },
        ));
    }

    // --- rail: flat quad ------------------------------------------------------------
    {
        let rail_ok = runtime
            .model_of("rail", 0)
            .map(|m| {
                m.quads.iter().any(|q| {
                    let b = q.bounds;
                    (b[4] - b[1]) < 2.0 // y-extent <= 2/16 (rail is 1/16 tall)
                })
            })
            .unwrap_or(false);
        lines.push((
            "rail".into(),
            rail_ok,
            if rail_ok { "flat quad present".into() } else { "no flat quad".into() },
        ));
    }

    // --- short_grass: cross -----------------------------------------------------------
    {
        let cross = runtime
            .model_of("short_grass", 0)
            .map(|m| m.quads.len() >= 2 && m.quads.iter().all(|q| q.tint.is_some()))
            .unwrap_or(false);
        lines.push((
            "short_grass".into(),
            cross,
            if cross { "cross quads, tinted".into() } else { "cross geometry/tint missing".into() },
        ));
    }

    // --- vine: multipart -----------------------------------------------------------------
    {
        let multipart = runtime
            .block("vine")
            .map(|b| {
                b.states
                    .iter()
                    .any(|s| matches!(s.appearance, CompiledAppearance::Multipart(_)))
            })
            .unwrap_or(false);
        lines.push((
            "vine".into(),
            multipart,
            if multipart { "multipart appearance".into() } else { "not multipart".into() },
        ));
    }

    // --- redstone_wire: multipart with property-dependent parts ----------------------------
    {
        let wire = runtime
            .block("redstone_wire")
            .map(|b| {
                b.states.len() > 1
                    && b.states
                        .iter()
                        .any(|s| matches!(s.appearance, CompiledAppearance::Multipart(_)))
            })
            .unwrap_or(false);
        lines.push((
            "redstone_wire".into(),
            wire,
            if wire { "multipart per connection state".into() } else { "not multipart".into() },
        ));
    }

    // --- water: animated -----------------------------------------------------------------------
    // The water blockstate points at a particle-only model (vanilla renders
    // fluids in engine code); the animated sprite is reachable via the
    // particle or a quad, so check both paths.
    {
        let frames = runtime
            .block("water")
            .and_then(|b| b.state(0))
            .and_then(|s| match &s.appearance {
                CompiledAppearance::Static(m) => m.first().map(|mi| mi.model),
                _ => None,
            })
            .and_then(|mid| runtime.model(mid))
            .and_then(|m| {
                let from_quad = m.quads.first().map(|q| q.sprite);
                from_quad.or(Some(m.particle))
            })
            .and_then(|sid| runtime.anim_frames(sid));
        let ok = frames.map(|f| f > 1).unwrap_or(false);
        lines.push((
            "water".into(),
            ok,
            frames
                .map(|f| format!("animated strip, {f} frames (via particle sprite)"))
                .unwrap_or_else(|| "no animation found".into()),
        ));
    }

    // --- atlas sanity: oak_log_top packed correctly ----------------------------------------------
    {
        let entry = store
            .get("minecraft", "block/oak_log_top")
            .and_then(|s| atlas.get("minecraft", "block/oak_log_top").map(|e| (s, e)));
        let ok = entry
            .map(|(s, e)| e.frame_w == s.tex.width && e.frame_h == s.tex.height)
            .unwrap_or(false);
        lines.push((
            "atlas".into(),
            ok,
            if ok { "oak_log_top packed with correct frame size".into() } else { "atlas entry mismatch".into() },
        ));
    }

    Ok(ValidationReport { lines })
}
