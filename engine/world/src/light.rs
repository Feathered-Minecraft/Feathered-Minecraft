//! Voxel light propagation (sky + block channels).
//!
//! This is the data source for the renderer's deferred lighting stage — the
//! Feathered analog of the lightmap.xy input that OptiFine/Iris packs (Noble
//! included) sample in their gbuffers programs. Vanilla packs receive the
//! lightmap from the game engine; Feathered owns the whole stack, so the
//! light grid is computed here from the same block data the mesher consumes.
//!
//! Semantics follow vanilla Minecraft closely enough to light the validation
//! scene identically:
//! * **Sky light** starts at 15 above the heightmap and attenuates by 1 per
//!   block of opaque material it passes through; it propagates in all 6
//!   directions from lit cells (so caves under overhangs get decreasing
//!   levels rather than hard darkness).
//! * **Block light** starts at each light emitter's level and attenuates by 1
//!   per block (manhattan flood fill).
//! * Non-occluding blocks (glass, plants, rails, fluids) pass light without
//!   attenuation — same rule the mesher's face-culling uses.
//!
//! Pure CPU, deterministic BFS with explicit queues (no external crates), so
//! the values are bit-identical across platforms and testable in isolation.

use crate::grid::World;
use crate::Registry;
use std::collections::VecDeque;

/// Light levels for one voxel: (sky, block), 0..=15 each.
pub type LightLevels = [u8; 2];

/// Named light emitters (vanilla levels). Noble reads `heldBlockLightValue`
/// and block light from the engine; Feathered derives emitters from the
/// compiled block registry by name, which covers the validation scene.
fn emitter_level(name: &str) -> u8 {
    match name {
        "torch" | "soul_torch" | "redstone_torch" | "lantern" | "sea_lantern" => 14,
        "glowstone" | "shroomlight" | "ochre_froglight" | "verdant_froglight"
        | "pearlescent_froglight" | "jack_o_lantern" | "campfire" | "soul_lantern" => 15,
        "end_rod" | "candle" | "beacon" | "conduit" | "respawn_anchor" | "magma_block" => 15,
        "fire" | "soul_fire" | "furnace" | "redstone_lamp" | "crying_obsidian" => 15,
        "lava" => 15,
        "light" => 15,
        _ => 0,
    }
}

/// Abstract world access for the light BFS. Implemented for the Phase-1 flat
/// `World` and, via the client, for a view over the chunked streaming world
/// (region-of-interest window around one chunk). Keeping this a trait lets
/// the same BFS drive both paths without duplicating the algorithm.
pub trait LightWorld {
    /// Block/state at (x, y, z); `None` = outside the world (treated as
    /// light-blocking so light does not leak across region borders).
    fn get(&self, x: i64, y: i64, z: i64) -> Option<(u32, u32)>;
}

impl LightWorld for World {
    fn get(&self, x: i64, y: i64, z: i64) -> Option<(u32, u32)> {
        World::get(self, x, y, z)
    }
}

impl LightWorld for crate::chunks::RegionView<'_> {
    fn get(&self, x: i64, y: i64, z: i64) -> Option<(u32, u32)> {
        crate::chunks::RegionView::get(self, x, y, z)
    }
}

/// The block/light query rules shared by both frontends.
pub struct LightRules<'a> {
    pub registry: &'a Registry,
}

impl<'a> LightRules<'a> {
    /// "Does this voxel block light?" — occlusion data is authoritative.
    pub fn blocks_light(&self, x: i64, y: i64, z: i64, world: &dyn LightWorld) -> Option<bool> {
        let (id, sid) = world.get(x, y, z)?;
        if id == 0 {
            return Some(false);
        }
        let block = self.registry.block_by_id(id)?;
        let state = block.state(sid)?;
        Some(state.occlusion.hides_neighbor())
    }

    pub fn emitter(&self, id: u32) -> u8 {
        self.registry
            .block_by_id(id)
            .map(|b| emitter_level(&b.name))
            .unwrap_or(0)
    }
}

/// Compute light for a rectangular window of voxels. The window is the area
/// that gets written (`out`) plus a `pad`-block margin that is *read* for
/// propagation context (so light from just outside the window flows in
/// one-sidedly). Returns the grid covering exactly `min..max`.
pub fn compute_window(
    world: &dyn LightWorld,
    rules: &LightRules,
    min: [i64; 3],
    max: [i64; 3],
) -> LightGrid {
    let [x0, y0, z0] = min;
    let [x1, y1, z1] = max;
    // An inverted window would wrap `usize` below and corrupt the whole
    // grid silently; fail loudly instead (the streaming world spans negative
    // coordinates, so a bad clamp is easy to introduce).
    assert!(
        x1 > x0 && y1 > y0 && z1 > z0,
        "inverted light window min={min:?} max={max:?}"
    );
    let sx = (x1 - x0) as usize;
    let sy = (y1 - y0) as usize;
    let sz = (z1 - z0) as usize;
    let len = sx * sy * sz;
    let mut levels = vec![[0u8, 0u8]; len];

    let idx = |x: i64, y: i64, z: i64| -> usize {
        let lx = (x - x0) as usize;
        let ly = (y - y0) as usize;
        let lz = (z - z0) as usize;
        (ly * sz + lz) * sx + lx
    };
    let in_bounds = |x: i64, y: i64, z: i64| -> bool {
        x >= x0 && x < x1 && y >= y0 && y < y1 && z >= z0 && z < z1
    };
    let blocks_light =
        |x: i64, y: i64, z: i64| -> bool { rules.blocks_light(x, y, z, world).unwrap_or(true) };

    // --- Sky light: column fill from the top of the window, then BFS.
    let mut queue = VecDeque::new();
    for z in z0..z1 {
        for x in x0..x1 {
            let mut level = 15u8;
            for y in (y0..y1).rev() {
                let blocked = blocks_light(x, y, z);
                if blocked {
                    level = 0;
                }
                levels[idx(x, y, z)][0] = level;
                if level > 1 {
                    queue.push_back((x, y, z, level));
                }
            }
        }
    }
    bfs(
        &mut levels,
        &mut queue,
        0,
        &blocks_light,
        &in_bounds,
        &idx,
        sx,
        sy,
        sz,
    );

    // --- Block light: seed emitters, BFS.
    let mut queue = VecDeque::new();
    for y in y0..y1 {
        for z in z0..z1 {
            for x in x0..x1 {
                let Some((id, _sid)) = world.get(x, y, z) else { continue };
                if id == 0 {
                    continue;
                }
                let level = rules.emitter(id);
                if level == 0 {
                    continue;
                }
                // Emitters shine even when embedded (torches); seed them
                // and all adjacent air so enclosed lanterns still glow.
                levels[idx(x, y, z)][1] = level;
                queue.push_back((x, y, z, level));
                for (dx, dy, dz) in NEIGHBORS_6 {
                    let (nx, ny, nz) = (x + dx, y + dy, z + dz);
                    if !in_bounds(nx, ny, nz) {
                        continue;
                    }
                    if levels[idx(nx, ny, nz)][1] < level - 1 {
                        levels[idx(nx, ny, nz)][1] = level - 1;
                        queue.push_back((nx, ny, nz, level - 1));
                    }
                }
            }
        }
    }
    bfs(
        &mut levels,
        &mut queue,
        1,
        &blocks_light,
        &in_bounds,
        &idx,
        sx,
        sy,
        sz,
    );

    LightGrid {
        levels,
        size: [sx, sy, sz],
        origin: [x0, y0, z0],
    }
}

const NEIGHBORS_6: [(i64, i64, i64); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

/// Shared BFS for both channels. Light spreads through non-light-blocking
/// voxels, dropping 1 per step; solid voxels keep their seeded value only.
fn bfs(
    levels: &mut [LightLevels],
    queue: &mut VecDeque<(i64, i64, i64, u8)>,
    channel: usize,
    blocks_light: &dyn Fn(i64, i64, i64) -> bool,
    in_bounds: &dyn Fn(i64, i64, i64) -> bool,
    idx: &dyn Fn(i64, i64, i64) -> usize,
    sx: usize,
    sy: usize,
    sz: usize,
) {
    let _ = (sx, sy, sz);
    while let Some((x, y, z, level)) = queue.pop_front() {
        let l = levels[idx(x, y, z)][channel];
        if l > level {
            continue; // superseded by a brighter seed
        }
        for (dx, dy, dz) in NEIGHBORS_6 {
            let (nx, ny, nz) = (x + dx, y + dy, z + dz);
            if !in_bounds(nx, ny, nz) {
                continue;
            }
            // Light never spreads INTO light-blocking voxels.
            if blocks_light(nx, ny, nz) {
                continue;
            }
            let next = level - 1;
            let i = idx(nx, ny, nz);
            if levels[i][channel] < next {
                levels[i][channel] = next;
                queue.push_back((nx, ny, nz, next));
            }
        }
    }
}

/// Flood-filled light grid. Phase 3 adds an `origin` so a grid can cover a
/// streaming window rather than the whole world; `get` stays absolute.
#[derive(Debug, Clone)]
pub struct LightGrid {
    /// (sky, block) per voxel, row-major over `size`, relative to `origin`.
    levels: Vec<LightLevels>,
    size: [usize; 3],
    /// World coordinate of voxel (0, 0, 0) of this grid.
    origin: [i64; 3],
}

impl LightGrid {
    /// Compute sky + block light for the whole flat world (Phase-1 scenes).
    pub fn compute(world: &World, registry: &Registry) -> LightGrid {
        let [sx, sy, sz] = world.size.map(|s| s as i64);
        compute_window(
            world,
            &LightRules { registry },
            [0, 0, 0],
            [sx, sy, sz],
        )
    }

    /// Light levels at one voxel; `None` outside the grid.
    pub fn get(&self, x: i64, y: i64, z: i64) -> Option<LightLevels> {
        let [sx, sy, sz] = self.size;
        let lx = (x - self.origin[0]) as usize;
        let ly = (y - self.origin[1]) as usize;
        let lz = (z - self.origin[2]) as usize;
        if lx >= sx || ly >= sy || lz >= sz {
            return None;
        }
        Some(self.levels[(ly * sz + lz) * sx + lx])
    }

    /// Sky light of the block ABOVE `y` (the face-light for a top face).
    pub fn sky_above(&self, x: i64, y: i64, z: i64) -> u8 {
        self.get(x, y + 1, z).map(|l| l[0]).unwrap_or(15)
    }
}

/// Per-chunk light for the streaming world: computes over the center chunk
/// padded by `PAD` blocks into the loaded neighborhood, so torch light from
/// neighbor chunks and sky under border overhangs are correct inside the
/// chunk (14 = max emitter radius; sky columns seed from the window top).
///
/// Only y is clamped (chunks span y 0..WORLD_H): x/z are unbounded — the
/// streaming world extends into negative chunk coordinates, and clamping a
/// negative min without its max would invert the window.
pub fn compute_chunk_light(
    world: &crate::ChunkedWorld,
    registry: &Registry,
    pos: crate::ChunkPos,
) -> LightGrid {
    const PAD: i64 = 14;
    let view = world.region_view(pos);
    let (min_c, max_c) = view.center_bounds();
    let min = [min_c[0] - PAD, 0, min_c[2] - PAD];
    let max = [max_c[0] + PAD, max_c[1], max_c[2] + PAD];
    compute_window(&view, &LightRules { registry }, min, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::World;

    fn registry_with(block_names: &[&str]) -> Registry {
        // Minimal registry: every requested name becomes a full-occlusion
        // block unless the name is a known transparent one.
        Registry::from_names_for_tests(block_names)
    }

    #[test]
    fn sky_light_falls_off_in_caves() {
        let mut world = World::new([3, 3, 3]);
        let reg = registry_with(&["stone"]);
        // Solid floor at y=0, air above.
        for z in 0..3i64 {
            for x in 0..3i64 {
                world.set(x as u32, 0, z as u32, 1, 0);
            }
        }
        let light = LightGrid::compute(&world, &reg);
        assert_eq!(light.get(1, 2, 1).unwrap()[0], 15);
        assert_eq!(light.get(1, 1, 1).unwrap()[0], 15);
        assert_eq!(light.get(1, 0, 1).unwrap()[0], 0, "inside solid stone");
    }

    #[test]
    fn block_light_spreads_from_torch() {
        let mut world = World::new([7, 3, 7]);
        let reg = registry_with(&["stone", "torch"]);
        for z in 0..7i64 {
            for x in 0..7i64 {
                world.set(x as u32, 0, z as u32, 1, 0); // stone floor
            }
        }
        world.set(3, 1, 3, 2, 0); // torch on the floor
        let light = LightGrid::compute(&world, &reg);
        assert_eq!(light.get(3, 1, 3).unwrap()[1], 14);
        assert_eq!(light.get(3, 1, 4).unwrap()[1], 13);
        assert_eq!(light.get(4, 1, 4).unwrap()[1], 12);
        assert_eq!(light.get(6, 1, 6).unwrap()[1], 8, "manhattan distance 6");
    }

    #[test]
    fn glass_does_not_block_light() {
        let mut world = World::new([3, 4, 1]);
        let reg = registry_with(&["glass"]);
        world.set(0, 0, 0, 1, 0);
        world.set(0, 1, 0, 1, 0); // glass "roof"
        world.set(0, 2, 0, 1, 0);
        let light = LightGrid::compute(&world, &reg);
        assert_eq!(light.get(0, 1, 0).unwrap()[0], 15, "glass passes sky light");
    }

    #[test]
    fn chunk_light_handles_negative_chunk_coordinates() {
        use crate::chunks::{Chunk, ChunkPos, ChunkedWorld};
        let reg = registry_with(&["stone"]);
        let mut world = ChunkedWorld::new();
        for cx in -3..=1 {
            for cz in -2..=2 {
                world.insert(Chunk::new(ChunkPos::new(cx, cz)));
            }
        }
        let stone = reg.block_id("stone").unwrap();
        // Solid ground (y 0..3) across the loaded region, open sky above.
        for x in -48..32 {
            for z in -32..32 {
                for y in 0..3 {
                    world.set(x, y, z, stone, 0);
                }
            }
        }
        // Regression: chunk (-2, 0) has light window x in [-46, -2). Clamping
        // the negative min to 0 without the matching max inverted the window
        // and overflowed the grid allocation.
        let light = compute_chunk_light(&world, &reg, ChunkPos::new(-2, 0));
        assert_eq!(
            light.get(-10, 5, 5),
            Some([15, 0]),
            "open sky above the ground at a negative chunk"
        );
        assert_eq!(
            light.get(-10, 1, 5),
            Some([0, 0]),
            "inside solid stone at a negative chunk"
        );
    }

    #[test]
    fn windowed_computation_matches_full_computation() {
        // A 10×6×10 world computed whole vs computed as two overlapping
        // windows: cells in the overlap must agree (deterministic BFS).
        let mut world = World::new([10, 6, 10]);
        let reg = registry_with(&["stone", "torch"]);
        for z in 0..10i64 {
            for x in 0..10i64 {
                world.set(x as u32, 0, z as u32, 1, 0);
            }
        }
        world.set(4, 1, 4, 2, 0); // torch
        let full = LightGrid::compute(&world, &reg);

        let rules = LightRules { registry: &reg };
        let half = compute_window(&world, &rules, [0, 0, 0], [10, 6, 5]);
        for y in 0..6i64 {
            for z in 0..5i64 {
                for x in 0..10i64 {
                    assert_eq!(
                        full.get(x, y, z),
                        half.get(x, y, z),
                        "window mismatch at {x},{y},{z}"
                    );
                }
            }
        }
    }
}
