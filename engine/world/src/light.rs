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

/// Flood-filled light grid, indexed like the world grid.
pub struct LightGrid {
    /// (sky, block) per voxel, row-major over world.size.
    levels: Vec<LightLevels>,
    size: [usize; 3],
}

impl LightGrid {
    /// Compute sky + block light for `world`. Emitters are looked up from the
    /// registry by block name (vanilla light levels); transparent behavior
    /// reuses each state's occlusion data — exactly the data the mesher uses.
    pub fn compute(world: &World, registry: &Registry) -> LightGrid {
        let [sx, sy, sz] = world.size.map(|s| s as usize);
        let len = sx * sy * sz;
        let mut levels = vec![[0u8, 0u8]; len];

        let idx = |x: usize, y: usize, z: usize| -> usize { (y * sz + z) * sx + x };

        // "Does this voxel block light?" — occlusion data is authoritative.
        let blocks_light = |x: i64, y: i64, z: i64| -> Option<bool> {
            let (id, sid) = world.get(x, y, z)?;
            if id == 0 {
                return Some(false);
            }
            let block = registry.block_by_id(id)?;
            let state = block.state(sid)?;
            Some(state.occlusion.hides_neighbor())
        };

        // --- Sky light: column fill from the top, then BFS lateral spread.
        let mut queue = VecDeque::new();
        for z in 0..sz {
            for x in 0..sx {
                let mut level = 15u8;
                for y in (0..sy).rev() {
                    let blocked = blocks_light(x as i64, y as i64, z as i64).unwrap_or(true);
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
        bfs(&mut levels, &mut queue, 0, &blocks_light, sx, sy, sz);

        // --- Block light: seed emitters, BFS.
        let mut queue = VecDeque::new();
        for y in 0..sy as i64 {
            for z in 0..sz as i64 {
                for x in 0..sx as i64 {
                    let Some((id, _sid)) = world.get(x, y, z) else { continue };
                    if id == 0 {
                        continue;
                    }
                    let Some(block) = registry.block_by_id(id) else { continue };
                    let level = emitter_level(&block.name);
                    if level == 0 {
                        continue;
                    }
                    // Emitters shine even when embedded (torches); seed them
                    // and all adjacent air so enclosed lanterns still glow.
                    let (ux, uy, uz) = (x as usize, y as usize, z as usize);
                    levels[idx(ux, uy, uz)][1] = level;
                    queue.push_back((ux, uy, uz, level));
                    for (dx, dy, dz) in NEIGHBORS_6 {
                        let nx = x + dx;
                        let ny = y + dy;
                        let nz = z + dz;
                        if nx < 0 || ny < 0 || nz < 0 {
                            continue;
                        }
                        let (nx, ny, nz) = (nx as usize, ny as usize, nz as usize);
                        if nx >= sx || ny >= sy || nz >= sz {
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
        bfs(&mut levels, &mut queue, 1, &blocks_light, sx, sy, sz);

        LightGrid { levels, size: [sx, sy, sz] }
    }

    /// Light levels at one voxel; `None` outside the grid.
    pub fn get(&self, x: i64, y: i64, z: i64) -> Option<LightLevels> {
        let [sx, sy, sz] = self.size;
        if x < 0 || y < 0 || z < 0 || x >= sx as i64 || y >= sy as i64 || z >= sz as i64 {
            return None;
        }
        Some(self.levels[((y as usize) * sz + z as usize) * sx + x as usize])
    }

    /// Sky light of the block ABOVE `y` (the face-light for a top face).
    pub fn sky_above(&self, x: i64, y: i64, z: i64) -> u8 {
        self.get(x, y + 1, z).map(|l| l[0]).unwrap_or(15)
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
    queue: &mut VecDeque<(usize, usize, usize, u8)>,
    channel: usize,
    blocks_light: &dyn Fn(i64, i64, i64) -> Option<bool>,
    sx: usize,
    sy: usize,
    sz: usize,
) {
    while let Some((x, y, z, level)) = queue.pop_front() {
        let l = levels[(y * sz + z) * sx + x][channel];
        if l > level {
            continue; // superseded by a brighter seed
        }
        for (dx, dy, dz) in NEIGHBORS_6 {
            let nx = x as i64 + dx;
            let ny = y as i64 + dy;
            let nz = z as i64 + dz;
            if nx < 0 || ny < 0 || nz < 0 {
                continue;
            }
            let (nx, ny, nz) = (nx as usize, ny as usize, nz as usize);
            if nx >= sx || ny >= sy || nz >= sz {
                continue;
            }
            // Light never spreads INTO light-blocking voxels.
            if blocks_light(nx as i64, ny as i64, nz as i64).unwrap_or(true) {
                continue;
            }
            let next = level - 1;
            let i = (ny * sz + nz) * sx + nx;
            if levels[i][channel] < next {
                levels[i][channel] = next;
                queue.push_back((nx, ny, nz, next));
            }
        }
    }
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
}
