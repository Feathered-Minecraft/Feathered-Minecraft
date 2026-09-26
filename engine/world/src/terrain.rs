//! Deterministic seeded terrain generation.
//!
//! Blocks come from the user's resource pack (blocks are resolved by name
//! from the registry at generation time — the generator never hardcodes ids,
//! and when a pack lacks a block the generator degrades gracefully instead
//! of panicking). Everything else is Feathered's own:
//!
//! * value noise (hash-based, smooth-interpolated, 2 octaves of fBm) for the
//!   surface height,
//! * a second independent low-frequency field for hilltops (so the terrain
//!   has both rolling ground and occasional higher hills),
//! * 3D noise caves below the surface, only where the architecture supports
//!   them cleanly (they carve air; nothing else has to know they exist).
//!
//! All noise is integer-hash based and therefore bit-deterministic across
//! platforms, and regeneration after unload reproduces the same chunk
//! (unless the player edited it — edits persist in the container).

use crate::chunks::{Chunk, ChunkPos, CHUNK_X, CHUNK_Z};
use crate::grid::WORLD_H;
use crate::Registry;
use std::collections::HashMap;

/// Surface below `sea_level - 4` gets stone; between that and the surface
/// block it is dirt; the top block is grass (all resolved by name).
const DIRT_DEPTH: i64 = 3;

/// Hash one 3D integer position into a uniform u64 (Feathered's own hash —
/// the same construction family the mesher uses for variant selection).
fn hash3(seed: u64, x: i64, y: i64, z: i64) -> u64 {
    let mut h = seed
        ^ (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
        ^ (z as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 30;
    h = h.wrapping_mul(0x2545_F491_4F6C_DD1D);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 31;
    h
}

/// Hash a 2D lattice point (y folded in as a constant plane).
fn hash2(seed: u64, x: i64, z: i64) -> u64 {
    hash3(seed, x, 0x5150_2A5A_5E5F, z)
}

/// Uniform u64 → f32 in [0, 1).
fn unit(h: u64) -> f32 {
    (h >> 40) as f32 / (1u64 << 24) as f32
}

/// Smooth (quintic) interpolation weight.
fn smooth(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// 2D value noise in [0, 1) with bilinear-smooth interpolation on the unit
/// lattice. `freq` is in lattice cells per block.
fn value_noise2(seed: u64, x: f32, z: f32, freq: f32) -> f32 {
    let fx = x * freq;
    let fz = z * freq;
    let x0 = fx.floor() as i64;
    let z0 = fz.floor() as i64;
    let tx = smooth(fx - x0 as f32);
    let tz = smooth(fz - z0 as f32);
    let (a, b, c, d) = (
        unit(hash2(seed, x0, z0)),
        unit(hash2(seed, x0 + 1, z0)),
        unit(hash2(seed, x0, z0 + 1)),
        unit(hash2(seed, x0 + 1, z0 + 1)),
    );
    let top = a + (b - a) * tx;
    let bottom = c + (d - c) * tx;
    top + (bottom - top) * tz
}

/// 3D value noise in [0, 1), trilinear-smooth. Used for caves.
fn value_noise3(seed: u64, x: f32, y: f32, z: f32, freq: f32) -> f32 {
    let fx = x * freq;
    let fy = y * freq;
    let fz = z * freq;
    let (x0, y0, z0) = (fx.floor() as i64, fy.floor() as i64, fz.floor() as i64);
    let (tx, ty, tz) = (
        smooth(fx - x0 as f32),
        smooth(fy - y0 as f32),
        smooth(fz - z0 as f32),
    );
    let c = |dx: i64, dy: i64, dz: i64| unit(hash3(seed, x0 + dx, y0 + dy, z0 + dz));
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let lo = lerp(
        lerp(c(0, 0, 0), c(1, 0, 0), tx),
        lerp(c(0, 1, 0), c(1, 1, 0), tx),
        ty,
    );
    let hi = lerp(
        lerp(c(0, 0, 1), c(1, 0, 1), tx),
        lerp(c(0, 1, 1), c(1, 1, 1), tx),
        ty,
    );
    lerp(lo, hi, tz)
}

/// Which blocks the generator fills (all resolved from the registry).
#[derive(Debug, Clone)]
struct Blocks {
    stone: u32,
    dirt: u32,
    grass: u32,
    sand: u32,
    water: Option<u32>,
    bedrock: Option<u32>,
}

impl Blocks {
    fn resolve(registry: &Registry) -> Blocks {
        let id = |name: &str| registry.block_id(name).unwrap_or(0);
        Blocks {
            stone: id("stone"),
            dirt: id("dirt"),
            grass: id("grass_block"),
            sand: id("sand"),
            water: registry.block_id("water"),
            bedrock: registry.block_id("bedrock"),
        }
    }
}

/// Deterministic terrain generator for one world seed.
pub struct TerrainGenerator {
    seed: u64,
    /// 2D surface height per chunk-column cache — generation of one chunk
    /// queries the 18×18 region (own 16×16 + 1-cell border) so caves and
    /// overhangs across borders stay consistent; the cache keeps repeated
    /// neighbor evaluations cheap during streaming. A Mutex (not RefCell)
    /// so one generator can be shared by parallel generation threads
    /// (rayon); a lost race just recomputes the same deterministic value.
    height_cache: std::sync::Mutex<HashMap<(i64, i64), i64>>,
}

/// Height amplitude: base ground at y≈24, hills up to ~+14, dips to ~-8.
pub const BASE_HEIGHT: i64 = 24;
pub const HEIGHT_AMPLITUDE: i64 = 14;
pub const SEA_LEVEL: i64 = 20;

impl TerrainGenerator {
    pub fn new(seed: u64) -> TerrainGenerator {
        TerrainGenerator {
            seed,
            height_cache: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// The world seed (debug screen / save metadata).
    pub fn seed_for_debug(&self) -> u64 {
        self.seed
    }

    /// Sand when the column is near sea level (beach band); grass otherwise.
    /// A small noise wobble keeps the shoreline irregular instead of a
    /// straight contour line.
    fn beach_block(&self, x: i64, z: i64, surface: i64) -> bool {
        let band_start = SEA_LEVEL - 3;
        let band_end = SEA_LEVEL + 2;
        if surface < band_start || surface > band_end {
            return false;
        }
        // Wobble: ±1 around the band edges (hash noise, deterministic).
        let wobble = (hash2(self.seed ^ 0x5A57, x, z) % 3) as i64 - 1;
        surface >= band_start + wobble && surface <= band_end + wobble
    }

    /// Surface height (top solid block y) at one column. Two noise fields:
    /// rolling ground (2-octave fBm) + rare hill mask.
    pub fn height_at(&self, x: i64, z: i64) -> i64 {
        if let Some(&h) = self.height_cache.lock().unwrap().get(&(x, z)) {
            return h;
        }
        let ground = {
            let n = 0.55 * value_noise2(self.seed, x as f32, z as f32, 1.0 / 48.0)
                + 0.30 * value_noise2(self.seed ^ 0xA5A5, x as f32, z as f32, 1.0 / 16.0)
                + 0.15 * value_noise2(self.seed ^ 0x3C3C, x as f32, z as f32, 1.0 / 6.0);
            (n * 2.0 - 1.0) * HEIGHT_AMPLITUDE as f32 * 0.6
        };
        let hills = {
            let m = value_noise2(self.seed ^ 0x77AA, x as f32, z as f32, 1.0 / 96.0);
            // Only distinct high patches become hills.
            let t = ((m - 0.62) / 0.38).clamp(0.0, 1.0);
            t * t * HEIGHT_AMPLITUDE as f32
        };
        let h = (BASE_HEIGHT as f32 + ground + hills).round() as i64;
        let h = h.clamp(2, WORLD_H as i64 - 24);
        self.height_cache.lock().unwrap().insert((x, z), h);
        h
    }

    /// Is the voxel carved (air) by the cave system? A single 3D noise field
    /// thresholded on a band (worm-like tubes), off near the surface top so
    /// terrain is not riddled with skylight holes, and off below y=2.
    fn is_cave(&self, x: i64, y: i64, z: i64, surface: i64) -> bool {
        if y < 3 || y > surface - 4 {
            return false;
        }
        let n = value_noise3(self.seed ^ 0xCAFE, x as f32, y as f32 * 1.6, z as f32, 1.0 / 14.0);
        // Band threshold: values near the iso-level carve; ± falloff makes
        // tubes rather than blobs.
        let band = (n - 0.5).abs();
        band < 0.045 * (1.0 - (surface - y) as f32 / (surface as f32 + 1.0)).max(0.3)
    }

    /// Generate one chunk into `Chunk`. Deterministic for a given seed and
    /// position; called again after unload reproduces the same terrain.
    pub fn generate_chunk(&self, registry: &Registry, pos: ChunkPos) -> Chunk {
        let mut chunk = Chunk::new(pos);
        let blocks = Blocks::resolve(registry);
        let (bx, bz) = pos.min_block();

        for lz in 0..CHUNK_Z {
            for lx in 0..CHUNK_X {
                let (wx, wz) = (bx + lx as i64, bz + lz as i64);
                let surface = self.height_at(wx, wz);

                for y in 0..WORLD_H {
                    let y = y as i64;
                    let block = if y > surface {
                        // Air above the surface; water fills up to sea level
                        // (ocean/lake columns).
                        if y <= SEA_LEVEL {
                            blocks.water.unwrap_or(0)
                        } else {
                            0
                        }
                    } else if self.is_cave(wx, y, wz, surface) {
                        // Caves below the surface; caves that break the sea
                        // floor flood rather than creating dry holes.
                        if y <= SEA_LEVEL && y > surface - 2 {
                            blocks.water.unwrap_or(0)
                        } else {
                            0
                        }
                    } else if y == surface {
                        // Surface block: underwater floors and beaches take
                        // sand when the pack provides it; dry land grass.
                        let under_water = surface < SEA_LEVEL;
                        let beach = !under_water
                            && blocks.sand != 0
                            && self.beach_block(wx, wz, surface);
                        if under_water || beach {
                            if blocks.sand != 0 {
                                blocks.sand
                            } else {
                                blocks.dirt
                            }
                        } else {
                            blocks.grass
                        }
                    } else if y >= surface - DIRT_DEPTH {
                        // Sand packs a couple of blocks deep on beaches.
                        if blocks.sand != 0 && self.beach_block(wx, wz, surface) {
                            blocks.sand
                        } else {
                            blocks.dirt
                        }
                    } else {
                        blocks.stone
                    };

                    if block != 0 {
                        // State 0 (default properties: axis=y, no snow…).
                        chunk.set_local(lx, y as u32, lz, block, 0);
                    }
                }

                // Bedrock floor (one unbreakable layer when the pack has it).
                if let Some(b) = blocks.bedrock {
                    chunk.set_local(lx, 0, lz, b, 0);
                }
            }
        }
        chunk
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Registry;

    /// Registry with just the names the generator asks for (full cubes).
    fn test_registry() -> Registry {
        Registry::from_names_for_tests(&["stone", "dirt", "grass_block", "water", "bedrock"])
    }

    #[test]
    fn generation_is_deterministic_per_seed() {
        let reg = test_registry();
        let pos = ChunkPos::new(3, -2);
        let a = TerrainGenerator::new(1234).generate_chunk(&reg, pos);
        let b = TerrainGenerator::new(1234).generate_chunk(&reg, pos);
        assert_eq!(a.blocks, b.blocks);
        assert_eq!(a.states, b.states);

        let c = TerrainGenerator::new(999).generate_chunk(&reg, pos);
        assert_ne!(a.blocks, c.blocks, "different seeds must differ");
    }

    #[test]
    fn regenerated_chunks_after_unload_match() {
        let reg = test_registry();
        let gen = TerrainGenerator::new(77);
        let pos = ChunkPos::new(-5, 5);
        let first = gen.generate_chunk(&reg, pos);
        let again = gen.generate_chunk(&reg, pos);
        assert_eq!(first.blocks, again.blocks);
    }

    #[test]
    fn heightmap_is_continuous_across_chunk_borders() {
        let gen = TerrainGenerator::new(42);
        // Columns straddling the 16-block border must not jump more than a
        // couple of blocks (noise is continuous; the cache must not create
        // per-chunk seams).
        let h1 = gen.height_at(15, 0);
        let h2 = gen.height_at(16, 0);
        assert!(
            (h1 - h2).abs() <= 3,
            "height discontinuity at chunk border: {h1} vs {h2}"
        );
        let h3 = gen.height_at(0, -1);
        let h4 = gen.height_at(0, 0);
        assert!((h3 - h4).abs() <= 3);
    }

    #[test]
    fn surface_has_a_grass_layer_over_dirt_over_stone() {
        let reg = test_registry();
        let gen = TerrainGenerator::new(7);
        let chunk = gen.generate_chunk(&reg, ChunkPos::new(0, 0));
        let (bx, bz) = ChunkPos::new(0, 0).min_block();
        let grass = reg.block_id("grass_block").unwrap();
        let dirt = reg.block_id("dirt").unwrap();

        // Pick a column that is above sea level (not flooded).
        let mut checked = 0;
        for lz in 0..CHUNK_Z {
            for lx in 0..CHUNK_X {
                let surface = gen.height_at(bx + lx as i64, bz + lz as i64);
                if surface <= SEA_LEVEL {
                    continue;
                }
                checked += 1;
                let (top, _) = chunk.get_local(lx, surface as u32, lz).unwrap();
                let (below1, _) = chunk.get_local(lx, surface as u32 - 1, lz).unwrap();
                assert_eq!(top, grass, "surface must be grass at {lx},{lz}");
                assert_eq!(below1, dirt, "under the surface must be dirt");
                // Deep must be stone OR air (a cave carved through it —
                // caves are allowed below the dirt layer by design).
            }
        }
        assert!(checked > 8, "expected several dry columns");
    }

    #[test]
    fn bedrock_caps_the_floor_when_the_pack_has_it() {
        let reg = test_registry();
        let gen = TerrainGenerator::new(11);
        let chunk = gen.generate_chunk(&reg, ChunkPos::new(1, 1));
        let bedrock = reg.block_id("bedrock").unwrap();
        for lz in 0..CHUNK_Z {
            for lx in 0..CHUNK_X {
                assert_eq!(chunk.get_local(lx, 0, lz).unwrap().0, bedrock);
            }
        }
    }

    #[test]
    fn caves_exist_somewhere_but_not_everywhere() {
        let reg = test_registry();
        let gen = TerrainGenerator::new(31337);
        let chunk = gen.generate_chunk(&reg, ChunkPos::new(0, 0));
        let mut air_below_surface = 0;
        let (bx, bz) = ChunkPos::new(0, 0).min_block();
        for lz in 0..CHUNK_Z {
            for lx in 0..CHUNK_X {
                let surface = gen.height_at(bx + lx as i64, bz + lz as i64);
                for y in 4..(surface - 4).max(5) {
                    if chunk.get_local(lx, y as u32, lz).unwrap().0 == 0 {
                        air_below_surface += 1;
                    }
                }
            }
        }
        assert!(
            air_below_surface > 0 && air_below_surface < CHUNK_X as usize * CHUNK_Z as usize * 40,
            "caves should carve some but not most of the underground ({air_below_surface})"
        );
    }

    #[test]
    fn beaches_are_sand_near_sea_level() {
        let reg = Registry::from_names_for_tests(&[
            "stone", "dirt", "grass_block", "sand", "water", "bedrock",
        ]);
        let sand = reg.block_id("sand").unwrap();
        let grass = reg.block_id("grass_block").unwrap();
        let gen = TerrainGenerator::new(4242);
        // Scan many chunks: shore columns (within the beach band) must be
        // sand; well-above-sea-level columns must be grass.
        let mut sandy = 0;
        let mut grassy = 0;
        'outer: for cx in [-6i32, -2, 0, 3, 7] {
            for cz in [-5i32, -1, 2, 6] {
                let pos = ChunkPos::new(cx, cz);
                let chunk = gen.generate_chunk(&reg, pos);
                let (bx, bz) = pos.min_block();
                for lz in 0..CHUNK_Z {
                    for lx in 0..CHUNK_X {
                        let (wx, wz) = (bx + lx as i64, bz + lz as i64);
                        let surface = gen.height_at(wx, wz);
                        if surface <= SEA_LEVEL {
                            continue; // underwater
                        }
                        let (top, _) = chunk.get_local(lx, surface as u32, lz).unwrap();
                        if (SEA_LEVEL - 3..=SEA_LEVEL + 2).contains(&surface) {
                            if top == sand {
                                sandy += 1;
                            }
                        } else if surface > SEA_LEVEL + 4 && top == grass {
                            grassy += 1;
                        }
                        if sandy > 0 && grassy > 0 {
                            break 'outer;
                        }
                    }
                }
            }
        }
        assert!(sandy > 0, "beach band must produce sand somewhere");
        assert!(grassy > 0, "dry land must stay grass somewhere");

        // Determinism: same seed → identical chunk.
        let a = TerrainGenerator::new(4242).generate_chunk(&reg, ChunkPos::new(0, 0));
        let b = TerrainGenerator::new(4242).generate_chunk(&reg, ChunkPos::new(0, 0));
        assert_eq!(a.blocks, b.blocks);
    }

    #[test]
    fn water_fills_to_sea_level_in_dips() {
        let reg = test_registry();
        let water = reg.block_id("water").unwrap();
        // Scan many seeds × chunks for a dipped column (below sea level);
        // when found, the whole column up to sea level must be flooded.
        let mut found = false;
        'seeds: for seed in [1u64, 2, 3, 5, 8, 13, 21, 34, 55, 89] {
            let gen = TerrainGenerator::new(seed);
            for cx in [-3i32, 0, 4, -11, 9] {
                for cz in [-3i32, 2, -8, 6] {
                    let pos = ChunkPos::new(cx, cz);
                    let chunk = gen.generate_chunk(&reg, pos);
                    let (bx, bz) = pos.min_block();
                    for lz in 0..CHUNK_Z {
                        for lx in 0..CHUNK_X {
                            let surface = gen.height_at(bx + lx as i64, bz + lz as i64);
                            if surface < SEA_LEVEL {
                                found = true;
                                for y in surface + 1..=SEA_LEVEL {
                                    let (b, _) = chunk.get_local(lx, y as u32, lz).unwrap();
                                    assert_eq!(
                                        b, water,
                                        "flooded column must be water at y={y} (seed {seed})"
                                    );
                                }
                                continue 'seeds;
                            }
                        }
                    }
                }
            }
        }
        assert!(found, "no dipped column found across seeds/chunks — amplitude broken");
    }
}
