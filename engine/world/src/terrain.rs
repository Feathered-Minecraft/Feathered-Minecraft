//! Deterministic seeded terrain generation — compatibility façade.
//!
//! The generator internals moved to [`crate::worldgen`] (Seed → Climate →
//! Density/Shape → Carvers → Surface → Features; see
//! `docs/WORLDGEN_ENTITIES_REPORT.md`). This module keeps the historical
//! `TerrainGenerator` name and public surface (`height_at`, `generate_chunk`,
//! `seed_for_debug`, the `BASE_HEIGHT`/`SEA_LEVEL` constants) so the
//! streamer, client and tests keep working unchanged.

use crate::chunks::{Chunk, ChunkPos};
use crate::Registry;

pub use crate::worldgen::shape::{BASE_HEIGHT, SEA_LEVEL};

/// Deterministic terrain generator for one world seed (delegates to the
/// worldgen pipeline).
pub struct TerrainGenerator {
    inner: crate::worldgen::WorldGenerator,
}

impl TerrainGenerator {
    pub fn new(seed: u64) -> TerrainGenerator {
        TerrainGenerator {
            inner: crate::worldgen::WorldGenerator::new(seed),
        }
    }

    /// The world seed (debug screen / save metadata).
    pub fn seed_for_debug(&self) -> u64 {
        self.inner.seed_for_debug()
    }

    /// Surface height (top solid block y) at one column.
    pub fn height_at(&self, x: i64, z: i64) -> i64 {
        self.inner.column(x, z).height
    }

    /// Biome name at a column (debug screen / future spawner rules).
    pub fn biome_name_at(&self, x: i64, z: i64) -> &'static str {
        self.inner.column(x, z).biome.name()
    }

    /// Generation stats (chunks, avg/max µs) for the debug screen.
    pub fn gen_stats(&self) -> (u64, u64, u64) {
        let n = self
            .inner
            .stats
            .chunks_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        let (avg, max) = self.inner.stats.summary();
        (n, avg, max)
    }

    /// Generate one chunk into `Chunk`. Deterministic for a given seed and
    /// position; called again after unload reproduces the same terrain.
    /// Negative chunk coordinates are fully supported (pure functions of
    /// `div_euclid`-derived origins).
    pub fn generate_chunk(&self, registry: &Registry, pos: ChunkPos) -> Chunk {
        self.inner.generate_chunk(registry, pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunks::{CHUNK_X, CHUNK_Z};
    use crate::grid::WORLD_H;
    use crate::Registry;

    /// Registry with just the names the generator asks for (full cubes).
    fn test_registry() -> Registry {
        Registry::from_names_for_tests(&[
            "stone",
            "dirt",
            "grass_block",
            "water",
            "bedrock",
            "sand",
            "gravel",
            "snow_block",
            "oak_log",
            "oak_leaves",
            "coal_ore",
            "iron_ore",
        ])
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
    fn negative_coordinates_generate_without_panics() {
        let reg = test_registry();
        let gen = TerrainGenerator::new(0xFEED);
        for cx in [-17i32, -1, 0, 1, 17] {
            for cz in [-9i32, -1, 0, 3, 9] {
                let chunk = gen.generate_chunk(&reg, ChunkPos::new(cx, cz));
                // Every chunk must be well-formed: some blocks, valid y range.
                assert!(!chunk.blocks.is_empty());
                assert!(chunk.blocks.iter().any(|&b| b != 0));
            }
        }
    }

    #[test]
    fn chunk_borders_are_seamless_deterministically() {
        let reg = test_registry();
        let gen = TerrainGenerator::new(0xB0A);
        // Generate the 2×2 chunk neighborhood twice (fresh generator each
        // time); the border columns must match across generators, proving
        // no per-chunk hidden state leaks into the shape.
        let mk = || -> Vec<Chunk> {
            let g = TerrainGenerator::new(0xB0A);
            [
                ChunkPos::new(-1, -1),
                ChunkPos::new(0, -1),
                ChunkPos::new(-1, 0),
                ChunkPos::new(0, 0),
            ]
            .iter()
            .map(|&p| g.generate_chunk(&reg, p))
            .collect()
        };
        let a = mk();
        let b = mk();
        for (ca, cb) in a.iter().zip(b.iter()) {
            assert_eq!(ca.blocks, cb.blocks);
        }
        // Explicit border continuity: height across the x=0 border jumps
        // no more than the noise's natural slope.
        let h1 = gen.height_at(-1, 0);
        let h2 = gen.height_at(0, 0);
        assert!((h1 - h2).abs() <= 6, "border cliff too sharp: {h1} vs {h2}");
    }

    #[test]
    fn surface_layering_by_biome() {
        let reg = test_registry();
        let gen = TerrainGenerator::new(7);
        let chunk = gen.generate_chunk(&reg, ChunkPos::new(0, 0));
        let grass = reg.block_id("grass_block").unwrap();
        let dirt = reg.block_id("dirt").unwrap();
        let water = reg.block_id("water").unwrap();

        let mut checked = 0;
        for lz in 0..CHUNK_Z {
            for lx in 0..CHUNK_X {
                let surface = gen.height_at(lx as i64, lz as i64);
                if surface <= SEA_LEVEL {
                    continue; // flooded columns differ by design
                }
                let (top, _) = chunk.get_local(lx, surface as u32, lz).unwrap();
                let (below1, _) = chunk.get_local(lx, (surface - 1) as u32, lz).unwrap();
                // Grass surface over dirt — unless a feature (tree trunk)
                // replaced it, which the biome layering allows.
                if top != grass && top != reg.block_id("oak_log").unwrap() {
                    // Snow biomes swap the surface block; accept any solid.
                    assert!(top != water, "dry column must not be water");
                }
                if below1 != dirt {
                    // Mountains swap sub-surface for stone; allowed.
                    assert!(below1 == dirt || below1 == reg.block_id("stone").unwrap());
                }
                checked += 1;
            }
        }
        assert!(checked > 8, "expected dry columns in chunk (0,0)");
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
        for lz in 0..CHUNK_Z {
            for lx in 0..CHUNK_X {
                let surface = gen.height_at(lx as i64, lz as i64);
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
    fn water_fills_to_sea_level_in_dips() {
        let reg = test_registry();
        let water = reg.block_id("water").unwrap();
        let mut found = false;
        'seeds: for seed in [1u64, 2, 3, 5, 8, 13, 21, 34, 55, 89] {
            let gen = TerrainGenerator::new(seed);
            for cx in [-3i32, 0, 4, -11, 9] {
                for cz in [-3i32, 2, -8, 6] {
                    let pos = ChunkPos::new(cx, cz);
                    let chunk = gen.generate_chunk(&reg, pos);
                    for lz in 0..CHUNK_Z {
                        for lx in 0..CHUNK_X {
                            let surface = gen.height_at(
                                pos.min_block().0 + lx as i64,
                                pos.min_block().1 + lz as i64,
                            );
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
        assert!(
            found,
            "no dipped column found across seeds/chunks — amplitude broken"
        );
    }

    #[test]
    fn heightmap_stays_within_world_bounds() {
        let gen = TerrainGenerator::new(8080);
        for x in -200..200 {
            for z in -50..50 {
                let h = gen.height_at(x, z);
                assert!(
                    h >= 2 && h < WORLD_H as i64 - 24,
                    "height {h} out of bounds"
                );
            }
        }
    }

    #[test]
    fn generation_stats_are_recorded() {
        let reg = test_registry();
        let gen = TerrainGenerator::new(606);
        let _ = gen.generate_chunk(&reg, ChunkPos::new(0, 0));
        let _ = gen.generate_chunk(&reg, ChunkPos::new(1, 0));
        let (n, _avg, max) = gen.gen_stats();
        assert!(n >= 2, "stats must count chunks (n={n})");
        assert!(max > 0);
    }
}
