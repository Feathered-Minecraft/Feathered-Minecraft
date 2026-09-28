//! Terrain shaping + carvers + surface rules + features.
//!
//! Stage layout (each a pure function of seed + coordinates + context):
//!
//!   climate → biome → (density) → terrain shape → carvers (caves)
//!   → surface rules → features (trees/boulders) → chunk finalize
//!
//! Deterministic for any (seed, x, z) — including negative coordinates —
//! because every stage is a pure function of hashed integers. Column
//! values (height, biome, surface depth) are cached in a mutex-guarded map
//! (rayon-safe; a lost race just recomputes the same value).
//!
//! Density model (simplified from the reference density-function idea):
//! the column's terrain height comes from three octave fields (base
//! continental shaping, hill/ridge detail, small roughness), blended by the
//! biome's height bias/scale. Caves carve where two 3D fields cross their
//! iso-bands (worm tubes + larger cheese pockets), never above the surface
//! skin, so skylight stays sane for the light engine.

use super::biomes::{Biome, Climate};
use super::noise::OctaveNoise;
use crate::chunks::{Chunk, ChunkPos, CHUNK_X, CHUNK_Z};
use crate::grid::WORLD_H;
use crate::Registry;
use std::collections::HashMap;
use std::sync::Mutex;

/// Base ground level and sea level (kept from the previous generator's
/// contract so the client camera/streamer constants still hold).
pub const BASE_HEIGHT: i64 = 24;
pub const SEA_LEVEL: i64 = 20;
/// Height headroom above base the shape may use (clamp bound).
const AMPLITUDE: f32 = 30.0;

/// Hash one 3D integer position into a uniform u64 (Feathered's own).
pub(crate) fn hash3(seed: u64, x: i64, y: i64, z: i64) -> u64 {
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

/// Uniform u64 → f32 in [0, 1).
pub(crate) fn unit(h: u64) -> f32 {
    (h >> 40) as f32 / (1u64 << 24) as f32
}

/// Blocks the generator fills, resolved by name (never hardcoded ids;
/// a pack without a block degrades that feature, never panics).
#[derive(Debug, Clone)]
pub(crate) struct Blocks {
    pub stone: u32,
    pub dirt: u32,
    pub grass: u32,
    pub sand: u32,
    pub snow: u32,
    pub gravel: u32,
    pub water: Option<u32>,
    pub bedrock: Option<u32>,
    pub log: Option<u32>,
    pub leaves: Option<u32>,
    pub coal_ore: Option<u32>,
    pub iron_ore: Option<u32>,
}

impl Blocks {
    fn resolve(registry: &Registry) -> Blocks {
        let id = |name: &str| registry.block_id(name).unwrap_or(0);
        Blocks {
            stone: id("stone"),
            dirt: id("dirt"),
            grass: id("grass_block"),
            sand: id("sand"),
            snow: id("snow_block"),
            gravel: id("gravel"),
            water: registry.block_id("water"),
            bedrock: registry.block_id("bedrock"),
            log: registry.block_id("oak_log"),
            leaves: registry.block_id("oak_leaves"),
            coal_ore: registry.block_id("coal_ore"),
            iron_ore: registry.block_id("iron_ore"),
        }
    }
}

/// Cached per-column context (height, biome, surface depth).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Column {
    pub height: i64,
    pub biome: Biome,
    /// Surface-rule depth noise (0..1): drives dirt/sand thickness wobble.
    pub surface_depth: f32,
}

/// The full Feathered world generator. Public API is `generate_chunk`
/// (unchanged from the previous generator's contract).
pub struct WorldGenerator {
    seed: u64,
    climate: Climate,
    base: OctaveNoise,
    hills: OctaveNoise,
    rough: OctaveNoise,
    cave_worm: OctaveNoise,
    cave_cheese: OctaveNoise,
    ore: OctaveNoise,
    columns: Mutex<HashMap<(i64, i64), Column>>,
    /// Debug counters (PHASE 2): generation stats, lock-free atomics.
    pub stats: GenStats,
}

/// Generation instrumentation (debug screen / benchmarks).
#[derive(Debug, Default)]
pub struct GenStats {
    pub chunks_generated: std::sync::atomic::AtomicU64,
    pub total_gen_us: std::sync::atomic::AtomicU64,
    pub max_gen_us: std::sync::atomic::AtomicU64,
}

impl GenStats {
    /// Average + max generation time in microseconds.
    pub fn summary(&self) -> (u64, u64) {
        let n = self
            .chunks_generated
            .load(std::sync::atomic::Ordering::Relaxed)
            .max(1);
        let total = self.total_gen_us.load(std::sync::atomic::Ordering::Relaxed);
        let max = self.max_gen_us.load(std::sync::atomic::Ordering::Relaxed);
        (total / n, max)
    }
}

impl WorldGenerator {
    pub fn new(seed: u64) -> WorldGenerator {
        WorldGenerator {
            seed,
            climate: Climate::new(seed),
            base: OctaveNoise::new(seed ^ 0xB45E, -5, &[1.0, 1.0, 0.9, 0.5]),
            hills: OctaveNoise::new(seed ^ 0x4111, -3, &[1.0, 0.8, 0.5, 0.3]),
            rough: OctaveNoise::new(seed ^ 0x05E5, -2, &[0.7, 0.5]),
            cave_worm: OctaveNoise::new(seed ^ 0xCAFE, -3, &[1.0, 1.0, 1.0]),
            cave_cheese: OctaveNoise::new(seed ^ 0x015E, -4, &[1.0, 1.0, 0.6]),
            ore: OctaveNoise::new(seed ^ 0x07E5, -3, &[1.0, 1.0]),
            columns: Mutex::new(HashMap::new()),
            stats: GenStats::default(),
        }
    }

    /// The world seed (debug screen / save metadata).
    pub fn seed_for_debug(&self) -> u64 {
        self.seed
    }

    /// Column context (height/biome/surface depth) — cached per (x, z).
    pub(crate) fn column(&self, x: i64, z: i64) -> Column {
        if let Some(&c) = self.columns.lock().unwrap().get(&(x, z)) {
            return c;
        }
        // Terrain shape: continental base + biome bias + scaled hills +
        // fine roughness. The biome is selected *before* height so ocean/
        // mountain biases can pull the shape; the height then feeds back
        // into ocean-vs-land confirmation (a column whose height ends up
        // well below sea level reads as ocean regardless of the band).
        let mut biome = self.climate.biome_at(x, z, BASE_HEIGHT, SEA_LEVEL);
        let base_n = self.base.sample2(x as f32, z as f32);
        let hills_n = self.hills.sample2(x as f32, z as f32);
        let rough_n = self.rough.sample2(x as f32, z as f32);

        let bias = biome.height_bias();
        let scale = biome.height_scale();
        let mut h =
            BASE_HEIGHT as f32 + base_n * 8.0 + bias + hills_n * 6.0 * scale + rough_n * 1.5;

        // Ocean confirmation: pull low-continentalness columns down.
        if biome == Biome::Ocean {
            h = h.min(SEA_LEVEL as f32 - 2.0 - base_n * 4.0);
        }
        // Mountains get real peaks.
        if biome == Biome::Mountains {
            let ridge = (hills_n + 0.35).clamp(0.0, 1.0);
            h += ridge * ridge * AMPLITUDE;
        }
        let height = (h.round() as i64).clamp(2, WORLD_H as i64 - 24);
        // Columns that ended below sea level read as ocean (beach handled
        // by surface rules); re-derive the biome for consistency.
        if height < SEA_LEVEL - 2 && biome != Biome::Ocean {
            biome = Biome::Ocean;
        }
        let surface_depth = unit(hash3(self.seed ^ 0x5EED, x, 77, z));
        let col = Column {
            height,
            biome,
            surface_depth,
        };
        self.columns.lock().unwrap().insert((x, z), col);
        col
    }

    /// Is (x, y, z) carved? Worm tubes (band-crossing worm field) + cheese
    /// pockets (large field above threshold), both closed near the surface
    /// skin and the world floor. Pure function of seed + coords.
    pub(crate) fn is_carved(&self, x: i64, y: i64, z: i64, surface: i64) -> bool {
        if y < 3 || y > surface - 3 {
            return false;
        }
        // Worm tubes: distance-to-iso on two offset fields crossing.
        let a = self.cave_worm.sample(x as f32, y as f32 * 1.7, z as f32);
        let b = self
            .cave_worm
            .sample((x + 31337) as f32, y as f32 * 1.7, (z - 13337) as f32);
        let tube = a * a + b * b < 0.012;
        // Cheese pockets: big caverns deep down only.
        let deep = y < surface - 12;
        let cheese = deep
            && self
                .cave_cheese
                .sample(x as f32 * 0.9, y as f32 * 1.4, z as f32 * 0.9)
                > 0.58;
        // Surface skin: solid crust of 3 blocks under grass stays intact
        // (handled by the y > surface - 3 guard above).
        tube || cheese
    }

    /// Ore substitution at (x, y, z): thin noise veins + hash sparkle.
    pub(crate) fn ore_at(&self, x: i64, y: i64, z: i64, b: &Blocks) -> u32 {
        let Some(coal) = b.coal_ore else {
            return b.stone;
        };
        let v = self.ore.sample(x as f32, y as f32, z as f32);
        if v > 0.62 && y < 40 {
            if let Some(iron) = b.iron_ore {
                if v > 0.78 && y < 24 {
                    return iron;
                }
            }
            return coal;
        }
        // Hash sparkle for single-block ores (breaks up vein edges).
        if v > 0.55 && unit(hash3(self.seed ^ 0x07, x, y, z)) > 0.995 && y < 40 {
            return coal;
        }
        b.stone
    }

    /// The surface block + subsurface for one column (surface rules).
    pub(crate) fn surface_pair(&self, _x: i64, _z: i64, col: &Column, b: &Blocks) -> (u32, u32) {
        match col.biome {
            Biome::Desert => {
                if b.sand != 0 {
                    (b.sand, b.sand)
                } else {
                    (b.sand, b.dirt)
                }
            }
            Biome::Ocean => {
                let top = if b.gravel != 0 { b.gravel } else { b.dirt };
                (top, b.dirt)
            }
            Biome::Snowfield => {
                if b.snow != 0 {
                    (b.snow, b.dirt)
                } else {
                    (b.grass, b.dirt)
                }
            }
            Biome::Mountains => {
                // Stone-y high peaks; grass only on the lower slopes.
                if col.height > BASE_HEIGHT + 14 {
                    (b.stone, b.stone)
                } else {
                    (b.grass, b.dirt)
                }
            }
            _ => (b.grass, b.dirt),
        }
    }

    /// Generate one chunk. Deterministic per (seed, pos) — the same call
    /// after unload reproduces the same chunk. Negative coordinates are
    /// handled by `div_euclid`-derived chunk origins and pure-function
    /// sampling everywhere.
    pub fn generate_chunk(&self, registry: &Registry, pos: ChunkPos) -> Chunk {
        let t0 = std::time::Instant::now();
        let mut chunk = Chunk::new(pos);
        let b = Blocks::resolve(registry);
        let (bx, bz) = pos.min_block();

        for lz in 0..CHUNK_Z {
            for lx in 0..CHUNK_X {
                let (wx, wz) = (bx + lx as i64, bz + lz as i64);
                let col = self.column(wx, wz);
                let surface = col.height;
                let (top, sub) = self.surface_pair(wx, wz, &col, &b);
                let depth = 2 + (col.surface_depth * 2.5) as i64;

                for y in 0..WORLD_H {
                    let y = y as i64;
                    let block = if y > surface {
                        if y <= SEA_LEVEL {
                            b.water.unwrap_or(0)
                        } else {
                            0
                        }
                    } else if self.is_carved(wx, y, wz, surface) {
                        // Flood carved cells that open under sea level near
                        // the surface; deeper caverns stay dry.
                        if y <= SEA_LEVEL - 6 {
                            0
                        } else if y <= SEA_LEVEL {
                            b.water.unwrap_or(0)
                        } else {
                            0
                        }
                    } else if y == surface {
                        top
                    } else if y >= surface - depth {
                        sub
                    } else {
                        self.ore_at(wx, y, wz, &b)
                    };
                    if block != 0 {
                        chunk.set_local(lx, y as u32, lz, block, 0);
                    }
                }

                if let Some(bed) = b.bedrock {
                    chunk.set_local(lx, 0, lz, bed, 0);
                }
            }
        }

        // Features: trees + boulders, placed by hashed lattice points so a
        // feature never depends on its chunk being generated first (a tree
        // near a border is generated identically by whichever chunk owns
        // its trunk; canopy cells reaching into a neighbor chunk are drawn
        // only if that chunk runs the same deterministic check — we clamp
        // canopies to the trunk's own chunk to keep this simple and exact).
        self.features(&mut chunk, &b);

        self.stats
            .chunks_generated
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let us = t0.elapsed().as_micros() as u64;
        self.stats
            .total_gen_us
            .fetch_add(us, std::sync::atomic::Ordering::Relaxed);
        self.stats
            .max_gen_us
            .fetch_max(us, std::sync::atomic::Ordering::Relaxed);
        chunk
    }

    /// Features: deterministic lattice placement. Each 5×5 lattice cell in
    /// the chunk rolls one hash: below a biome-dependent threshold a tree
    /// (forest) or boulder (plains) is placed. Placement only writes inside
    /// this chunk, so no cross-chunk ordering exists.
    fn features(&self, chunk: &mut Chunk, b: &Blocks) {
        let (Some(log), Some(leaves)) = (b.log, b.leaves) else {
            return;
        };
        let (bx, bz) = chunk.pos.min_block();
        // Lattice cells intersecting this chunk (5-block lattice).
        let cx0 = bx.div_euclid(5);
        let cz0 = bz.div_euclid(5);
        let cx1 = (bx + CHUNK_X as i64 - 1).div_euclid(5);
        let cz1 = (bz + CHUNK_Z as i64 - 1).div_euclid(5);
        for cx in cx0..=cx1 {
            for cz in cz0..=cz1 {
                let jitter_x = (hash3(self.seed ^ 0x7EE, cx, 1, cz) % 5) as i64;
                let jitter_z = (hash3(self.seed ^ 0x7EE, cx, 2, cz) % 5) as i64;
                let wx = cx * 5 + jitter_x;
                let wz = cz * 5 + jitter_z;
                // Only trunks physically inside this chunk are built.
                let lx = wx - bx;
                let lz = wz - bz;
                if !(0..CHUNK_X as i64).contains(&lx) || !(0..CHUNK_Z as i64).contains(&lz) {
                    continue;
                }
                let col = self.column(wx, wz);
                let roll = unit(hash3(self.seed ^ 0xFA7, cx, 3, cz));
                let density = match col.biome {
                    Biome::Forest => 0.55,
                    Biome::Plains => 0.06,
                    Biome::Snowfield => 0.12,
                    Biome::Desert | Biome::Ocean | Biome::Mountains
                        if col.height > BASE_HEIGHT + 14 =>
                    {
                        0.0
                    }
                    _ => 0.0,
                };
                if roll >= density || col.height <= SEA_LEVEL {
                    continue;
                }
                // Trunk height 4..6, canopy centered at the top.
                let h = 4 + (hash3(self.seed ^ 0x71, cx, 4, cz) % 3) as i64;
                let base_y = col.height + 1;
                for dy in 0..h {
                    let y = base_y + dy;
                    if (y as u32) < WORLD_H {
                        chunk.set_local(lx as u32, y as u32, lz as u32, log, 0);
                    }
                }
                // Canopy: two layers around the top.
                for dy in [h - 2, h - 1] {
                    let y = base_y + dy;
                    if y as u32 >= WORLD_H {
                        continue;
                    }
                    for dx in -2i64..=2 {
                        for dz in -2i64..=2 {
                            let (nx, nz) = (lx + dx, lz + dz);
                            if !(0..CHUNK_X as i64).contains(&nx)
                                || !(0..CHUNK_Z as i64).contains(&nz)
                            {
                                continue;
                            }
                            if dx.abs() + dz.abs() > 3 {
                                continue;
                            }
                            let idx = chunk.get_local(nx as u32, y as u32, nz as u32);
                            if idx.map(|(bl, _)| bl == 0).unwrap_or(false) {
                                chunk.set_local(nx as u32, y as u32, nz as u32, leaves, 0);
                            }
                        }
                    }
                }
                // Cap block on top.
                let top = base_y + h;
                if (top as u32) < WORLD_H {
                    chunk.set_local(lx as u32, top as u32, lz as u32, leaves, 0);
                }
            }
        }
    }
}
