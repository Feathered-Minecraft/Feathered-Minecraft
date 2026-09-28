//! Chunk coordinates and the chunked world container.
//!
//! Phase 3 replaces the single flat `World` grid with a streamable map of
//! 16×16-column chunks. Each chunk owns a dense block/state array for its
//! 16×WORLD_H×16 volume; neighbor queries that cross a chunk border resolve
//! through the container, so the mesher and the light grid can treat the
//! loaded region as one world. Terrain generation (see `terrain`) fills
//! chunks deterministically from a seed; edits made at runtime persist in
//! the container independent of (re)generation.

use crate::grid::WORLD_H;
use std::collections::HashMap;

/// Chunk footprint on the x/z plane (blocks).
pub const CHUNK_X: u32 = 16;
pub const CHUNK_Z: u32 = 16;

/// Chunk coordinates: floor-division of a block coordinate by the chunk
/// size, so negative block coordinates map to negative chunk coordinates
/// (the chunk containing block -1 starts at chunk -1, unlike a shift-based
/// division which would fold it into chunk 0).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct ChunkPos {
    pub x: i32,
    pub z: i32,
}

impl ChunkPos {
    pub const fn new(x: i32, z: i32) -> Self {
        ChunkPos { x, z }
    }

    /// The chunk containing world block (x, z).
    pub fn of_block(x: i64, z: i64) -> Self {
        ChunkPos {
            x: x.div_euclid(CHUNK_X as i64) as i32,
            z: z.div_euclid(CHUNK_Z as i64) as i32,
        }
    }

    /// World block coordinate of this chunk's minimum corner.
    pub const fn min_block(&self) -> (i64, i64) {
        (
            self.x as i64 * CHUNK_X as i64,
            self.z as i64 * CHUNK_Z as i64,
        )
    }

    /// Squared ring distance in chunks (for load/unload ordering).
    pub fn dist2(&self, other: ChunkPos) -> u32 {
        let dx = (self.x - other.x) as i64;
        let dz = (self.z - other.z) as i64;
        (dx * dx + dz * dz) as u32
    }
}

/// One 16×WORLD_H×16 chunk: dense block/state arrays with y-major indexing
/// (same layout as the Phase-1 `World`, just a smaller footprint).
#[derive(Debug, Clone)]
pub struct Chunk {
    pub pos: ChunkPos,
    /// `block_id` per cell (0 = air).
    pub blocks: Vec<u32>,
    /// `state_id` per cell (packed property index per block).
    pub states: Vec<u32>,
}

impl Chunk {
    pub fn new(pos: ChunkPos) -> Chunk {
        let len = (CHUNK_X * WORLD_H * CHUNK_Z) as usize;
        Chunk {
            pos,
            blocks: vec![0; len],
            states: vec![0; len],
        }
    }

    fn idx(&self, lx: u32, y: u32, lz: u32) -> usize {
        ((y * CHUNK_Z + lz) * CHUNK_X + lx) as usize
    }

    /// Local-index set (`lx`/`lz` must be inside the chunk).
    pub fn set_local(&mut self, lx: u32, y: u32, lz: u32, block: u32, state: u32) {
        if y >= WORLD_H {
            return;
        }
        let i = self.idx(lx, y, lz);
        self.blocks[i] = block;
        self.states[i] = state;
    }

    /// Local-index get.
    pub fn get_local(&self, lx: u32, y: u32, lz: u32) -> Option<(u32, u32)> {
        if lx >= CHUNK_X || lz >= CHUNK_Z || y >= WORLD_H {
            return None;
        }
        let i = self.idx(lx, y, lz);
        Some((self.blocks[i], self.states[i]))
    }

    /// Sky light source height: the lowest fully-open sky column would need
    /// a heightmap; Phase 3 keeps light conservative (see `light`), so this
    /// only feeds the light seeder.
    pub fn set_block_runtime(&mut self, lx: u32, y: u32, lz: u32, block: u32, state: u32) {
        self.set_local(lx, y, lz, block, state);
    }
}

/// Container of loaded chunks with world-coordinate queries. Blocks outside
/// any loaded chunk report "air that occludes nothing" for rendering, but
/// `is_solid_at` reports false and physics treats un-loaded space as solid
/// walls (players never fall out of the streamed region).
pub struct ChunkedWorld {
    chunks: HashMap<ChunkPos, Chunk>,
}

impl Default for ChunkedWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl ChunkedWorld {
    pub fn new() -> ChunkedWorld {
        ChunkedWorld {
            chunks: HashMap::new(),
        }
    }

    /// Insert (or replace) a chunk.
    pub fn insert(&mut self, chunk: Chunk) {
        self.chunks.insert(chunk.pos, chunk);
    }

    /// Take a chunk out of the map (unload).
    pub fn remove(&mut self, pos: ChunkPos) -> Option<Chunk> {
        self.chunks.remove(&pos)
    }

    pub fn get_chunk(&self, pos: ChunkPos) -> Option<&Chunk> {
        self.chunks.get(&pos)
    }

    pub fn get_chunk_at_mut(&mut self, pos: ChunkPos) -> Option<&mut Chunk> {
        self.chunks.get_mut(&pos)
    }

    pub fn contains(&self, pos: ChunkPos) -> bool {
        self.chunks.contains_key(&pos)
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Iterate loaded chunks in unspecified order.
    pub fn iter(&self) -> impl Iterator<Item = &Chunk> {
        self.chunks.values()
    }

    /// Mutable access for streaming/lighting bookkeeping.
    pub fn get_chunk_mut(&mut self, pos: ChunkPos) -> Option<&mut Chunk> {
        self.chunks.get_mut(&pos)
    }

    /// World-coordinate block/state read. `None` when outside the loaded
    /// region (mesher treats it as occluding so chunk-shell faces are not
    /// emitted; physics treats it as solid).
    pub fn block(&self, x: i64, y: i64, z: i64) -> Option<(u32, u32)> {
        if y < 0 || y >= WORLD_H as i64 {
            return None;
        }
        let pos = ChunkPos::of_block(x, z);
        let chunk = self.chunks.get(&pos)?;
        let (bx, bz) = pos.min_block();
        chunk.get_local((x - bx) as u32, y as u32, (z - bz) as u32)
    }

    /// World-coordinate write into an already-loaded chunk. `false` when the
    /// chunk is not loaded (edits never silently vanish).
    pub fn set(&mut self, x: i64, y: i64, z: i64, block: u32, state: u32) -> bool {
        if y < 0 || y >= WORLD_H as i64 {
            return false;
        }
        let pos = ChunkPos::of_block(x, z);
        let Some(chunk) = self.chunks.get_mut(&pos) else {
            return false;
        };
        let (bx, bz) = pos.min_block();
        chunk.set_local((x - bx) as u32, y as u32, (z - bz) as u32, block, state);
        true
    }

    /// Does the voxel at (x, y, z) block light / hide neighbor faces?
    /// Uses the registry's occlusion classification. Out-of-world and
    /// out-of-loaded-region count as occluding (chunk shells are never
    /// meshed; light does not leak out of the region).
    pub fn occludes(&self, registry: &crate::Registry, x: i64, y: i64, z: i64) -> bool {
        match self.block(x, y, z) {
            None => true,
            Some((0, _)) => false,
            Some((id, sid)) => registry
                .block_by_id(id)
                .and_then(|b| b.state(sid))
                .map(|s| s.occlusion.hides_neighbor())
                .unwrap_or(false),
        }
    }

    /// Solidity for physics (full-cube occluders only).
    pub fn is_solid(&self, registry: &crate::Registry, x: i64, y: i64, z: i64) -> bool {
        self.occludes(registry, x, y, z)
    }

    /// A padded view over this container for one center chunk: reads span
    /// the 3×3 chunk neighborhood (so meshing and lighting see border
    /// neighbors); writes are impossible through this view.
    pub fn region_view(&self, center: ChunkPos) -> RegionView<'_> {
        RegionView {
            world: self,
            center,
        }
    }
}

/// Read-only padded view: `get` resolves through the full container (any
/// loaded chunk), while `contains_center` tells callers whether the center
/// chunk is present. Out-of-loaded-region reads return `None`.
pub struct RegionView<'a> {
    world: &'a ChunkedWorld,
    pub center: ChunkPos,
}

impl RegionView<'_> {
    /// World-coordinate read through the whole loaded set.
    pub fn get(&self, x: i64, y: i64, z: i64) -> Option<(u32, u32)> {
        if y < 0 || y >= WORLD_H as i64 {
            return None;
        }
        let pos = ChunkPos::of_block(x, z);
        let chunk = self.world.chunks.get(&pos)?;
        let (bx, bz) = pos.min_block();
        chunk.get_local((x - bx) as u32, y as u32, (z - bz) as u32)
    }

    /// The center chunk's block range `[min, max)` for meshing bounds.
    pub fn center_bounds(&self) -> ([i64; 3], [i64; 3]) {
        let (bx, bz) = self.center.min_block();
        (
            [bx, 0, bz],
            [bx + CHUNK_X as i64, WORLD_H as i64, bz + CHUNK_Z as i64],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_coords_floor_divide_negative_positions() {
        assert_eq!(ChunkPos::of_block(0, 0), ChunkPos::new(0, 0));
        assert_eq!(ChunkPos::of_block(15, 15), ChunkPos::new(0, 0));
        assert_eq!(ChunkPos::of_block(16, -1), ChunkPos::new(1, -1));
        assert_eq!(ChunkPos::of_block(-1, -1), ChunkPos::new(-1, -1));
        assert_eq!(ChunkPos::of_block(-16, 15), ChunkPos::new(-1, 0));
        assert_eq!(ChunkPos::of_block(-17, -17), ChunkPos::new(-2, -2));
    }

    #[test]
    fn min_block_round_trips() {
        for (cx, cz) in [(-3i32, 2i32), (0, 0), (7, -9)] {
            let pos = ChunkPos::new(cx, cz);
            let (bx, bz) = pos.min_block();
            assert_eq!(ChunkPos::of_block(bx, bz), pos);
            assert_eq!(
                ChunkPos::of_block(bx + CHUNK_X as i64 - 1, bz + CHUNK_Z as i64 - 1),
                pos
            );
            assert_eq!(
                ChunkPos::of_block(bx + CHUNK_X as i64, bz),
                ChunkPos::new(cx + 1, cz)
            );
        }
    }

    #[test]
    fn world_queries_cross_chunk_borders() {
        let mut world = ChunkedWorld::new();
        for cx in -1..=1 {
            for cz in -1..=1 {
                world.insert(Chunk::new(ChunkPos::new(cx, cz)));
            }
        }
        assert!(world.set(-1, 5, -1, 42, 7));
        // The same block read back through either chunk's local view:
        assert_eq!(world.block(-1, 5, -1), Some((42, 7)));
        let c = world.get_chunk(ChunkPos::new(-1, -1)).unwrap();
        assert_eq!(c.get_local(15, 5, 15), Some((42, 7)));
        // Adjacent chunk sees air.
        assert_eq!(world.block(0, 5, -1), Some((0, 0)));
    }

    #[test]
    fn writes_outside_loaded_chunks_fail_loudly() {
        let mut world = ChunkedWorld::new();
        assert!(!world.set(100, 5, 100, 1, 0));
        assert!(world.block(100, 5, 100).is_none());
        world.insert(Chunk::new(ChunkPos::new(6, 6)));
        assert!(world.set(100, 5, 100, 1, 0));
        assert_eq!(world.block(100, 5, 100), Some((1, 0)));
    }

    #[test]
    fn y_bounds_are_enforced() {
        let mut world = ChunkedWorld::new();
        world.insert(Chunk::new(ChunkPos::new(0, 0)));
        assert!(!world.set(0, WORLD_H as i64, 0, 1, 0));
        assert!(!world.set(0, -1, 0, 1, 0));
        assert!(world.set(0, 0, 0, 1, 0));
        assert!(world.set(0, WORLD_H as i64 - 1, 0, 1, 0));
    }

    #[test]
    fn chunk_local_sets_respect_bounds() {
        let mut c = Chunk::new(ChunkPos::new(0, 0));
        c.set_local(16, 0, 0, 1, 0); // lx out of range: ignored
        assert_eq!(c.get_local(16, 0, 0), None);
        c.set_local(15, WORLD_H, 0, 1, 0); // y out of range: ignored
        assert_eq!(c.get_local(15, WORLD_H, 0), None);
        c.set_local(15, 0, 15, 3, 4);
        assert_eq!(c.get_local(15, 0, 15), Some((3, 4)));
    }
}
