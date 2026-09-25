//! World grid: a flat block-state array with neighbor queries.
//! Phase 1 uses one flat world (no section streaming).

/// World dimensions in blocks (Phase 1 validation scene size).
pub const WORLD_X: u32 = 32;
pub const WORLD_Y: u32 = 16;
pub const WORLD_Z: u32 = 32;

/// A dense world grid of (block_id, state_id) pairs.
pub struct World {
    pub size: [u32; 3],
    /// `block_id` per cell (0 = air).
    pub blocks: Vec<u32>,
    /// `state_id` per cell (packed property index per block).
    pub states: Vec<u32>,
}

impl World {
    pub fn new(size: [u32; 3]) -> World {
        World {
            size,
            blocks: vec![0; (size[0] * size[1] * size[2]) as usize],
            states: vec![0; (size[0] * size[1] * size[2]) as usize],
        }
    }

    pub fn index(&self, x: u32, y: u32, z: u32) -> Option<usize> {
        if x >= self.size[0] || y >= self.size[1] || z >= self.size[2] {
            return None;
        }
        Some(((y * self.size[2] + z) * self.size[0] + x) as usize)
    }

    pub fn set(&mut self, x: u32, y: u32, z: u32, block: u32, state: u32) {
        if let Some(i) = self.index(x, y, z) {
            self.blocks[i] = block;
            self.states[i] = state;
        }
    }

    pub fn get(&self, x: i64, y: i64, z: i64) -> Option<(u32, u32)> {
        if x < 0 || y < 0 || z < 0 {
            return None;
        }
        let i = self.index(x as u32, y as u32, z as u32)?;
        Some((self.blocks[i], self.states[i]))
    }
}
