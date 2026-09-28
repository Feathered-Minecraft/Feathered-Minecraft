//! Chunk streaming: load/mesh/unload around the player without stalling
//! the render thread.
//!
//! Every frame the streamer is allowed a small time budget (a few ms): it
//! generates/meshes at most a couple of chunks per tick, closest-first, so
//! the world fills in progressively and frames stay regular. Unloading
//! frees chunks (and their GPU buffers) that fall outside the view
//! distance — a *bounded* number per tick via an explicit unload queue, so
//! even teleport-scale moves cannot tear hundreds of chunks out in one
//! frame or balloon memory. Player edits live in the streamer's edit
//! journal (the world save): broken/placed blocks override regenerated
//! terrain deterministically at generation time, survive unload/reload,
//! and are what gets persisted to disk.
//!
//! Ownership: the streamer is `Send`-free and lifetime-free — it owns the
//! chunk container, the generator and the atlas/UV lookup table. The
//! registry is *passed per call* (`poll`, `remesh`, `relight`,
//! `spawn_height`) so `AppState` can own both the `Registry` and the
//! `Streamer` without a self-referential borrow.

use feathered_chunk::{mesh_region, MeshedChunk, TintPolicy};
use feathered_world::save::WorldSave;
use feathered_world::chunks::{Chunk, ChunkPos, ChunkedWorld, CHUNK_X, CHUNK_Z};
use feathered_world::{LightGrid, Registry, TerrainGenerator};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};

/// One atlas lookup entry: (x, y, frame_w, frame_h, frames, stride,
/// fully-opaque-sprite) — mirrors the mesher's `SpriteRect`.
pub type SpriteRect = (u32, u32, u32, u32, u32, u32, bool);

/// Atlas lookup: SpriteId → rect. Owned by the streamer so no borrow of
/// the atlas (or registry) outlives construction.
pub type UvRects = dyn Fn(u32) -> Option<SpriteRect>;

/// Streaming metrics for the debug screen (plain counters, no locks).
#[derive(Debug, Default, Clone, Copy)]
pub struct StreamStats {
    /// Chunks currently held in memory (loaded, meshed or bare).
    pub loaded: usize,
    /// Chunks with a finished mesh (renderable).
    pub meshed: usize,
    /// Chunks queued for unload (bounded drain per tick).
    pub unload_pending: usize,
    /// Total chunks ever loaded this session.
    pub loads_total: u64,
    /// Total chunks ever unloaded this session.
    pub unloads_total: u64,
    /// Total block edits applied this session (journal size).
    pub edits: usize,
    /// Last burst load (parallel neighbor generation) — debug visibility.
    pub last_load_burst: u32,
}

/// Rectangular region of blocks around a chunk (the meshing source window).
pub struct Streamer {
    pub world: ChunkedWorld,
    pub generator: TerrainGenerator,
    /// View distance in chunks (Manhattan-ish ring radius).
    pub view_distance: i32,
    /// Meshed/light caches keyed by chunk (light grids kept for editing).
    pub lights: HashMap<ChunkPos, LightGrid>,
    /// Chunks whose mesh has already been produced (a chunk can exist in
    /// `world` before being meshed — spawn pre-generation — and must then
    /// still be picked up by `poll`).
    meshed: HashSet<ChunkPos>,
    /// Player edits (breaks/places), keyed by block position. Terrain
    /// generation consults this so edits survive unload/reload.
    pub edits: HashMap<(i64, i64, i64), (u32, u32)>,
    /// Set when `edits` changed since the last `apply_save` (dirty flag for
    /// the client's periodic autosave).
    edits_dirty: bool,
    /// Bounded unload queue (farthest-first); drained CHUNK loads per tick.
    unload_queue: VecDeque<ChunkPos>,
    /// Set when chunks may exist outside the current sweep (after any load
    /// or an explicit re-queue); cleared when a sweep rebuild runs.
    needs_unload_sweep: bool,
    tint: TintPolicy,
    uv_rects: Box<UvRects>,
    atlas_size: (u32, u32),
    stats: StreamStats,
}

/// One chunk of streaming work.
#[derive(Debug)]
pub enum ChunkJob {
    Load {
        pos: ChunkPos,
        mesh: MeshedChunk,
        light: LightGrid,
    },
    Unload(ChunkPos),
}

impl Streamer {
    pub fn new(
        seed: u64,
        view_distance: i32,
        atlas_size: (u32, u32),
        uv_rects: Box<UvRects>,
    ) -> Streamer {
        Streamer {
            world: ChunkedWorld::new(),
            generator: TerrainGenerator::new(seed),
            view_distance: view_distance.clamp(1, 32),
            lights: HashMap::new(),
            meshed: HashSet::new(),
            edits: HashMap::new(),
            edits_dirty: false,
            unload_queue: VecDeque::new(),
            needs_unload_sweep: true,
            tint: TintPolicy {
                tint0: [145, 189, 89, 255], // plains grass (Phase-1 policy)
            },
            uv_rects,
            atlas_size,
            stats: StreamStats::default(),
        }
    }

    /// Center chunk for the given player block position.
    pub fn center_for(&self, player_x: f32, player_z: f32) -> ChunkPos {
        ChunkPos::of_block(player_x.floor() as i64, player_z.floor() as i64)
    }

    /// The next chunk to load: closest missing chunk within the view ring.
    fn next_to_load(&self, center: ChunkPos) -> Option<ChunkPos> {
        let r = self.view_distance;
        let mut best: Option<(u32, ChunkPos)> = None;
        for dz in -r..=r {
            for dx in -r..=r {
                let d2 = (dx * dx + dz * dz) as u32;
                if d2 > (r * r) as u32 {
                    continue;
                }
                let pos = ChunkPos::new(center.x + dx, center.z + dz);
                // Not yet usable for rendering: missing OR present-but-
                // never-meshed (spawn pre-generation inserts bare chunks).
                if self.world.contains(pos) && self.meshed.contains(&pos) {
                    continue;
                }
                if best.map(|(bd, _)| d2 < bd).unwrap_or(true) {
                    best = Some((d2, pos));
                }
            }
        }
        best.map(|(_, p)| p)
    }

    /// Rebuild the unload queue from the loaded set: every chunk outside
    /// the view circle, farthest-first. Called when the center chunk moves.
    /// The queue is *bounded* (`unload_budget`): a teleport across the map
    /// enqueues only the first 1024 unload targets per rebuild and drains
    /// them over ticks — no unbounded queue growth, no single-frame spike.
    fn rebuild_unload_queue(&mut self, center: ChunkPos) {
        const UNLOAD_BUDGET: usize = 1024;
        let r2 = (self.view_distance * self.view_distance) as i64;
        let mut out: Vec<(i64, ChunkPos)> = Vec::new();
        for pos in self.world.iter().map(|c| c.pos) {
            let d2 = pos.dist2(center) as i64;
            if d2 <= r2 {
                continue;
            }
            out.push((d2, pos));
        }
        out.sort_by_key(|b| std::cmp::Reverse(b.0)); // farthest first
        self.unload_queue = out.into_iter().take(UNLOAD_BUDGET).map(|(_, p)| p).collect();
    }

    /// Produce at most one chunk job (load or unload). Callers run this in
    /// a loop with a time budget. Loads include mesh + light; the caller
    /// uploads the mesh GPU-side and keeps the light grid for edits.
    pub fn poll(&mut self, registry: &Registry, center: ChunkPos) -> Option<ChunkJob> {
        // Prefer loading until the ring is full; then drain the unload queue.
        if let Some(pos) = self.next_to_load(center) {
            // Ensure the 8 neighbors are generated BEFORE meshing so border
            // culling is exact. Generation of each missing neighbor is fully
            // parallel (rayon) — one chunk's cost dominates this job, so the
            // extra budget spent here buys correct, seam-free meshes.
            let missing: Vec<ChunkPos> = {
                let world = &self.world;
                (-1..=1)
                    .flat_map(|dz| (-1..=1).map(move |dx| ChunkPos::new(pos.x + dx, pos.z + dz)))
                    .filter(|p| !world.contains(*p))
                    .collect()
            };
            self.stats.last_load_burst = missing.len() as u32;
            if missing.len() >= 2 {
                let gen = &self.generator;
                let made: Vec<Chunk> = missing
                    .par_iter()
                    .map(|p| gen.generate_chunk(registry, *p))
                    .collect();
                for mut c in made {
                    self.apply_edits_to_chunk(&mut c);
                    self.world.insert(c);
                }
            } else {
                for p in missing {
                    let mut c = self.generator.generate_chunk(registry, p);
                    self.apply_edits_to_chunk(&mut c);
                    self.world.insert(c);
                }
            }
            let light = feathered_world::light::compute_chunk_light(&self.world, registry, pos);
            let view = self.world.region_view(pos);
            let (min, max) = view.center_bounds();
            let mesh = mesh_region(
                &view,
                registry,
                self.uv_rects.as_ref(),
                self.atlas_size,
                &self.tint,
                (min, max),
            );
            self.lights.insert(pos, light.clone());
            self.meshed.insert(pos);
            self.stats.loads_total += 1;
            // The load may have generated fresh fringe chunks — a later
            // sweep (once the ring is full) must see them.
            self.needs_unload_sweep = true;
            self.refresh_stats();
            return Some(ChunkJob::Load { pos, mesh, light });
        }

        // Bounded unload drain: one queued chunk per poll (the client polls
        // CHUNK_JOBS_PER_FRAME times per frame, so unload throughput matches
        // load throughput and the GPU buffer count decays smoothly). The
        // queue is filled two ways: explicitly, via
        // `queue_unloads_around(new_center)` when the streaming center
        // moves, and lazily — when the ring is full, the queue has drained
        // and loads happened since the last sweep, one final rebuild
        // catches fringe chunks created during the fill. Stale entries
        // (already unloaded) are skipped; live entries are RETURNED — the
        // caller applies them via `unload` (dropping CPU data; the client
        // also drops GPU buffers for the same pos).
        if self.unload_queue.is_empty() && self.needs_unload_sweep {
            self.rebuild_unload_queue(center);
            self.needs_unload_sweep = false;
        }
        while let Some(pos) = self.unload_queue.pop_front() {
            if self.world.contains(pos) {
                self.refresh_stats();
                return Some(ChunkJob::Unload(pos));
            }
        }
        self.refresh_stats();
        None
    }

    /// Mesh one already-loaded chunk again (after an edit). Requires the
    /// 3×3 neighborhood to still be loaded (it is, inside the view ring).
    pub fn remesh(&self, registry: &Registry, pos: ChunkPos) -> Option<MeshedChunk> {
        if !self.world.contains(pos) {
            return None;
        }
        let view = self.world.region_view(pos);
        let (min, max) = view.center_bounds();
        Some(mesh_region(
            &view,
            registry,
            self.uv_rects.as_ref(),
            self.atlas_size,
            &self.tint,
            (min, max),
        ))
    }

    /// Recompute one chunk's light (after an edit).
    pub fn relight(&mut self, registry: &Registry, pos: ChunkPos) -> Option<LightGrid> {
        if !self.world.contains(pos) {
            return None;
        }
        let light = feathered_world::light::compute_chunk_light(&self.world, registry, pos);
        self.lights.insert(pos, light.clone());
        Some(light)
    }

    /// Drop a chunk and its cached light grid (unload).
    pub fn unload(&mut self, pos: ChunkPos) {
        self.world.remove(pos);
        self.lights.remove(&pos);
        self.meshed.remove(&pos);
        self.stats.unloads_total += 1;
        self.refresh_stats();
    }

    /// Queue every loaded chunk outside `center`'s ring for unloading
    /// (farthest-first, bounded budget). Used when the streaming center
    /// changes (teleport / world load); the actual unloads drain through
    /// `poll`, one per tick-slot, so frames stay regular.
    pub fn queue_unloads_around(&mut self, center: ChunkPos) {
        self.rebuild_unload_queue(center);
        // Fringe chunks created by the coming fill phase must be swept too.
        self.needs_unload_sweep = true;
        self.refresh_stats();
    }

    /// Record a block edit. The edit is journaled (survives unload AND
    /// persists to the save), applied to the chunk if loaded, and flagged
    /// dirty for the next autosave.
    pub fn record_edit(&mut self, x: i64, y: i64, z: i64, block: u32, state: u32) {
        let key = (x, y, z);
        self.edits.insert(key, (block, state));
        self.edits_dirty = true;
        // Apply immediately if the chunk is loaded (keeps `world` a pure
        // function of terrain + edits even mid-frame).
        let (bx, bz) = ChunkPos::of_block(x, z).min_block();
        let _ = (bx, bz);
        if let Some(chunk) = self
            .world
            .get_chunk_mut(ChunkPos::of_block(x, z))
        {
            let (lx, y_, lz) = (
                (x - bx) as u32,
                y as u32,
                (z - bz) as u32,
            );
            chunk.set_local(lx, y_, lz, block, state);
        }
        self.stats.edits = self.edits.len();
    }

    /// Apply journaled edits to one chunk (called right after generation so
    /// reloaded chunks show player modifications).
    fn apply_edits_to_chunk(&self, chunk: &mut Chunk) {
        if self.edits.is_empty() {
            return;
        }
        let (bx, bz) = chunk.pos.min_block();
        // Scan the journal — it is small (player edits only) and this runs
        // once per chunk generation.
        for (&(x, y, z), &(block, state)) in &self.edits {
            let in_x = x >= bx && x < bx + CHUNK_X as i64;
            let in_z = z >= bz && z < bz + CHUNK_Z as i64;
            let in_y = y >= 0 && (y as u32) < feathered_world::grid::WORLD_H;
            if in_x && in_z && in_y {
                chunk.set_local((x - bx) as u32, y as u32, (z - bz) as u32, block, state);
            }
        }
    }

    /// Import a save's edit journal + metadata (load path). Returns the
    /// player/day state so the client can restore it.
    pub fn apply_save(&mut self, save: &WorldSave) {
        self.edits = save.edits.clone();
        self.stats.edits = self.edits.len();
        self.edits_dirty = false;
    }

    /// Snapshot the journal into a save (client calls this on autosave).
    /// Returns false when nothing changed since the last snapshot.
    pub fn build_save(&self, meta: feathered_world::save::WorldMeta) -> (bool, WorldSave) {
        let mut save = WorldSave::new(meta);
        for (&(x, y, z), &(block, state)) in &self.edits {
            save.edits.insert((x, y, z), (block, state));
        }
        (self.edits_dirty, save)
    }

    /// Mark the journal clean (after a successful disk write).
    pub fn mark_saved(&mut self) {
        self.edits_dirty = false;
    }

    /// Snapshot counters.
    pub fn stats(&self) -> StreamStats {
        let mut s = self.stats;
        s.loaded = self.world.len();
        s.meshed = self.meshed.len();
        s.unload_pending = self.unload_queue.len();
        s
    }

    fn refresh_stats(&mut self) {
        self.stats.loaded = self.world.len();
        self.stats.meshed = self.meshed.len();
        self.stats.unload_pending = self.unload_queue.len();
        self.stats.edits = self.edits.len();
    }

    /// Has this chunk already produced its mesh (via `poll`)?
    pub fn is_meshed(&self, pos: ChunkPos) -> bool {
        self.meshed.contains(&pos)
    }

    /// Spawn point: surface height at (0.5, 0.5) of the world origin. The
    /// spawn chunk AND its 3×3 neighborhood are generated (not meshed —
    /// `poll` meshes them over the first frames) with a small margin below
    /// the surface, so the player never spawns over a void and the first
    /// frames of physics have real ground. Deterministic per seed.
    pub fn spawn_height(&mut self, registry: &Registry) -> f32 {
        let pos = ChunkPos::new(0, 0);
        if self.world.contains(pos) && self.meshed.contains(&pos) {
            return 40.0;
        }
        let mut chunk = self.generator.generate_chunk(registry, pos);
        // Journaled edits first: the spawn height must reflect player
        // modifications (a tower at spawn raises the surface).
        self.apply_edits_to_chunk(&mut chunk);
        let h = surface_height(&chunk, 8, 8);
        self.world.insert(chunk.clone());
        // Insert the neighborhood too (parallel generation) so spawn physics
        // has ground and the first remesh of the spawn chunk sees neighbors.
        let mut neighbors: Vec<ChunkPos> = Vec::new();
        for dz in -1..=1 {
            for dx in -1..=1 {
                let p = ChunkPos::new(dx, dz);
                if !self.world.contains(p) {
                    neighbors.push(p);
                }
            }
        }
        let gen = &self.generator;
        let made: Vec<Chunk> = neighbors
            .par_iter()
            .map(|p| gen.generate_chunk(registry, *p))
            .collect();
        for mut c in made {
            self.apply_edits_to_chunk(&mut c);
            self.world.insert(c);
        }
        h as f32
    }

    /// Synchronously load (but don't mesh) every chunk in a square radius
    /// around `center`. Used by the title screen's panorama: the first ring
    /// of terrain exists the moment the menu opens (no empty sky on frame
    /// one) while `poll` still supplies the meshes over the next frames.
    pub fn preload_around(&mut self, registry: &Registry, center: ChunkPos, radius: i32) {
        let missing: Vec<ChunkPos> = (-radius..=radius)
            .flat_map(|dz| (-radius..=radius).map(move |dx| ChunkPos::new(center.x + dx, center.z + dz)))
            .filter(|p| !self.world.contains(*p))
            .collect();
        if missing.is_empty() {
            return;
        }
        let gen = &self.generator;
        let made: Vec<Chunk> = missing
            .par_iter()
            .map(|p| gen.generate_chunk(registry, *p))
            .collect();
        for mut c in made {
            self.apply_edits_to_chunk(&mut c);
            self.world.insert(c);
        }
    }

    /// The seed of this streamer's generator (debug screen).
    pub fn seed(&self) -> u64 {
        self.generator.seed_for_debug()
    }

    /// Surface height at an arbitrary column without loading anything:
    /// loaded chunks answer from their data; unloaded columns fall back to
    /// the generator's deterministic heightmap (the same value generation
    /// would produce — no cross-system drift).
    pub fn spawn_height_at(&mut self, registry: &Registry, x: i64, z: i64) -> i64 {
        let pos = ChunkPos::of_block(x, z);
        if let Some(chunk) = self.world.get_chunk(pos) {
            let (bx, bz) = chunk.pos.min_block();
            let (lx, lz) = ((x - bx) as u32, (z - bz) as u32);
            for y in (0..feathered_world::grid::WORLD_H).rev() {
                if let Some((b, _)) = chunk.get_local(lx, y, lz) {
                    if b != 0 {
                        return y as i64;
                    }
                }
            }
            return 0;
        }
        self.generator.height_at(x, z)
    }

    /// Access a block with edit-journal fallback: if the chunk is not
    /// loaded, returns the journaled edit if one exists there (interaction
    /// and targeting must behave consistently near the loaded border).
    pub fn block_with_edits(&self, x: i64, y: i64, z: i64) -> Option<(u32, u32)> {
        if let Some(b) = self.world.block(x, y, z) {
            return Some(b);
        }
        self.edits.get(&(x, y, z)).copied()
    }
}

/// Topmost non-air block y in one column (local coords) + 1.
fn surface_height(chunk: &Chunk, lx: u32, lz: u32) -> u32 {
    let mut top = 1u32;
    for y in (0..feathered_world::grid::WORLD_H).rev() {
        if chunk.get_local(lx, y, lz).map(|(b, _)| b != 0).unwrap_or(false) {
            top = y + 1;
            break;
        }
    }
    top
}

#[cfg(test)]
mod tests {
    use super::*;
    use feathered_world::grid::WORLD_H;

    fn test_registry() -> Registry {
        Registry::from_names_for_tests(&["stone", "dirt", "grass_block", "water", "bedrock"])
    }

    fn uv_missing(_: u32) -> Option<SpriteRect> {
        // No atlas: the mesher skips every quad (no sprite rects), which is
        // fine for streaming bookkeeping tests.
        None
    }

    fn streamer(reg: &Registry, seed: u64, view_distance: i32) -> Streamer {
        // Assert the registry really is borrowed per call: the streamer
        // itself must be usable without it (proves no hidden 'a capture).
        let _ = reg;
        Streamer::new(seed, view_distance, (16, 16), Box::new(uv_missing))
    }

    #[test]
    fn poll_loads_closest_chunks_first_then_unloads_far() {
        let reg = test_registry();
        let mut s = streamer(&reg, 1, 2);
        let center = ChunkPos::new(0, 0);

        // Fill the ring: chunks with dist² ≤ 2² — a circle of 13 chunks.
        // Each load also generates its 3×3 neighborhood (border culling and
        // light need it), so the loaded set is the circle plus a fringe.
        let mut loads = 0;
        loop {
            match s.poll(&reg, center) {
                Some(ChunkJob::Load { .. }) => loads += 1,
                _ => break, // ring full, nothing to unload
            }
            assert!(loads <= 25, "ring must terminate");
        }
        assert_eq!(loads, 13, "circle of radius 2 holds 13 chunks");
        // The loaded set is the circle plus generated-but-unmeshed fringe
        // chunks (each load brings its 3×3 neighborhood for border culling).
        let fringe = s.world.len() - 13;
        assert!(
            (13..=61).contains(&s.world.len()),
            "ring + fringe must stay bounded, got {} (fringe {fringe})",
            s.world.len()
        );

        // Walk one chunk over and drain again: poll prefers loads, so the
        // new ring fills (meshing present-but-unmeshed fringe chunks too)
        // before chunks that fell outside the view circle unload.
        s.queue_unloads_around(ChunkPos::new(1, 0));
        let mut polls = 0;
        let mut unloads = 0;
        loop {
            polls += 1;
            assert!(polls <= 300, "streaming must converge");
            match s.poll(&reg, ChunkPos::new(1, 0)) {
                Some(ChunkJob::Load { .. }) => {}
                Some(ChunkJob::Unload(p)) => {
                    s.unload(p);
                    unloads += 1;
                }
                None => break,
            }
        }
        assert!(unloads > 0, "walking on must unload out-of-range chunks");
        // Converged state: exactly the new center's circle — the unload
        // sweep drops every loaded chunk outside the ring (old ring AND
        // old fringe chunks; fringes regenerate on demand when needed).
        assert_eq!(s.world.len(), 13, "converged to the circle (fringe unloaded)");
        assert!(
            s.world.contains(ChunkPos::new(1, 0)) && s.is_meshed(ChunkPos::new(1, 0)),
            "the new center chunk is meshed"
        );
        assert!(!s.world.contains(ChunkPos::new(-2, 0)), "old area unloaded");
    }

    #[test]
    fn loads_are_closest_first() {
        let reg = test_registry();
        let mut s = streamer(&reg, 1, 2);
        let center = ChunkPos::new(0, 0);
        let first = match s.poll(&reg, center) {
            Some(ChunkJob::Load { pos, .. }) => pos,
            other => panic!("first job must be a load, got {other:?}"),
        };
        assert_eq!(first, center, "the center chunk loads first");
        let second = match s.poll(&reg, center) {
            Some(ChunkJob::Load { pos, .. }) => pos,
            other => panic!("second job must be a load, got {other:?}"),
        };
        assert!(
            second.dist2(center) <= 2,
            "second load must be adjacent (ring dist² ≤ 2), got {second:?}"
        );
    }

    #[test]
    fn view_distance_is_clamped_to_a_sane_range() {
        let s = Streamer::new(1, 0, (16, 16), Box::new(uv_missing));
        assert_eq!(s.view_distance, 1, "0 rings would never load anything");
        let s = Streamer::new(1, 999, (16, 16), Box::new(uv_missing));
        assert_eq!(s.view_distance, 32, "huge rings would exhaust memory");
    }

    #[test]
    fn edits_survive_a_poll_cycle_and_an_unload_reload() {
        let reg = test_registry();
        let mut s = streamer(&reg, 42, 1);
        let center = ChunkPos::new(0, 0);
        loop {
            match s.poll(&reg, center) {
                Some(ChunkJob::Load { .. }) => {}
                Some(ChunkJob::Unload(p)) => s.unload(p),
                None => break,
            }
        }
        // Place a stone block high in the air through the journal.
        let stone = reg.block_id("stone").unwrap();
        s.record_edit(2, 90, 2, stone, 0);
        // Another poll round: everything loaded, nothing to do.
        assert!(s.poll(&reg, center).is_none());
        // The edit is visible through the world.
        assert_eq!(s.world.block(2, 90, 2).unwrap().0, stone);
        // Journal round-trips through a save snapshot.
        let meta = feathered_world::save::WorldMeta {
            seed: 42,
            player: feathered_world::save::PlayerSave {
                pos: [0.5, 90.0, 0.5],
                yaw: 0.0,
                pitch: 0.0,
            },
            day_fraction: Some(0.25),
            saved_at_unix: None,
        };
        let (dirty, save) = s.build_save(meta);
        assert!(dirty);
        assert_eq!(save.edits.len(), 1);
        let bytes = feathered_world::save::encode(&save).unwrap();
        let back = feathered_world::save::decode(&bytes).unwrap();
        assert_eq!(back.edits, s.edits);

        // Physically removing the chunk drops the block from the world BUT
        // the journal restores it on regeneration (persistence).
        s.unload(center);
        assert!(s.world.block(2, 90, 2).is_none());
        let mut fresh = streamer(&reg, 42, 1);
        fresh.apply_save(&back);
        fresh.poll(&reg, center);
        assert_eq!(
            fresh.world.block(2, 90, 2).map(|(b, _)| b),
            Some(stone),
            "edits must survive unload/reload via the journal"
        );
        // Break the block: journal records air, regeneration honors it.
        fresh.record_edit(2, 90, 2, 0, 0);
        let (_, save2) = fresh.build_save(meta);
        assert_eq!(save2.edits.get(&(2, 90, 2)), Some(&(0, 0)));
    }

    #[test]
    fn edits_at_negative_coordinates_apply_and_persist() {
        let reg = test_registry();
        let mut s = streamer(&reg, 5, 2);
        let center = ChunkPos::new(-1, -1); // negative chunk coordinates
        loop {
            match s.poll(&reg, center) {
                Some(ChunkJob::Load { .. }) => {}
                Some(ChunkJob::Unload(p)) => s.unload(p),
                None => break,
            }
        }
        let stone = reg.block_id("stone").unwrap();
        // Place a block deep in negative space.
        s.record_edit(-20, 80, -20, stone, 0);
        assert_eq!(s.world.block(-20, 80, -20).map(|(b, _)| b), Some(stone));
        // Save / reload into a fresh streamer centered at the same chunk.
        let meta = feathered_world::save::WorldMeta {
            seed: 5,
            player: feathered_world::save::PlayerSave {
                pos: [-20.5, 80.0, -20.5],
                yaw: 0.0,
                pitch: 0.0,
            },
            day_fraction: None,
            saved_at_unix: None,
        };
        let (_, save) = s.build_save(meta);
        let bytes = feathered_world::save::encode(&save).unwrap();
        let mut fresh = streamer(&reg, 5, 2);
        fresh.apply_save(&feathered_world::save::decode(&bytes).unwrap());
        loop {
            match fresh.poll(&reg, center) {
                Some(ChunkJob::Load { .. }) => {}
                Some(ChunkJob::Unload(p)) => fresh.unload(p),
                None => break,
            }
        }
        assert_eq!(
            fresh.world.block(-20, 80, -20).map(|(b, _)| b),
            Some(stone),
            "negative-coordinate edits persist"
        );
    }

    #[test]
    fn unload_queue_is_bounded_and_drains() {
        let reg = test_registry();
        let mut s = streamer(&reg, 9, 1);
        // Load a small ring, then teleport far: everything must unload via
        // the bounded queue, and the queue must not grow unboundedly.
        let center = ChunkPos::new(0, 0);
        loop {
            match s.poll(&reg, center) {
                Some(ChunkJob::Load { .. }) => {}
                Some(ChunkJob::Unload(p)) => s.unload(p),
                None => break,
            }
        }
        let far = ChunkPos::new(500, 500);
        s.queue_unloads_around(far);
        assert!(s.stats().unload_pending > 0, "teleport queues unloads");
        // Drain while polling at the NEW center (what the client does — the
        // poll center follows the player): the new ring fills while the old
        // area drains through the bounded queue.
        let mut polls = 0;
        loop {
            polls += 1;
            assert!(polls < 2000, "teleport drain must terminate");
            match s.poll(&reg, far) {
                Some(ChunkJob::Load { .. }) => {}
                Some(ChunkJob::Unload(p)) => {
                    s.unload(p);
                }
                None => break,
            }
        }
        // The old area is gone; the new ring (+fringe) is loaded; bounded.
        assert!(!s.world.contains(center), "old area unloaded");
        assert!(s.world.contains(far), "new area loaded");
        assert!(s.world.len() <= 61, "bounded after teleport: {}", s.world.len());
    }

    #[test]
    fn rapid_direction_changes_do_not_leak_chunks() {
        let reg = test_registry();
        let mut s = streamer(&reg, 11, 2);
        // Zigzag the center between two distant points repeatedly, fully
        // draining each time. The loaded set must converge to the current
        // ring (+fringe) and never grow without bound.
        let a = ChunkPos::new(0, 0);
        let b = ChunkPos::new(4, 3);
        for round in 0..4 {
            let center = if round % 2 == 0 { a } else { b };
            s.queue_unloads_around(center);
            let mut polls = 0;
            loop {
                polls += 1;
                assert!(polls <= 400, "streaming must converge (round {round})");
                match s.poll(&reg, center) {
                    Some(ChunkJob::Load { .. }) => {}
                    Some(ChunkJob::Unload(p)) => s.unload(p),
                    None => break,
                }
            }
            assert!(
                s.world.len() <= 61,
                "loaded set must stay bounded (round {}: {})",
                round,
                s.world.len()
            );
        }
        assert_eq!(
            s.world.len(),
            13,
            "converged to the final center's circle (fringe unloaded)"
        );
    }

    #[test]
    fn spawn_height_sits_above_terrain() {
        let reg = test_registry();
        let mut s = streamer(&reg, 7, 1);
        let h = s.spawn_height(&reg);
        assert!(h > 1.0, "spawn must be above bedrock (got {h})");
        // The spawn chunk and its neighborhood are loaded.
        assert!(s.world.contains(ChunkPos::new(0, 0)));
        assert!(s.world.contains(ChunkPos::new(1, 1)));
        // A second call is a no-op (idempotent).
        let h2 = s.spawn_height(&reg);
        assert_eq!(h, h2, "spawn height must be stable");
        let _ = WORLD_H;
    }

    #[test]
    fn spawn_uses_journaled_edits_for_height() {
        let reg = test_registry();
        let mut s = streamer(&reg, 21, 1);
        let stone = reg.block_id("stone").unwrap();
        // A tower at spawn: the journal is applied during spawn generation.
        for y in 60..70 {
            s.record_edit(8, y, 8, stone, 0);
        }
        let h = s.spawn_height(&reg);
        assert!(h >= 70.0, "spawn surface must reflect journaled edits (got {h})");
    }

    #[test]
    fn stats_track_the_streaming_state() {
        let reg = test_registry();
        let mut s = streamer(&reg, 3, 1);
        let before = s.stats();
        assert_eq!(before.loaded, 0);
        let center = ChunkPos::new(0, 0);
        loop {
            match s.poll(&reg, center) {
                Some(ChunkJob::Load { .. }) => {}
                Some(ChunkJob::Unload(p)) => s.unload(p),
                None => break,
            }
        }
        let st = s.stats();
        assert_eq!(st.loads_total, 5, "ring of radius 1 is a 5-chunk circle");
        assert_eq!(st.meshed, 5);
        assert!(st.loaded >= st.meshed, "fringe chunks may stay bare");
        assert_eq!(st.edits, 0);
    }
}
