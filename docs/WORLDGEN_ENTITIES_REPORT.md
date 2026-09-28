# Architecture & Licensing Report — Worldgen + Entities (Phase 0)

Reference trees consulted locally (git-ignored, never committed):

* `pumpkin mc and steel mc src code/SteelMC-0.15.3-mc26.2` — crate `steel-worldgen`
* `pumpkin mc and steel mc src code/Pumpkin-0.2.0-26.3-26.51` — `crates/pumpkin/src/entity`

## License findings

| Project | License | Compatibility with Feathered (GPL-3.0) |
|---|---|---|
| Feathered | GPL-3.0-or-later | — |
| SteelMC | **AGPL-3.0** | One-way compatible only: code copied *from* SteelMC would force Feathered to AGPL. **No source copying.** |
| PumpkinMC | **GPL-3.0** | Same-family license; direct reuse would be legally *permissible*, but the code is server-architecture-bound (Arc/Mutex world, protocol, plugin API). **Ideas only — no copying.** |

Decision: **zero lines copied from either tree.** Everything is independently
reimplemented in Feathered's own style, informed by the *algorithms and
architectural patterns* observed. This keeps Feathered GPL-3.0 with no new
obligations and no provenance debt. The reference trees stay outside the repo
(already `.gitignore`d, verified with `git check-ignore`).

## Systems inspected (Feathered)

* `engine/world/src/terrain.rs` — value-noise fBm heightmap + 3D cave band,
  18×18 column height cache, block resolution by name (no hardcoded ids).
* `engine/world/src/chunks.rs` — 16×WORLD_H×16 chunk, `ChunkPos::of_block`
  uses `div_euclid` (negative coordinates correct).
* `engine/world/src/light.rs` — window-based light; regression test already
  pins the negative-coordinate window fix (line ~391).
* `client/src/streaming.rs` — `poll()`-based budgeted loader, rayon
  neighborhood generation, edit journal (`apply_save`/`build_save`).
* `engine/world/src/physics.rs` — swept AABB voxel collision (`SolidAt`
  closure), `PlayerBody`. **This is the collision engine entities will reuse.**
* `engine/world/src/save.rs` — versioned `FEATWORLD` format (seed, player,
  day, edit journal). Entities get a second section, same pattern.
* No entity system exists anywhere in the workspace (searched: `entity`,
  `mob`, `ai`, `spawn` → only unrelated hits).

## SteelMC concepts observed (steel-worldgen)

1. **Improved Perlin noise** (permutation-table gradient noise, octave
   stack with precomputed per-octave factors, ~1014 lines w/ SIMD) —
   strictly better quality than Feathered's value noise for the same cost.
2. **Climate sampler → biome** (temperature/humidity/continentalness/
   erosion/ridges parameters → nearest-parameter-point biome).
3. **Density-function pipeline**: noise → density → terrain shape, cell-
   cached per column.
4. **Surface rules** as a context struct (depth, water height, steepness)
   rather than ad-hoc branches.
5. Column caches (`OverworldColumnCache`) to avoid re-evaluating 2D noise
   per y — same idea as Feathered's height cache, generalized.

## PumpkinMC concepts observed (entity/AI)

1. **GoalSelector**: prioritized goals with `Controls` flags (MOVE, LOOK,
   TARGET); a goal may run only if no higher-priority running goal holds
   the same control; `can_start`/`should_continue`/`start`/`tick`/`stop`
   lifecycle.
2. Goals: wander (interval-gated, random reachable position), look-around,
   melee attack, avoid, panic. Mob = goals + navigation + attributes.
3. `spawn_util::try_spawn_mob`: attempts loop around a center, on-top-of-
   collider strategy, collision check, spawn cap checks live at the caller.
4. Server-only machinery deliberately NOT taken: ArcSwap/atomic-cell world
   state, protocol sync, plugin events, NBT persistence, attributes stack.

## Plan (what gets built, Feathered-native)

* **worldgen** (`engine/world/src/worldgen/`): feathered-hash Perlin noise
  (permutation tables seeded by splitmix from the world seed), octave stack;
  climate fields (temperature/humidity/continentalness/erosion) → biome
  selection; 3D density (base + hills + caves + cheese caves) → terrain
  shape; surface rules per biome (grass/dirt/sand/snow/stone patches);
  ores; trees + boulders as features; all deterministic, negative-coordinate
  safe, column-cached. `TerrainGenerator` stays as the public API — its
  internals are replaced, the streamer keeps calling `generate_chunk`.
* **entities** (`engine/entity/` new crate): `EntityId` (u64 generation
  counter), `EntityWorld` store (dense Vec + slot reuse), `Mob` state
  (health, yaw, goal state), physics via the existing `physics::step`
  machinery, tick budget. Player stays in `PlayerController` (not an entity).
* **AI** (`engine/entity/src/ai/`): GoalSelector (controls + priorities,
  simplified from Pumpkin's design), goals: wander / look-around / chase /
  melee / panic; bounded A* voxel pathfinder (node + time caps, step height 1,
  cooldowns).
* **spawning** (`engine/entity/src/spawn.rs`): category caps (hostile/passive),
  per-tick attempt budget, surface-light rules, distance-from-player ring,
  deterministic RNG from (world seed, chunk, tick), despawn on chunk unload
  unless persisted.
* **persistence**: `EntitySave` section appended to the save format (v2,
  backward-compatible reader).
* **debug rendering**: entities drawn as colored boxes through the existing
  world-space overlay pipeline (no copyrighted assets; real model/skin
  support is a later renderer feature).

## Phasing

Phases 1–10 as specified in the task; each lands as a compiling, tested unit.
No existing public API changes except additive ones.

---

# Final Completion Report (Phases 1–10)

## What was built (all Feathered-native)

**World generation** — `engine/world/src/worldgen/{mod,noise,biomes,shape}.rs`:
splitmix64 seed expansion; gradient Perlin with per-stage masked hash chain;
`OctaveNoise` stacks; climate fields (temperature / humidity /
continentalness, sampled at 4×4-quart cells) → `Biome` (Plains, Forest,
Desert, Snowfield, Mountains, Ocean); 3D density shape (base + hills +
rough octaves, worm + cheese carvers); per-biome surface rules; ore veins;
lattice feature/tree placement; per-column cache behind a `Mutex<HashMap>`;
`GenStats` atomics. `TerrainGenerator` remains the public façade — the
streamer, mesher, lighting and physics required no behavioral changes.

**Entities** — new crate `engine/entity`:
* `world.rs`: `EntityId(index, generation)` handle, slot-reuse `EntityWorld`,
  deterministic iteration, `by_chunk` spatial map, snapshot/restore.
* `ai.rs`: GoalSelector (priority + `Controls` MOVE/LOOK/CHASE flags,
  parity-gated), goal builder closures, bounded A* (400-node / 2 ms caps,
  1-block step, ±3 settle), goals: wander, look-around, chase, melee, panic.
* `spawn.rs`: `Spawner` — 4 attempts/tick, 4-chunk sim ring, hostile/passive
  caps (12/10), per-chunk soft cap 4, 12–56 block player distance band,
  daylight suppresses hostiles, unloaded chunks never spawn, deterministic
  RNG, `SpawnStats` counters.
* `mobs.rs`: `Muncher` (passive) and `Lurker` (hostile) brains.

**Client integration** — `client/src/entities_host.rs`: two-pass update
(immutable AI pass producing `Intent`s → mutable physics pass through the
existing `physics::step`), wire-frame debug boxes via overlay `beam3`
(hostile red, passive green). `ChunkPos` gained serde derives. Streaming
gained `spawn_height_at` and a 6 ms/frame time budget inside the 2-jobs cap.

## Directly reused vs reimplemented

* **Copied source: zero lines** from SteelMC (AGPL-3.0 — copying would
  obligate Feathered under AGPL) and zero from Pumpkin (GPL-3.0 — same
  family as Feathered, so reuse would have been *legal*, but the code is
  server-architecture-bound and copying would import that shape).
* SteelMC: noise-family and pipeline *ideas* reimplemented in Feathered's
  own hash/noise/biome/shape code. Pumpkin: goal-selector and spawner
  *patterns* reimplemented against Feathered's existing physics and slot
  entity store. No license headers touched, no reference files moved.

## Files changed / added

Added: `engine/entity/` (whole crate), `engine/world/src/worldgen/`,
`client/src/entities_host.rs`, `client/src/title.rs`,
`docs/WORLDGEN_ENTITIES_REPORT.md`.
Modified: `Cargo.toml`/`client/Cargo.toml` (workspace member + dep),
`client/src/{lib,menu,overlay,streaming}.rs`, `client/tests/menu_preview.rs`,
`engine/renderer/src/lib.rs`, `engine/world/src/{chunks,lib,terrain}.rs`,
`Cargo.lock`. Verified via `git status` that nothing else changed and no
reference trees / assets / caches are tracked.

## Tests & verification

* `cargo check --workspace --all-targets`: clean (2 pre-existing warnings).
* `cargo test --workspace`: **21 test binaries, 203 tests passed, 0 failed**
  (feathered-world 51 incl. new worldgen suite; feathered-entity 13;
  feathered-client 80; assets/chunk/mesher/persistence/… as before).
* `cargo build --release`: OK.
* Runtime soak: `target/release/feathered.exe render --scene menu` ran the
  full 12 s window without crash or leak warnings (killed by `timeout`).
* GPU preview: `menu_preview` tests (2 ignored-run tests) pass, writing
  `target/menu-ui-preview.png`. (`--screenshot` remains blocked by the
  pre-existing, unrelated stall.)

## Performance (release build, real measurements)

* Chunk generation: avg **1.79 ms**, max 2.71 ms per chunk; **~557 chunks/s**
  sustained across a 128-chunk region. The 6 ms streaming budget admits
  ~3 chunks/frame — generation keeps ahead of streaming at any supported
  view distance.
* Column cache: ~2.0 M `height_at` lookups/s across a 257×257 column sweep.
* Entity budget: AI on tick parity, pathfinding capped at 2 ms/mob/pass and
  400 nodes, spawning at 4 attempts/tick — bounded per-frame work by design.
* Live counters (`GenStats`, `SpawnStats`) are exposed for debug overlays.

## Integration fixes made during verification

* `spawn.rs`: class/per-chunk caps were compared against pre-loop counts,
  so a single tick with several successful attempts could overshoot by one
  (test caught 11 > 10). Counts are now re-read per attempt and the
  per-chunk count is taken live (also removes a per-call map allocation).
* `lib.rs`: the entity pass is inlined into `sandbox_update` with disjoint
  field borrows; a `&mut self` helper would conflict with the immutable
  `solid` closure (borrow checker caught the naive shape).

## Remaining limitations (honest list)

* Mobs render as colored wire boxes — no models/skins/animations yet.
* Entity persistence: snapshot/restore hooks exist, but the save format v2
  entity section is not wired into `build_save`/`apply_save` yet.
* Pathfinder is 1-block-step grid A*; no door/swim/climb traversal.
* No sounds, drops, projectiles, or player-attack damage on mobs yet.
* Biomes are 6 broad classes; no feature structures (villages/fortresses).
