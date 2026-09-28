//! feathered-client — winit window, input, player controller, world
//! streaming, block interaction and the render loop.
//!
//! Phase 3: a playable first-person voxel sandbox. Modules:
//! * [`input`] — keyboard/mouse state → movement intent,
//! * [`controller`] — mouse-look + delta-time physics movement,
//! * [`streaming`] — time-budgeted chunk load/mesh/unload around the player,
//! * [`interaction`] — targeting (voxel raycast), break/place with cooldowns,
//! * [`inventory`] — hotbar + slot list,
//! * [`daycycle`] — time-of-day clock driving the renderer's sun.
//!
//! `scene = validation` still runs the Phase-1 ten-block scene with the
//! fly camera (tests and screenshots pin its behavior); the default is the
//! generated sandbox.

pub mod controller;
pub mod controls;
pub mod daycycle;
pub mod entities_host;
pub mod font;
pub mod input;
pub mod interaction;
pub mod inventory;
pub mod menu;
pub mod overlay;
pub mod profile;
pub mod settings;
pub mod streaming;
pub mod title;
pub mod ui;

use std::path::PathBuf;
use std::sync::Arc;

use feathered_assets::cache::cached_to_atlas;
use feathered_world::chunks::ChunkPos;
use feathered_world::grid::World;
use feathered_world::Registry;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowAttributes, WindowId};

pub struct RunOptions {
    pub pack_dir: PathBuf,
    pub cache_path: PathBuf,
    /// When set, capture one frame to this PNG and exit (headless validation).
    pub screenshot: Option<PathBuf>,
    /// Render quality preset (Low/Medium/High/Ultra).
    pub quality: feathered_renderer::RenderQuality,
    /// Enabled shader pack id, if any (from feathered-packs). Informational:
    /// the renderer keeps its pipeline modular across shader packs.
    pub shader_pack: Option<String>,
    /// Directory of the active shader *pack*, when one is enabled. The client
    /// translates its configuration (feathered-packs bridge) into the
    /// renderer's generic `ShaderEffectConfig` — the renderer never sees pack
    /// files or pack-specific data.
    pub shader_config_path: Option<PathBuf>,
    /// Sandbox terrain seed (default 20260926).
    pub seed: u64,
    /// Streaming view distance in chunks (default 6).
    pub view_distance: i32,
    /// Mouse sensitivity multiplier (default 1.0).
    pub sensitivity: f32,
    /// Camera field of view in degrees (default 70; menu settings override
    /// for menu-launched worlds).
    pub fov: f32,
    /// `sandbox` (default) or `validation` (Phase-1 fly-camera scene).
    pub scene: SceneKind,
    /// Day length in seconds (0 = frozen sun at the shader config's angle).
    pub day_length: f32,
    /// World directory for saves (default `./world`). Player edits,
    /// position and time persist here; `--world <dir>` selects another.
    pub world_dir: PathBuf,
    /// Human-readable pack label for the debug screen.
    pub pack_label: String,
    /// Human-readable shader-pack label for the debug screen.
    pub shader_label: Option<String>,
    /// Autosave interval in seconds of gameplay (0 disables autosave;
    /// the world still saves on exit).
    pub save_interval: f32,
    /// Whether the HUD starts visible (F1 toggles at runtime).
    pub hud_default: bool,
    /// Root directory containing world subdirectories + menus' settings
    /// (default `./worlds`). `world_dir` remains the singleplayer override.
    pub worlds_dir: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneKind {
    /// Title screen → world select → play (the normal game flow).
    Menu,
    Sandbox,
    Validation,
}

impl Default for RunOptions {
    fn default() -> Self {
        RunOptions {
            pack_dir: PathBuf::from("texture/assets"),
            cache_path: PathBuf::from("target/feathered-cache.bin"),
            screenshot: None,
            quality: feathered_renderer::RenderQuality::Medium,
            shader_pack: None,
            shader_config_path: None,
            seed: 20260926,
            view_distance: 6,
            sensitivity: 1.0,
            fov: 70.0,
            scene: SceneKind::Sandbox,
            day_length: daycycle::DEFAULT_DAY_SECONDS,
            world_dir: PathBuf::from("world"),
            pack_label: String::new(),
            shader_label: None,
            save_interval: 30.0,
            hud_default: true,
            worlds_dir: PathBuf::from("worlds"),
        }
    }
}

/// One frame of streaming work budget: at most this many chunk jobs per
/// event-loop pass (generation+mesh of one 16×128×16 chunk is the dominant
/// cost; 2/frame at 60 fps ≈ 120 chunks/s fill rate).
const CHUNK_JOBS_PER_FRAME: usize = 2;

/// Simulation tick rate driving `state.tick` (the renderer's anim clock).
const TICKS_PER_SECOND: f32 = 20.0;

/// Per-frame streaming time budget (seconds). Chunks load greedily up to
/// this slice each frame — smooth movement across chunk borders instead of
/// multi-frame hitches.
const STREAM_BUDGET_SECS: f32 = 6.0 / 1000.0;

/// Title-screen sun phase (fraction of a day; 0.08 = low golden-hour sun,
/// matching the reference mock's sunset backdrop). Pinned via the
/// renderer's sun-phase override while menus are up.
const MENU_SUN_PHASE: f32 = 0.08;
/// Title panorama drift: degrees per second the camera yaws while menus
/// are up (a full circle every 3 minutes — slow enough to read as still,
/// fast enough to notice).
const MENU_PANORAMA_YAW_RATE: f32 = 2.0f32.to_radians();
/// Title panorama eye height (blocks above the surface at spawn).
const MENU_PANORAMA_EYE: f32 = 34.0;

enum GameMode {
    /// Title/menus: no world loaded yet (sandbox state below is inert).
    /// A chosen world queues on `App::pending_world` (not here) so menu
    /// borrows never fight the app-level dispatch. Boxed: the menu state is
    /// far larger than the other variants.
    Menu { menu: Box<menu::MenuState> },
    /// Phase-1 validation scene (fly camera, fixed world). The world/mesh
    /// live on `AppState` (`mesh` is rendered every frame; `world` is only
    /// needed at setup).
    Validation,
    /// Generated sandbox (streaming chunks, physics, interaction).
    Sandbox,
}

struct AppState {
    /// The window is shared with the renderer: the renderer consumes an
    /// `Arc` clone when creating the wgpu surface, and the client keeps its
    /// own clone alive for cursor capture / title updates. (Moving the only
    /// handle into the renderer would leave every window-control path dead.)
    window: Option<Arc<dyn Window>>,
    /// Declared before `window`-related state so the surface is dropped
    /// before the native window at teardown.
    renderer: Option<feathered_renderer::Renderer>,
    registry: Registry,
    /// World mesh for the validation scene (empty for the sandbox, which
    /// renders exclusively through the streamer's per-chunk uploads).
    mesh: feathered_chunk::MeshedChunk,
    camera: feathered_renderer::Camera,
    input: input::InputState,
    mouse_captured: bool,
    focused: bool,
    last_tick: std::time::Instant,
    tick: u64,
    /// One-shot capture: render frame N, save to disk, exit.
    screenshot: Option<(u32, PathBuf)>, // (frames remaining, path)
    // --- sandbox state ---
    mode: GameMode,
    controller: controller::PlayerController,
    streamer: streaming::Streamer,
    hotbar: inventory::Inventory,
    interact: interaction::Interaction,
    day: daycycle::DayCycle,
    /// Last interaction feedback line (printed to the title bar).
    feedback: Option<(std::time::Instant, String)>,
    /// pending break/place intent while the mouse is held.
    mouse_held: [bool; 2], // [left = break, right = place]
    /// Targeted block this frame (for feedback).
    target: Option<interaction::Target>,
    // --- sandbox polish state ---
    /// Pause: stops simulation (unfocus or Esc-to-pause); rendering keeps
    /// presenting the last frame's view so the pause overlay reads.
    paused: bool,
    /// HUD visible (F1 toggles; screenshots keep the HUD state as-is).
    hud_visible: bool,
    /// F3 debug screen.
    debug_visible: bool,
    /// Frame-time EMA + fps counter for the debug screen.
    frame_dt_ema: f32,
    fps_ema: f32,
    /// Break particles (in-world, colored from the broken block).
    particles: Vec<overlay::Particle>,
    /// Break progress cached for the HUD (from the interaction driver).
    break_progress: f32,
    /// Autosave clock (seconds of accumulated gameplay since last save).
    save_timer: f32,
    /// Sprint-FOV easing state (current extra FOV, radians).
    fov_kick: f32,
    /// Streaming center last seen (to re-queue unloads on chunk change).
    last_center: ChunkPos,
    /// Loaded worlds directory (save/load root).
    world_dir: Option<PathBuf>,
    /// Active pack / shader pack labels (debug screen).
    pack_label: String,
    shader_label: Option<String>,
    /// Autosave interval (seconds of gameplay; 0 = off; save on exit too).
    save_interval: f32,
    /// Menu settings (fov/sensitivity/view/day/hud) applied to worlds
    /// started from the menu; CLI flags override at construction.
    settings: settings::Settings,
    /// Entity world (mobs) — Phase 3 of the entity system.
    entities: feathered_entity::EntityWorld,
    /// Per-mob AI scratch (keyed by entity index) + goal selectors.
    brains: std::collections::HashMap<u32, (feathered_entity::ai::GoalSelector, [f32; 4])>,
    /// Mob spawner (deterministic, category-capped).
    spawner: feathered_entity::spawn::Spawner,
    /// Sim tick counter (AI parity + spawn cadence).
    sim_tick: u64,
}

impl AppState {
    /// The streaming center for the player's current position.
    fn stream_center(&self) -> ChunkPos {
        self.streamer
            .center_for(self.controller.body.pos[0], self.controller.body.pos[2])
    }

    /// Update the target raycast for this frame.
    fn update_target(&mut self) {
        let eye = self.controller.body.eye();
        let dir = self.camera.forward();
        let world = &self.streamer.world;
        let targetable = move |x: i64, y: i64, z: i64| match world.block(x, y, z) {
            Some((0, _)) | None => false,
            Some(_) => true,
        };
        self.target = interaction::current_target(eye, dir, &targetable);
    }

    /// Capture the cursor for mouse-look (grab + hide).
    fn capture_cursor(&mut self) {
        self.mouse_captured = true;
        self.paused = false;
        if let Some(w) = &self.window {
            let _ = w.set_cursor_grab(winit::window::CursorGrabMode::Confined);
            w.set_cursor_visible(false);
        }
    }

    /// Release the cursor (grab off + visible). Simulation keeps running
    /// unless the caller also sets `paused`.
    fn release_cursor(&mut self) {
        self.mouse_captured = false;
        self.input.clear();
        self.mouse_held = [false, false];
        if let Some(w) = &self.window {
            let _ = w.set_cursor_grab(winit::window::CursorGrabMode::None);
            w.set_cursor_visible(true);
        }
    }

    /// Run streaming jobs within a per-frame TIME budget (not a fixed job
    /// count): one chunk job can cost 10ms+ (3×3 generation + mesh + light),
    /// so a count budget hitched the frame every time the player crossed a
    /// chunk border while walking. Jobs run until the budget is spent; the
    /// job cap remains as a hard safety ceiling.
    fn pump_streaming(&mut self) {
        let center = self.stream_center();
        let started = std::time::Instant::now();
        for _ in 0..CHUNK_JOBS_PER_FRAME {
            match self.streamer.poll(&self.registry, center) {
                Some(streaming::ChunkJob::Load { pos, mut mesh, .. }) => {
                    if let Some(light) = self.streamer.lights.get(&pos) {
                        apply_light_to_mesh(&mut mesh, light, pos);
                    }
                    if let Some(r) = &mut self.renderer {
                        r.set_chunk(pos, mesh);
                    }
                }
                Some(streaming::ChunkJob::Unload(pos)) => {
                    if let Some(r) = &mut self.renderer {
                        r.drop_chunk(pos);
                    }
                    self.streamer.unload(pos);
                }
                None => break,
            }
            // Time budget: stop mid-loop when the frame's streaming slice is
            // spent (checked after each completed job).
            if started.elapsed().as_secs_f32() >= STREAM_BUDGET_SECS {
                break;
            }
        }
    }

    /// Re-mesh + re-light the chunk containing (x, z) after an edit, plus
    /// neighbor chunks when the edit touches a border (their culling or
    /// border light changes).
    fn remesh_around(&mut self, x: i64, z: i64) {
        let mut dirty: Vec<ChunkPos> = vec![ChunkPos::of_block(x, z)];
        // Border-adjacent chunks when within 1 block of an edge.
        let pos = ChunkPos::of_block(x, z);
        let (bx, bz) = pos.min_block();
        let (lx, lz) = (x - bx, z - bz);
        if lx == 0 {
            dirty.push(ChunkPos::new(pos.x - 1, pos.z));
        }
        if lx == 15 {
            dirty.push(ChunkPos::new(pos.x + 1, pos.z));
        }
        if lz == 0 {
            dirty.push(ChunkPos::new(pos.x, pos.z - 1));
        }
        if lz == 15 {
            dirty.push(ChunkPos::new(pos.x, pos.z + 1));
        }
        for p in dirty {
            if !self.streamer.world.contains(p) || !self.streamer.is_meshed(p) {
                continue;
            }
            self.streamer.relight(&self.registry, p);
            if let Some(mut mesh) = self.streamer.remesh(&self.registry, p) {
                if let Some(light) = self.streamer.lights.get(&p) {
                    apply_light_to_mesh(&mut mesh, light, p);
                }
                if let Some(r) = &mut self.renderer {
                    r.set_chunk(p, mesh);
                }
            }
        }
    }

    /// Menu-mode per-frame work: stream the title panorama's terrain,
    /// drift the camera, and sync hover → title selection. The streamer is
    /// the inert default until the first world session built one — menus
    /// render fine without it (empty world).
    fn menu_update(&mut self, dt: f32) {
        self.pump_streaming();
        self.camera.yaw = (self.camera.yaw + MENU_PANORAMA_YAW_RATE * dt) % std::f32::consts::TAU;
        // Hover selects (keyboard and mouse share one selection index);
        // the stored layout matches what was drawn this frame.
        if let GameMode::Menu { menu } = &mut self.mode {
            if menu.screen == menu::Screen::Title {
                let sel = menu.title_sel.min(2);
                if let Some(l) = &menu.last_title_layout {
                    for i in 0..l.rows.len() {
                        let r = l.row_hit_rect(i, i == sel);
                        if ui::hit(r, menu.cursor) {
                            menu.title_sel = i;
                            break;
                        }
                    }
                }
            }
        }
    }

    fn sandbox_update(&mut self, dt: f32) {
        // Streaming first (ground must exist before physics probes it).
        self.pump_streaming();
        // Re-queue unloads when the streaming center crosses a chunk border.
        let center = self.stream_center();
        if center != self.last_center {
            self.streamer.queue_unloads_around(center);
            self.last_center = center;
        }

        // Mouse look.
        let (dx, dy) = self.input.take_mouse();
        let mut cam = self.camera.clone();
        self.controller.look(dx, dy, &mut cam);
        self.camera = cam;

        // Movement + physics. The solid closure borrows only the registry
        // and the chunk world (disjoint field borrows — the controller keeps
        // mutable access to its own body and the camera).
        let intent = self.input.move_intent();
        let registry = &self.registry;
        let world = &self.streamer.world;
        let solid = move |x: i64, y: i64, z: i64| match world.block(x, y, z) {
            // Unloaded space is SOLID: it walls the player in rather than
            // letting them fall out of the streamed region.
            None => true,
            Some((0, _)) => false,
            Some((id, sid)) => registry
                .block_by_id(id)
                .and_then(|b| b.state(sid))
                .map(|s| s.occlusion.hides_neighbor())
                .unwrap_or(false),
        };
        let mut cam = self.camera;
        self.controller.update(dt, &intent, &mut cam, &solid);
        self.camera = cam;

        // Entities: spawn (budgeted, capped) → AI goals → physics. Inlined
        // (not a helper method) so the mutable field borrows below stay
        // disjoint from the `solid` closure's immutable registry/world
        // captures — a `&mut self` method call would conflict with them.
        let player_pos = self.controller.body.pos;
        let player_chunk = ChunkPos::of_block(player_pos[0] as i64, player_pos[2] as i64);
        let center = self.stream_center();
        let view = self.settings.view_distance;
        let daylight = self.day.elevation_sin();
        // All closures capture disjoint field borrows (world / generator)
        // — no whole-`self` borrow is held across the EntityHost call.
        let loaded = |p: ChunkPos| self.streamer.world.contains(p);
        let generator = &self.streamer.generator;
        let surface_at = move |x: i64, z: i64| generator.height_at(x, z);
        entities_host::EntityHost::update(
            dt,
            &mut self.sim_tick,
            &mut self.entities,
            &mut self.brains,
            &mut self.spawner,
            player_pos,
            player_chunk,
            daylight,
            center,
            &loaded,
            &surface_at,
            &solid,
            view,
        );

        // Sprint FOV feedback: ease the kick toward the desired value.
        let want_kick = if intent.sprint && intent.is_moving() {
            controller::SPRINT_FOV_KICK
        } else {
            0.0
        };
        let rate = (controller::SPRINT_FOV_KICK / 0.15) * dt; // ~0.15 s ease
        self.fov_kick = if want_kick > self.fov_kick {
            (self.fov_kick + rate).min(want_kick)
        } else {
            (self.fov_kick - rate).max(want_kick)
        };
        self.camera.fov_y = 70.0f32.to_radians() + self.fov_kick;

        // Day cycle drives the renderer's sun (real sun direction + sky).
        self.day.advance(dt);
        if let Some(r) = &mut self.renderer {
            r.set_day_fraction(Some(self.day.fraction));
        }

        // Interaction: cooldown tick, then act on held buttons.
        self.interact.tick(dt);
        self.update_target();
        self.break_progress = 0.0;
        if let Some(target) = self.target.clone() {
            if self.mouse_held[0] {
                // Mining: progress accumulates while aiming at the same block.
                // The block id is read once per frame up front (avoids holding
                // an immutable borrow while the break closure mutates).
                let aimed = self
                    .streamer
                    .block_with_edits(target.hit.x, target.hit.y, target.hit.z)
                    .map(|(b, _)| b);
                let ev = self.interact.update_break(
                    &target,
                    dt,
                    &move |_, _, _| aimed,
                    &mut |x, y, z| {
                        // Only break inside loaded chunks; never the bottom layer.
                        if y <= 0 {
                            return false;
                        }
                        let world = &mut self.streamer.world;
                        world.block(x, y, z).map(|(b, _)| b != 0).unwrap_or(false)
                            && world.set(x, y, z, 0, 0)
                    },
                );
                self.break_progress = self.interact.progress();
                if let Some(interaction::InteractionEvent::Broke(x, y, z)) = ev {
                    self.remesh_around(x, z);
                    self.record_edit(x, y, z, 0, 0);
                    // Particle burst colored from the block's sprite tint.
                    let color = self.block_color(x, y, z);
                    overlay::spawn_burst(
                        &mut self.particles,
                        x,
                        y,
                        z,
                        color,
                        (x as u64) ^ ((y as u64) << 21) ^ ((z as u64) << 42),
                    );
                    self.feedback = Some((
                        std::time::Instant::now(),
                        format!("broke block at {x},{y},{z}"),
                    ));
                }
            } else {
                self.interact.reset_progress();
            }
            if self.mouse_held[1] {
                if let Some(block) = self.hotbar.selected_block().map(str::to_string) {
                    let feet = self.controller.body.pos;
                    let ev = self.interact.try_place(
                        &target,
                        &block,
                        &self.registry,
                        feet,
                        feathered_world::grid::WORLD_H,
                        &mut |x, y, z, b, s| {
                            // Cell must be empty and inside a loaded chunk.
                            let world = &mut self.streamer.world;
                            matches!(world.block(x, y, z), Some((0, _))) && world.set(x, y, z, b, s)
                        },
                    );
                    if let Some(interaction::InteractionEvent::Placed(x, y, z)) = ev {
                        self.remesh_around(x, z);
                        let id = self.registry.block_id(&block).unwrap_or(0);
                        self.record_edit(x, y, z, id, 0);
                        self.feedback = Some((
                            std::time::Instant::now(),
                            format!("placed {block} at {x},{y},{z}"),
                        ));
                    }
                }
            }
        } else {
            self.interact.reset_progress();
        }

        // Hotbar wheel.
        let wheel = self.input.take_wheel();
        if wheel != 0 {
            self.hotbar.cycle(wheel);
            if let Some(block) = self.hotbar.selected_block() {
                self.feedback = Some((std::time::Instant::now(), format!("selected: {block}")));
            }
        }

        // Break particles.
        overlay::step_particles(&mut self.particles, dt);

        // Autosave.
        self.save_timer += dt;
        let interval = self.save_interval;
        if interval > 0.0 && self.save_timer >= interval {
            self.save_world();
        }

        self.tick += (dt * TICKS_PER_SECOND) as u64;
    }

    /// Record an edit into the streaming journal (persistence).
    fn record_edit(&mut self, x: i64, y: i64, z: i64, block: u32, state: u32) {
        self.streamer.record_edit(x, y, z, block, state);
    }

    /// Deterministic particle tint for the block at (x, y, z): derive a
    /// stable pseudo-color from the block's particle sprite id (cheap and
    /// pack-dependent without touching GPU data).
    fn block_color(&self, x: i64, y: i64, z: i64) -> [u8; 4] {
        const FALLBACK: [u8; 4] = [125, 125, 125, 255];
        let Some((id, sid)) = self.streamer.block_with_edits(x, y, z) else {
            return FALLBACK;
        };
        let Some(def) = self.registry.block_by_id(id) else {
            return FALLBACK;
        };
        let name = def.name.clone();
        let particle_sprite = self
            .registry
            .model_of(&name, sid)
            .map(|m| m.particle)
            .unwrap_or(feathered_assets::models::SpriteId(u32::MAX));
        // Hash the sprite id into a mid-tone, block-distinct color.
        let h = (particle_sprite.0 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let r = 90 + (h & 0x3F) as u8;
        let g = 90 + ((h >> 8) & 0x3F) as u8;
        let b = 90 + ((h >> 16) & 0x3F) as u8;
        [r, g, b, 255]
    }

    /// Snapshot the session into a `WorldSave` (metadata + edit journal).
    fn build_world_save(&self) -> feathered_world::save::WorldSave {
        let meta = feathered_world::save::WorldMeta {
            seed: self.streamer.seed(),
            player: feathered_world::save::PlayerSave {
                pos: [
                    self.controller.body.pos[0] as f64,
                    self.controller.body.pos[1] as f64,
                    self.controller.body.pos[2] as f64,
                ],
                yaw: self.camera.yaw,
                pitch: self.camera.pitch,
            },
            day_fraction: Some(self.day.fraction),
            saved_at_unix: Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
            ),
        };
        self.streamer.build_save(meta).1
    }

    /// Atomic-save the world (journal + metadata). Feedback on failure only
    /// — a failed autosave must never crash gameplay.
    fn save_world(&mut self) -> bool {
        let Some(dir) = self.world_dir.clone() else {
            return false;
        };
        let save = self.build_world_save();
        match feathered_world::save::save_to_dir(&dir, &save) {
            Ok(_) => {
                self.streamer.mark_saved();
                self.save_timer = 0.0;
                true
            }
            Err(e) => {
                self.feedback = Some((std::time::Instant::now(), format!("save failed: {e}")));
                false
            }
        }
    }

    /// Transition Menu → Sandbox: build the streamer for `world_dir` (seed
    /// from its save), restore the journal, and spawn the player. Mirrors the
    /// setup done for `--scene sandbox` so both entry paths behave the same.
    fn start_world(
        &mut self,
        world_dir: PathBuf,
        fallback_seed: u64,
        atlas: &feathered_assets::atlas::Atlas,
    ) {
        // Seed + journal come from the save the menu materialized.
        let mut seed = fallback_seed;
        let mut restore: Option<feathered_world::save::PlayerSave> = None;
        let mut loaded_day: Option<Option<f32>> = None;
        // Applied to the NEW streamer below (apply_save only fills the
        // edit journal; the old/menu streamer is replaced in between).
        let mut loaded_save: Option<feathered_world::save::WorldSave> = None;
        if feathered_world::save::exists(&world_dir) {
            match feathered_world::save::load_from_dir(&world_dir) {
                Ok(save) => {
                    seed = save.meta.seed;
                    restore = Some(save.meta.player);
                    loaded_day = Some(save.meta.day_fraction);
                    loaded_save = Some(save);
                    println!("world: loaded edit journal from {}", world_dir.display());
                }
                Err(e) => eprintln!("world: save unreadable ({e}) — starting fresh"),
            }
        }
        let view = self.settings.view_distance;
        let uv_table = feathered_renderer::sprite_uv_table(&self.registry, atlas);
        let (w, h) = (atlas.width, atlas.height);
        self.streamer = streaming::Streamer::new(
            seed,
            view,
            (w, h),
            Box::new(move |sprite_id: u32| uv_table(sprite_id)),
        );
        if let Some(save) = &loaded_save {
            self.streamer.apply_save(save);
        }
        // Spawn on the surface (or the saved position).
        let spawn_h = self.streamer.spawn_height(&self.registry);
        let spawn: [f32; 3] = match restore {
            Some(p) => [p.pos[0] as f32, p.pos[1] as f32, p.pos[2] as f32],
            None => [0.5, spawn_h + 0.01, 0.5],
        };
        let (yaw, pitch) = restore.map(|p| (p.yaw, p.pitch)).unwrap_or((0.0, 0.0));
        self.controller = controller::PlayerController::new(spawn, self.settings.sensitivity);
        self.camera = feathered_renderer::Camera {
            pos: self.controller.body.eye(),
            yaw,
            pitch,
            fov_y: self.settings.fov.to_radians(),
            aspect: self.camera.aspect,
            near: 0.1,
            far: 400.0,
        };
        // Day clock per settings (a loaded save restores its fraction on top).
        // Also release the title screen's pinned sun so gameplay follows the
        // day cycle again.
        self.day = daycycle::DayCycle::new(self.settings.day_length);
        if let Some(f) = loaded_day {
            self.day.set_fraction(f);
        }
        if let Some(r) = &mut self.renderer {
            r.set_sun_phase_override(None);
            // Stale panorama meshes must never survive the transition; the
            // new session's streamer re-uploads as it loads.
            r.clear_chunks();
        }
        self.hotbar = inventory::Inventory::new();
        self.interact = interaction::Interaction::default();
        self.particles.clear();
        self.target = None;
        self.break_progress = 0.0;
        self.save_timer = 0.0;
        self.fov_kick = 0.0;
        self.paused = false;
        self.hud_visible = self.settings.hud;
        self.debug_visible = false;
        self.mouse_held = [false, false];
        self.last_center = ChunkPos::new(i32::MAX, i32::MAX); // forces first re-queue
        self.world_dir = Some(world_dir);
        self.mode = GameMode::Sandbox;
        // Playable immediately: capture the cursor for mouse-look (Esc
        // releases + pauses; E re-captures).
        self.capture_cursor();
    }

    /// Build the screen-space HUD draw list (crosshair, hotbar, progress,
    /// feedback line, debug screen). Pure authoring — the renderer uploads.
    fn build_hud(&self) -> feathered_renderer::HudDraw {
        let mut draw = overlay::HudLists::default();
        if !self.hud_visible && !self.debug_visible {
            return convert_hud(draw);
        }
        let (w, h) = (
            self.window
                .as_ref()
                .map(|w| w.surface_size().width as f32)
                .unwrap_or(1280.0),
            self.window
                .as_ref()
                .map(|w| w.surface_size().height as f32)
                .unwrap_or(720.0),
        );

        // Entity debug boxes (world-space overlay; Phase 6 renderer).
        entities_host::EntityHost::render_debug(&self.entities, &mut draw.world);

        // Crosshair only when a target or aim makes sense (always in the
        // sandbox; the validation scene has no HUD by design).
        if self.hud_visible {
            overlay::build_crosshair(&mut draw.screen, w, h);
            if self.break_progress > 0.0 {
                overlay::build_progress(&mut draw.screen, w, h, self.break_progress);
            }
            let rects = overlay::build_hotbar(&mut draw.screen, w, h, self.hotbar.selected);
            // Slot numbers + selected block name.
            for (i, r) in rects.iter().enumerate() {
                let label = (i + 1).to_string();
                let glyphs = overlay::centered_text(&label, *r, 1.5);
                for (ch, pos) in glyphs {
                    font::draw_text(
                        &mut draw.screen,
                        &ch.to_string(),
                        pos[0],
                        pos[1],
                        1.5,
                        [200, 200, 200, 220],
                    );
                }
            }
            if let Some(name) = self.hotbar.selected_block() {
                font::draw_text_shadow(
                    &mut draw.screen,
                    name,
                    w / 2.0 - font::text_width(name, 2.0) / 2.0,
                    h - 46.0 - 14.0 - 24.0,
                    2.0,
                    [240, 240, 240, 235],
                );
            }
            // Feedback line (recent interaction).
            if let Some((t, msg)) = &self.feedback {
                if t.elapsed().as_secs_f32() < 3.0 {
                    font::draw_text_shadow(
                        &mut draw.screen,
                        msg,
                        w / 2.0 - font::text_width(msg, 1.5) / 2.0,
                        h / 2.0 + 44.0,
                        1.5,
                        [230, 230, 230, 220],
                    );
                }
            }
            if self.paused {
                font::draw_text_shadow(
                    &mut draw.screen,
                    "PAUSED",
                    w / 2.0 - font::text_width("PAUSED", 4.0) / 2.0,
                    h / 2.0 - 60.0,
                    4.0,
                    [255, 220, 120, 240],
                );
                let hint = "CLICK OR PRESS E TO RESUME - ESC AGAIN QUITS";
                font::draw_text_shadow(
                    &mut draw.screen,
                    hint,
                    w / 2.0 - font::text_width(hint, 1.5) / 2.0,
                    h / 2.0 - 20.0,
                    1.5,
                    [225, 225, 230, 230],
                );
            }
        }

        // F3 debug screen (top-left stack).
        if self.debug_visible {
            let pos = self.controller.body.pos;
            let chunk = ChunkPos::of_block(pos[0].floor() as i64, pos[2].floor() as i64);
            let st = self.streamer.stats();
            let fps = self.fps_ema;
            let lines = [
                format!(
                    "FEATHERED {fps:.0} FPS ({:.1} MS)",
                    self.frame_dt_ema * 1000.0
                ),
                format!("XYZ {:.2} / {:.2} / {:.2}", pos[0], pos[1], pos[2]),
                format!(
                    "CHUNK {} {}  IN {:?} {:?}",
                    chunk.x,
                    chunk.z,
                    ((pos[0].floor() as i64) & 15),
                    ((pos[2].floor() as i64) & 15)
                ),
                format!(
                    "CHUNKS LOADED {} MESHED {} UNLOAD-Q {}",
                    st.loaded, st.meshed, st.unload_pending
                ),
                format!(
                    "LOADS {} UNLOADS {} EDITS {}",
                    st.loads_total, st.unloads_total, st.edits
                ),
                format!(
                    "GPU CHUNKS {}",
                    self.renderer
                        .as_ref()
                        .map(|r| r.streamed_chunks())
                        .unwrap_or(0)
                ),
                format!("SEED {}", self.streamer.seed()),
                format!(
                    "TIME {:.3} ({})",
                    self.day.fraction,
                    if self.day.is_night() { "NIGHT" } else { "DAY" }
                ),
                format!(
                    "QUALITY {:?}",
                    self.renderer
                        .as_ref()
                        .map(|r| r.settings().quality)
                        .unwrap_or(feathered_renderer::RenderQuality::Medium)
                ),
                format!("PACK {}", self.pack_label),
                format!(
                    "SHADER {}",
                    self.shader_label.as_deref().unwrap_or("(none)")
                ),
                format!(
                    "SAVE {}",
                    if self.save_timer > 0.0 {
                        "PENDING"
                    } else {
                        "OK"
                    }
                ),
            ];
            for (i, line) in lines.iter().enumerate() {
                font::draw_text_shadow(
                    &mut draw.screen,
                    line,
                    8.0,
                    8.0 + i as f32 * 14.0,
                    1.5,
                    [235, 235, 235, 230],
                );
            }
        }
        convert_hud(draw)
    }

    fn validation_update(&mut self, dt: f32) {
        let (mut dx, mut dz, mut dy) = (0.0f32, 0.0f32, 0.0f32);
        if self.input.key_down(KeyCode::KeyW) {
            dz += 1.0;
        }
        if self.input.key_down(KeyCode::KeyS) {
            dz -= 1.0;
        }
        if self.input.key_down(KeyCode::KeyA) {
            dx -= 1.0;
        }
        if self.input.key_down(KeyCode::KeyD) {
            dx += 1.0;
        }
        if self.input.key_down(KeyCode::Space) {
            dy += 1.0;
        }
        if self.input.key_down(KeyCode::ShiftLeft) {
            dy -= 1.0;
        }

        let (sin_y, cos_y) = self.camera.yaw.sin_cos();
        let forward = [sin_y, 0.0, -cos_y];
        let right = [cos_y, 0.0, sin_y];
        for (i, axis) in [forward, right].iter().enumerate() {
            let amount = if i == 0 { dz } else { dx };
            self.camera.pos[i] += axis[i] * amount * 8.0 * dt;
        }
        self.camera.pos[1] += dy * 8.0 * dt;
        self.tick += (dt * TICKS_PER_SECOND) as u64;
    }
}

/// Convert client-authored HUD lists into the renderer's POD mirror types
/// (the crates are deliberately decoupled; the layout is const-asserted on
/// both sides).
fn convert_hud(lists: overlay::HudLists) -> feathered_renderer::HudDraw {
    let conv = |l: overlay::TriList| -> feathered_renderer::client_overlay::TriList {
        feathered_renderer::client_overlay::TriList {
            vertices: l
                .vertices
                .iter()
                .map(|v| feathered_renderer::client_overlay::HudVertex {
                    pos: v.pos,
                    color: v.color,
                    px: v.px,
                    uv: v.uv,
                })
                .collect(),
            indices: l.indices,
        }
    };
    feathered_renderer::HudDraw {
        screen: conv(lists.screen),
        world: conv(lists.world),
    }
}

/// Overwrite each vertex's baked light with the voxel light of the block the
/// vertex sits in (sky, block — levels 0..15 stored raw, like
/// `build_lighting_meshes` does for the validation scene).
fn apply_light_to_mesh(
    mesh: &mut feathered_chunk::MeshedChunk,
    grid: &feathered_world::LightGrid,
    pos: ChunkPos,
) {
    let (bx, bz) = pos.min_block();
    for layer in [&mut mesh.opaque, &mut mesh.cutout, &mut mesh.translucent] {
        for v in &mut layer.vertices {
            let x = (v.pos[0].floor() as i64).clamp(bx, bx + 15);
            let z = (v.pos[2].floor() as i64).clamp(bz, bz + 15);
            let y = (v.pos[1].floor() as i64).clamp(0, feathered_world::grid::WORLD_H as i64 - 1);
            v.light = grid.get(x, y, z).unwrap_or([15, 0]);
        }
    }
}

/// Launch the game (sandbox by default, `--scene validation` for Phase 1).
pub fn run(opts: RunOptions) -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    let logo = std::fs::read("logo.png").ok().and_then(|bytes| {
        image::load_from_memory(&bytes)
            .ok()
            .map(|img| img.to_rgba8())
            .map(|rgba| {
                let (w, h) = rgba.dimensions();
                // Crop transparent canvas padding so the feather fills its
                // box like the mock's art.
                let (rgba, w, h) = menu::crop_to_alpha(rgba.into_raw(), w, h);
                (w, h, rgba, w as f32 / h as f32)
            })
    });
    let background = std::fs::read("background.png").ok().and_then(|bytes| {
        image::load_from_memory(&bytes).ok().map(|img| {
            let rgba = img.to_rgba8();
            let (w, h) = rgba.dimensions();
            (w, h, rgba.into_raw())
        })
    });
    let app = App {
        state: None,
        opts,
        atlas: None,
        pending_world: None,
        logo,
        background,
        logo_applied: false,
    };
    event_loop.run_app(app)?;
    Ok(())
}

struct App {
    state: Option<AppState>,
    opts: RunOptions,
    /// Parsed texture atlas, kept for menu → world transitions (the
    /// streamer needs sprite UVs when a world starts from the menu).
    atlas: Option<std::sync::Arc<feathered_assets::atlas::Atlas>>,
    /// World chosen in the menu, started on the next event-loop pass.
    pending_world: Option<(PathBuf, u64)>,
    /// Title-screen logo art (pixels + aspect), uploaded to the renderer
    /// once the overlay initializes.
    logo: Option<(u32, u32, Vec<u8>, f32)>,
    /// Full-screen menu background (background.png), same lifecycle.
    background: Option<(u32, u32, Vec<u8>)>,
    logo_applied: bool,
}

impl ApplicationHandler for App {
    fn can_create_surfaces(&mut self, event_loop: &dyn ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = WindowAttributes::default()
            .with_title("Feathered — Phase 3 sandbox")
            .with_surface_size(winit::dpi::LogicalSize::new(1280.0, 720.0));
        // Shared ownership (see AppState.window): renderer + client both
        // hold the window alive; the surface outlives input usage.
        let window: Arc<dyn Window> =
            Arc::from(event_loop.create_window(attrs).expect("create window"));

        // Load cache (already compiled by `feathered compile-pack`).
        let blob = std::fs::read(&self.opts.cache_path).unwrap_or_else(|e| {
            panic!(
                "cannot read cache {}: {e} (run `feathered compile-pack` first)",
                self.opts.cache_path.display()
            )
        });
        let (_, payload) = feathered_assets::cache::decode(&blob).expect("cache decode");
        let registry = Registry::from_compiled(payload.pack);
        let atlas = std::sync::Arc::new(cached_to_atlas(&payload.atlas, registry.sprite_names()));
        self.atlas = Some(atlas.clone());

        let size = window.surface_size();
        let renderer = pollster::block_on(feathered_renderer::Renderer::new(
            window.clone(),
            size.width.max(1),
            size.height.max(1),
            &atlas,
            feathered_renderer::build_anim_slots(&registry, &atlas),
            feathered_renderer::RenderSettings {
                quality: self.opts.quality,
                shader_pack: self.opts.shader_pack.clone(),
            },
        ));
        let mut renderer = renderer;

        // Shader-pack configuration bridge (unchanged from Phase 2).
        if let Some(pack_dir) = &self.opts.shader_config_path {
            match feathered_packs::translate_shader_config(pack_dir, self.opts.quality) {
                Some(translated) => {
                    if !translated.unsupported.is_empty() {
                        println!(
                            "shader pack: {} option(s) recognized but not supported: {}",
                            translated.unsupported.len(),
                            translated
                                .unsupported
                                .iter()
                                .map(|(k, _)| *k)
                                .collect::<Vec<_>>()
                                .join(", ")
                        );
                    }
                    renderer.apply_shader_config(translated.config);
                }
                None => {
                    eprintln!(
                        "shader pack: no translatable configuration found in {}; using the {q:?} preset",
                        pack_dir.display(),
                        q = self.opts.quality
                    );
                }
            }
        }

        let aspect = 1280.0 / 720.0;

        // Day fraction restored from a world save (sandbox only).
        let mut day_fraction_restore: Option<Option<f32>> = None;

        // Controls: load the keymap (defaults + user overrides). A missing
        // file is written with the defaults so users have a template.
        let controls_path = self.opts.worlds_dir.join("controls.json");
        let (controls, loaded) = controls::Controls::load(&controls_path);
        if !loaded {
            controls.save(&controls_path);
        }

        let (mode, mesh, camera, controller, streamer) = match self.opts.scene {
            SceneKind::Validation => {
                let world = build_validation_world(&registry);
                let mesh = feathered_renderer::build_meshes(&world, &registry, &atlas);
                let light_grid = feathered_world::LightGrid::compute(&world, &registry);
                let lighting_mesh = feathered_renderer::build_lighting_meshes(
                    &world,
                    &registry,
                    &atlas,
                    &light_grid,
                );
                renderer.set_world_lighting(light_grid, lighting_mesh);
                let camera = feathered_renderer::Camera {
                    pos: [16.0, 14.0, 44.0],
                    yaw: 0.0,
                    pitch: -0.28,
                    fov_y: 70.0_f32.to_radians(),
                    aspect,
                    near: 0.1,
                    far: 400.0,
                };
                let controller =
                    controller::PlayerController::new([0.5, 60.0, 0.5], self.opts.sensitivity);
                (GameMode::Validation, mesh, camera, controller, None)
            }
            SceneKind::Menu => {
                // Menus first: streamer/controller stay inert defaults until
                // a world is chosen (start_world rebuilds them for real).
                // The menu scene follows settings.json (CLI --quality still
                // wins for explicit --scene sandbox/validation launches).
                let settings_path = settings::Settings::path_for(&self.opts.worlds_dir);
                let (settings, _) = settings::Settings::load(&settings_path);
                let quality = match settings.quality.as_str() {
                    "Low" => feathered_renderer::RenderQuality::Low,
                    "High" => feathered_renderer::RenderQuality::High,
                    "Ultra" => feathered_renderer::RenderQuality::Ultra,
                    _ => feathered_renderer::RenderQuality::Medium,
                };
                renderer.set_quality(quality);
                // Golden-hour panorama behind the title (the mock's sunset
                // backdrop; the menu wash keeps the left panel readable).
                // Absolute phase override: the menu pins the sun regardless
                // of the day cycle / pack config; cleared when a world starts.
                renderer.set_sun_phase_override(Some(MENU_SUN_PHASE));
                let panorama_view = settings.view_distance.max(3);
                let profile =
                    profile::ProfileStore::new(self.opts.worlds_dir.join("profile")).load();
                let mut menu = menu::MenuState::new(&self.opts.worlds_dir, settings, profile);
                // The title screen draws its own logo art (logo.png) over
                // the user-supplied background (background.png) + fade.
                menu.logo_art = self.logo.as_ref().map(|l| l.3);
                menu.background = self.background.is_some();
                // Title panorama: stream real terrain around the spawn area
                // (same generator the sandbox uses) and park the camera high
                // above it for the mock's landscape backdrop. The first ring
                // is generated synchronously so the title never opens on an
                // empty sky; the rest streams in via menu_update.
                let uv_table = feathered_renderer::sprite_uv_table(&registry, &atlas);
                let mut streamer = streaming::Streamer::new(
                    self.opts.seed,
                    panorama_view,
                    (atlas.width, atlas.height),
                    Box::new(move |sprite_id: u32| uv_table(sprite_id)),
                );
                let eye_h = streamer.spawn_height(&registry) as f32 + MENU_PANORAMA_EYE;
                streamer.preload_around(&registry, ChunkPos::new(0, 0), 3);
                let camera = feathered_renderer::Camera {
                    pos: [0.5, eye_h, 0.5],
                    yaw: 0.0,
                    pitch: -0.38,
                    fov_y: 70.0_f32.to_radians(),
                    aspect,
                    near: 0.1,
                    far: 400.0,
                };
                let controller =
                    controller::PlayerController::new([0.5, eye_h, 0.5], self.opts.sensitivity);
                (
                    GameMode::Menu {
                        menu: Box::new(menu),
                    },
                    Default::default(),
                    camera,
                    controller,
                    Some(streamer),
                )
            }
            SceneKind::Sandbox => {
                // The streamer owns its world, generator and atlas table; the
                // registry is passed per call, so nothing borrows `self`.
                let uv_table = feathered_renderer::sprite_uv_table(&registry, &atlas);
                let mut streamer = streaming::Streamer::new(
                    self.opts.seed,
                    self.opts.view_distance,
                    (atlas.width, atlas.height),
                    // Box<dyn Fn>: wrap the Arc in a forwarding closure so
                    // the box holds a plain callable (Box<Arc<dyn Fn>> is
                    // not itself a Fn).
                    Box::new(move |sprite_id: u32| uv_table(sprite_id)),
                );
                // World load: restore the edit journal + player/day state
                // when a save exists; otherwise start fresh at spawn.
                let mut loaded_day: Option<Option<f32>> = None;
                let mut restore = None;
                if feathered_world::save::exists(&self.opts.world_dir) {
                    match feathered_world::save::load_from_dir(&self.opts.world_dir) {
                        Ok(save) => {
                            if save.meta.seed == self.opts.seed {
                                println!(
                                    "world: loaded {} edit(s) from {}",
                                    save.edits.len(),
                                    self.opts.world_dir.display()
                                );
                                restore = Some(save.meta.player);
                                loaded_day = Some(save.meta.day_fraction);
                                streamer.apply_save(&save);
                            } else {
                                eprintln!(
                                    "world: save seed {} != --seed {} — starting fresh (delete {} to keep this world)",
                                    save.meta.seed,
                                    self.opts.seed,
                                    self.opts.world_dir.display()
                                );
                            }
                        }
                        Err(e) => eprintln!("world: save unreadable ({e}) — starting fresh"),
                    }
                }
                // Generate the spawn neighborhood and measure the surface
                // height (RunOptions.seed feeds the deterministic terrain).
                let spawn_h = streamer.spawn_height(&registry);
                // Feet on the surface block + epsilon: the first physics
                // step settles the player onto the ground.
                let spawn: [f32; 3] = match restore {
                    Some(p) => [p.pos[0] as f32, p.pos[1] as f32, p.pos[2] as f32],
                    None => [0.5, spawn_h + 0.01, 0.5],
                };
                let (yaw, pitch) = restore.map(|p| (p.yaw, p.pitch)).unwrap_or((0.0, 0.0));
                let controller = controller::PlayerController::new(spawn, self.opts.sensitivity);
                let camera = feathered_renderer::Camera {
                    pos: controller.body.eye(),
                    yaw,
                    pitch,
                    fov_y: 70.0_f32.to_radians(),
                    aspect,
                    near: 0.1,
                    far: 400.0,
                };
                if let Some(f) = loaded_day {
                    // Applied to AppState's clock below.
                    day_fraction_restore = Some(f);
                }
                (
                    GameMode::Sandbox,
                    Default::default(),
                    camera,
                    controller,
                    Some(streamer),
                )
            }
        };

        self.state = Some(AppState {
            window: Some(window),
            renderer: Some(renderer),
            registry,
            mesh,
            camera,
            input: {
                let mut inp = input::InputState::default();
                inp.controls = controls;
                inp
            },
            mouse_captured: false,
            focused: true,
            last_tick: std::time::Instant::now(),
            tick: 0,
            screenshot: self.opts.screenshot.clone().map(|p| (3, p)),
            mode,
            controller,
            streamer: streamer.unwrap_or_else(|| {
                // The validation scene never touches the streamer; an empty
                // one is harmless and keeps AppState total.
                streaming::Streamer::new(0, 1, (16, 16), Box::new(|_| None))
            }),
            hotbar: inventory::Inventory::new(),
            interact: interaction::Interaction::default(),
            day: {
                let mut d = daycycle::DayCycle::new(self.opts.day_length);
                if let Some(f) = day_fraction_restore {
                    d.set_fraction(f);
                }
                d
            },
            feedback: None,
            mouse_held: [false, false],
            target: None,
            paused: false,
            hud_visible: self.opts.hud_default,
            debug_visible: false,
            frame_dt_ema: 1.0 / 60.0,
            fps_ema: 60.0,
            particles: Vec::new(),
            break_progress: 0.0,
            save_timer: 0.0,
            fov_kick: 0.0,
            last_center: ChunkPos::new(i32::MAX, i32::MAX), // forces first re-queue
            world_dir: self.opts.world_dir.clone().into(),
            pack_label: self.opts.pack_label.clone(),
            shader_label: self.opts.shader_label.clone(),
            save_interval: self.opts.save_interval,
            settings: {
                // Load once here (menus keep the same copy live); CLI flags
                // override the saved values for direct sandbox launches.
                let (mut s, _) =
                    settings::Settings::load(&settings::Settings::path_for(&self.opts.worlds_dir));
                if self.opts.scene == SceneKind::Sandbox {
                    s.sensitivity = self.opts.sensitivity;
                    s.fov = self.opts.fov;
                    s.view_distance = self.opts.view_distance;
                    s.day_length = self.opts.day_length;
                    s.hud = self.opts.hud_default;
                }
                s
            },
            entities: feathered_entity::EntityWorld::new(),
            brains: std::collections::HashMap::new(),
            spawner: feathered_entity::spawn::Spawner::new(self.opts.seed),
            sim_tick: 0,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &dyn ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => {
                // Window closed (X button): persist the sandbox first.
                if matches!(state.mode, GameMode::Sandbox) {
                    state.save_world();
                }
                event_loop.exit();
            }
            WindowEvent::SurfaceResized(size) => {
                if let Some(r) = &mut state.renderer {
                    r.resize(size.width.max(1), size.height.max(1));
                }
                state.camera.aspect = size.width as f32 / size.height.max(1) as f32;
            }
            WindowEvent::Focused(focused) => {
                state.focused = focused;
                if !focused {
                    // Pause behavior: stop simulating, release keys/buttons
                    // and give the cursor back so focus changes are clean.
                    state.paused = true;
                    state.input.clear();
                    state.mouse_held = [false, false];
                    if state.mouse_captured {
                        state.mouse_captured = false;
                        if let Some(w) = &state.window {
                            let _ = w.set_cursor_grab(winit::window::CursorGrabMode::None);
                            w.set_cursor_visible(true);
                        }
                    }
                } else {
                    state.paused = false;
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                // Menu mode: keys drive the forms (Esc back, Enter submit,
                // Backspace delete, printable chars type). No world hotkeys.
                // Menu mode: keys drive the forms (Esc back, Enter submit,
                // Backspace delete, printable chars type). No world hotkeys.
                if matches!(state.mode, GameMode::Menu { .. }) {
                    let mut menu_action = None;
                    if let GameMode::Menu { menu } = &mut state.mode {
                        let on_title = menu.screen == menu::Screen::Title;
                        if pressed {
                            match event.physical_key {
                                PhysicalKey::Code(KeyCode::Escape) => menu.escape(),
                                PhysicalKey::Code(KeyCode::Enter)
                                | PhysicalKey::Code(KeyCode::NumpadEnter) => {
                                    if on_title {
                                        // Title: Enter activates the selection.
                                        menu_action = Some(menu.title_activate());
                                    } else {
                                        menu_action = Some(menu.confirm(&self.opts.worlds_dir));
                                    }
                                }
                                PhysicalKey::Code(KeyCode::Backspace) => menu.backspace(),
                                _ => {}
                            }
                            // Title-only nav keys (arrow up/down + Space).
                            if on_title {
                                match event.physical_key {
                                    PhysicalKey::Code(KeyCode::ArrowDown)
                                    | PhysicalKey::Code(KeyCode::KeyS) => menu.title_move(1),
                                    PhysicalKey::Code(KeyCode::ArrowUp)
                                    | PhysicalKey::Code(KeyCode::KeyW) => menu.title_move(-1),
                                    PhysicalKey::Code(KeyCode::Space) => {
                                        menu_action = Some(menu.title_activate());
                                    }
                                    _ => {}
                                }
                            }
                        }
                        if let Some(text) = event.text.as_ref() {
                            for ch in text.chars() {
                                if !ch.is_control() {
                                    menu.type_char(ch);
                                }
                            }
                        }
                    }
                    if let Some(action) = menu_action {
                        if let Some(p) =
                            App::handle_menu_action(&self.opts, state, event_loop, action)
                        {
                            self.pending_world = Some(p);
                        }
                    }
                    return;
                }
                if pressed {
                    // Actions resolved through the configurable keymap.
                    if let PhysicalKey::Code(code) = event.physical_key {
                        match state.input.controls.action_of(code) {
                            Some(controls::Action::MouseCapture) => {
                                if state.mouse_captured {
                                    state.release_cursor();
                                    state.paused = true;
                                } else {
                                    state.capture_cursor();
                                }
                            }
                            Some(controls::Action::CycleQuality) => {
                                if let Some(r) = &mut state.renderer {
                                    let next = r.settings().quality.next();
                                    r.set_quality(next);
                                    println!("render quality: {next:?}");
                                }
                            }
                            Some(controls::Action::ToggleHud) => {
                                state.hud_visible = !state.hud_visible;
                            }
                            Some(controls::Action::BreakDebug) => {
                                state.debug_visible = !state.debug_visible;
                            }
                            Some(controls::Action::Hotbar1) => state.hotbar.select(0),
                            Some(controls::Action::Hotbar2) => state.hotbar.select(1),
                            Some(controls::Action::Hotbar3) => state.hotbar.select(2),
                            Some(controls::Action::Hotbar4) => state.hotbar.select(3),
                            Some(controls::Action::Hotbar5) => state.hotbar.select(4),
                            Some(controls::Action::Hotbar6) => state.hotbar.select(5),
                            Some(controls::Action::Hotbar7) => state.hotbar.select(6),
                            Some(controls::Action::Hotbar8) => state.hotbar.select(7),
                            Some(controls::Action::Hotbar9) => state.hotbar.select(8),
                            _ => {}
                        }
                    }
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Escape) => {
                            // First Esc: release cursor + pause (PAUSED hint
                            // on screen). Second Esc: save + exit.
                            if state.mouse_captured {
                                state.release_cursor();
                                state.paused = true;
                            } else if state.paused {
                                state.save_world();
                                event_loop.exit();
                            } else {
                                event_loop.exit();
                            }
                        }
                        _ => {}
                    }
                }
                if let PhysicalKey::Code(code) = event.physical_key {
                    state.input.set_key(code, pressed);
                }
            }
            WindowEvent::PointerMoved { position, .. } => {
                // Menus track the cursor for hover feedback (physical px).
                if let GameMode::Menu { menu, .. } = &mut state.mode {
                    menu.cursor = (position.x as f32, position.y as f32);
                }
                if state.mouse_captured {
                    if let Some(w) = &state.window {
                        let size = w.surface_size();
                        let center = (
                            position.x - size.width as f64 / 2.0,
                            position.y - size.height as f64 / 2.0,
                        );
                        state.input.mouse_dx += center.0 as f32;
                        state.input.mouse_dy += center.1 as f32;
                        let _ = w.set_cursor_position(winit::dpi::Position::Physical(
                            winit::dpi::PhysicalPosition::new(
                                size.width as i32 / 2,
                                size.height as i32 / 2,
                            ),
                        ));
                    }
                }
            }
            WindowEvent::PointerButton {
                state: btn_state,
                button,
                ..
            } => {
                let pressed = btn_state == ElementState::Pressed;
                // Paused sandbox: any click resumes capture (standard FPS
                // behavior; the PAUSED hint says so).
                if pressed && !state.mouse_captured && matches!(state.mode, GameMode::Sandbox) {
                    state.capture_cursor();
                }
                // Menu clicks (physical px positions; draw hit-rects are in
                // the same space since the menu authors at surface size).
                let mut menu_action = None;
                if pressed {
                    if let GameMode::Menu { menu } = &mut state.mode {
                        if let winit::event::ButtonSource::Mouse(mb) = button {
                            if mb == MouseButton::Left {
                                let p = (menu.cursor.0, menu.cursor.1);
                                menu_action = Some(menu.click(p, &self.opts.worlds_dir));
                            }
                        }
                    }
                }
                if let Some(action) = menu_action {
                    if let Some(p) = App::handle_menu_action(&self.opts, state, event_loop, action)
                    {
                        self.pending_world = Some(p);
                    }
                }
                if let winit::event::ButtonSource::Mouse(mb) = button {
                    // Break/place only while captured — clicking with the
                    // cursor free (paused) must not punch blocks.
                    let gate = state.mouse_captured;
                    match mb {
                        MouseButton::Left => state.mouse_held[0] = pressed && gate,
                        MouseButton::Right => state.mouse_held[1] = pressed && gate,
                        _ => {}
                    }
                    if gate && pressed && mb == MouseButton::Left && state.target.is_none() {
                        // Clicked with capture but no block in reach: give a
                        // subtle feedback beat via the crosshair feedback line.
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let lines = match &delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
                    winit::event::MouseScrollDelta::PixelDelta(p) => (p.y / 24.0) as f32,
                    // #[non_exhaustive] future variants: ignore.
                    _ => 0.0,
                };
                // Scroll up (positive) = next slot.
                if lines > 0.0 {
                    state.input.wheel_steps += 1;
                } else if lines < 0.0 {
                    state.input.wheel_steps -= 1;
                }
            }
            WindowEvent::RedrawRequested => {}
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &dyn ActiveEventLoop) {
        let Some(state) = &mut self.state else { return };
        let now = std::time::Instant::now();
        let dt = now.duration_since(state.last_tick).as_secs_f32();
        state.last_tick = now;

        // Frame-time clamp: after alt-tab or a stall, simulate at most
        // MAX_FRAME_DT (the controller subdivides; this protects the
        // streaming budget too).
        let dt = dt.min(controller::MAX_FRAME_DT);

        // FPS/frame-time EMA for the debug screen.
        if dt > 0.0 {
            state.frame_dt_ema = state.frame_dt_ema * 0.9 + dt * 0.1;
            state.fps_ema = state.fps_ema * 0.9 + (1.0 / dt) * 0.1;
        }

        // Paused (unfocused or Esc) → no simulation, but still present.
        if state.focused && !state.paused {
            match &state.mode {
                GameMode::Validation => state.validation_update(dt),
                GameMode::Sandbox => state.sandbox_update(dt),
                GameMode::Menu { .. } => state.menu_update(dt), // panorama backdrop
            }
        }

        // Menu → world transition: rebuild streaming around the chosen save.
        let pending = self.pending_world.take();
        if let Some((dir, seed)) = pending {
            if let Some(atlas) = self.atlas.clone() {
                state.start_world(dir, seed, &atlas);
                if let Some(w) = &state.window {
                    w.set_title("Feathered — sandbox");
                }
            }
        }

        // Upload the title logo once the overlay state exists (set_overlay
        // initializes it lazily; the first frame may miss the art).
        if !self.logo_applied {
            if state
                .renderer
                .as_ref()
                .map(|r| r.overlay_ready())
                .unwrap_or(false)
            {
                if let Some(r) = &mut state.renderer {
                    if let Some((w, h, rgba, _)) = &self.logo {
                        r.set_logo_texture(*w, *h, rgba.clone());
                    }
                    if let Some((w, h, rgba)) = &self.background {
                        r.set_menu_background(*w, *h, rgba.clone());
                    }
                    self.logo_applied = true;
                }
            }
        }

        // HUD + world overlays (screen-space authoring is cheap; skip when
        // the HUD is fully hidden).
        if let GameMode::Menu { menu, .. } = &mut state.mode {
            let (w, h) = (
                state
                    .window
                    .as_ref()
                    .map(|w| w.surface_size().width as f32)
                    .unwrap_or(1280.0),
                state
                    .window
                    .as_ref()
                    .map(|w| w.surface_size().height as f32)
                    .unwrap_or(720.0),
            );
            let mut lists = overlay::HudLists::default();
            menu.draw(&mut lists.screen, w, h);
            let draw = convert_hud(lists);
            if let Some(r) = &mut state.renderer {
                r.set_overlay(draw);
            }
        } else if state.hud_visible || state.debug_visible {
            let hud = state.build_hud();
            if let Some(r) = &mut state.renderer {
                r.set_overlay(hud);
            }
        }

        // Feedback line in the title (duplicates the in-HUD line for
        // window-switchers).
        if let Some((t, msg)) = &state.feedback {
            if t.elapsed().as_secs_f32() < 3.0 {
                if let Some(w) = &state.window {
                    w.set_title(&format!("Feathered — sandbox — {msg}"));
                }
            }
        }

        // One-shot screenshot: capture after 3 warmup frames, then exit.
        if let Some((remaining, path)) = &mut state.screenshot {
            if *remaining > 0 {
                *remaining -= 1;
            } else if let Some(r) = &mut state.renderer {
                r.render_capture(&state.mesh, &state.camera, state.tick);
                match r.capture_last_frame() {
                    Ok(rgba) => {
                        let (w, h) = r.frame_size();
                        let img = image::RgbaImage::from_fn(w, h, |x, y| {
                            let i = (y * w + x) as usize * 4;
                            image::Rgba([rgba[i], rgba[i + 1], rgba[i + 2], 255])
                        });
                        if let Err(e) = img.save(&*path) {
                            eprintln!("screenshot save failed: {e}");
                        } else {
                            println!("screenshot: {} ({}x{})", path.display(), w, h);
                        }
                    }
                    Err(e) => eprintln!("screenshot capture failed: {e}"),
                }
                // The screenshot path bypasses CloseRequested/Esc: persist
                // the sandbox here too so capture runs leave a valid world.
                if matches!(state.mode, GameMode::Sandbox) {
                    state.save_world();
                }
                event_loop.exit();
            }
        }

        if state.screenshot.is_none() {
            if let Some(r) = &mut state.renderer {
                r.render(&state.mesh, &state.camera, state.tick);
            }
        }
    }

    // NOTE: this winit beta exposes no `exiting` hook — save-on-exit happens
    // explicitly at each exit point (Esc quit below, CloseRequested above).
}

impl App {
    /// Route a menu action: quits, live quality changes, delete
    /// confirmations. Returns the world to start (queued by the caller on
    /// `App::pending_world`, started next event-loop pass). An associated fn
    /// (not a method): callers already hold `&mut self.state`.
    fn handle_menu_action(
        opts: &RunOptions,
        state: &mut AppState,
        event_loop: &dyn ActiveEventLoop,
        action: menu::MenuAction,
    ) -> Option<(PathBuf, u64)> {
        match action {
            menu::MenuAction::None => {}
            menu::MenuAction::Quit => {
                event_loop.exit();
            }
            menu::MenuAction::PlayWorld(dir) => {
                let seed = feathered_world::save::load_from_dir(&dir)
                    .map(|s| s.meta.seed)
                    .map_err(|e| eprintln!("world: save unreadable ({e})"))
                    .unwrap_or(opts.seed);
                return Some((dir, seed));
            }
            menu::MenuAction::DeleteWorld(dir) => {
                match std::fs::remove_dir_all(&dir) {
                    Ok(_) => {
                        eprintln!("world deleted: {}", dir.display());
                    }
                    Err(e) => {
                        eprintln!("could not delete {}: {e}", dir.display());
                        if let GameMode::Menu { menu, .. } = &mut state.mode {
                            menu.toast = format!("Delete failed: {e}");
                        }
                        return None;
                    }
                }
                if let GameMode::Menu { menu, .. } = &mut state.mode {
                    menu.scan_worlds(&opts.worlds_dir);
                    menu.world_sel = usize::MAX;
                    menu.toast = "World deleted".into();
                }
            }
            menu::MenuAction::JoinServer(_) => {
                if let GameMode::Menu { menu, .. } = &mut state.mode {
                    menu.toast =
                        "Multiplayer is not implemented yet — servers save for later".into();
                }
            }
            menu::MenuAction::OpenUrl(url) => {
                // Safety gate: only the title footer's own URLs may open.
                if menu::title::ALLOWED_URLS.contains(&url.as_str()) {
                    open_external(&url);
                } else {
                    eprintln!("blocked non-whitelisted URL: {url}");
                }
            }
            menu::MenuAction::QualityChanged(q) => {
                if let Some(r) = &mut state.renderer {
                    let quality = match q.as_str() {
                        "Low" => feathered_renderer::RenderQuality::Low,
                        "High" => feathered_renderer::RenderQuality::High,
                        "Ultra" => feathered_renderer::RenderQuality::Ultra,
                        _ => feathered_renderer::RenderQuality::Medium,
                    };
                    r.set_quality(quality);
                }
                state.settings.quality = q;
            }
        }
        None
    }
}

/// Open a whitelisted external URL through the OS default handler.
/// Cross-platform, no shell interpolation of the URL (argument array on
/// Windows; direct exec elsewhere).
fn open_external(url: &str) {
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("cmd")
            .args(["/c", "start", "", url])
            .spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(url).spawn();
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let _ = std::process::Command::new("xdg-open").arg(url).spawn();
    }
}

/// The ten-block validation scene (Phase 1 world — preserved verbatim).
fn build_validation_world(registry: &Registry) -> World {
    let id = |name: &str| {
        registry
            .block_id(name)
            .unwrap_or_else(|| panic!("block {name} missing"))
    };
    let mut world = World::new([32, 16, 32]);

    let stone = id("stone");
    for x in 0..32 {
        for z in 0..32 {
            world.set(x, 0, z, stone, 0);
        }
    }

    let log = id("oak_log");
    world.set(4, 1, 4, log, 0);
    world.set(6, 1, 4, log, 1);
    world.set(8, 1, 4, log, 2);

    let grass = id("grass_block");
    world.set(4, 1, 8, grass, 0);
    world.set(6, 1, 8, grass, 1);

    let glass = id("glass");
    for y in 1..4 {
        world.set(12, y, 4, glass, 0);
        world.set(13, y, 4, glass, 0);
    }

    let torch = id("torch");
    world.set(4, 1, 12, torch, 0);

    let rail = id("rail");
    for x in 6..10 {
        world.set(x, 1, 12, rail, 0);
    }

    let short_grass = id("short_grass");
    world.set(12, 1, 8, short_grass, 0);
    world.set(13, 1, 9, short_grass, 0);
    world.set(14, 1, 8, short_grass, 0);

    let vine = id("vine");
    for y in 1..4 {
        world.set(16, y, 8, stone, 0);
    }
    world.set(17, 2, 8, vine, 0);
    world.set(17, 3, 8, vine, 0);

    let wire = id("redstone_wire");
    for x in 12..18 {
        world.set(x, 1, 14, wire, 0);
    }

    let water = id("water");
    for x in 20..26 {
        for z in 4..10 {
            world.set(x, 1, z, water, 0);
        }
    }

    world
}
