//! feathered-client — winit window, input, fly camera.

use std::path::PathBuf;

use feathered_assets::cache::cached_to_atlas;
use feathered_world::grid::World;
use feathered_world::Registry;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowAttributes, WindowId};

pub struct RunOptions {
    pub pack_dir: PathBuf,
    pub cache_path: PathBuf,
    /// When set, capture one frame to this PNG and exit (headless validation).
    pub screenshot: Option<PathBuf>,
}

const MOVE_SPEED: f32 = 8.0;
const MOUSE_SENS: f32 = 0.0025;

struct AppState {
    window: Option<Box<dyn Window>>,
    renderer: Option<feathered_renderer::Renderer>,
    #[allow(dead_code)] // kept for Phase 2 (block editing / interaction)
    registry: Registry,
    #[allow(dead_code)]
    atlas: feathered_assets::atlas::Atlas,
    mesh: feathered_chunk::MeshedChunk,
    camera: feathered_renderer::Camera,
    keys: std::collections::HashSet<KeyCode>,
    mouse_captured: bool,
    last_tick: std::time::Instant,
    tick: u64,
    /// One-shot capture: render frame N, save to disk, exit.
    screenshot: Option<(u32, PathBuf)>, // (frames remaining, path)
}

impl AppState {
    fn handle_key(&mut self, key: KeyCode, pressed: bool) {
        if pressed {
            self.keys.insert(key);
        } else {
            self.keys.remove(&key);
        }
    }

    fn update(&mut self, dt: f32) {
        let (mut dx, mut dy, mut dz) = (0.0f32, 0.0f32, 0.0f32);
        if self.keys.contains(&KeyCode::KeyW) { dz += 1.0; }
        if self.keys.contains(&KeyCode::KeyS) { dz -= 1.0; }
        if self.keys.contains(&KeyCode::KeyA) { dx -= 1.0; }
        if self.keys.contains(&KeyCode::KeyD) { dx += 1.0; }
        if self.keys.contains(&KeyCode::Space) { dy += 1.0; }
        if self.keys.contains(&KeyCode::ShiftLeft) { dy -= 1.0; }

        let (sin_y, cos_y) = self.camera.yaw.sin_cos();
        let forward = [sin_y, 0.0, -cos_y];
        let right = [cos_y, 0.0, sin_y];
        for (i, axis) in [forward, right].iter().enumerate() {
            let amount = if i == 0 { dz } else { dx };
            self.camera.pos[i] += axis[i] * amount * MOVE_SPEED * dt;
        }
        self.camera.pos[1] += dy * MOVE_SPEED * dt;
    }
}

struct App {
    state: Option<AppState>,
    cache_path: PathBuf,
    screenshot: Option<PathBuf>,
}

impl ApplicationHandler for App {
    fn can_create_surfaces(&mut self, event_loop: &dyn ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = WindowAttributes::default()
            .with_title("Feathered — Phase 1 validation scene")
            .with_surface_size(winit::dpi::LogicalSize::new(1280.0, 720.0));
        let window = event_loop.create_window(attrs).expect("create window");

        // Load cache (already compiled by `feathered compile-pack`).
        let blob = std::fs::read(&self.cache_path).unwrap_or_else(|e| {
            panic!(
                "cannot read cache {}: {e} (run `feathered compile-pack` first)",
                self.cache_path.display()
            )
        });
        let (_, payload) = feathered_assets::cache::decode(&blob).expect("cache decode");
        let registry = Registry::from_compiled(payload.pack);
        let atlas = cached_to_atlas(&payload.atlas, registry.sprite_names());

        let size = window.surface_size();
        let renderer = pollster::block_on(feathered_renderer::Renderer::new(
            window,
            size.width.max(1),
            size.height.max(1),
            &atlas,
            feathered_renderer::build_anim_slots(&registry, &atlas),
        ));

        let world = build_validation_world(&registry);
        let mesh = feathered_renderer::build_meshes(&world, &registry, &atlas);
        eprintln!(
            "[dbg] mesh counts: opaque {}v/{}i, cutout {}v/{}i, translucent {}v/{}i",
            mesh.opaque.vertices.len(), mesh.opaque.indices.len(),
            mesh.cutout.vertices.len(), mesh.cutout.indices.len(),
            mesh.translucent.vertices.len(), mesh.translucent.indices.len()
        );
        let vp = feathered_renderer::Camera { pos: [16.0, 10.0, 30.0], yaw: 0.0, pitch: -0.3, fov_y: 70.0_f32.to_radians(), aspect: 1280.0/720.0, near: 0.1, far: 400.0 }.view_proj();
        eprintln!("[dbg] view_proj row0={:?} row3={:?}", vp[0], vp[3]);

        self.state = Some(AppState {
            window: None, // wgpu owns the surface; redraws come from about_to_wait
            renderer: Some(renderer),
            registry,
            atlas,
            mesh,
            camera: feathered_renderer::Camera {
                pos: [16.0, 14.0, 44.0],
                yaw: 0.0,
                pitch: -0.28,
                fov_y: 70.0_f32.to_radians(),
                aspect: 1280.0 / 720.0,
                near: 0.1,
                far: 400.0,
            },
            keys: Default::default(),
            mouse_captured: false,
            last_tick: std::time::Instant::now(),
            tick: 0,
            screenshot: self.screenshot.clone().map(|p| (1, p)),
        });
    }

    fn window_event(&mut self, event_loop: &dyn ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::SurfaceResized(size) => {
                if let Some(r) = &mut state.renderer {
                    r.resize(size.width.max(1), size.height.max(1));
                }
                state.camera.aspect = size.width as f32 / size.height.max(1) as f32;
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Escape) => event_loop.exit(),
                        PhysicalKey::Code(KeyCode::KeyE) => {
                            state.mouse_captured = !state.mouse_captured;
                            if let Some(w) = &state.window {
                                let _ = w.set_cursor_grab(
                                    if state.mouse_captured {
                                        winit::window::CursorGrabMode::Confined
                                    } else {
                                        winit::window::CursorGrabMode::None
                                    },
                                );
                            }
                        }
                        _ => {}
                    }
                }
                if let PhysicalKey::Code(code) = event.physical_key {
                    state.handle_key(code, event.state == ElementState::Pressed);
                }
            }
            WindowEvent::PointerMoved { position, .. } => {
                if state.mouse_captured {
                    if let Some(w) = &state.window {
                        let size = w.surface_size();
                        let center = (position.x - size.width as f64 / 2.0,
                                      position.y - size.height as f64 / 2.0);
                        state.camera.yaw += center.0 as f32 * MOUSE_SENS;
                        state.camera.pitch -= center.1 as f32 * MOUSE_SENS;
                        state.camera.pitch = state
                            .camera
                            .pitch
                            .clamp(-1.5, 1.5);
                        let _ = w.set_cursor_position(winit::dpi::Position::Physical(
                            winit::dpi::PhysicalPosition::new(
                                size.width as i32 / 2,
                                size.height as i32 / 2,
                            ),
                        ));
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                // Rendering happens in about_to_wait; nothing to do here.
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &dyn ActiveEventLoop) {
        // Continuous redraw: Phase 1 renders every loop iteration.
        let Some(state) = &mut self.state else { return };
        let now = std::time::Instant::now();
        let dt = now.duration_since(state.last_tick).as_secs_f32();
        state.last_tick = now;
        state.tick += (dt * 20.0) as u64;
        state.update(dt);

        // One-shot screenshot: capture after 3 warmup frames, then exit.
        if let Some((remaining, path)) = &mut state.screenshot {
            if *remaining > 0 {
                *remaining -= 1;
            } else if let Some(r) = &mut state.renderer {
                r.render_capture(&state.mesh, &state.camera, state.tick);
                match r.capture_last_frame() {
                    Ok(rgba) => {
                        let (w, h) = r.frame_size();
                        // Offscreen capture target is Rgba8Unorm: rows map
                        // straight into the RGBA PNG.
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
                event_loop.exit();
            }
        }

        if state.screenshot.is_none() {
            if let Some(r) = &mut state.renderer {
                let _ = r.render(&state.mesh, &state.camera, state.tick);
            }
        }
    }
}

/// The ten-block validation scene (Phase 1 world).
fn build_validation_world(registry: &Registry) -> World {
    let id = |name: &str| registry.block_id(name).unwrap_or_else(|| panic!("block {name} missing"));
    let mut world = World::new([32, 16, 32]);

    // Stone ground.
    let stone = id("stone");
    for x in 0..32 {
        for z in 0..32 {
            world.set(x, 0, z, stone, 0);
        }
    }

    // Oak logs: all three axes.
    let log = id("oak_log");
    world.set(4, 1, 4, log, 0); // axis=y
    world.set(6, 1, 4, log, 1); // axis=x (schema order: x,y,z -> state 1)
    world.set(8, 1, 4, log, 2); // axis=z

    // Grass blocks (normal + snowy via variant ordering).
    let grass = id("grass_block");
    world.set(4, 1, 8, grass, 0);
    world.set(6, 1, 8, grass, 1);

    // Glass pane wall (translucent check).
    let glass = id("glass");
    for y in 1..4 {
        world.set(12, y, 4, glass, 0);
        world.set(13, y, 4, glass, 0);
    }

    // Torch.
    let torch = id("torch");
    world.set(4, 1, 12, torch, 0);

    // Rail.
    let rail = id("rail");
    for x in 6..10 {
        world.set(x, 1, 12, rail, 0);
    }

    // Short grass (cross).
    let short_grass = id("short_grass");
    world.set(12, 1, 8, short_grass, 0);
    world.set(13, 1, 9, short_grass, 0);
    world.set(14, 1, 8, short_grass, 0);

    // Vine on a stone pillar (multipart).
    let vine = id("vine");
    for y in 1..4 {
        world.set(16, y, 8, stone, 0);
    }
    world.set(17, 2, 8, vine, 0);
    world.set(17, 3, 8, vine, 0);

    // Redstone wire (multipart, property states).
    let wire = id("redstone_wire");
    for x in 12..18 {
        world.set(x, 1, 14, wire, 0);
    }

    // Water pool (animated).
    let water = id("water");
    for x in 20..26 {
        for z in 4..10 {
            world.set(x, 1, z, water, 0);
        }
    }

    world
}

/// Launch the validation scene.
pub fn run(opts: RunOptions) -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    let app = App {
        state: None,
        cache_path: opts.cache_path,
        screenshot: opts.screenshot,
    };
    event_loop.run_app(app)?;
    Ok(())
}
