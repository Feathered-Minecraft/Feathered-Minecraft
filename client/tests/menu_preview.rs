//! Headless previews of the client's menu screens (GPU iteration loop).
//!
//! `menu_title_preview` renders the real `MenuState::draw` output at
//! 1280×720 and dumps target/menu-ui-preview.png so the title layout can be
//! compared against the reference mock without launching a window.
//! `textured_quads_sample_the_logo` pins the textured-quad path: with a
//! tiny two-tone texture uploaded, a textured quad must show texel colors
//! (not the flat tint, not nothing).

use feathered_renderer::{Camera, HudDraw, RenderQuality, RenderSettings};

fn headless_renderer() -> feathered_renderer::Renderer {
    // The atlas only feeds terrain pipelines; menus are pure overlay, so a
    // 1×1 empty atlas is enough (no pack files needed).
    let atlas = feathered_assets::atlas::Atlas {
        width: 1,
        height: 1,
        pixels: vec![0; 4],
        mips: Vec::new(),
        entries: Default::default(),
        animations: Default::default(),
    };
    pollster::block_on(feathered_renderer::Renderer::headless(
        1280,
        720,
        &atlas,
        Default::default(),
        RenderSettings { quality: RenderQuality::Medium, shader_pack: None },
    ))
}

fn save_png(path: &str, frame: &[u8], w: u32, h: u32) {
    // Workspace root (client/ is one level deep): two climbs from the
    // manifest dir land on <root>/target.
    let full = format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), path);
    if let Some(parent) = std::path::Path::new(&full).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let img = image::RgbaImage::from_fn(w, h, |x, y| {
        let i = ((y * w + x) * 4) as usize;
        image::Rgba([frame[i], frame[i + 1], frame[i + 2], 255])
    });
    img.save(&full).expect("dump png");
}

const W: u32 = 1280;
const H: u32 = 720;

/// Render the real title screen exactly as the app would author it.
#[test]
#[ignore = "requires a GPU adapter (wgpu)"]
fn menu_title_preview() {
    let mut r = headless_renderer();
    let dir = std::env::temp_dir().join(format!("feathered-ui-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let mut menu = feathered_client::menu::MenuState::new(
        &dir,
        feathered_client::settings::Settings::default(),
        feathered_client::profile::Profile::default(),
    );
    menu.cursor = (400.0, 500.0); // hover the singleplayer button
    // Logo art: the real logo.png when present (same lookup as the app's
    // run()), else a synthetic orange gradient so the test stays hermetic.
    let logo_path = format!("{}/../../logo.png", env!("CARGO_MANIFEST_DIR"));
    match std::fs::read(&logo_path)
        .ok()
        .and_then(|bytes| image::load_from_memory(&bytes).ok().map(|i| i.to_rgba8()))
    {
        Some(rgba) => {
            let (lw, lh) = rgba.dimensions();
            let (px, lw, lh) = feathered_client::menu::crop_to_alpha(rgba.into_raw(), lw, lh);
            menu.logo_art = Some(lw as f32 / lh as f32);
            r.set_overlay(HudDraw::default());
            r.set_logo_texture(lw, lh, px);
        }
        None => {
            let mut tex = vec![0u8; 64 * 64 * 4];
            for (i, px) in tex.chunks_exact_mut(4).enumerate() {
                let x = (i % 64) as f32 / 64.0;
                let y = (i / 64) as f32 / 64.0;
                px[0] = (120.0 + 135.0 * x) as u8;
                px[1] = (40.0 + 80.0 * y) as u8;
                px[2] = 64;
                px[3] = (255.0 * (0.25 + 0.75 * x * y)) as u8;
            }
            menu.logo_art = Some(1.0);
            r.set_overlay(HudDraw::default());
            r.set_logo_texture(64, 64, tex);
        }
    }

    let mut lists = feathered_client::overlay::HudLists::default();
    menu.draw(&mut lists.screen, W as f32, H as f32);
    let draw = HudDraw {
        screen: convert(&lists.screen),
        world: convert(&lists.world),
    };
    r.render_capture(&feathered_chunk::MeshedChunk::default(), &sample_camera(), 0);
    r.set_overlay(draw);
    // Second capture consumes the pending overlay set after render_capture.
    r.render_capture(&feathered_chunk::MeshedChunk::default(), &sample_camera(), 0);
    let frame = r.capture_last_frame().expect("frame");
    assert_eq!(r.frame_size(), (W, H));
    save_png("target/menu-ui-preview.png", &frame, W, H);

    // Fade panel: left edge near-black; right side lighter (fade eases out
    // over the sky clear color). No background.png in tests → sky shows.
    let px = |x: u32, y: u32| {
        let i = ((y * W + x) * 4) as usize;
        [frame[i], frame[i + 1], frame[i + 2]]
    };
    let left = px(8, 8);
    let right = px(W - 60, H - 60);
    let ll = left.iter().map(|&c| c as u32).sum::<u32>();
    let rl = right.iter().map(|&c| c as u32).sum::<u32>();
    assert!(
        ll < 100,
        "fade left edge should be near-black, got {left:?}"
    );
    assert!(
        rl > ll,
        "fade should ease out to the right (left {left:?} vs right {right:?})"
    );
    // Feather band upper-left: brighter than the panel (white art + tint).
    let feather = px(120, 120);
    let fl = feather.iter().map(|&c| c as u32).sum::<u32>();
    assert!(
        fl > ll + 40,
        "feather art should brighten the panel, got {feather:?}"
    );
}

/// Textured quads must sample the uploaded texture (tint × texel).
#[test]
#[ignore = "requires a GPU adapter (wgpu)"]
fn textured_quads_sample_the_logo() {
    let mut r = headless_renderer();
    // Left half black, right half white, alpha 255.
    let mut tex = vec![0u8; 8 * 8 * 4];
    for y in 0..8 {
        for x in 0..8 {
            let i = (y * 8 + x) * 4;
            let v = if x < 4 { 0 } else { 255 };
            tex[i..i + 4].copy_from_slice(&[v, v, v, 255]);
        }
    }
    r.set_overlay(HudDraw::default());
    r.set_logo_texture(8, 8, tex);

    let mut list = feathered_client::overlay::TriList::default();
    // Textured quad covering the left half of the frame.
    list.textured_quad([0.0, 360.0], [640.0, 360.0], [640.0, 720.0], [0.0, 720.0], [255, 255, 255, 255]);
    // Flat sentinel quad on the right half (must stay flat: sentinel UV).
    list.quad([640.0, 0.0], [1280.0, 0.0], [1280.0, 360.0], [640.0, 360.0], [0, 200, 0, 255]);
    let draw = HudDraw { screen: convert(&list), world: Default::default() };
    r.render_capture(&feathered_chunk::MeshedChunk::default(), &sample_camera(), 0);
    r.set_overlay(draw);
    r.render_capture(&feathered_chunk::MeshedChunk::default(), &sample_camera(), 0);
    let frame = r.capture_last_frame().expect("frame");
    let px = |x: u32, y: u32| {
        let i = ((y * W + x) * 4) as usize;
        [frame[i], frame[i + 1], frame[i + 2]]
    };
    // Textured half: sky clear color × black texels (left) vs white texels
    // (right side of the textured quad).
    let tl = px(100, 400);
    let tr = px(600, 400);
    assert!(tl.iter().all(|&c| c < 10), "black texels, got {tl:?}");
    assert!(tr.iter().all(|&c| c > 245), "white texels, got {tr:?}");
    // Flat half: green, unaffected by the texture.
    let flat = px(900, 180);
    assert!(flat[1] > 180 && flat[0] < 30, "flat quad stays flat, got {flat:?}");
}

fn convert(
    l: &feathered_client::overlay::TriList,
) -> feathered_renderer::client_overlay::TriList {
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
        indices: l.indices.clone(),
    }
}

fn sample_camera() -> Camera {
    Camera {
        pos: [0.0, 80.0, 0.0],
        yaw: 0.0,
        pitch: 0.0,
        fov_y: 70.0f32.to_radians(),
        aspect: W as f32 / H as f32,
        near: 0.1,
        far: 400.0,
    }
}
