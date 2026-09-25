//! Atlas construction. All sprites referenced by block models are packed into
//! one RGBA8 texture. Every sprite (and every frame of an animation strip) is
//! blitted with a 1-texel clamp-extended gutter so sampling and mip generation
//! never bleed between neighbors.
//!
//! Animated sprites occupy one logical slot of `frames` rows; the renderer
//! offsets UVs down the strip by `frame_stride = frame_h + GUTTER` per frame
//! instead of expanding frames into separate atlas entries.

use crate::error::{AssetError, AssetResult};
use crate::sprites::{Sprite, SpriteStore, SpriteRef};
use std::collections::HashMap;

pub const GUTTER: u32 = 1;

/// Where one sprite lives inside the atlas texture.
#[derive(Debug, Clone, Copy)]
pub struct AtlasEntry {
    /// Top-left texel of the sprite's *frame 0* (gutter already inset).
    pub x: u32,
    pub y: u32,
    pub frame_w: u32,
    pub frame_h: u32,
    /// Animation rows stacked in this slot (1 = still sprite).
    pub frames: u32,
    /// Vertical distance between frame origins (= frame_h + GUTTER).
    pub frame_stride: u32,
    /// Horizontal pitch between slots in the atlas.
    pub slot_stride: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layer {
    Opaque,
    Cutout,
    Translucent,
}

/// Compact runtime animation descriptor (tick-driven).
#[derive(Debug, Clone)]
pub struct AnimTable {
    /// Game ticks each frame is shown.
    pub frametime: u32,
    /// Playback order (indices into the strip); empty = sequential 0..n-1.
    pub frames: Vec<u16>,
    pub interpolate: bool,
}

impl AnimTable {
    /// Strip row shown at `tick`.
    pub fn frame_at(&self, tick: u64, strip_frames: u32) -> u32 {
        let seq_len = if self.frames.is_empty() {
            strip_frames
        } else {
            self.frames.len() as u32
        };
        let idx = ((tick / self.frametime.max(1) as u64) % seq_len.max(1) as u64) as usize;
        if self.frames.is_empty() {
            idx as u32
        } else {
            self.frames[idx] as u32
        }
    }
}

/// The compiled atlas: pixels, mip chain, layout and animation timelines.
pub struct Atlas {
    pub width: u32,
    pub height: u32,
    /// RGBA8, mip level 0.
    pub pixels: Vec<u8>,
    /// Mip images starting at level 1 (length = mip_levels - 1).
    pub mips: Vec<Vec<u8>>,
    pub entries: HashMap<SpriteRef, AtlasEntry>,
    pub animations: HashMap<SpriteRef, AnimTable>,
}

impl Atlas {
    pub fn mip_levels(&self) -> u32 {
        self.mips.len() as u32 + 1
    }

    pub fn get(&self, ns: &str, name: &str) -> Option<&AtlasEntry> {
        self.entries.get(&(ns.to_string(), name.to_string()))
    }

    pub fn anim(&self, ns: &str, name: &str) -> Option<&AnimTable> {
        self.animations.get(&(ns.to_string(), name.to_string()))
    }
}

/// Simple shelf packer: rows of increasing height, sprites tall-first.
struct ShelfPacker {
    width: u32,
    y: u32,
    row_h: u32,
    row_x: u32,
}

impl ShelfPacker {
    fn new(width: u32) -> Self {
        Self { width, y: 0, row_h: 0, row_x: 0 }
    }

    fn place(&mut self, w: u32, h: u32) -> Option<(u32, u32)> {
        let cell_w = w + 2 * GUTTER;
        let cell_h = h + 2 * GUTTER;
        if cell_w > self.width {
            return None;
        }
        if self.row_x + cell_w > self.width {
            self.y += self.row_h;
            self.row_h = 0;
            self.row_x = 0;
        }
        let (x, y) = (self.row_x, self.y);
        self.row_x += cell_w;
        self.row_h = self.row_h.max(cell_h);
        Some((x + GUTTER, y + GUTTER))
    }
}

/// Build the atlas from the sprites the block models reference.
///
/// `sprite_names` must be the resolved closure of every texture referenced by
/// every block model (the model compiler produces this list).
pub fn build(store: &SpriteStore, sprite_names: &[SpriteRef]) -> AssetResult<Atlas> {
    // Slot geometry per sprite: full strip height (frames stacked), guttered.
    let slot_w = |s: &Sprite| s.tex.width + 2 * GUTTER;
    let slot_h = |s: &Sprite| s.tex.height + 2 * GUTTER;

    let mut sorted: Vec<&SpriteRef> = sprite_names.iter().collect();
    sorted.sort();
    let mut items: Vec<&&SpriteRef> = sorted.iter().collect();
    items.sort_by_key(|name| {
        store
            .get(&name.0, &name.1)
            .map(|s| -(slot_h(s) as i64))
            .unwrap_or(0)
    });

    // Pack into growing square atlases.
    let mut placed: Vec<(SpriteRef, AtlasEntry)> = Vec::new();
    let mut atlas_w = 512u32;
    let mut atlas_h = 512u32;
    'grow: for _attempt in 0..6 {
        atlas_w = 512u32 << (_attempt.min(2) as u32);
        atlas_h = atlas_w;
        let mut packer = ShelfPacker::new(atlas_w);
        placed.clear();
        for name in &items {
            let sprite = store.get(&name.0, &name.1).ok_or_else(|| AssetError {
                path: format!("{}:{}", name.0, name.1),
                message: "sprite missing from store".into(),
            })?;
            let (w, h) = (slot_w(sprite), slot_h(sprite));
            let Some((x, y)) = packer.place(w, h) else {
                // out of width — grow and restart
                continue 'grow;
            };
            if y + h > atlas_h {
                continue 'grow;
            }
            let frames = sprite.frame_count();
            let frame_h = if frames > 1 { sprite.frame_h } else { sprite.tex.height };
            placed.push((
                (**name).clone(),
                AtlasEntry {
                    x,
                    y,
                    frame_w: sprite.frame_w,
                    frame_h,
                    frames,
                    frame_stride: frame_h + GUTTER,
                    slot_stride: w,
                },
            ));
        }
        break;
    }

    if placed.len() != items.len() {
        return crate::error::err("atlas", "failed to pack all sprites (atlas too large)");
    }

    let mut pixels = vec![0u8; (atlas_w * atlas_h * 4) as usize];

    let mut animations = HashMap::new();
    for (name, entry) in &placed {
        let sprite = store.get(&name.0, &name.1).unwrap();
        if entry.frames > 1 {
            // Blit every frame as its own guttered sub-slot (stacked rows).
            for f in 0..entry.frames {
                let y_off = f * entry.frame_stride;
                blit_frame_guttered(
                    &mut pixels,
                    atlas_w,
                    atlas_h,
                    &sprite.tex.rgba,
                    sprite.tex.width,
                    sprite.tex.height,
                    entry.frame_h,
                    f,
                    entry.x,
                    entry.y + y_off,
                );
            }
            let a = sprite.animation.as_ref();
            animations.insert(
                name.clone(),
                AnimTable {
                    frametime: a.map(|x| x.frametime).unwrap_or(1),
                    frames: a.map(|x| x.frames.clone()).unwrap_or_default(),
                    interpolate: a.map(|x| x.interpolate).unwrap_or(false),
                },
            );
        } else {
            blit_frame_guttered(
                &mut pixels,
                atlas_w,
                atlas_h,
                &sprite.tex.rgba,
                sprite.tex.width,
                sprite.tex.height,
                entry.frame_h,
                0,
                entry.x,
                entry.y,
            );
        }
    }

    // Alpha-aware mip chain: cutout sprites must not smear into halos.
    // Conservative policy: alpha-weighted average with alpha-sharpening
    // (mean alpha over 0.5 -> opaque; else 0) for mips >= 1. Textures with
    // `mipmap_strategy: mean` (glass, redstone dust) are marked later via
    // metadata if we adopt per-sprite mip policies; the sharpened default is
    // visually correct for leaves/grass and acceptable for glass.
    let mip_count = ((atlas_w.min(atlas_h) as u32).trailing_zeros()).clamp(1, 3);
    let mut mips = Vec::with_capacity(mip_count as usize);
    let mut prev = pixels.clone();
    let (mut w, mut h) = (atlas_w, atlas_h);
    for _ in 0..mip_count {
        w = (w / 2).max(1);
        h = (h / 2).max(1);
        let next = downsample_cutout(&prev, w * 2, h * 2, w, h);
        mips.push(next.clone());
        prev = next;
    }

    Ok(Atlas {
        width: atlas_w,
        height: atlas_h,
        pixels,
        mips,
        entries: placed.into_iter().collect(),
        animations,
    })
}

/// Blit frame `frame` of a strip with a clamp-extended gutter around it.
#[allow(clippy::too_many_arguments)]
fn blit_frame_guttered(
    dst: &mut [u8],
    dst_w: u32,
    dst_h: u32,
    src: &[u8],
    src_w: u32,
    src_h: u32,
    frame_h: u32,
    frame: u32,
    x0: u32,
    y0: u32,
) {
    let src_y0 = frame * frame_h;
    for dy in 0..frame_h + 2 * GUTTER {
        for dx in 0..src_w + 2 * GUTTER {
            let sx = (dx as i64 - GUTTER as i64).clamp(0, src_w as i64 - 1) as u32;
            let sy_raw = (dy as i64 - GUTTER as i64).clamp(0, frame_h as i64 - 1) as u32;
            let sy = (src_y0 + sy_raw).min(src_h.max(1) - 1);
            let si = ((sy * src_w + sx) * 4) as usize;
            let px = x0 + dx - GUTTER;
            let py = y0 + dy - GUTTER;
            if px >= dst_w || py >= dst_h {
                continue;
            }
            let di = ((py * dst_w + px) * 4) as usize;
            if let (Some(s), Some(d)) = (src.get(si..si + 4), dst.get_mut(di..di + 4)) {
                d.copy_from_slice(s);
            }
        }
    }
}

/// 2x2 box filter, alpha-weighted. Output alpha is sharpened to 0 or 255 so
/// cutout textures keep their silhouette through the mip chain; RGB is the
/// alpha-weighted average of contributing texels (transparent texels do not
/// darken the result).
pub fn downsample_cutout(src: &[u8], w: u32, h: u32, dw: u32, dh: u32) -> Vec<u8> {
    let mut out = vec![0u8; (dw * dh * 4) as usize];
    for dy in 0..dh {
        for dx in 0..dw {
            let (sx, sy) = (dx * 2, dy * 2);
            let (mut r, mut g, mut b, mut a, mut n) = (0u32, 0u32, 0u32, 0u32, 0u32);
            for oy in 0..2 {
                for ox in 0..2 {
                    let x = (sx + ox).min(w - 1);
                    let y = (sy + oy).min(h - 1);
                    let i = ((y * w + x) * 4) as usize;
                    let alpha = src[i + 3] as u32;
                    r += src[i] as u32 * alpha;
                    g += src[i + 1] as u32 * alpha;
                    b += src[i + 2] as u32 * alpha;
                    a += alpha;
                    n += 1;
                }
            }
            let di = ((dy * dw + dx) * 4) as usize;
            // Sharpen: any transparency kills the texel (conservative cutout).
            let alpha_out = if a > 0 && a >= n * 255 / 2 { 255 } else { 0 };
            if a > 0 {
                out[di] = (r / a) as u8;
                out[di + 1] = (g / a) as u8;
                out[di + 2] = (b / a) as u8;
            }
            out[di + 3] = alpha_out;
        }
    }
    out
}
