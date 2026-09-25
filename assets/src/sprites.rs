//! Sprite loading: decode every texture referenced by atlases, attach
//! `.mcmeta` metadata, and register animation strips.

use crate::error::{AssetError, AssetResult};
use crate::meta::{Animation, AnimationRaw};
use crate::texture::DecodedTexture;
use crate::{meta, pack::PackIndex};
use rayon::prelude::*;
use std::collections::HashMap;

/// Identifies a texture inside the pack, e.g. `("minecraft", "block/stone")`.
pub type SpriteRef = (String, String);

/// A sprite whose strip layout has been resolved from its mcmeta + dimensions.
#[derive(Debug, Clone)]
pub struct Sprite {
    pub name: SpriteRef,
    pub tex: DecodedTexture,
    /// (frame width, frame height). Equal to (w, h) for non-animated sprites.
    pub frame_w: u32,
    pub frame_h: u32,
    /// Animation metadata when the mcmeta declares an animation section.
    pub animation: Option<Animation>,
}

impl Sprite {
    /// Number of animation frames (1 for still textures).
    pub fn frame_count(&self) -> u32 {
        match &self.animation {
            Some(a) => a.frame_count(self.tex.height / self.frame_h.max(1)),
            None => 1,
        }
    }

    pub fn is_animated(&self) -> bool {
        self.frame_count() > 1
    }
}

/// All textures decoded from the pack.
pub struct SpriteStore {
    pub sprites: HashMap<SpriteRef, Sprite>,
    /// Sprite names in sorted order (stable atlas packing).
    pub sorted: Vec<SpriteRef>,
}

impl SpriteStore {
    /// Load every texture under `textures/` for the given namespaces.
    pub fn load(index: &PackIndex) -> AssetResult<SpriteStore> {
        let texture_files: Vec<((String, String), std::path::PathBuf)> = index
            .files
            .iter()
            .filter(|(_, path)| path.to_string_lossy().replace('\\', "/").contains("/textures/"))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let loaded: Vec<(SpriteRef, Sprite)> = texture_files
            .par_iter()
            .filter_map(|((ns, path), _file)| {
                // `path` is the indexed stem (extension stripped by discovery):
                // `textures/block/stone` or `textures/block/fire_0.png.mcmeta`.
                let name = path.strip_prefix("textures/").unwrap_or(path.as_str());
                // Skip sidecars (their stem ends in `.png`), fonts, and any
                // non-image payloads.
                if name.ends_with(".mcmeta")
                    || name.ends_with(".png")
                    || name.starts_with("font/")
                {
                    return None;
                }
                match load_sprite(index, ns, name) {
                    Ok(s) => Some(((ns.clone(), name.to_string()), s)),
                    Err(e) => {
                        eprintln!("asset warning: {e}");
                        None
                    }
                }
            })
            .collect();

        let mut sprites = HashMap::new();
        for (name, sprite) in loaded {
            sprites.insert(name, sprite);
        }
        let mut sorted: Vec<SpriteRef> = sprites.keys().cloned().collect();
        sorted.sort();
        Ok(SpriteStore { sprites, sorted })
    }

    pub fn get(&self, ns: &str, name: &str) -> Option<&Sprite> {
        self.sprites.get(&(ns.to_string(), name.to_string()))
    }
}

/// Decode one texture and resolve its animation strip layout.
fn load_sprite(index: &PackIndex, ns: &str, name: &str) -> AssetResult<Sprite> {
    let file = index.get(ns, &format!("textures/{name}")).ok_or_else(|| AssetError {
        path: format!("{ns}:textures/{name}"),
        message: "texture file missing".into(),
    })?;

    // Sidecar lookup. Discovery strips one extension, so `foo.png.mcmeta`
    // lives under the index key `textures/foo.png`.
    let sidecar_key = format!("textures/{name}.png");
    let mcmeta_path = index.get(ns, &sidecar_key).filter(|p| {
        p.to_string_lossy().ends_with(".mcmeta")
    });
    let (tex_meta, anim_raw) = match mcmeta_path {
        Some(p) => {
            let json = std::fs::read_to_string(p).map_err(|e| AssetError {
                path: p.display().to_string(),
                message: format!("cannot read mcmeta: {e}"),
            })?;
            crate::meta::parse_mcmeta(&p.display().to_string(), &json)?
        }
        None => (meta::TextureMeta::default(), None),
    };

    let tex = crate::texture::decode_png(file, tex_meta)?;

    // Resolve the animation strip geometry.
    let (frame_w, frame_h, animation) = match anim_raw {
        Some(raw) => {
            let fw = raw.width.unwrap_or(tex.width);
            let fh = raw.height.unwrap_or(tex.width); // default: square frames
            if fh == 0 || tex.height % fh != 0 || tex.width % fw != 0 {
                return crate::error::err(
                    &format!("{ns}:textures/{name}"),
                    format!(
                        "animation strip {}x{} does not divide into {}x{} frames",
                        tex.width, tex.height, fw, fh
                    ),
                );
            }
            let anim = finalize_animation(raw, tex.height / fh);
            (fw, fh, Some(anim))
        }
        None => (tex.width, tex.height, None),
    };

    Ok(Sprite { name: (ns.to_string(), name.to_string()), tex, frame_w, frame_h, animation })
}

/// Test-visible wrapper for animation finalization.
pub fn frame_count_for_test(raw: &AnimationRaw, strip_frames: u32) -> u32 {
    finalize_animation(raw.clone(), strip_frames).frame_count(strip_frames)
}

/// Validate frame indices and fold per-frame overrides into the Animation.
fn finalize_animation(raw: AnimationRaw, strip_frames: u32) -> Animation {
    // Per-frame time overrides are only used by vanilla's interpolated
    // "advanced" animations; Phase 1 keeps global frametime and validates the
    // frame list.
    if let Some(times) = raw.frame_times() {
        let frames: Vec<u16> = times.iter().map(|(i, _)| *i).collect();
        let valid = frames.iter().all(|i| (*i as u32) < strip_frames);
        if valid {
            return Animation {
                frametime: raw.frametime.unwrap_or(1),
                frames,
                interpolate: raw.interpolate.unwrap_or(false),
            };
        }
        // Fall through: invalid frame list -> sequential fallback.
    }
    raw.into_animation()
}

#[allow(dead_code)]
fn _anim_type_witness(_: &AnimationRaw, _: &Animation) {}
