//! Texture metadata decoded from `.mcmeta` sidecar files.
//!
//! Two independent sections exist in the pack:
//! * `animation` — vertical strip layout, frame order and timing.
//! * `texture` — sampling/mip-generation flags (blur, clamp, mip strategies).

use serde::Deserialize;

/// How mip levels are generated for a texture with transparent pixels.
/// Values observed in the 26.3 pack: `strict_cutout`, `dark_cutout`, `mean`.
/// Exact vanilla algorithms are undocumented; our implementations are
/// approximations chosen from the names (see atlas.rs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MipStrategy {
    /// Plain box-filter average of RGBA (also used for fully opaque textures).
    #[default]
    Mean,
    /// A mip texel that touched any transparent pixel becomes fully transparent.
    StrictCutout,
    /// A mip texel that touched any transparent pixel blends toward opaque black
    /// (prevents cutout halos from brightening at distance).
    DarkCutout,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TextureMeta {
    /// Linear filtering requested (glint, vignette).
    pub blur: bool,
    /// Clamp-to-edge sampling requested (entity shadow).
    pub clamp: bool,
    /// Alpha value added to the cutout threshold (cactus, kelp, tripwire).
    pub alpha_cutoff_bias: f32,
    /// Mip generation policy.
    pub mip_strategy: MipStrategy,
}

/// Frame timing/order of a vertical-strip animation.
#[derive(Debug, Clone)]
pub struct Animation {
    /// Game ticks each frame is shown (default 1). Interpolated animations
    /// blend across this duration.
    pub frametime: u32,
    /// Playback order; indices into the strip rows. Empty = sequential.
    pub frames: Vec<u16>,
    /// Linear-blend toward the next frame instead of hard cuts.
    pub interpolate: bool,
}

impl Animation {
    /// Number of frames the animation plays through.
    pub fn frame_count(&self, strip_rows: u32) -> u32 {
        if self.frames.is_empty() {
            strip_rows
        } else {
            self.frames.len() as u32
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct AnimationRaw {
    pub frametime: Option<u32>,
    pub interpolate: Option<bool>,
    /// Raw frame entries: integers or `{"index": n, "time": t}` objects.
    pub frames: Option<Vec<serde_json::Value>>,
    /// Declared frame height in texels (default: image width).
    pub height: Option<u32>,
    /// Declared frame width in texels (default: image width).
    pub width: Option<u32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct AnimationJson {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    frametime: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    interpolate: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    frames: Option<Vec<serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    height: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    width: Option<u32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TextureJson {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    blur: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    clamp: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    mip_strategy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    alpha_cutoff_bias: Option<f32>,
}

#[derive(Debug, Default, Deserialize)]
struct McmetaJson {
    #[serde(default)]
    animation: Option<AnimationJson>,
    #[serde(default)]
    texture: Option<TextureJson>,
}

impl AnimationRaw {
    /// Resolve the generic `{ "index": N, "time": T }` frame objects into
    /// `(index, time)` pairs used by interpolating animations.
    pub fn frame_times(&self) -> Option<Vec<(u16, u32)>> {
        self.frames.as_ref()?;
        let mut out = Vec::new();
        for v in self.frames.as_ref().unwrap() {
            match v {
                serde_json::Value::Number(n) => out.push((n.as_u64()? as u16, 0)),
                serde_json::Value::Object(o) => {
                    let index = o.get("index")?.as_u64()? as u16;
                    let time = o.get("time").and_then(|t| t.as_u64()).unwrap_or(0) as u32;
                    out.push((index, time));
                }
                _ => return None,
            }
        }
        Some(out)
    }

    pub fn into_animation(self) -> Animation {
        Animation {
            frametime: self.frametime.unwrap_or(1),
            frames: self
                .frames
                .map(|f| {
                    f.into_iter()
                        .filter_map(|v| v.as_u64().map(|n| n as u16))
                        .collect()
                })
                .unwrap_or_default(),
            interpolate: self.interpolate.unwrap_or(false),
        }
    }
}

/// Parse a `.mcmeta` sidecar JSON document.
pub fn parse_mcmeta(path: &str, json: &str) -> AssetResult<(TextureMeta, Option<AnimationRaw>)> {
    let doc: McmetaJson = serde_json::from_str(json)
        .map_err(|e| crate::error::AssetError { path: path.into(), message: e.to_string() })?;
    let mut meta = TextureMeta {
        alpha_cutoff_bias: doc.texture.as_ref().and_then(|t| t.alpha_cutoff_bias).unwrap_or(0.0),
        ..TextureMeta::default()
    };
    if let Some(tex) = &doc.texture {
        if tex.blur.unwrap_or(false) {
            meta.blur = true;
        }
        if tex.clamp.unwrap_or(false) {
            meta.clamp = true;
        }
        meta.mip_strategy = match tex.mip_strategy.as_deref() {
            Some("strict_cutout") => MipStrategy::StrictCutout,
            Some("dark_cutout") => MipStrategy::DarkCutout,
            Some("mean") | None => MipStrategy::Mean,
            Some(other) => {
                return crate::error::err(
                    path,
                    format!("unknown mipmap_strategy `{other}`"),
                )
            }
        };
    }
    let anim = doc.animation.map(|a| AnimationRaw {
        frametime: a.frametime,
        interpolate: a.interpolate,
        frames: a.frames,
        height: a.height,
        width: a.width,
    });
    Ok((meta, anim))
}

use crate::error::AssetResult;
