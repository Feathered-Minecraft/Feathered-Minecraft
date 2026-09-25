//! PNG decoding. Every texture in the pack is normalized to RGBA8 here so no
//! downstream stage ever sees indexed, grayscale or 16-bit pixel data.

use crate::error::{AssetError, AssetResult};
use image::DynamicImage;

/// A decoded texture: RGBA8, row-major, top-to-bottom.
#[derive(Debug, Clone)]
pub struct DecodedTexture {
    pub width: u32,
    pub height: u32,
    /// `width * height * 4` bytes.
    pub rgba: Vec<u8>,
    /// Metadata parsed from the `.mcmeta` sidecar, if one exists.
    pub meta: crate::meta::TextureMeta,
}

impl DecodedTexture {
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * self.width + x) * 4) as usize;
        [
            self.rgba[i],
            self.rgba[i + 1],
            self.rgba[i + 2],
            self.rgba[i + 3],
        ]
    }

    /// True when every pixel satisfies r≈g≈b (a tint candidate).
    pub fn is_grayscale_content(&self) -> bool {
        self.rgba
            .chunks_exact(4)
            .all(|p| p[0].abs_diff(p[1]) <= 2 && p[1].abs_diff(p[2]) <= 2)
    }
}

/// Decode a PNG (or any image crate supported format) into `DecodedTexture`.
pub fn decode_png(path: &Path, meta: crate::meta::TextureMeta) -> AssetResult<DecodedTexture> {
    let bytes = std::fs::read(path).map_err(|e| AssetError {
        path: path.display().to_string(),
        message: format!("cannot read file: {e}"),
    })?;
    let img = image::load_from_memory(&bytes).map_err(|e| AssetError {
        path: path.display().to_string(),
        message: format!("image decode failed: {e}"),
    })?;
    let (width, height, rgba) = match img {
        DynamicImage::ImageRgba8(buf) => {
            let (w, h) = buf.dimensions();
            (w, h, buf.into_raw())
        }
        other => {
            let buf = other.to_rgba8();
            let (w, h) = buf.dimensions();
            (w, h, buf.into_raw())
        }
    };
    Ok(DecodedTexture { width, height, rgba, meta })
}

pub use std::path::Path;
