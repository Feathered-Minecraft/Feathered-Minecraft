//! Tint system. Grayscale-content sprites are multiplied by tint colors at
//! mesh time. Colors come from:
//! * `textures/colormap/grass.png`, `foliage.png`, `dry_foliage.png`
//!   (256x256, indexed by (temperature, downfall) biome coordinates), or
//! * fixed constant colors for non-biome tints (water, redstone, etc.).

use crate::error::{AssetError, AssetResult};
use crate::sprites::SpriteStore;
use crate::texture::DecodedTexture;

/// Which colormap a tinted face samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Colormap {
    Grass,
    Foliage,
    DryFoliage,
}

/// A resolved tint color (sRGB, 0-255).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TintColor(pub [u8; 3]);

/// The tint lookup tables.
pub struct TintSystem {
    grass: Option<DecodedTexture>,
    foliage: Option<DecodedTexture>,
    dry_foliage: Option<DecodedTexture>,
}

impl TintSystem {
    /// Load the three colormaps from the pack (optional — defaults kick in).
    pub fn load(store: &SpriteStore) -> Self {
        let get = |name: &str| {
            store
                .get("minecraft", &format!("colormap/{name}"))
                .map(|s| s.tex.clone())
        };
        Self {
            grass: get("grass"),
            foliage: get("foliage"),
            dry_foliage: get("dry_foliage"),
        }
    }

    /// Sample a colormap at biome coordinates (both 0..1). Phase 1 uses fixed
    /// "plains" coordinates (0.5, 0.5) chosen by the validation scene.
    pub fn sample(&self, cm: Colormap, temperature: f32, downfall: f32) -> TintColor {
        let tex = match cm {
            Colormap::Grass => self.grass.as_ref(),
            Colormap::Foliage => self.foliage.as_ref(),
            Colormap::DryFoliage => self.dry_foliage.as_ref(),
        };
        let Some(tex) = tex else {
            return match cm {
                Colormap::Grass => TintColor([145, 189, 89]),   // plains fallback
                Colormap::Foliage => TintColor([119, 171, 47]),
                Colormap::DryFoliage => TintColor([169, 146, 82]),
            };
        };
        let clamp01 = |v: f32| v.clamp(0.0, 1.0);
        // Vanilla colormap lookup: x = 1 - temp, y = 1 - (downfall * temp).
        let t = clamp01(temperature);
        let d = clamp01(downfall) * t;
        let x = ((1.0 - t) * 255.0) as u32;
        let y = ((1.0 - d) * 255.0) as u32;
        let px = tex.pixel(x.min(255), y.min(255));
        TintColor([px[0], px[1], px[2]])
    }

    /// Fixed plains-biome tint (validation scene default).
    pub fn plains_grass(&self) -> TintColor {
        self.sample(Colormap::Grass, 0.8, 0.4)
    }

    pub fn plains_foliage(&self) -> TintColor {
        self.sample(Colormap::Foliage, 0.8, 0.4)
    }

    pub fn plains_dry_foliage(&self) -> TintColor {
        self.sample(Colormap::DryFoliage, 0.8, 0.4)
    }
}

/// Constant tints for non-biome tinted sprites (Phase 1 set).
pub mod constants {
    use super::TintColor;

    /// Water overlay blue (vanilla water tint at plains).
    pub const WATER: TintColor = TintColor([63, 118, 228]);
    /// Redstone dust at signal strength 0 (dark red) — full red at 15.
    pub const REDSTONE_OFF: TintColor = TintColor([75, 0, 0]);
    pub const REDSTONE_ON: TintColor = TintColor([255, 0, 0]);
}

#[allow(unused)]
fn _unused_witness(_: ()) {}

/// Verify a sprite is actually grayscale (tint candidate) — validation helper.
pub fn assert_tint_candidate(
    store: &SpriteStore,
    ns: &str,
    name: &str,
) -> AssetResult<bool> {
    let sprite = store.get(ns, name).ok_or_else(|| AssetError {
        path: format!("{ns}:{name}"),
        message: "sprite not found".into(),
    })?;
    Ok(sprite.tex.is_grayscale_content())
}
