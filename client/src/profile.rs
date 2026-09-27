//! Player profile: display name + skin (Minecraft-format PNG library).
//!
//! Skins are stored as plain PNG files in `<data>/feathered/profile/skins/`
//! and parsed in memory for preview. Both classic layouts are supported:
//! **64×64** (modern, includes overlay layers) and **64×32** (legacy). The
//! parser reads only vanilla-format regions — head (8,8,8×8) and body
//! (16,20,8×12 for 64×64; 16,16,8×12 for 64×32) — and averages texel
//! colors into the avatar's blocky front-view mesh. No pack or online
//! assets are involved: users drop their own legally-obtained skin PNG
//! into the skins folder.

use image::GenericImageView;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Parse errors (kept stringly for the menu's feedback line).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkinError {
    Io(String),
    NotPng,
    BadDimensions(u32, u32),
}

impl std::fmt::Display for SkinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SkinError::Io(e) => write!(f, "cannot read skin: {e}"),
            SkinError::NotPng => write!(f, "not a PNG file"),
            SkinError::BadDimensions(w, h) => {
                write!(f, "unsupported skin size {w}x{h} (need 64x64 or 64x32)")
            }
        }
    }
}

/// RGBA texel grid of the regions we render.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skin {
    /// Head 8×8 (from 8,8).
    pub head: Vec<[u8; 4]>,
    /// Body 8×12 (front torso strip; layout-aware).
    pub body: Vec<[u8; 4]>,
    /// Legs 8×12 (front leg strip; layout-aware).
    pub legs: Vec<[u8; 4]>,
}

/// Column-averaged solid colors for quick menu drawing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkinPalette {
    /// Head average (hat overlay included when non-transparent).
    pub head: [u8; 4],
    /// Torso average.
    pub body: [u8; 4],
    /// Legs average.
    pub legs: [u8; 4],
}

impl Skin {
    /// Parse a Minecraft skin from PNG bytes.
    pub fn from_png(bytes: &[u8]) -> Result<Skin, SkinError> {
        let img = image::load_from_memory(bytes).map_err(|_| SkinError::NotPng)?;
        let (w, h) = img.dimensions();
        if !((w == 64 && h == 64) || (w == 64 && h == 32)) {
            return Err(SkinError::BadDimensions(w, h));
        }
        let rgba = img.to_rgba8();
        let get = |x: u32, y: u32| -> [u8; 4] {
            let p = rgba.get_pixel(x, y).0;
            // Fully transparent texels read as "absent".
            if p[3] == 0 { [0, 0, 0, 0] } else { p }
        };

        // Head: 8×8 at (8,8) — same in both layouts.
        let mut head = Vec::with_capacity(64);
        for y in 0..8u32 {
            for x in 0..8u32 {
                head.push(get(8 + x, 8 + y));
            }
        }

        // Vanilla layout (identical in 64×64 and 64×32 for these regions):
        // torso front 8×12 at (20,20); right-leg front 4×12 at (4,20).
        // 64×64 adds the LEFT-leg front 4×12 at (20,52) — the 64×32 legacy
        // format mirrors the right leg instead.
        let mut body = Vec::with_capacity(8 * 12);
        for y in 0..12u32 {
            for x in 0..8u32 {
                body.push(get(20 + x, 20 + y));
            }
        }
        let mut legs = Vec::with_capacity(8 * 12);
        for y in 0..12u32 {
            for x in 0..4u32 {
                legs.push(get(4 + x, 20 + y)); // right leg
            }
            for x in 0..4u32 {
                legs.push(if h == 64 { get(20 + x, 52 + y) } else { get(4 + x, 20 + y) });
            }
        }
        Ok(Skin { head, body, legs })
    }

    /// Average non-transparent color of a region (with alpha coverage).
    fn avg(texels: &[[u8; 4]]) -> [u8; 4] {
        let (mut r, mut g, mut b, mut n) = (0u32, 0u32, 0u32, 0u32);
        for t in texels {
            if t[3] > 0 {
                r += t[0] as u32;
                g += t[1] as u32;
                b += t[2] as u32;
                n += 1;
            }
        }
        if n == 0 {
            [180, 180, 180, 255]
        } else {
            [
                (r / n.max(1)) as u8,
                (g / n.max(1)) as u8,
                (b / n.max(1)) as u8,
                255,
            ]
        }
    }

    /// Palette for the avatar preview (head uses top half for "face").
    pub fn palette(&self) -> SkinPalette {
        let head_face: Vec<[u8; 4]> = {
            // Rows 2..6 are the face band of the 8×8 head.
            (2..6).flat_map(|y| (0..8).map(move |x| y * 8 + x))
                .map(|i| self.head[i])
                .collect()
        };
        SkinPalette {
            head: Self::avg(&head_face),
            body: Self::avg(&self.body),
            legs: Self::avg(&self.legs),
        }
    }
}

/// The saved profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Profile {
    /// Player name shown in menus (and future multiplayer).
    pub name: String,
    /// File name (inside `skins/`) of the active skin.
    pub skin: Option<String>,
}

impl Default for Profile {
    fn default() -> Profile {
        Profile {
            name: "Player".into(),
            skin: None,
        }
    }
}

/// Profile store: `<dir>/profile.json` + `<dir>/skins/*.png`.
pub struct ProfileStore {
    dir: PathBuf,
}

impl ProfileStore {
    pub fn new(dir: PathBuf) -> ProfileStore {
        ProfileStore { dir }
    }

    fn json(&self) -> PathBuf {
        self.dir.join("profile.json")
    }

    pub fn skins_dir(&self) -> PathBuf {
        self.dir.join("skins")
    }

    /// Load the profile (missing/corrupt → default).
    pub fn load(&self) -> Profile {
        std::fs::read(self.json())
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default()
    }

    /// Save the profile (best effort).
    pub fn save(&self, p: &Profile) -> bool {
        let _ = std::fs::create_dir_all(&self.dir);
        match serde_json::to_vec_pretty(p) {
            Ok(bytes) => std::fs::write(self.json(), bytes).is_ok(),
            Err(_) => false,
        }
    }

    /// List available skin PNGs (file names, sorted).
    pub fn list_skins(&self) -> Vec<String> {
        let mut out = Vec::new();
        if let Ok(rd) = std::fs::read_dir(self.skins_dir()) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if name.to_ascii_lowercase().ends_with(".png") {
                    out.push(name);
                }
            }
        }
        out.sort();
        out
    }

    /// Import a PNG: copies it into the library under `name.png` and
    /// validates the format. Returns the stored file name.
    pub fn import_skin(&self, name: &str, png_bytes: &[u8]) -> Result<String, SkinError> {
        // Validate before storing.
        Skin::from_png(png_bytes)?;
        let _ = std::fs::create_dir_all(self.skins_dir());
        let file_name = format!("{}.png", sanitize(name));
        std::fs::write(self.skins_dir().join(&file_name), png_bytes)
            .map_err(|e| SkinError::Io(e.to_string()))?;
        Ok(file_name)
    }

    /// Load a skin's PNG bytes from the library.
    pub fn read_skin(&self, file_name: &str) -> Result<Vec<u8>, SkinError> {
        std::fs::read(self.skins_dir().join(file_name)).map_err(|e| SkinError::Io(e.to_string()))
    }

    /// Delete a skin from the library.
    pub fn delete_skin(&self, file_name: &str) -> bool {
        std::fs::remove_file(self.skins_dir().join(file_name)).is_ok()
    }
}

/// Sanitize a skin file stem (drops path separators and odd chars but
/// keeps the single trailing extension handling to the caller).
fn sanitize(name: &str) -> String {
    // Strip a known extension first so "x.png" doesn't become "xpng".
    let stem = name.strip_suffix(".png").unwrap_or(name);
    let cleaned: String = stem
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-' || *c == ' ')
        .collect();
    cleaned.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a valid 64×64 PNG: head = red, body = green, legs = blue.
    fn sample_skin_64() -> Vec<u8> {
        let mut px = vec![[0u8, 0, 0, 0]; 64 * 64];
        let paint = |px: &mut [[u8; 4]], x: u32, y: u32, w: u32, h: u32, c: [u8; 4]| {
            for y in y..y + h {
                for x in x..x + w {
                    px[(y * 64 + x) as usize] = c;
                }
            }
        };
        paint(&mut px, 8, 8, 8, 8, [255, 0, 0, 255]); // head
        paint(&mut px, 20, 20, 8, 12, [0, 255, 0, 255]); // body
        paint(&mut px, 4, 20, 4, 12, [0, 0, 255, 255]); // right leg
        paint(&mut px, 20, 52, 4, 12, [0, 0, 200, 255]); // left leg
        let img = image::RgbaImage::from_fn(64, 64, |x, y| {
            let p = px[(y * 64 + x) as usize];
            image::Rgba(p)
        });
        let mut buf = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(img)
            .write_to(&mut buf, image::ImageFormat::Png)
            .unwrap();
        buf.into_inner()
    }

    /// Legacy 64×32 layout variant.
    fn sample_skin_32() -> Vec<u8> {
        let mut px = vec![[0u8, 0, 0, 0]; 64 * 32];
        let paint = |px: &mut [[u8; 4]], x: u32, y: u32, w: u32, h: u32, c: [u8; 4]| {
            for y in y..y + h {
                for x in x..x + w {
                    px[(y * 64 + x) as usize] = c;
                }
            }
        };
        paint(&mut px, 8, 8, 8, 8, [255, 0, 0, 255]); // head
        paint(&mut px, 20, 20, 8, 12, [0, 255, 0, 255]); // body
        paint(&mut px, 4, 20, 4, 12, [0, 0, 255, 255]); // right leg (mirrored left)
        let img = image::RgbaImage::from_fn(64, 32, |x, y| {
            let p = px[(y * 64 + x) as usize];
            image::Rgba(p)
        });
        let mut buf = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(img)
            .write_to(&mut buf, image::ImageFormat::Png)
            .unwrap();
        buf.into_inner()
    }

    #[test]
    fn parses_modern_and_legacy_layouts() {
        let s64 = Skin::from_png(&sample_skin_64()).unwrap();
        assert_eq!(s64.head[0], [255, 0, 0, 255]);
        assert_eq!(s64.body[0], [0, 255, 0, 255]);
        assert_eq!(s64.legs[0], [0, 0, 255, 255]);

        let s32 = Skin::from_png(&sample_skin_32()).unwrap();
        assert_eq!(s32.head[0], [255, 0, 0, 255]);
        assert_eq!(s32.body[0], [0, 255, 0, 255]);
        assert_eq!(s32.legs[0], [0, 0, 255, 255]);
    }

    #[test]
    fn rejects_bad_sizes_and_non_png() {
        assert_eq!(Skin::from_png(b"hello"), Err(SkinError::NotPng));
        // 32×32 PNG (wrong size).
        let img = image::RgbaImage::from_fn(32, 32, |_, _| image::Rgba([0u8; 4]));
        let mut buf = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(img)
            .write_to(&mut buf, image::ImageFormat::Png)
            .unwrap();
        let bytes = buf.into_inner();
        assert_eq!(
            Skin::from_png(&bytes),
            Err(SkinError::BadDimensions(32, 32))
        );
    }

    #[test]
    fn palette_averages_regions() {
        let s = Skin::from_png(&sample_skin_64()).unwrap();
        let p = s.palette();
        assert_eq!(p.head, [255, 0, 0, 255]);
        assert_eq!(p.body, [0, 255, 0, 255]);
        // Two legs: bright blue + darker blue average.
        assert_eq!(p.legs, [0, 0, 227, 255]);
    }

    #[test]
    fn store_round_trips_and_validates() {
        let dir = std::env::temp_dir().join(format!("feathered-profile-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = ProfileStore::new(dir.clone());

        assert_eq!(store.load(), Profile::default());
        let mut p = Profile::default();
        p.name = "FeatheredFan".into();
        p.skin = Some("steve.png".into());
        assert!(store.save(&p));
        assert_eq!(store.load(), p);

        // Import a real skin.
        let stored = store.import_skin("my skin!.png", &sample_skin_64()).unwrap();
        assert_eq!(stored, "my skin.png", "sanitized name");
        assert_eq!(store.list_skins(), vec!["my skin.png".to_string()]);
        assert!(store.read_skin("my skin.png").is_ok());
        // Invalid content rejected.
        assert!(store.import_skin("bad", b"nope").is_err());

        assert!(store.delete_skin("my skin.png"));
        assert!(store.list_skins().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
