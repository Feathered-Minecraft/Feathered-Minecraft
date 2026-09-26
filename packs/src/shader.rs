//! Shader-pack manager.
//!
//! Shader packs are independent from resource packs: they change *how* the
//! world is rendered (Feathered's wgpu pipelines) rather than *what* is
//! rendered (the resource pack's atlas).
//!
//! Licensing: shader packs are third-party works with their own licenses.
//! Feathered does **not** assume a shader pack is GPL-compatible, never
//! relicenses third-party shader code as Feathered code, and only records
//! where the pack's own license file lives so the user can read it.

use crate::settings::PackSettings;
use crate::SourceInfo;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Which established shader-pack convention a pack follows. Feathered's
/// pipeline is modular: the engine adapts to the pack's configuration file
/// rather than hardcoding one pack (e.g. "Noble-style") into the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShaderLayout {
    /// `shaders/` with `*.fsh`/`*.vsh` program files (OptiFine/Iris style).
    OptifineStyle,
    /// `shaders/` with `*.glsl`/`*.gsh` and pass includes (Iris 1.6+/Canvas style).
    GlslPasses,
    /// A `shaders/` directory was found but its contents fit neither profile.
    Unknown,
}

/// What a shader pack claims to be compatible with (parsed from its own
/// `shaders.properties` when present). Nothing here is a guarantee.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineCompat {
    pub engine: String,
    pub min_version: Option<String>,
}

/// One installed shader pack.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShaderPack {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    /// Detected layout profile (drives which config file Feathered reads).
    pub profile: ShaderLayout,
    /// Compatibility claims from the pack's own metadata, verbatim.
    pub engines: Vec<EngineCompat>,
    /// For `Zip`: extracted directory. For `Folder`: the referenced folder.
    pub dir: PathBuf,
    pub kind: ShaderKind,
    /// Where the pack came from + its license pointer (never a claim).
    pub source: SourceInfo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShaderKind {
    Zip,
    Folder,
}

impl ShaderPack {
    /// Path of the pack's own license file, if it shipped one. Feathered
    /// preserves it verbatim and never edits or relicenses it.
    pub fn license_file(&self) -> Option<PathBuf> {
        ["LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING", "license.txt"]
            .iter()
            .map(|n| self.dir.join(n))
            .find(|p| p.is_file())
    }

    /// Human-readable license note. Deliberately vague unless the pack
    /// declares otherwise: we do not guess licenses on the user's behalf.
    pub fn license_summary(&self) -> String {
        match (&self.source.license_note, self.license_file()) {
            (Some(note), _) => note.clone(),
            (None, Some(file)) => format!(
                "Third-party shader pack — its own license applies (see {})",
                file.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
            ),
            (None, None) => "Third-party shader pack — its own license applies \
                (no license file found; check the pack's distribution page)"
                .into(),
        }
    }
}

/// Problems found while validating a shader pack. Non-fatal ones don't block
/// installation — shader packs are heterogeneous by nature.
#[derive(Debug, Default)]
pub struct ShaderValidation {
    pub fatal: Option<String>,
    pub warnings: Vec<String>,
    pub profile: Option<ShaderLayout>,
    pub engines: Vec<EngineCompat>,
}

/// Manager rooted at `<data>/feathered/shaders`.
pub struct ShaderPackManager {
    base: PathBuf,
}

const META_FILE: &str = "feathered.shader.json";

impl ShaderPackManager {
    /// Default manager rooted at the OS data dir. Honors the
    /// `FEATHERED_DATA_DIR` override (portable installs, tests, CI).
    pub fn open() -> Result<Self, std::io::Error> {
        if let Ok(dir) = std::env::var("FEATHERED_DATA_DIR") {
            return Ok(Self::open_at(dir));
        }
        let base = dirs::data_dir()
            .ok_or_else(|| std::io::Error::other("no OS data directory"))?
            .join("feathered")
            .join("shaders");
        Ok(Self { base })
    }

    /// Manager rooted at an explicit directory. The argument is the Feathered
    /// data root; shader packs live in `<root>/shaders`.
    pub fn open_at(base: impl Into<PathBuf>) -> Self {
        Self {
            base: base.into().join("shaders"),
        }
    }

    pub fn base_dir(&self) -> &Path {
        &self.base
    }

    fn settings_path(&self) -> PathBuf {
        self.base.parent().unwrap_or(&self.base).join("settings.json")
    }

    pub fn settings(&self) -> PackSettings {
        crate::settings::load_or_create(&self.settings_path()).unwrap_or_default()
    }

    pub fn save_settings(&self, s: &PackSettings) -> Result<(), String> {
        crate::settings::save(&self.settings_path(), s).map_err(|e| e.message)
    }

    /// Install from a ZIP archive (extracted under the data dir; original
    /// file untouched).
    pub fn import_zip(&self, zip_path: &Path) -> Result<ShaderPack, String> {
        let file = std::fs::File::open(zip_path).map_err(|e| e.to_string())?;
        let mut archive = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;

        let validation = Self::validate_entries(&mut archive)?;
        if let Some(fatal) = validation.fatal {
            return Err(format!("{zip_path:?}: {fatal}"));
        }

        let top = top_level_dir(&mut archive);
        let dest = self.new_pack_dir(&stem(zip_path));
        std::fs::create_dir_all(&dest).map_err(|e| e.to_string())?;

        for i in 0..archive.len() {
            let mut entry = archive.by_index(i).map_err(|e| e.to_string())?;
            if entry.is_dir() {
                continue;
            }
            let enclosed = entry
                .enclosed_name()
                .ok_or_else(|| format!("unsafe archive entry: {}", entry.name()))?
                .to_path_buf();
            let rel = match (&top, &enclosed) {
                (Some(t), e) if e.starts_with(t) => e.strip_prefix(t).unwrap_or(e).to_path_buf(),
                _ => enclosed,
            };
            if rel.as_os_str().is_empty() {
                continue;
            }
            let target = dest.join(&rel);
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            let mut out = std::fs::File::create(&target).map_err(|e| e.to_string())?;
            std::io::copy(&mut entry, &mut out).map_err(|e| e.to_string())?;
        }

        let profile = detect_profile(&dest).unwrap_or(ShaderLayout::Unknown);
        let pack = ShaderPack {
            id: shader_id(&dest),
            name: stem(zip_path),
            description: validation
                .engines
                .first()
                .map(|e| format!("targets {} (from shaders.properties)", e.engine)),
            profile,
            engines: validation.engines,
            dir: dest.clone(),
            kind: ShaderKind::Zip,
            source: SourceInfo {
                origin: zip_path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| zip_path.display().to_string()),
                license_note: None,
            },
        };
        write_meta(&dest, &pack)?;
        Ok(pack)
    }

    /// Install by reference to a folder (not copied).
    pub fn import_folder(&self, dir: &Path) -> Result<ShaderPack, String> {
        if !dir.is_dir() {
            return Err(format!("not a directory: {}", dir.display()));
        }
        let layout = detect_profile(dir);
        if layout.is_none() {
            return Err(format!(
                "{}: no shaders/ directory found (not a shader pack)",
                dir.display()
            ));
        }
        let canonical = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
        let origin = canonical.display().to_string();
        let profile = layout.unwrap_or(ShaderLayout::Unknown);
        let engines = parse_engine_claims(&canonical.join("shaders/shaders.properties"));
        let pack = ShaderPack {
            id: shader_id(&canonical),
            name: dir
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| canonical.display().to_string()),
            description: engines
                .first()
                .map(|e| format!("targets {} (from shaders.properties)", e.engine)),
            profile,
            engines,
            dir: canonical,
            kind: ShaderKind::Folder,
            source: SourceInfo {
                origin,
                license_note: None,
            },
        };
        std::fs::create_dir_all(&self.base).map_err(|e| e.to_string())?;
        write_json(&self.base.join(format!("{}.folder.json", pack.id)), &pack)
            .map_err(|e| e.to_string())?;
        Ok(pack)
    }

    /// Structural validation over archive entries before extracting.
    fn validate_entries(archive: &mut zip::ZipArchive<std::fs::File>) -> Result<ShaderValidation, String> {
        let mut validation = ShaderValidation::default();
        let mut has_shaders_dir = false;
        let mut shader_files = 0usize;
        for i in 0..archive.len() {
            let entry = archive.by_index(i).map_err(|e| e.to_string())?;
            if entry.is_dir() {
                continue;
            }
            let name = entry.name().to_string();
            if name.starts_with('/') || name.contains("..") || Path::new(&name).is_absolute() {
                return Err(format!("unsafe archive entry: {name}"));
            }
            if components_contain(&name, "shaders") {
                has_shaders_dir = true;
                let lower = name.to_lowercase();
                if lower.ends_with(".fsh") || lower.ends_with(".vsh") {
                    shader_files += 1;
                } else if lower.ends_with(".glsl") || lower.ends_with(".gsh") {
                    shader_files += 1;
                } else if lower.ends_with("shaders.properties") {
                    validation.engines = parse_properties_text(
                        &std::io::read_to_string(entry).unwrap_or_default(),
                    );
                }
            }
        }
        if !has_shaders_dir {
            validation.fatal = Some("no shaders/ directory in archive".into());
        } else if shader_files == 0 {
            validation.warnings
                .push("shaders/ exists but contains no recognizable shader files".into());
        }
        Ok(validation)
    }

    pub fn list(&self) -> Result<Vec<ShaderPack>, String> {
        let mut out = Vec::new();
        let entries = match std::fs::read_dir(&self.base) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
            Err(e) => return Err(e.to_string()),
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let meta_path = if path.is_dir() {
                path.join(META_FILE)
            } else if path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.ends_with(".folder.json"))
                .unwrap_or(false)
            {
                path.clone()
            } else {
                continue;
            };
            let Ok(text) = std::fs::read_to_string(&meta_path) else {
                continue;
            };
            let Ok(pack) = serde_json::from_str::<ShaderPack>(&text) else {
                continue;
            };
            let alive = detect_profile(&pack.dir).is_some();
            if alive {
                out.push(pack);
            }
        }
        out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
        Ok(out)
    }

    pub fn prune_missing(&self) -> Result<usize, String> {
        let entries = std::fs::read_dir(&self.base).map_err(|e| e.to_string())?;
        let mut removed = 0;
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if !name.ends_with(".folder.json") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let Ok(pack) = serde_json::from_str::<ShaderPack>(&text) else {
                continue;
            };
            if detect_profile(&pack.dir).is_none() {
                let _ = std::fs::remove_file(&path);
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// Enable a shader pack (persists in settings). Returns the previous
    /// active pack id, if any.
    pub fn enable(&self, pack: &ShaderPack) -> Result<Option<String>, String> {
        let mut settings = self.settings();
        let prev = settings.active_shader.replace(pack.id.clone());
        self.save_settings(&settings)?;
        Ok(prev)
    }

    /// Disable shaders entirely (settings-only; files are never touched).
    pub fn disable(&self) -> Result<Option<String>, String> {
        let mut settings = self.settings();
        let prev = settings.active_shader.take();
        self.save_settings(&settings)?;
        Ok(prev)
    }

    /// The currently enabled pack, if installed.
    pub fn active(&self) -> Option<ShaderPack> {
        let id = self.settings().active_shader?;
        self.list().ok()?.into_iter().find(|p| p.id == id)
    }

    /// Remove a ZIP-extracted pack's files; folder packs lose only metadata.
    pub fn uninstall(&self, pack: &ShaderPack) -> Result<(), String> {
        match pack.kind {
            ShaderKind::Zip => {
                if pack.dir.is_dir() {
                    std::fs::remove_dir_all(&pack.dir).map_err(|e| e.to_string())?;
                }
            }
            ShaderKind::Folder => {
                let meta = self.base.join(format!("{}.folder.json", pack.id));
                if meta.is_file() {
                    std::fs::remove_file(&meta).map_err(|e| e.to_string())?;
                }
            }
        }
        if self.settings().active_shader.as_deref() == Some(pack.id.as_str()) {
            self.disable()?;
        }
        Ok(())
    }

    fn new_pack_dir(&self, name: &str) -> PathBuf {
        let mut hasher = Sha256::new();
        hasher.update(name.as_bytes());
        hasher.update(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos().to_le_bytes())
                .unwrap_or([0; 16]),
        );
        let hash = format!("{:x}", hasher.finalize());
        self.base.join(format!("{}-{}", crate::resource::slug_of(name), &hash[..8]))
    }
}

/// Locate the `shaders/` directory for a pack root (searches two levels).
pub fn find_shaders_dir(root: &Path) -> Option<PathBuf> {
    let direct = root.join("shaders");
    if direct.is_dir() {
        return Some(direct);
    }
    if let Ok(entries) = std::fs::read_dir(root) {
        for entry in entries.flatten() {
            let nested = entry.path().join("shaders");
            if nested.is_dir() {
                return Some(nested);
            }
        }
    }
    None
}

/// Detect the layout profile from the files inside `shaders/`.
pub fn detect_profile(root: &Path) -> Option<ShaderLayout> {
    let shaders = find_shaders_dir(root)?;
    let mut optifine = 0;
    let mut glsl = 0;
    if let Ok(entries) = std::fs::read_dir(&shaders) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_lowercase();
            if name.ends_with(".fsh") || name.ends_with(".vsh") {
                optifine += 1;
            } else if name.ends_with(".glsl") || name.ends_with(".gsh") {
                glsl += 1;
            }
        }
    }
    Some(match (optifine, glsl) {
        (0, 0) => ShaderLayout::Unknown,
        (f, g) if f >= g => ShaderLayout::OptifineStyle,
        _ => ShaderLayout::GlslPasses,
    })
}

/// Read engine-compatibility claims from `shaders.properties`, verbatim.
fn parse_engine_claims(props: &Path) -> Vec<EngineCompat> {
    match std::fs::read_to_string(props) {
        Ok(text) => parse_properties_text(&text),
        Err(_) => Vec::new(),
    }
}

/// Parse the subset of `shaders.properties` keys that describe target engines.
/// Handles both the synthetic/conventional `requires`/`profile.requires.*`
/// keys and the real Iris scheme (`iris.features.required`,
/// `iris.features.optional`, likewise `optifine.*`), which real packs such as
/// Noble use (verified against BelmuTM/Noble `shaders.properties`).
fn parse_properties_text(text: &str) -> Vec<EngineCompat> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim();
        let (engine, version): (String, Option<String>) = match key {
            "profile.requires" | "requires" | "supports" => (value.to_string(), None),
            k if k.starts_with("profile.requires.") => (
                k.trim_start_matches("profile.requires.").to_string(),
                Some(value.to_string()),
            ),
            k if k.starts_with("iris.features.") || k.starts_with("optifine.features.") => {
                // `iris.features.required = SSBO COMPUTE_SHADERS` — a feature
                // requirement for that engine, recorded as its own entry so
                // importers can see what the pack demands.
                (k.to_string(), Some(value.to_string()))
            }
            _ => continue,
        };
        out.push(EngineCompat {
            engine,
            min_version: version,
        });
    }
    out
}

/// True when any path component equals `want` or ends with it (covers
/// wrapper folders like `FancyShaders/shaders/...` and bare `shaders/...`).
fn components_contain(name: &str, want: &str) -> bool {
    name.split(['/', '\\'])
        .any(|c| c.eq_ignore_ascii_case(want) || c.to_lowercase().ends_with(want))
}

fn stem(p: &Path) -> String {
    p.file_stem()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "shader".into())
}

fn top_level_dir(archive: &mut zip::ZipArchive<std::fs::File>) -> Option<PathBuf> {
    let mut first: Option<String> = None;
    for i in 0..archive.len() {
        let Ok(entry) = archive.by_index(i) else { continue };
        let name = entry.name();
        if entry.is_dir() || name.starts_with('.') {
            continue;
        }
        let Some((head, rest)) = name.split_once('/') else {
            return None;
        };
        if rest.is_empty() {
            continue;
        }
        let head = head.to_string();
        match &first {
            None => first = Some(head),
            Some(f) if *f != head => return None,
            _ => {}
        }
    }
    first.map(PathBuf::from)
}

fn shader_id(dir: &Path) -> String {
    format!(
        "{}-{}",
        crate::resource::slug_of(&dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()),
        {
            let mut hasher = Sha256::new();
            if let Ok(bytes) = std::fs::read(dir.join("shaders").join("shaders.properties")) {
                hasher.update(&bytes);
            }
            let out = format!("{:x}", hasher.finalize());
            out[..8].to_string()
        }
    )
}

fn write_meta(dir: &Path, pack: &ShaderPack) -> Result<(), String> {
    write_json(&dir.join(META_FILE), pack).map_err(|e| e.to_string())
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let text = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    std::fs::write(path, text).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests;
