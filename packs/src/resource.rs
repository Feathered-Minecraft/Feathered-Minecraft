//! Resource-pack manager: discovery, import, validation, switching.
//!
//! Invariants:
//! * The user's original files are never modified — ZIPs are extracted into a
//!   Feathered-owned data directory and folders are referenced, never moved.
//! * A pack is only listed if it structurally validates: it must contain
//!   `pack.mcmeta` and namespace folders.
//! * Every pack keeps its own license/attribution metadata; Feathered makes
//!   no licensing claims about user-supplied packs.

use crate::settings::PackSettings;
use crate::SourceInfo;
use feathered_assets::pack::PackIndex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Component, Path, PathBuf};

/// Kind of source the pack was imported from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackKind {
    /// Imported from a user ZIP archive (extracted into the data dir).
    Zip,
    /// Referenced in place (the folder stays wherever the user put it).
    Folder,
}

/// Metadata about one installed resource pack.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePack {
    /// Unique id: stable slug of the pack name plus a short content hash.
    pub id: String,
    /// Display name (folder/zip name, or `pack.mcmeta` description's first line).
    pub name: String,
    /// Description line from `pack.mcmeta` (JSON text components flattened).
    pub description: Option<String>,
    /// `pack_format` from `pack.mcmeta` (e.g. 97 for the 26.3 default pack).
    pub pack_format: Option<u32>,
    /// Best-effort mapping of the format to a Minecraft release.
    pub detected_version: Option<String>,
    /// How the pack was imported.
    pub kind: PackKind,
    /// For `Zip`: extracted directory inside the data dir.
    /// For `Folder`: the absolute path the user pointed at.
    pub dir: PathBuf,
    /// Where it came from + optional license note (attribution).
    pub source: SourceInfo,
    /// SHA-256 over `pack.mcmeta` — cheap staleness/fingerprint signal.
    pub mcmeta_hash: String,
}

impl ResourcePack {
    /// Human line summarizing the detected version, e.g. "26.3 (format 97)".
    pub fn version_label(&self) -> String {
        match (self.detected_version.as_deref(), self.pack_format) {
            (Some(v), Some(f)) => format!("{v} (format {f})"),
            (None, Some(f)) => format!("unknown (format {f})"),
            (Some(v), None) => v.to_string(),
            (None, None) => "unknown".into(),
        }
    }
}

/// What an import produced, for UI/reporting.
#[derive(Debug)]
pub struct ImportReport {
    pub pack: ResourcePack,
    /// Number of files copied/extracted (folders report 0 — nothing copied).
    pub files_copied: u64,
}

/// Result of structurally validating a pack directory.
#[derive(Debug)]
pub struct PackValidation {
    pub valid: bool,
    pub problems: Vec<String>,
    pub pack_format: Option<u32>,
    pub description: Option<String>,
}

/// Failure modes surfaced to the UI as friendly strings.
#[derive(Debug)]
pub enum ImportError {
    NotFound(PathBuf),
    NotAPack(PathBuf),
    /// A ZIP entry tried to escape the destination (zip-slip attack).
    UnsafePath(PathBuf, String),
    Io(PathBuf, String),
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImportError::NotFound(p) => write!(f, "not found: {}", p.display()),
            ImportError::NotAPack(p) => write!(
                f,
                "{}: not a resource pack (missing pack.mcmeta and no namespace folders)",
                p.display()
            ),
            ImportError::UnsafePath(p, m) => write!(f, "unsafe archive entry in {}: {m}", p.display()),
            ImportError::Io(p, m) => write!(f, "{}: {m}", p.display()),
        }
    }
}

impl std::error::Error for ImportError {}

/// Detects the Minecraft release associated with a `pack_format` number.
/// Covers the formats Feathered targets; anything newer/older reports the
/// format number itself. Data: `data/data.js` in the Grasscall snapshot.
pub fn pack_format_to_version(format: u32) -> Option<&'static str> {
    Some(match format {
        97 => "26.3",
        96 => "26.2",
        95 => "26.1",
        94 => "1.21.11",
        93 => "1.21.9",
        88 => "1.21.6",
        81 => "1.21.5",
        71 => "1.21.4",
        64 => "1.21.2",
        61 => "1.21",
        57 => "1.20.5",
        46 => "1.20",
        34 => "1.19.4",
        22 => "1.19",
        18 => "1.18",
        15 => "1.17",
        8 => "1.16",
        6 => "1.15",
        5 => "1.14",
        4 => "1.13",
        _ => return None,
    })
}

/// Manager over Feathered's pack data directory (default:
/// `<OS data dir>/feathered/packs`).
pub struct ResourcePackManager {
    base: PathBuf,
}

impl ResourcePackManager {
    /// Default manager rooted at the OS data dir. Honors the
    /// `FEATHERED_DATA_DIR` override (portable installs, tests, CI).
    pub fn open() -> Result<Self, ImportError> {
        if let Ok(dir) = std::env::var("FEATHERED_DATA_DIR") {
            return Ok(Self::open_at(dir));
        }
        let base = dirs::data_dir()
            .ok_or_else(|| ImportError::Io(PathBuf::from("<data-dir>"), "no OS data directory".into()))?
            .join("feathered")
            .join("packs");
        Ok(Self { base })
    }

    /// Manager rooted at an explicit directory (used by tests and
    /// `--data-dir`). The argument is the Feathered data root; packs live in
    /// `<root>/packs` and settings in `<root>/settings.json`.
    pub fn open_at(base: impl Into<PathBuf>) -> Self {
        Self {
            base: base.into().join("packs"),
        }
    }

    /// Root of the pack data directory (UI's "Open Resource Pack Folder").
    pub fn base_dir(&self) -> &Path {
        &self.base
    }

    /// Create the directory skeleton and the default settings file.
    pub fn initialize(&self) -> Result<PackSettings, ImportError> {
        std::fs::create_dir_all(&self.base).map_err(|e| self.io(&self.base, e))?;
        let settings = crate::settings::load_or_create(&self.settings_path())
            .map_err(|e| ImportError::Io(self.settings_path(), e.message))?;
        Ok(settings)
    }

    fn settings_path(&self) -> PathBuf {
        self.base.parent().unwrap_or(&self.base).join("settings.json")
    }

    pub fn settings(&self) -> PackSettings {
        crate::settings::load_or_create(&self.settings_path()).unwrap_or_default()
    }

    pub fn save_settings(&self, settings: &PackSettings) -> Result<(), ImportError> {
        crate::settings::save(&self.settings_path(), settings)
            .map_err(|e| ImportError::Io(self.settings_path(), e.message))
    }

    fn io(&self, path: &Path, e: std::io::Error) -> ImportError {
        ImportError::Io(path.to_path_buf(), e.to_string())
    }

    /// List installed packs (sorted by name). ZIP packs keep their metadata
    /// inside the extracted directory; folder packs keep a `.folder.json`
    /// sidecar here. Packs whose source vanished are skipped.
    pub fn list(&self) -> Result<Vec<ResourcePack>, ImportError> {
        let mut out = Vec::new();
        let entries = match std::fs::read_dir(&self.base) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
            Err(e) => return Err(self.io(&self.base, e)),
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
            let Ok(pack) = serde_json::from_str::<ResourcePack>(&text) else {
                continue;
            };
            let exists = match pack.kind {
                PackKind::Zip => pack.dir.is_dir(),
                PackKind::Folder => pack.dir.is_dir(),
            };
            if exists {
                out.push(pack);
            }
        }
        out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
        Ok(out)
    }

    /// Remove stale metadata for folder packs whose source vanished.
    /// (ZIP packs live entirely inside the data dir — if the extracted
    /// directory is gone, so is its metadata.)
    pub fn prune_missing(&self) -> Result<usize, ImportError> {
        let mut removed = 0;
        let Ok(entries) = std::fs::read_dir(&self.base) else {
            return Ok(0);
        };
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
            let Ok(pack) = serde_json::from_str::<ResourcePack>(&text) else {
                continue;
            };
            if !pack.dir.is_dir() {
                let _ = std::fs::remove_file(&path);
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// Import a ZIP resource pack. The archive is extracted into
    /// `<data>/packs/<slug>-<hash8>/`; the original file is untouched.
    pub fn import_zip(&self, zip_path: &Path) -> Result<ImportReport, ImportError> {
        let file = std::fs::File::open(zip_path).map_err(|e| self.io(zip_path, e))?;
        let mut archive =
            zip::ZipArchive::new(file).map_err(|e| self.io(zip_path, e.into()))?;

        // Validate structure BEFORE extracting anything to disk.
        let validation = Self::validate_entries(&mut archive, zip_path)?;
        if !validation.valid {
            return Err(ImportError::NotAPack(zip_path.to_path_buf()));
        }

        // Top-level folder inside the ZIP (e.g. "MyPack/..."), if any.
        let top = top_level_dir(&mut archive);
        let dest = self.new_pack_dir(&zip_name(zip_path));
        std::fs::create_dir_all(&dest).map_err(|e| self.io(&dest, e))?;

        let mut files_copied = 0u64;
        for i in 0..archive.len() {
            let mut entry = archive.by_index(i).map_err(|e| self.io(zip_path, e.into()))?;
            if entry.is_dir() {
                continue;
            }
            let enclosed = match entry.enclosed_name() {
                Some(p) => p.to_path_buf(),
                None => {
                    return Err(ImportError::UnsafePath(
                        zip_path.to_path_buf(),
                        entry.name().to_string(),
                    ))
                }
            };
            // Strip the ZIP's top-level folder so the extracted dir *is* the
            // pack root (`pack.mcmeta` sits directly inside `pack.dir`).
            let rel = match (&top, &enclosed) {
                (Some(t), e) if e.starts_with(t) => e.strip_prefix(t).unwrap_or(e).to_path_buf(),
                _ => enclosed,
            };
            if rel.as_os_str().is_empty() {
                continue;
            }
            let target = dest.join(&rel);
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent).map_err(|e| self.io(parent, e))?;
            }
            let mut out = std::fs::File::create(&target).map_err(|e| self.io(&target, e))?;
            std::io::copy(&mut entry, &mut out).map_err(|e| self.io(&target, e))?;
            files_copied += 1;
        }

        // Defense in depth: the extracted tree must itself validate.
        let mcmeta = dest.join("pack.mcmeta");
        if !mcmeta.is_file() {
            return Err(ImportError::NotAPack(dest));
        }

        let (pack_format, description) = match read_mcmeta(&mcmeta) {
            Some(m) => (m.pack_format, m.description),
            None => (validation.pack_format, validation.description),
        };
        let pack = ResourcePack {
            id: pack_id(&dest),
            name: zip_name(zip_path),
            description,
            pack_format,
            detected_version: pack_format.and_then(pack_format_to_version).map(str::to_string),
            kind: PackKind::Zip,
            dir: dest.clone(),
            source: SourceInfo {
                origin: zip_path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| zip_path.display().to_string()),
                license_note: None,
            },
            mcmeta_hash: hash_file(&mcmeta),
        };
        write_meta(&dest, &pack)?;
        Ok(ImportReport { pack, files_copied })
    }

    /// Import (reference) a folder resource pack. The folder is not copied —
    /// metadata records its absolute path.
    pub fn import_folder(&self, dir: &Path) -> Result<ImportReport, ImportError> {
        if !dir.is_dir() {
            return Err(ImportError::NotFound(dir.to_path_buf()));
        }
        let validation = validate_folder(dir)?;
        if !validation.valid {
            return Err(ImportError::NotAPack(dir.to_path_buf()));
        }
        let canonical = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
        let origin = canonical.display().to_string();
        let mcmeta = canonical.join("pack.mcmeta");
        // Prefer a meaningful display name: generic basenames (`assets`,
        // `texture`, `resourcepacks`, ...) fall back to the nearest
        // non-generic ancestor directory name.
        let mut display_name = canonical
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| canonical.display().to_string());
        let mut ancestor = canonical.parent();
        while is_generic_dir_name(&display_name) {
            match ancestor.and_then(|p| p.file_name()) {
                Some(parent_name) => {
                    display_name = parent_name.to_string_lossy().into_owned();
                    ancestor = ancestor.and_then(|p| p.parent());
                }
                None => break,
            }
        }
        let (pack_format, description) = read_mcmeta(&mcmeta)
            .map(|m| (m.pack_format, m.description))
            .unwrap_or((validation.pack_format, validation.description));
        let pack = ResourcePack {
            id: pack_id(&canonical),
            name: display_name,
            description,
            pack_format,
            detected_version: pack_format.and_then(pack_format_to_version).map(str::to_string),
            kind: PackKind::Folder,
            dir: canonical,
            source: SourceInfo {
                origin,
                license_note: None,
            },
            mcmeta_hash: mcmeta.is_file().then(|| hash_file(&mcmeta)).unwrap_or_default(),
        };
        self.write_meta_for_folder(&pack)?;
        Ok(ImportReport { pack, files_copied: 0 })
    }

    fn write_meta_for_folder(&self, pack: &ResourcePack) -> Result<(), ImportError> {
        std::fs::create_dir_all(&self.base).map_err(|e| self.io(&self.base, e))?;
        let meta_path = self.base.join(format!("{}.folder.json", pack.id));
        write_json(&meta_path, pack)
    }

    /// Scan a directory tree (e.g. the user's `.minecraft/resourcepacks`)
    /// and import every valid pack found there as a folder reference.
    pub fn import_all_folders_under(&self, base: &Path) -> Result<Vec<ImportReport>, ImportError> {
        let mut reports = Vec::new();
        let mut stack = vec![base.to_path_buf()];
        while let Some(dir) = stack.pop() {
            let entries = match std::fs::read_dir(&dir) {
                Ok(e) => e,
                Err(_) => continue,
            };
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    if p.join("pack.mcmeta").is_file() {
                        if let Ok(r) = self.import_folder(&p) {
                            reports.push(r);
                        }
                    } else {
                        stack.push(p);
                    }
                } else if p.extension().map(|e| e.eq_ignore_ascii_case("zip")).unwrap_or(false) {
                    if let Ok(r) = self.import_zip(&p) {
                        reports.push(r);
                    }
                }
            }
        }
        Ok(reports)
    }

    /// Build the asset-compiler input index for a pack (what `compile-pack`
    /// consumes). Works identically for ZIP-extracted and folder packs.
    pub fn pack_index(&self, pack: &ResourcePack) -> Result<PackIndex, ImportError> {
        feathered_assets::pack::discover(&pack.dir).map_err(|e| ImportError::Io(pack.dir.clone(), e.message))
    }

    /// Delete an imported (ZIP) pack's extracted directory and metadata.
    /// Folder packs: only the metadata is removed; the original folder is
    /// left exactly as the user had it.
    pub fn uninstall(&self, pack: &ResourcePack) -> Result<(), ImportError> {
        match pack.kind {
            PackKind::Zip => {
                if pack.dir.is_dir() {
                    std::fs::remove_dir_all(&pack.dir).map_err(|e| self.io(&pack.dir, e))?;
                }
            }
            PackKind::Folder => {
                let meta = self.base.join(format!("{}.folder.json", pack.id));
                if meta.is_file() {
                    std::fs::remove_file(&meta).map_err(|e| self.io(&meta, e))?;
                }
            }
        }
        Ok(())
    }
}

impl ResourcePackManager {
    // helpers shared by import paths

    fn new_pack_dir(&self, name: &str) -> PathBuf {
        // Unique dir: slug + 8 hex of hash(name + timestamp).
        let mut hasher = Sha256::new();
        hasher.update(name.as_bytes());
        hasher.update(std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos().to_le_bytes())
            .unwrap_or([0; 16]));
        let hash = format!("{:x}", hasher.finalize());
        self.base.join(format!("{}-{}", slug_of(name), &hash[..8]))
    }

    fn validate_entries(
        archive: &mut zip::ZipArchive<std::fs::File>,
        zip_path: &Path,
    ) -> Result<PackValidation, ImportError> {
        let mut has_mcmeta = false;
        let mut namespace_dirs: Vec<String> = Vec::new();
        for i in 0..archive.len() {
            let entry = archive.by_index(i).map_err(|e| zip_err(zip_path, e))?;
            if entry.is_dir() {
                continue;
            }
            let name = entry.name().to_string();
            let Some(rel) = sanitize_entry(&name) else {
                return Err(ImportError::UnsafePath(zip_path.to_path_buf(), name));
            };
            // The pack root sits either at the archive root or under a
            // wrapper folder. Its depth is where the first root marker
            // (`assets/`, `pack.mcmeta`, `pack.png`) appears.
            let comps: Vec<&str> = rel.split('/').collect();
            let Some(root_depth) = comps
                .iter()
                .position(|c| *c == "assets" || *c == "pack.png" || c.ends_with(".mcmeta"))
            else {
                continue;
            };
            if comps.get(root_depth) == Some(&"pack.mcmeta") {
                has_mcmeta = true;
            }
            if comps.get(root_depth) == Some(&"assets") {
                if let Some(ns) = comps.get(root_depth + 1) {
                    if !ns.is_empty() {
                        namespace_dirs.push((*ns).to_string());
                    }
                }
            }
        }
        let mut problems = Vec::new();
        if !has_mcmeta {
            problems.push("missing pack.mcmeta".into());
        }
        if namespace_dirs.is_empty() {
            problems.push("no namespace folders (expected e.g. assets/minecraft/...)".into());
        }
        Ok(PackValidation {
            valid: has_mcmeta && !namespace_dirs.is_empty(),
            problems,
            pack_format: None,
            description: None,
        })
    }
}

/// Known top-level entries that are files-or-metadata, not namespaces.
/// Directories under `assets/` that are not namespaces (icons, lang indexes,
/// and similar metadata). Anything else under `assets/` counts as a namespace.
const NON_NAMESPACE_DIRS: &[&str] = &["icons", "lang", "sounds"];

/// Known non-pack files/dirs that may sit at the pack root and are never
/// namespaces themselves.
const KNOWN_PACK_FILES: &[&str] = &[
    "pack.mcmeta", "pack.png", "assets", "data", "overlay", "credits.txt", "license", "license.txt", "README.md", "README.txt",
];

/// Parse `pack.mcmeta`: pack_format (+ supported_formats), description as text.
pub fn read_mcmeta(path: &Path) -> Option<Mcmeta> {
    let text = std::fs::read_to_string(path).ok()?;
    parse_mcmeta_text(&text)
}

/// Parse mcmeta content from a string (testable without the filesystem).
pub fn parse_mcmeta_text(text: &str) -> Option<Mcmeta> {
    let v: serde_json::Value = serde_json::from_str(text).ok()?;
    let pack = v.get("pack")?;
    let pack_format = pack
        .get("pack_format")
        .and_then(|f| f.as_u64())
        .map(|f| f as u32);
    // Prefer a formats range covering the current pack_format if present.
    let supported = pack.get("supported_formats").and_then(|s| match s {
        serde_json::Value::Array(a) => match (a.first(), a.get(1)) {
            (Some(serde_json::Value::Number(a)), Some(serde_json::Value::Number(b))) => {
                Some((a.as_u64()? as u32, b.as_u64()? as u32))
            }
            _ => None,
        },
        serde_json::Value::Object(o) => match (o.get("min_inclusive"), o.get("max_inclusive")) {
            (Some(serde_json::Value::Number(a)), Some(serde_json::Value::Number(b))) => {
                Some((a.as_u64()? as u32, b.as_u64()? as u32))
            }
            _ => None,
        },
        _ => None,
    });
    let description = pack
        .get("description")
        .map(description_text)
        .filter(|s| !s.is_empty());
    Some(Mcmeta {
        pack_format,
        supported_formats: supported,
        description,
    })
}

/// Flatten a JSON text component to plain text (string, array, or object).
fn description_text(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => strip_section_codes(s),
        serde_json::Value::Array(a) => a.iter().map(description_text).collect::<Vec<_>>().join(""),
        serde_json::Value::Object(o) => match o.get("text").and_then(|t| t.as_str()) {
            Some(t) => {
                let mut out = strip_section_codes(t);
                if let Some(extra) = o.get("extra") {
                    out.push_str(&description_text(extra));
                }
                out
            }
            None => {
                if let Some(extra) = o.get("extra") {
                    description_text(extra)
                } else {
                    String::new()
                }
            }
        },
        _ => String::new(),
    }
}

/// Strip legacy `§x` color codes from descriptions.
fn strip_section_codes(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\u{a7}' {
            chars.next(); // skip the code char
        } else {
            out.push(c);
        }
    }
    out
}

#[derive(Debug, Clone)]
pub struct Mcmeta {
    pub pack_format: Option<u32>,
    pub supported_formats: Option<(u32, u32)>,
    pub description: Option<String>,
}

/// Validate a folder as a resource pack root: it needs `pack.mcmeta` and a
/// structure that looks like a pack. Two layouts are recognized:
/// * standard (`assets/<namespace>/…`) — imported/vanilla-style packs;
/// * Feathered-local (`<namespace>/…` directly at the root) — e.g. a pack
///   root that *is* an `assets` folder.
pub fn validate_folder(dir: &Path) -> Result<PackValidation, ImportError> {
    if !dir.is_dir() {
        return Err(ImportError::NotFound(dir.to_path_buf()));
    }
    let mcmeta = dir.join("pack.mcmeta");
    let has_mcmeta = mcmeta.is_file();
    let mut namespace_dirs = Vec::new();
    let assets = dir.join("assets");
    if assets.is_dir() {
        // Standard layout: namespaces live under `assets/` (excluding
        // known non-namespace metadata folders like `icons/`).
        if let Ok(entries) = std::fs::read_dir(&assets) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                if entry.path().is_dir() && !NON_NAMESPACE_DIRS.contains(&name.as_str()) {
                    namespace_dirs.push(name);
                }
            }
        }
    } else if let Ok(entries) = std::fs::read_dir(dir) {
        // Local layout: root-level directories are namespaces (excluding
        // known pack metadata files/dirs).
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if entry.path().is_dir() && !KNOWN_PACK_FILES.contains(&name.as_str()) {
                namespace_dirs.push(name);
            }
        }
    }
    let mut problems = Vec::new();
    if !has_mcmeta {
        problems.push("missing pack.mcmeta".into());
    }
    if namespace_dirs.is_empty() {
        problems.push("no namespace folders (expected e.g. assets/minecraft/...)".into());
    }
    let (pack_format, description) = read_mcmeta(&mcmeta)
        .map(|m| (m.pack_format, m.description))
        .unwrap_or((None, None));
    Ok(PackValidation {
        valid: has_mcmeta && !namespace_dirs.is_empty(),
        problems,
        pack_format,
        description,
    })
}

/// Ensure `path` is a safe relative subpath (ZIP-slip guard). The `base`
/// parameter documents intent for callers; the check itself is depth-based:
/// the path must never escape above the root at any point.
pub fn is_within(_base: &Path, path: &Path) -> bool {
    let mut depth: i32 = 0;
    for c in path.components() {
        match c {
            Component::Prefix(_) | Component::RootDir => return false,
            Component::CurDir => {}
            Component::ParentDir => {
                depth -= 1;
                if depth < 0 {
                    return false;
                }
            }
            Component::Normal(_) => depth += 1,
        }
    }
    true
}

// ---------------------------------------------------------------------------
// filesystem helpers
// ---------------------------------------------------------------------------

const META_FILE: &str = "feathered.pack.json";

fn zip_err(path: &Path, e: zip::result::ZipError) -> ImportError {
    ImportError::Io(path.to_path_buf(), e.to_string())
}

/// Filesystem-safe lowercase slug used in pack directory and id names.
pub(crate) fn slug_of(name: &str) -> String {
    let mut out = String::new();
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if matches!(c, ' ' | '-' | '_' | '.') && !out.ends_with('-') {
            out.push('-');
        }
    }
    out.trim_matches('-').to_string()
}

fn zip_name(p: &Path) -> String {
    p.file_stem()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "pack".into())
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
            return None; // a file at the archive root: no wrapper folder
        };
        if rest.is_empty() {
            continue;
        }
        let head = head.to_string();
        match &first {
            None => first = Some(head),
            Some(f) => {
                if *f != head {
                    return None;
                }
            }
        }
    }
    first.map(PathBuf::from)
}

/// Reject pathologically-structured archive entry names and normalize them to
/// a relative path. Absolute paths and parent-escapes are refused outright.
fn sanitize_entry(name: &str) -> Option<String> {
    if name.starts_with('/') || name.starts_with('\\') || name.contains(":/") {
        return None;
    }
    let p = Path::new(name);
    if p.is_absolute() || !is_within(Path::new(""), p) {
        return None;
    }
    let normalized: Vec<&str> = name
        .split(['/', '\\'])
        .filter(|s| !s.is_empty() && *s != ".")
        .collect();
    if normalized.is_empty() {
        return None;
    }
    Some(normalized.join("/"))
}

fn pack_id(dir: &Path) -> String {
    format!("{}-{}", slug_of(&dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()), &hash_file_short(dir))
}

/// Directory names too generic to use as a pack display name.
fn is_generic_dir_name(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "assets" | "texture" | "textures" | "resourcepacks" | "packs" | "pack" | "shaders"
    )
}

fn hash_file(path: &Path) -> String {
    let mut hasher = Sha256::new();
    if let Ok(bytes) = std::fs::read(path) {
        hasher.update(&bytes);
    }
    format!("{:x}", hasher.finalize())
}

fn hash_file_short(dir: &Path) -> String {
    let mut hasher = Sha256::new();
    if let Ok(bytes) = std::fs::read(dir.join("pack.mcmeta")) {
        hasher.update(&bytes);
    }
    let out = format!("{:x}", hasher.finalize());
    out[..8].to_string()
}

fn write_meta(dir: &Path, pack: &ResourcePack) -> Result<(), ImportError> {
    write_json(&dir.join(META_FILE), pack)
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), ImportError> {
    let text = serde_json::to_string_pretty(value).map_err(|e| ImportError::Io(path.to_path_buf(), e.to_string()))?;
    std::fs::write(path, text).map_err(|e| ImportError::Io(path.to_path_buf(), e.to_string()))
}

#[cfg(test)]
mod tests;
