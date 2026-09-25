//! Filesystem discovery: locate pack roots and build a `namespace:path` index
//! of every file under them.

use crate::error::{AssetError, AssetResult};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// One resource-pack root (e.g. `texture/assets`).
#[derive(Debug, Clone)]
pub struct PackRoot {
    /// Directory that directly contains `pack.mcmeta` / namespace folders.
    pub root: PathBuf,
    /// Human description from `pack.mcmeta`, if present.
    pub description: Option<String>,
}

/// Index of every file in every pack, keyed by `namespace:path` (no extension).
#[derive(Debug, Default)]
pub struct PackIndex {
    /// `("minecraft", "block/stone")` -> `texture/assets/minecraft/textures/block/stone.png`
    /// The stored path is the actual file on disk (extension included).
    pub files: HashMap<(String, String), PathBuf>,
    /// Packs scanned, in priority order (first = highest).
    pub packs: Vec<PackRoot>,
}

impl PackIndex {
    pub fn get(&self, namespace: &str, path: &str) -> Option<&PathBuf> {
        self.files.get(&(namespace.to_string(), path.to_string()))
    }
}

fn read_pack_description(mcmeta: &Path) -> Option<String> {
    let text = std::fs::read_to_string(mcmeta).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    v.get("pack")
        .and_then(|p| p.get("description"))
        .and_then(|d| d.as_str())
        .map(|s| s.to_string())
}

/// Discover pack roots under `base`. A directory containing `pack.mcmeta` or
/// `.mcassetsroot` is a pack root; namespaces are its direct child directories.
pub fn discover(base: &Path) -> AssetResult<PackIndex> {
    let mut packs = Vec::new();
    let mut stack = vec![base.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir).map_err(|e| AssetError {
            path: dir.display().to_string(),
            message: format!("cannot read directory: {e}"),
        })?;
        let mut has_mcmeta = false;
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().map(|n| n == "pack.mcmeta").unwrap_or(false) {
                has_mcmeta = true;
            }
        }
        if has_mcmeta {
            packs.push(PackRoot {
                description: read_pack_description(&dir.join("pack.mcmeta")),
                root: dir,
            });
        }
    }
    if packs.is_empty() {
        return crate::error::err(
            &base.display().to_string(),
            "no resource pack found (no pack.mcmeta)",
        );
    }
    // Sort for deterministic ordering; deepest path first is irrelevant here,
    // alphabetical keeps output stable.
    packs.sort_by(|a, b| a.root.cmp(&b.root));

    let mut index = PackIndex { packs, ..Default::default() };
    for pack in &index.packs.clone() {
        for ns_entry in std::fs::read_dir(&pack.root)
            .map_err(|e| AssetError {
                path: pack.root.display().to_string(),
                message: format!("cannot read pack root: {e}"),
            })?
            .flatten()
        {
            let ns_path = ns_entry.path();
            if !ns_path.is_dir() {
                continue;
            }
            let namespace = ns_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_string();
            for entry in walk(&ns_path) {
                let rel = entry
                    .strip_prefix(&ns_path)
                    .unwrap_or(&entry)
                    .to_string_lossy()
                    .replace('\\', "/");
                if rel.starts_with('.') {
                    continue;
                }
                // Strip the final extension: `textures/block/stone.png` -> `textures/block/stone`
                let stem = match rel.rsplit_once('.') {
                    Some((stem, ext)) if !ext.is_empty() && !stem.ends_with('.') => stem.to_string(),
                    _ => rel.clone(),
                };
                let key = (namespace.clone(), stem);
                // `.png.mcmeta` and `.png` both stem to `<name>.png`; the
                // mcmeta sidecar must win so animations resolve regardless of
                // filesystem walk order.
                match index.files.get_mut(&key) {
                    None => {
                        index.files.insert(key, entry.clone());
                    }
                    Some(existing) => {
                        let existing_str = existing.to_string_lossy();
                        let new_str = entry.to_string_lossy().into_owned();
                        let existing_is_mcmeta = existing_str.ends_with(".mcmeta");
                        let new_is_mcmeta = new_str.ends_with(".mcmeta");
                        if new_is_mcmeta && !existing_is_mcmeta {
                            *existing = entry.clone();
                        }
                    }
                }
            }
        }
    }
    Ok(index)
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.push(p);
            }
        }
    }
    out
}
