//! Binary cache format.
//!
//! Layout (big-endian):
//! ```text
//! "FEAT" (4 bytes magic)
//! u32 format_version
//! u32 compiler_version
//! [u8; 32] content digest (SHA-256 over sorted (path, bytes) of every input)
//! u64 payload_length
//! payload (bincode-serialized CompiledPack + Atlas)
//! ```
//! Stale/incompatible caches are rejected before deserialization and rebuilt —
//! a new compiler never misreads an old binary layout.

use crate::error::{AssetError, AssetResult};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const MAGIC: &[u8; 4] = b"FEAT";
/// Bump when the compiled data structures change shape.
pub const FORMAT_VERSION: u32 = 1;
/// Bump when compiler output semantics change (same structs, different rules).
pub const COMPILER_VERSION: u32 = 1;

/// What gets serialized into the cache payload.
#[derive(Serialize, Deserialize)]
pub struct CachePayload {
    pub pack: crate::compiled::CompiledPack,
    pub atlas: CachedAtlas,
}

/// Atlas data with mip chain flattened for serialization.
#[derive(Serialize, Deserialize)]
pub struct CachedAtlas {
    pub width: u32,
    pub height: u32,
    pub mip0: Vec<u8>,
    pub mips: Vec<Vec<u8>>,
    /// (sprite key index, x, y, frame_w, frame_h, frames, frame_stride, slot_stride)
    pub entries: Vec<(u32, u32, u32, u32, u32, u32, u32, u32)>,
}

/// SHA-256 over every (relative path, file bytes) in sorted order.
pub fn content_digest(files: &[(String, Vec<u8>)]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for (path, bytes) in files {
        hasher.update(path.as_bytes());
        hasher.update(&[0]);
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }
    hasher.finalize().into()
}

/// Serialize payload + header to bytes.
pub fn encode(digest: [u8; 32], payload: &CachePayload) -> AssetResult<Vec<u8>> {
    let body = bincode::serialize(payload).map_err(|e| AssetError {
        path: "<cache>".into(),
        message: format!("bincode serialize failed: {e}"),
    })?;
    let mut out = Vec::with_capacity(body.len() + 52);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_be_bytes());
    out.extend_from_slice(&COMPILER_VERSION.to_be_bytes());
    out.extend_from_slice(&digest);
    out.extend_from_slice(&(body.len() as u64).to_be_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}

/// Decode + validate a cache file. Errors name the reason for rejection.
pub fn decode(bytes: &[u8]) -> AssetResult<([u8; 32], CachePayload)> {
    if bytes.len() < 52 {
        return crate::error::err("<cache>", "cache too small (missing header)");
    }
    if &bytes[0..4] != MAGIC {
        return crate::error::err("<cache>", "bad magic (not a feathered cache)");
    }
    let fmt_ver = u32::from_be_bytes(bytes[4..8].try_into().unwrap());
    let comp_ver = u32::from_be_bytes(bytes[8..12].try_into().unwrap());
    if fmt_ver != FORMAT_VERSION {
        return crate::error::err(
            "<cache>",
            format!("format version {fmt_ver} != supported {FORMAT_VERSION}"),
        );
    }
    if comp_ver != COMPILER_VERSION {
        return crate::error::err(
            "<cache>",
            format!("compiler version {comp_ver} != supported {COMPILER_VERSION}"),
        );
    }
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&bytes[12..44]);
    let payload_len = u64::from_be_bytes(bytes[44..52].try_into().unwrap()) as usize;
    if bytes.len() != 52 + payload_len {
        return crate::error::err(
            "<cache>",
            format!("payload length {payload_len} != file remainder {}", bytes.len() - 52),
        );
    }
    let payload: CachePayload = bincode::deserialize(&bytes[52..]).map_err(|e| AssetError {
        path: "<cache>".into(),
        message: format!("payload deserialize failed: {e}"),
    })?;
    Ok((digest, payload))
}

/// Round-trip helpers used by the pack compiler.
pub fn atlas_to_cached(atlas: &crate::atlas::Atlas, sprite_names: &[(String, String)]) -> CachedAtlas {
    let mut index_of = std::collections::HashMap::new();
    for (i, n) in sprite_names.iter().enumerate() {
        index_of.insert(n, i as u32);
    }
    let mut entries = Vec::new();
    let mut sorted: Vec<_> = atlas.entries.iter().collect();
    sorted.sort_by_key(|(k, _)| index_of.get(k).copied().unwrap_or(u32::MAX));
    for (name, e) in sorted {
        let key = index_of.get(name).copied().unwrap_or(u32::MAX);
        entries.push((key, e.x, e.y, e.frame_w, e.frame_h, e.frames, e.frame_stride, e.slot_stride));
    }
    CachedAtlas {
        width: atlas.width,
        height: atlas.height,
        mip0: atlas.pixels.clone(),
        mips: atlas.mips.clone(),
        entries,
    }
}

pub fn cached_to_atlas(cached: &CachedAtlas, sprite_names: &[(String, String)]) -> crate::atlas::Atlas {
    let mut entries = std::collections::HashMap::new();
    for (key, x, y, fw, fh, frames, fstride, sstride) in &cached.entries {
        if let Some(name) = sprite_names.get(*key as usize) {
            entries.insert(
                name.clone(),
                crate::atlas::AtlasEntry {
                    x: *x,
                    y: *y,
                    frame_w: *fw,
                    frame_h: *fh,
                    frames: *frames,
                    frame_stride: *fstride,
                    slot_stride: *sstride,
                },
            );
        }
    }
    crate::atlas::Atlas {
        width: cached.width,
        height: cached.height,
        pixels: cached.mip0.clone(),
        mips: cached.mips.clone(),
        entries,
        animations: std::collections::HashMap::new(), // rebuilt from pack.animations
    }
}
