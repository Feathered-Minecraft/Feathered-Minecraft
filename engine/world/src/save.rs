//! World persistence: metadata + a compact journal of player edits.
//!
//! Only *modified* blocks are stored — terrain is deterministic from the
//! seed, so untouched chunks regenerate identically and are never written.
//! A modified block is stored as one (position → (block, state)) override;
//! on load the overrides are replayed onto freshly generated chunks by the
//! streaming layer (they survive unload/reload within a session too).
//!
//! Format (little-endian, versioned):
//! ```text
//! "FEATWORLD" (9 bytes magic) · u32 format version
//! metadata section: u32 len · serde_json bytes
//!   { seed, player { pos, yaw, pitch }, day_fraction, saved_at_unix }
//! edits section: u32 edit count · per edit:
//!   i64 x · i64 y · i64 z · u32 block · u32 state
//! ```
//!
//! Writes are atomic: the payload is serialized to a temp file in the same
//! directory and renamed over the target, so an interruption (crash, power
//! loss) can never leave a half-written save — readers see either the old
//! or the new file. The temp rename is retried briefly on Windows, where a
//! concurrent reader can momentarily hold the destination open.
//!
//! No networking, no external crates beyond the workspace's serde pair.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Current on-disk format version. Bump on any layout change; loaders must
/// reject (or migrate) older versions explicitly.
pub const FORMAT_VERSION: u32 = 1;

/// Magic identifying a Feathered world save.
pub const MAGIC: &[u8; 9] = b"FEATWORLD";

/// Player state saved with the world.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PlayerSave {
    pub pos: [f64; 3],
    pub yaw: f32,
    pub pitch: f32,
}

/// World metadata (the "header": everything except block edits).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WorldMeta {
    pub seed: u64,
    pub player: PlayerSave,
    /// Time-of-day fraction 0..1 (0 sunrise, 0.25 noon, 0.5 sunset, 0.75
    /// midnight). `None` = frozen sun.
    pub day_fraction: Option<f32>,
    /// Wall-clock save time (unix seconds) — informational only.
    pub saved_at_unix: Option<u64>,
}

/// One persisted block edit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Edit {
    pub x: i64,
    pub y: i64,
    pub z: i64,
    pub block: u32,
    pub state: u32,
}

/// The save format itself: metadata + the edit set. (Not the on-disk byte
/// layout — see the module docs for that.)
#[derive(Debug, Clone, PartialEq)]
pub struct WorldSave {
    pub meta: WorldMeta,
    /// (x, y, z) → (block, state). A (0, 0) value means "the player broke
    /// the block here" — terrain regenerates air at that position only if
    /// the generator would have placed air; the override wins either way.
    pub edits: HashMap<(i64, i64, i64), (u32, u32)>,
}

impl WorldSave {
    pub fn new(meta: WorldMeta) -> WorldSave {
        WorldSave {
            meta,
            edits: HashMap::new(),
        }
    }

    /// Record one edit (breaks record (0, 0)).
    pub fn set_edit(&mut self, x: i64, y: i64, z: i64, block: u32, state: u32) {
        self.edits.insert((x, y, z), (block, state));
    }
}

// ---------------------------------------------------------------------------
// serialization
// ---------------------------------------------------------------------------

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_i64(out: &mut Vec<u8>, v: i64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn read_u32(src: &[u8], off: &mut usize) -> Option<u32> {
    let b = src.get(*off..*off + 4)?;
    *off += 4;
    Some(u32::from_le_bytes(b.try_into().ok()?))
}

fn read_i64(src: &[u8], off: &mut usize) -> Option<i64> {
    let b = src.get(*off..*off + 8)?;
    *off += 8;
    Some(i64::from_le_bytes(b.try_into().ok()?))
}

/// Serialize a save into the versioned binary format.
pub fn encode(save: &WorldSave) -> Result<Vec<u8>, String> {
    let meta = serde_json::to_vec(&save.meta).map_err(|e| e.to_string())?;
    let mut out = Vec::with_capacity(MAGIC.len() + 8 + meta.len() + save.edits.len() * 24);
    out.extend_from_slice(MAGIC);
    put_u32(&mut out, FORMAT_VERSION);
    put_u32(&mut out, meta.len() as u32);
    out.extend_from_slice(&meta);
    // Sorted for a deterministic byte layout (stable across saves when
    // nothing changed — makes save files diffable and hash-stable).
    let mut edits: Vec<(&(i64, i64, i64), &(u32, u32))> = save.edits.iter().collect();
    edits.sort_by_key(|(k, _)| (**k, ()));
    put_u32(&mut out, edits.len() as u32);
    for (&(x, y, z), &(block, state)) in edits {
        put_i64(&mut out, x);
        put_i64(&mut out, y);
        put_i64(&mut out, z);
        put_u32(&mut out, block);
        put_u32(&mut out, state);
    }
    Ok(out)
}

/// Deserialize a save; rejects wrong magic, wrong version, truncated data
/// and trailing garbage (a corrupt file fails loudly instead of loading
/// half a world).
pub fn decode(bytes: &[u8]) -> Result<WorldSave, String> {
    if bytes.len() < MAGIC.len() + 8 {
        return Err("save too short".into());
    }
    if &bytes[..MAGIC.len()] != MAGIC {
        return Err("not a Feathered world save (bad magic)".into());
    }
    let mut off = MAGIC.len();
    let version = read_u32(bytes, &mut off).ok_or("truncated version")?;
    if version != FORMAT_VERSION {
        return Err(format!("unsupported save format version {version} (expected {FORMAT_VERSION})"));
    }
    let meta_len = read_u32(bytes, &mut off).ok_or("truncated meta length")? as usize;
    let meta_end = off
        .checked_add(meta_len)
        .filter(|&e| e <= bytes.len())
        .ok_or("truncated metadata")?;
    let meta: WorldMeta = serde_json::from_slice(&bytes[off..meta_end]).map_err(|e| e.to_string())?;
    off = meta_end;

    let count = read_u32(bytes, &mut off).ok_or("truncated edit count")? as usize;
    let mut edits = HashMap::with_capacity(count);
    for _ in 0..count {
        let x = read_i64(bytes, &mut off).ok_or("truncated edit")?;
        let y = read_i64(bytes, &mut off).ok_or("truncated edit")?;
        let z = read_i64(bytes, &mut off).ok_or("truncated edit")?;
        let block = read_u32(bytes, &mut off).ok_or("truncated edit")?;
        let state = read_u32(bytes, &mut off).ok_or("truncated edit")?;
        if block == 0 && state != 0 {
            return Err(format!("invalid edit (air with state {state}) at {x},{y},{z}"));
        }
        edits.insert((x, y, z), (block, state));
    }
    if off != bytes.len() {
        return Err("trailing bytes after edit section".into());
    }
    Ok(WorldSave { meta, edits })
}

// ---------------------------------------------------------------------------
// atomic file IO
// ---------------------------------------------------------------------------

/// Serialize + write atomically: temp file in the same directory, fsync,
/// rename over the destination. Returns the temp path used (for tests).
pub fn save_to_dir(dir: &Path, save: &WorldSave) -> Result<PathBuf, String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let path = dir.join("world.feathered");
    let tmp = dir.join("world.feathered.tmp");
    let bytes = encode(save)?;

    // Write + fsync so the data is on disk before the rename publishes it.
    {
        let mut f = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
        f.write_all(&bytes).map_err(|e| e.to_string())?;
        f.sync_all().map_err(|e| e.to_string())?;
    }

    // Rename may transiently fail on Windows if an antivirus/reader holds
    // the destination open — retry briefly before giving up.
    let mut err = None;
    for attempt in 0..10u32 {
        match std::fs::rename(&tmp, &path) {
            Ok(()) => {
                err = None;
                break;
            }
            Err(e) => {
                err = Some(e.to_string());
                if attempt == 9 {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(20 * (attempt + 1) as u64));
            }
        }
    }
    if let Some(e) = err {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("atomic rename failed: {e}"));
    }

    // Best-effort directory fsync (not supported everywhere).
    if let Ok(d) = std::fs::File::open(dir) {
        let _ = d.sync_all();
    }
    Ok(tmp)
}

/// Load from `dir/world.feathered`.
pub fn load_from_dir(dir: &Path) -> Result<WorldSave, String> {
    let path = dir.join("world.feathered");
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    decode(&bytes)
}

/// True when `dir` contains a usable save (no error reason — just a check).
pub fn exists(dir: &Path) -> bool {
    load_from_dir(dir).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(seed: u64) -> WorldMeta {
        WorldMeta {
            seed,
            player: PlayerSave {
                pos: [1.5, 42.0, -3.25],
                yaw: 1.0,
                pitch: -0.5,
            },
            day_fraction: Some(0.3),
            saved_at_unix: Some(1_700_000_000),
        }
    }

    #[test]
    fn round_trips_metadata_and_edits() {
        let mut save = WorldSave::new(meta(20260926));
        save.set_edit(0, 40, 0, 1, 0);
        save.set_edit(-100_000, 5, -100_000, 9, 3); // negative coords
        save.set_edit(1, 41, 2, 0, 0); // a break (air override)
        let bytes = encode(&save).unwrap();
        let back = decode(&bytes).unwrap();
        assert_eq!(back, save, "full save round-trip");
    }

    #[test]
    fn encoding_is_deterministic_and_stable() {
        let mut a = WorldSave::new(meta(7));
        a.set_edit(3, 4, 5, 1, 0);
        let mut b = WorldSave::new(meta(7));
        // Insert in a different order; the encoder sorts, so bytes match.
        b.set_edit(-9, 4, 5, 2, 1);
        b.set_edit(3, 4, 5, 1, 0);
        a.set_edit(-9, 4, 5, 2, 1);
        assert_eq!(encode(&a).unwrap(), encode(&b).unwrap());
        // And re-encoding the decoded save is byte-identical (stable format).
        let back = decode(&encode(&a).unwrap()).unwrap();
        assert_eq!(encode(&back).unwrap(), encode(&a).unwrap());
    }

    #[test]
    fn decode_rejects_corruption() {
        let mut save = WorldSave::new(meta(1));
        save.set_edit(0, 0, 0, 1, 0);
        let bytes = encode(&save).unwrap();

        // Bad magic.
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert!(decode(&bad).is_err());
        // Truncation.
        assert!(decode(&bytes[..bytes.len() - 3]).is_err());
        assert!(decode(&bytes[..12]).is_err());
        // Wrong version.
        let mut bad = bytes.clone();
        let voff = MAGIC.len();
        bad[voff..voff + 4].copy_from_slice(&99u32.to_le_bytes());
        assert!(decode(&bad).is_err());
        // Trailing garbage.
        let mut bad = bytes.clone();
        bad.push(0xAB);
        assert!(decode(&bad).is_err());
        // Empty.
        assert!(decode(&[]).is_err());
    }

    #[test]
    fn decode_rejects_air_edits_with_a_state() {
        let bytes = {
            let mut save = WorldSave::new(meta(1));
            // Hand-craft: write the edit through the encoder path but with a
            // bogus (0, 5) pair — encode does not police it, decode does.
            save.set_edit(0, 0, 0, 0, 5);
            encode(&save).unwrap()
        };
        assert!(decode(&bytes).is_err(), "air edits must carry state 0");
    }

    #[test]
    fn atomic_save_overwrites_and_survives_a_reader() {
        let dir = std::env::temp_dir().join(format!("feathered-save-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let mut v1 = WorldSave::new(meta(1));
        v1.set_edit(0, 0, 0, 1, 0);
        save_to_dir(&dir, &v1).unwrap();
        assert!(exists(&dir));

        // A "reader" holds the file open (Windows rename contention) — the
        // retry loop must still publish v2.
        let _held = std::fs::File::open(dir.join("world.feathered")).unwrap();
        let mut v2 = WorldSave::new(meta(2));
        v2.set_edit(5, 5, 5, 3, 0);
        save_to_dir(&dir, &v2).unwrap();

        let loaded = load_from_dir(&dir).unwrap();
        assert_eq!(loaded.meta.seed, 2, "new save wins");
        assert!(loaded.edits.contains_key(&(5, 5, 5)));
        // No temp litter left behind.
        assert!(!dir.join("world.feathered.tmp").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_missing_file_reports_cleanly() {
        let dir = std::env::temp_dir().join(format!("feathered-save-missing-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        assert!(load_from_dir(&dir).is_err());
        assert!(!exists(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn json_f64_precision_survives_player_positions() {
        // Positions go through serde_json as f64: assert a realistic value
        // round-trips exactly (it is stored as f64, not truncated).
        let mut m = meta(3);
        m.player.pos = [12_345.678_901_234_5, -0.5, 0.1];
        let mut save = WorldSave::new(m);
        save.set_edit(0, 0, 0, 1, 0);
        let back = decode(&encode(&save).unwrap()).unwrap();
        assert_eq!(back.meta.player.pos, m.player.pos);
    }
}
