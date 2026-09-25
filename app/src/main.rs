//! feathered — Phase 1 CLI.
//!
//! Subcommands:
//! * `compile-pack [--required] [--pack-dir <dir>] [--out <cache>]`
//!   Compile the resource pack into the binary cache (FEAT header + bincode).
//! * `validate`  Run the milestone validation checks against the compiled cache.
//! * `render`    Launch the ten-block validation scene (winit + wgpu).

use std::path::PathBuf;

mod validate;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("help");

    let result = match cmd {
        "compile-pack" => compile_pack(&args[2..]),
        "validate" => validate(&args[2..]),
        "render" => render(&args[2..]),
        _ => {
            print_help();
            Ok(())
        }
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn print_help() {
    println!(
        "feathered — Minecraft-compatible client (Phase 1)\n\n\
         USAGE:\n  \
         feathered compile-pack [--required] [--pack-dir <dir>] [--out <file>]\n  \
         feathered validate [--pack-dir <dir>] [--cache <file>]\n  \
         feathered render [--pack-dir <dir>] [--cache <file>] [--screenshot <file>]\n\n\
         Default pack dir: ./texture/assets  Default cache: ./target/feathered-cache.bin"
    );
}

fn default_cache() -> PathBuf {
    PathBuf::from("target/feathered-cache.bin")
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// Compile the resource pack. `--required` compiles only the ten validation
/// blocks (fast iteration); default compiles the whole namespace.
fn compile_pack(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let pack_dir = PathBuf::from(arg_value(args, "--pack-dir").unwrap_or_else(|| "texture/assets".into()));
    let out = PathBuf::from(arg_value(args, "--out").unwrap_or_else(|| default_cache().into_os_string().into_string().unwrap()));
    let required_only = args.iter().any(|a| a == "--required");

    let t0 = std::time::Instant::now();
    let index = feathered_assets::pack::discover(&pack_dir)?;
    println!(
        "discovered {} pack(s): {}",
        index.packs.len(),
        index
            .packs
            .iter()
            .filter_map(|p| p.description.clone())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("indexed {} files", index.files.len());

    let (pack, atlas, stats) = feathered_assets::compiler::compile_pack(&index, required_only)?;

    // Content digest over every indexed file.
    let mut digest_files: Vec<(String, Vec<u8>)> = Vec::new();
    for ((ns, path), file) in &index.files {
        if let Ok(bytes) = std::fs::read(file) {
            digest_files.push((format!("{ns}:{path}"), bytes));
        }
    }
    digest_files.sort();
    let digest = feathered_assets::cache::content_digest(&digest_files);

    let payload = feathered_assets::cache::CachePayload {
        atlas: feathered_assets::cache::atlas_to_cached(&atlas, &pack.sprite_names),
        pack,
    };
    let blob = feathered_assets::cache::encode(digest, &payload)?;
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&out, &blob)?;

    println!(
        "compiled: {} sprites, {} models, {} blocks, {} animations, skipped {} blocks",
        stats.sprites, stats.models, stats.blocks, stats.animations, stats.skipped_blocks
    );
    println!(
        "atlas: {}x{} ({} mips)",
        stats.atlas_size.0,
        stats.atlas_size.1,
        atlas.mip_levels()
    );
    println!(
        "cache: {} ({} KB, digest {:02x}{:02x}...) in {:.2?}",
        out.display(),
        blob.len() / 1024,
        digest[0],
        digest[1],
        t0.elapsed()
    );
    Ok(())
}

/// Load the compiled cache (rebuilding if missing/stale) and run validation.
fn validate(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let pack_dir = PathBuf::from(arg_value(args, "--pack-dir").unwrap_or_else(|| "texture/assets".into()));
    let cache_path = PathBuf::from(arg_value(args, "--cache").unwrap_or_else(|| default_cache().into_os_string().into_string().unwrap()));

    let blob = std::fs::read(&cache_path)?;
    let (_, payload) = feathered_assets::cache::decode(&blob)
        .map_err(|e| format!("cache {} unusable: {e} (run compile-pack)", cache_path.display()))?;

    let runtime = feathered_world::Registry::from_compiled(payload.pack);
    let atlas = feathered_assets::cache::cached_to_atlas(&payload.atlas, &runtime.sprite_names());

    let index = feathered_assets::pack::discover(&pack_dir)?;
    let store = feathered_assets::sprites::SpriteStore::load(&index)?;

    let report = self::validate::validate(&runtime, &atlas, &store)?;
    report.print();
    if report.failures() > 0 {
        Err(format!("validation failed: {} failures", report.failures()).into())
    } else {
        println!("\nVALIDATION PASSED");
        Ok(())
    }
}

/// Launch the ten-block validation scene.
fn render(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    feathered_client::run(feathered_client::RunOptions {
        pack_dir: PathBuf::from(arg_value(args, "--pack-dir").unwrap_or_else(|| "texture/assets".into())),
        cache_path: PathBuf::from(arg_value(args, "--cache").unwrap_or_else(|| default_cache().into_os_string().into_string().unwrap())),
        screenshot: arg_value(args, "--screenshot").map(PathBuf::from),
    })
}
