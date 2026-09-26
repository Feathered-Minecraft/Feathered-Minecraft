//! feathered — Phase 2 CLI.
//!
//! Subcommands:
//! * `packs`       — resource-pack management (list/import/use/uninstall/…)
//! * `shaders`     — shader-pack management (list/import/enable/disable/…)
//! * `compile-pack` — compile the selected pack into the binary cache
//! * `validate`    — run the milestone validation checks
//! * `render`      — launch the validation scene
//!
//! First launch: when no pack is installed/selected, Feathered explains that
//! it ships no game assets and offers to import one — it never downloads.

use std::path::PathBuf;

mod first_run;
mod validate;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("help");

    let result = match cmd {
        "first-run" => first_run(),
        "packs" => packs(&args[2..]),
        "shaders" => shaders(&args[2..]),
        "compile-pack" => compile_pack(&args[2..]),
        "validate" => validate(&args[2..]),
        "render" => render(&args[2..]),
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        _ => {
            print_help();
            Err(format!("unknown command: {cmd}").into())
        }
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn print_help() {
    println!(
        "feathered — an independent Minecraft-compatible engine (Phase 2)\n\
\n\
Feathered ships with no game assets: you supply a resource pack.\n\
\n\
USAGE:\n  \
feathered first-run                     Interactive first-launch setup (pack import + optional shaders)\n  \
feathered packs list|import <path>|use <id>|uninstall <id>|open-folder|browse\n  \
feathered shaders list|import <path>|enable <id>|disable|uninstall <id>\n  \
feathered compile-pack [--required] [--pack-dir <dir>] [--out <cache>]\n  \
feathered validate [--pack-dir <dir>] [--cache <file>]\n  \
feathered render [--pack-dir <dir>] [--cache <file>] [--screenshot <file>] [--quality low|medium|high|ultra]\n\n\
Default pack dir: ./texture/assets   Default cache: ./target/feathered-cache.bin\n\
Quality presets: low (half-res), medium (default), high (+post), ultra (max)."
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

fn arg_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn parse_quality(args: &[String]) -> feathered_packs::Quality {
    match arg_value(args, "--quality").as_deref().map(str::to_ascii_lowercase).as_deref() {
        Some("low") => feathered_packs::Quality::Low,
        Some("high") => feathered_packs::Quality::High,
        Some("ultra") => feathered_packs::Quality::Ultra,
        _ => feathered_packs::Quality::Medium,
    }
}

/// Map the settings-side preset onto the renderer's engine-side preset.
fn render_quality(q: feathered_packs::Quality) -> feathered_renderer::RenderQuality {
    match q {
        feathered_packs::Quality::Low => feathered_renderer::RenderQuality::Low,
        feathered_packs::Quality::Medium => feathered_renderer::RenderQuality::Medium,
        feathered_packs::Quality::High => feathered_renderer::RenderQuality::High,
        feathered_packs::Quality::Ultra => feathered_renderer::RenderQuality::Ultra,
    }
}

// ---------------------------------------------------------------------------
// first-launch flow
// ---------------------------------------------------------------------------

/// Interactive first launch: explains the no-assets policy and imports a pack.
fn first_run() -> Result<(), Box<dyn std::error::Error>> {
    first_run::run()
}

// ---------------------------------------------------------------------------
// resource packs
// ---------------------------------------------------------------------------

fn packs(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mgr = feathered_packs::ResourcePackManager::open()?;
    mgr.initialize()?;
    let action = args.first().map(String::as_str).unwrap_or("list");

    match action {
        "list" => {
            let packs = mgr.list()?;
            if packs.is_empty() {
                println!("no resource packs installed.");
                println!("import one with: feathered packs import <file-or-folder>");
            }
            for p in &packs {
                println!(
                    "{}  {:<24} {:<12} [{}]  {}",
                    p.id,
                    p.name,
                    p.version_label(),
                    match p.kind {
                        feathered_packs::PackKind::Zip => "zip",
                        feathered_packs::PackKind::Folder => "folder",
                    },
                    p.description.as_deref().unwrap_or("")
                );
            }
            let settings = mgr.settings();
            match settings.active_pack {
                Some(ref id) => println!("active pack: {id}"),
                None => println!("active pack: (none selected — run `feathered first-run` or `packs use <id>`)"),
            }
        }
        "import" => {
            let Some(path) = args.get(1) else {
                return Err("usage: feathered packs import <path>".into());
            };
            let p = PathBuf::from(path);
            let report = if p.is_dir() {
                mgr.import_folder(&p)?
            } else {
                mgr.import_zip(&p)?
            };
            println!(
                "imported \"{}\" ({}, {})",
                report.pack.name,
                report.pack.version_label(),
                if report.files_copied > 0 {
                    format!("{} files copied", report.files_copied)
                } else {
                    "referenced in place".into()
                }
            );
            if let Some(note) = &report.pack.source.license_note {
                println!("license note: {note}");
            }
        }
        "use" => {
            let Some(id) = args.get(1) else {
                return Err("usage: feathered packs use <id>".into());
            };
            let packs = mgr.list()?;
            let pack = packs
                .iter()
                .find(|p| p.id == *id || p.id.starts_with(id.as_str()) || p.name.eq_ignore_ascii_case(id))
                .ok_or_else(|| format!("no installed pack matches \"{id}\" (see `feathered packs list`)"))?;
            let mut settings = mgr.settings();
            settings.active_pack = Some(pack.id.clone());
            settings.pack_format = pack.pack_format;
            settings.onboarded = true;
            mgr.save_settings(&settings)?;
            println!("active pack: {} ({})", pack.name, pack.version_label());
            println!("compile it with: feathered compile-pack");
        }
        "uninstall" => {
            let Some(id) = args.get(1) else {
                return Err("usage: feathered packs uninstall <id>".into());
            };
            let packs = mgr.list()?;
            let pack = packs
                .iter()
                .find(|p| p.id.starts_with(id.as_str()) || p.name.eq_ignore_ascii_case(id))
                .ok_or_else(|| format!("no installed pack matches \"{id}\""))?;
            mgr.uninstall(pack)?;
            let mut settings = mgr.settings();
            if settings.active_pack.as_deref() == Some(pack.id.as_str()) {
                settings.active_pack = None;
                mgr.save_settings(&settings)?;
            }
            println!("uninstalled: {}", pack.name);
        }
        "open-folder" => {
            let settings = mgr.settings();
            std::fs::create_dir_all(mgr.base_dir())?;
            println!("resource-pack folder: {}", mgr.base_dir().display());
            if !settings.onboarded {
                println!("(zip files placed here are found by `feathered packs list` after import)");
            }
            open_in_explorer(mgr.base_dir());
        }
        "browse" => {
            println!("Feathered does not host or download packs. These external sites distribute packs under their own licenses:\n");
            for (label, url) in feathered_packs::BROWSE_LINKS {
                println!("  {label}\n    {url}");
            }
            println!("\n{}", feathered_packs::DEFAULT_TEMPLATE_NOTE);
            println!("Feathered itself remains GPL-3.0 — third-party packs keep their own licenses.");
        }
        "scan" => {
            // Bulk-import from a Minecraft-style resourcepacks directory.
            let Some(path) = args.get(1) else {
                return Err("usage: feathered packs scan <resourcepacks-dir>".into());
            };
            let reports = mgr.import_all_folders_under(&PathBuf::from(path))?;
            println!("imported {} pack(s)", reports.len());
            for r in &reports {
                println!("  {} ({})", r.pack.name, r.pack.version_label());
            }
        }
        other => {
            return Err(format!(
                "unknown packs action: {other} (list|import|use|uninstall|open-folder|browse|scan)"
            )
            .into());
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// shader packs
// ---------------------------------------------------------------------------

fn shaders(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mgr = feathered_packs::ShaderPackManager::open()?;
    let action = args.first().map(String::as_str).unwrap_or("list");

    match action {
        "list" => {
            let packs = mgr.list()?;
            if packs.is_empty() {
                println!("no shader packs installed.");
                println!("import one with: feathered shaders import <file-or-folder>");
            }
            for p in &packs {
                let active = mgr.settings().active_shader.as_deref() == Some(p.id.as_str());
                println!(
                    "{}  {:<24} {:<14} {}{}",
                    p.id,
                    p.name,
                    format!("{:?}", p.profile).to_lowercase(),
                    p.license_summary(),
                    if active { "  [ENABLED]" } else { "" }
                );
            }
        }
        "import" => {
            let Some(path) = args.get(1) else {
                return Err("usage: feathered shaders import <path>".into());
            };
            let p = PathBuf::from(path);
            let pack = if p.is_dir() {
                mgr.import_folder(&p)?
            } else {
                mgr.import_zip(&p)?
            };
            println!("imported shader pack \"{}\" ({:?})", pack.name, pack.profile);
            println!("  {}", pack.license_summary());
        }
        "enable" => {
            let Some(id) = args.get(1) else {
                return Err("usage: feathered shaders enable <id>".into());
            };
            let packs = mgr.list()?;
            let pack = packs
                .iter()
                .find(|p| p.id == *id || p.id.starts_with(id.as_str()) || p.name.eq_ignore_ascii_case(id))
                .ok_or_else(|| format!("no installed shader pack matches \"{id}\""))?;
            mgr.enable(pack)?;
            println!("enabled: {} ({:?})", pack.name, pack.profile);
            println!("note: {}", pack.license_summary());
        }
        "disable" => {
            mgr.disable()?;
            println!("shaders disabled (pipeline back to the plain preset).");
        }
        "uninstall" => {
            let Some(id) = args.get(1) else {
                return Err("usage: feathered shaders uninstall <id>".into());
            };
            let packs = mgr.list()?;
            let pack = packs
                .iter()
                .find(|p| p.id.starts_with(id.as_str()) || p.name.eq_ignore_ascii_case(id))
                .ok_or_else(|| format!("no installed shader pack matches \"{id}\""))?;
            mgr.uninstall(pack)?;
            println!("uninstalled: {}", pack.name);
        }
        other => {
            return Err(format!("unknown shaders action: {other} (list|import|enable|disable|uninstall)").into());
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// compile / validate / render
// ---------------------------------------------------------------------------

/// Resolve the pack directory to compile: the selected pack's folder when the
/// manager has an active pack, otherwise the classic local `--pack-dir`.
fn resolve_pack_dir(args: &[String]) -> Result<(PathBuf, Option<String>), Box<dyn std::error::Error>> {
    let mgr = feathered_packs::ResourcePackManager::open()?;
    if let Ok(settings) = std::fs::read_to_string(settings_path(&mgr)) {
        if let Ok(settings) = serde_json::from_str::<feathered_packs::PackSettings>(&settings) {
            if let Some(id) = &settings.active_pack {
                if let Ok(packs) = mgr.list() {
                    if let Some(pack) = packs.iter().find(|p| &p.id == id) {
                        if !arg_flag(args, "--pack-dir") {
                            return Ok((pack.dir.clone(), Some(pack.name.clone())));
                        }
                    }
                }
            }
        }
    }
    let dir = PathBuf::from(arg_value(args, "--pack-dir").unwrap_or_else(|| "texture/assets".into()));
    Ok((dir, None))
}

fn settings_path(mgr: &feathered_packs::ResourcePackManager) -> PathBuf {
    mgr.base_dir().parent().unwrap_or(mgr.base_dir()).join("settings.json")
}

fn compile_pack(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let (pack_dir, pack_name) = resolve_pack_dir(args)?;
    let out = PathBuf::from(
        arg_value(args, "--out").unwrap_or_else(|| default_cache().into_os_string().into_string().unwrap()),
    );
    let required_only = arg_flag(args, "--required");

    if let Some(name) = &pack_name {
        println!("compiling selected pack: {name}");
    }
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
        stats.atlas_size.0, stats.atlas_size.1, atlas.mip_levels()
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

fn validate(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let (pack_dir, _) = resolve_pack_dir(args)?;
    let cache_path = PathBuf::from(
        arg_value(args, "--cache").unwrap_or_else(|| default_cache().into_os_string().into_string().unwrap()),
    );

    let blob = std::fs::read(&cache_path)?;
    let (_, payload) = feathered_assets::cache::decode(&blob)
        .map_err(|e| format!("cache {} unusable: {e} (run compile-pack)", cache_path.display()))?;

    let runtime = feathered_world::Registry::from_compiled(payload.pack);
    let atlas = feathered_assets::cache::cached_to_atlas(&payload.atlas, runtime.sprite_names());

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

fn render(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    // Auto-compile when the cache is missing (first run after pack selection).
    let cache_path = PathBuf::from(
        arg_value(args, "--cache").unwrap_or_else(|| default_cache().into_os_string().into_string().unwrap()),
    );
    if !cache_path.exists() {
        println!("no compiled cache found — compiling current pack first...");
        compile_pack(args)?;
    }

    let (pack_dir, _) = resolve_pack_dir(args)?;
    let shader_mgr = feathered_packs::ShaderPackManager::open()?;
    let active_shader = shader_mgr.active();
    let active_shader_id = shader_mgr
        .settings()
        .active_shader
        .filter(|_| active_shader.is_some());
    // The pack's directory feeds the configuration bridge (its
    // shaders.properties is translated onto the renderer's generic stage
    // knobs; pack GLSL is never executed — see docs/SHADER_COMPATIBILITY.md).
    let shader_config_path = active_shader.as_ref().map(|p| p.dir.clone());
    let _ = std::env::var("FEATHERED_MARK_ATMO"); // debug hooks live in the renderer
    if let (Some(id), Some(p)) = (&active_shader_id, &active_shader) {
        println!(
            "shader pack: {id} ({} profile) — configuration translated, GLSL not executed",
            match p.profile {
                feathered_packs::shader::ShaderLayout::OptifineStyle => "OptiFine-style",
                feathered_packs::shader::ShaderLayout::GlslPasses => "GLSL-passes",
                feathered_packs::shader::ShaderLayout::Unknown => "unknown",
            }
        );
    }

    feathered_client::run(feathered_client::RunOptions {
        pack_dir,
        cache_path,
        screenshot: arg_value(args, "--screenshot").map(PathBuf::from),
        quality: render_quality(parse_quality(args)),
        shader_pack: active_shader_id,
        shader_config_path,
    })
}

/// Open a folder in the OS file manager (best effort; CLI still prints it).
fn open_in_explorer(path: &std::path::Path) {
    #[cfg(target_os = "windows")]
    let _ = std::process::Command::new("explorer").arg(path).spawn();
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(path).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let _ = std::process::Command::new("xdg-open").arg(path).spawn();
}
