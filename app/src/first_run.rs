//! First-launch onboarding.
//!
//! Normal users should never be dumped into a folder picker without context.
//! This flow explains *why* Feathered needs a pack, what is legal to use, and
//! then walks through: import → select → (optional) shaders → compile.

use std::io::Write as _;
use std::path::PathBuf;

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let packs = feathered_packs::ResourcePackManager::open()?;
    let settings = packs.initialize()?;

    println!("Welcome to Feathered Minecraft!");
    println!();
    println!("Feathered is an independent Minecraft-compatible engine. It does NOT");
    println!("include any Minecraft assets (textures, models, sounds, ...). Those");
    println!("belong to Mojang/Microsoft and are not ours to distribute.");
    println!();
    println!("To see the world, Feathered needs a resource pack that YOU provide:");
    println!("  * a legally obtained resource pack you already have, or");
    println!("  * one downloaded from an external site (their license applies), or");
    println!("  * an extracted copy of the vanilla resources from your own game.");
    println!();

    if settings.onboarded || settings.active_pack.is_some() {
        println!("A resource pack is already set up (run `feathered packs list`).");
        println!("Running setup again will let you switch packs.");
        println!();
    }

    // Show what's already installed, if anything.
    let installed = packs.list()?;
    if !installed.is_empty() {
        println!("Currently installed packs:");
        for p in &installed {
            println!("  * {} — {}", p.name, p.version_label());
        }
        println!();
    }

    println!("What would you like to do?");
    println!("  1) Import a resource pack (ZIP file or folder on this computer)");
    println!("  2) Open the Feathered resource-pack folder (drop packs in there)");
    println!("  3) Browse external pack websites (CurseForge, Modrinth, ...)");
    if PathBuf::from("texture/assets/pack.mcmeta").is_file() {
        println!("  4) Use the local pack in ./texture/assets (detected)");
    }
    print!("Choose an option (1-4): ");
    std::io::stdout().flush()?;

    let Some(choice) = read_line()?.map(|s| s.trim().to_string()) else {
        println!("\nSetup cancelled.");
        return Ok(());
    };

    let report = match choice.as_str() {
        "1" => {
            print!("Path to the ZIP file or folder: ");
            std::io::stdout().flush()?;
            let Some(path) = read_line()?.map(|s| s.trim().to_string()) else {
                println!("Setup cancelled.");
                return Ok(());
            };
            let p = PathBuf::from(expand_tilde(&path));
            let r = if p.is_dir() {
                packs.import_folder(&p)?
            } else {
                match packs.import_zip(&p) {
                    Ok(r) => r,
                    Err(e) => {
                        println!("\nThat file could not be imported as a resource pack:");
                        println!("  {e}");
                        println!("Tip: a valid pack contains pack.mcmeta and namespace folders like assets/minecraft/.");
                        return Ok(());
                    }
                }
            };
            println!(
                "\nImported \"{}\" ({}) — original file untouched.",
                r.pack.name,
                r.pack.version_label()
            );
            r
        }
        "2" => {
            std::fs::create_dir_all(packs.base_dir())?;
            println!();
            println!("Feathered's pack folder is:");
            println!("  {}", packs.base_dir().display());
            println!();
            println!("Drop resource-pack ZIPs or folders there, then run");
            println!("  feathered packs import <path>");
            println!("for each pack you want registered.");
            crate::open_in_explorer(packs.base_dir());
            return Ok(());
        }
        "3" => {
            println!();
            println!("Feathered does not host or download packs. Popular external sites:");
            for (label, url) in feathered_packs::BROWSE_LINKS {
                println!("  {label}\n    {url}");
            }
            println!();
            println!("{}", feathered_packs::DEFAULT_TEMPLATE_NOTE);
            println!();
            println!("Once you have downloaded a pack, run `feathered first-run` again");
            println!("or `feathered packs import <file>` to install it.");
            return Ok(());
        }
        "4" if PathBuf::from("texture/assets/pack.mcmeta").is_file() => {
            let r = packs.import_folder(&PathBuf::from("texture/assets"))?;
            println!(
                "\nImported \"{}\" ({}) — referenced in place, nothing copied.",
                r.pack.name,
                r.pack.version_label()
            );
            r
        }
        _ => {
            println!("No option selected — nothing changed.");
            return Ok(());
        }
    };

    // Select the imported pack.
    let mut settings = packs.settings();
    settings.active_pack = Some(report.pack.id.clone());
    settings.pack_format = report.pack.pack_format;
    settings.onboarded = true;
    packs.save_settings(&settings)?;
    println!("Set \"{}\" as the active pack.", report.pack.name);

    // Optional shader import.
    println!();
    print!("Import a shader pack now as well? (y/N): ");
    std::io::stdout().flush()?;
    if let Some(y) = read_line()? {
        if y.trim().eq_ignore_ascii_case("y") {
            print!("Path to the shader ZIP or folder: ");
            std::io::stdout().flush()?;
            if let Some(path) = read_line()?.map(|s| s.trim().to_string()) {
                let shaders = feathered_packs::ShaderPackManager::open()?;
                let p = PathBuf::from(expand_tilde(&path));
                let result = if p.is_dir() {
                    shaders.import_folder(&p)
                } else {
                    shaders.import_zip(&p)
                };
                match result {
                    Ok(pack) => {
                        println!("Imported shader pack \"{}\".", pack.name);
                        println!("  {}", pack.license_summary());
                        println!("(Shaders stay off until you run `feathered shaders enable <id>`.)");
                    }
                    Err(e) => println!("Could not import shader pack: {e}"),
                }
            }
        }
    }

    // Compile right away so `render` just works.
    println!();
    println!("Compiling the resource pack (this can take a minute for large packs)...");
    crate::compile_pack(&[])?;
    println!();
    println!("All set! Launch the game with:");
    println!("  feathered render");
    Ok(())
}

fn read_line() -> Result<Option<String>, Box<dyn std::error::Error>> {
    let mut buf = String::new();
    match std::io::stdin().read_line(&mut buf) {
        Ok(0) => Ok(None), // EOF (non-interactive)
        Ok(_) => Ok(Some(buf)),
        Err(e) => Err(e.into()),
    }
}

/// Minimal `~/` expansion so users can paste home-relative paths.
fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest).to_string_lossy().into_owned();
        }
    }
    path.to_string()
}
