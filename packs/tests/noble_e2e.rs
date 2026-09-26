//! End-to-end: the REAL Noble pack (local clone, git-ignored) goes through
//! the real ShaderPackManager import path and the real translation bridge.
//! Skipped when the clone is absent (fresh checkouts / CI) — the synthetic
//! properties tests in `shader_config`'s unit tests cover the bridge logic
//! itself; this test proves the genuine upstream file flows through both.
//!
//! The Noble clone is consulted read-only; nothing from it is copied into
//! the repository (GPL-3.0 — see docs/SHADER_COMPATIBILITY.md §1).

use feathered_packs::shader::ShaderPackManager;
use feathered_renderer::RenderQuality;

/// Where the investigation clone lives (workspace-relative).
fn noble_clone() -> Option<std::path::PathBuf> {
    for p in ["target/noble-upstream", "../target/noble-upstream"] {
        let dir = std::path::Path::new(p).join("shaders");
        if dir.join("shaders.properties").exists() {
            return Some(std::path::Path::new(p).to_path_buf());
        }
    }
    None
}

#[test]
fn real_noble_properties_translate_through_the_manager() {
    let Some(src) = noble_clone() else {
        eprintln!("skipping: no local Noble clone at target/noble-upstream");
        return;
    };
    let tmp = tempfile::TempDir::new().expect("tmp");
    let mgr = ShaderPackManager::open_at(tmp.path());

    // 1) The manager accepts the genuine pack folder (layout + engines).
    let pack = mgr
        .import_folder(&src)
        .expect("the real Noble pack imports");
    assert!(
        pack.engines.iter().any(|e| e.engine.starts_with("iris.features")),
        "Noble declares iris.features.required; got {:?}",
        pack.engines
    );

    // 2) Its shaders.properties translates onto the generic effect config.
    let translated = feathered_packs::translate_shader_config(&pack.dir, RenderQuality::Ultra)
        .expect("Noble's properties translate");
    // The upstream default profile enables the heavy stages.
    assert!(translated.config.shadows.is_some(), "Noble enables shadows by default");
    assert!(translated.config.atmosphere.is_some(), "Noble enables atmosphere by default");
    // And reports what cannot be translated instead of faking it.
    assert!(
        !translated.unsupported.is_empty(),
        "Noble has known-untranslatable options (TAA/compute/…); they must be reported"
    );
    println!(
        "translated {} unsupported + {} unknown options from the real pack",
        translated.unsupported.len(),
        translated.unknown.len()
    );

    // 3) The staged pipeline accepts the translated config (validates every
    // knob landed inside the renderer's accepted ranges).
    let staged = translated.config.uses_staged_pipeline();
    assert!(staged, "translated Noble config must drive the staged pipeline");
}
