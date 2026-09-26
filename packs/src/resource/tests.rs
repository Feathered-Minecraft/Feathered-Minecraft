//! Tests for resource-pack import, validation, version detection, switching,
//! and cache generation. Fixtures are built in tempdirs — no real game assets
//! are used anywhere in this suite.

use super::*;
use std::io::Write;

// ---------------------------------------------------------------------------
// fixture helpers
// ---------------------------------------------------------------------------

const MCMETA_26_3: &str = r#"{"pack":{"pack_format":97,"description":"Feathered test pack"}}"#;
const MCMETA_1_21: &str = r#"{"pack":{"pack_format":61,"description":["Chapter ",{"text":"one","color":"red"}]}}"#;

fn write_pack_dir(root: &Path, name: &str, mcmeta: &str) -> PathBuf {
    let dir = root.join(name);
    std::fs::create_dir_all(dir.join("assets/minecraft/textures/block")).unwrap();
    std::fs::create_dir_all(dir.join("assets/minecraft/blockstates")).unwrap();
    std::fs::write(dir.join("pack.mcmeta"), mcmeta).unwrap();
    std::fs::write(
        dir.join("assets/minecraft/blockstates/stone.json"),
        r#"{"variants":{"":{"model":"minecraft:block/stone"}}}"#,
    )
    .unwrap();
    std::fs::write(dir.join("assets/minecraft/textures/block/stone.png"), b"\x89PNG fake").unwrap();
    dir
}

fn write_zip(path: &Path, add: &[(&str, &[u8])]) {
    let file = std::fs::File::create(path).unwrap();
    let mut zip = zip::ZipWriter::new(file);
    let opts: zip::write::SimpleFileOptions = Default::default();
    for (name, bytes) in add {
        zip.start_file(*name, opts).unwrap();
        zip.write_all(bytes).unwrap();
    }
    zip.finish().unwrap();
}

fn zip_entries(name: &str, mcmeta: &str) -> Vec<(String, Vec<u8>)> {
    vec![
        (format!("{name}/pack.mcmeta"), mcmeta.as_bytes().to_vec()),
        (
            format!("{name}/assets/minecraft/blockstates/stone.json"),
            br#"{"variants":{"":{"model":"minecraft:block/stone"}}}"#.to_vec(),
        ),
        (format!("{name}/assets/minecraft/textures/block/stone.png"), b"\x89PNG fake".to_vec()),
    ]
}

// ---------------------------------------------------------------------------
// metadata / version detection
// ---------------------------------------------------------------------------

#[test]
fn mcmeta_parse_and_version_detection() {
    let m = parse_mcmeta_text(MCMETA_26_3).unwrap();
    assert_eq!(m.pack_format, Some(97));
    assert_eq!(m.description.as_deref(), Some("Feathered test pack"));
    assert_eq!(pack_format_to_version(97), Some("26.3"));
    assert_eq!(pack_format_to_version(61), Some("1.21"));
    assert_eq!(pack_format_to_version(9999), None);

    let m2 = parse_mcmeta_text(MCMETA_1_21).unwrap();
    assert_eq!(m2.pack_format, Some(61));
    // JSON text components flatten to plain text.
    assert_eq!(m2.description.as_deref(), Some("Chapter one"));
}

#[test]
fn supported_formats_range_is_parsed() {
    let m = parse_mcmeta_text(
        r#"{"pack":{"pack_format":46,"supported_formats":{"min_inclusive":40,"max_inclusive":999},"description":"x"}}"#,
    )
    .unwrap();
    assert_eq!(m.pack_format, Some(46));
    assert_eq!(m.supported_formats, Some((40, 999)));
}

// ---------------------------------------------------------------------------
// folder import
// ---------------------------------------------------------------------------

#[test]
fn import_folder_validates_and_references_in_place() {
    let tmp = tempfile::tempdir().unwrap();
    let mgr = ResourcePackManager::open_at(tmp.path().join("data"));
    mgr.initialize().unwrap();

    let src = write_pack_dir(tmp.path(), "CoolPack", MCMETA_26_3);
    let report = mgr.import_folder(&src).unwrap();

    assert_eq!(report.pack.name, "CoolPack");
    assert_eq!(report.pack.pack_format, Some(97));
    assert_eq!(report.pack.detected_version.as_deref(), Some("26.3"));
    assert_eq!(report.pack.kind, PackKind::Folder);
    assert_eq!(report.files_copied, 0, "folders must not be copied");
    // Original file untouched, still at its original location.
    assert!(src.join("pack.mcmeta").is_file());
    // Metadata sidecar lives in the data dir.
    assert!(tmp
        .path()
        .join("data/packs")
        .read_dir()
        .unwrap()
        .flatten()
        .any(|e| e.file_name().to_string_lossy().ends_with(".folder.json")));

    let listed = mgr.list().unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].id, report.pack.id);
}

#[test]
fn import_rejects_directory_without_mcmeta() {
    let tmp = tempfile::tempdir().unwrap();
    let mgr = ResourcePackManager::open_at(tmp.path().join("data"));
    let bad = tmp.path().join("notapack");
    std::fs::create_dir_all(bad.join("assets/minecraft")).unwrap();
    match mgr.import_folder(&bad) {
        Err(ImportError::NotAPack(_)) => {}
        other => panic!("expected NotAPack, got {other:?}"),
    }
}

#[test]
fn import_rejects_missing_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let mgr = ResourcePackManager::open_at(tmp.path().join("data"));
    assert!(matches!(
        mgr.import_folder(&tmp.path().join("nope")),
        Err(ImportError::NotFound(_))
    ));
}

// ---------------------------------------------------------------------------
// zip import
// ---------------------------------------------------------------------------

#[test]
fn import_zip_extracts_and_validates() {
    let tmp = tempfile::tempdir().unwrap();
    let mgr = ResourcePackManager::open_at(tmp.path().join("data"));
    mgr.initialize().unwrap();

    let zip_path = tmp.path().join("MyPack.zip");
    let entries = zip_entries("MyPack", MCMETA_26_3);
    let refs: Vec<(&str, &[u8])> = entries.iter().map(|(n, b)| (n.as_str(), b.as_slice())).collect();
    write_zip(&zip_path, &refs);

    let report = mgr.import_zip(&zip_path).unwrap();
    assert_eq!(report.pack.kind, PackKind::Zip);
    assert_eq!(report.pack.name, "MyPack");
    assert_eq!(report.pack.pack_format, Some(97));
    assert!(report.files_copied >= 3);
    // The wrapper folder was stripped: pack.mcmeta is directly in pack.dir.
    assert!(report.pack.dir.join("pack.mcmeta").is_file());
    assert!(report.pack.dir.join("assets/minecraft/blockstates/stone.json").is_file());
    // Original ZIP untouched.
    assert!(zip_path.is_file());

    let listed = mgr.list().unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].id, report.pack.id);
}

#[test]
fn import_zip_without_wrapper_folder_works() {
    let tmp = tempfile::tempdir().unwrap();
    let mgr = ResourcePackManager::open_at(tmp.path().join("data"));
    let zip_path = tmp.path().join("Flat.zip");
    write_zip(
        &zip_path,
        &[
            ("pack.mcmeta", MCMETA_26_3.as_bytes()),
            ("assets/minecraft/textures/block/x.png", b"x"),
        ],
    );
    let report = mgr.import_zip(&zip_path).unwrap();
    assert!(report.pack.dir.join("pack.mcmeta").is_file());
}

#[test]
fn import_zip_rejects_non_pack_archive() {
    let tmp = tempfile::tempdir().unwrap();
    let mgr = ResourcePackManager::open_at(tmp.path().join("data"));
    let zip_path = tmp.path().join("Random.zip");
    write_zip(&zip_path, &[("docs/readme.txt", b"hello")]);
    assert!(matches!(
        mgr.import_zip(&zip_path),
        Err(ImportError::NotAPack(_))
    ));
}

#[test]
fn import_zip_blocks_zip_slip() {
    let tmp = tempfile::tempdir().unwrap();
    let mgr = ResourcePackManager::open_at(tmp.path().join("data"));
    let zip_path = tmp.path().join("Evil.zip");
    // An entry that tries to escape the destination directory.
    write_zip(
        &zip_path,
        &[
            ("pack.mcmeta", MCMETA_26_3.as_bytes()),
            ("assets/minecraft/x.png", b"x"),
            ("../evil.txt", b"gotcha"),
        ],
    );
    match mgr.import_zip(&zip_path) {
        Err(ImportError::UnsafePath(_, name)) => assert!(name.contains("..")),
        other => panic!("expected UnsafePath, got {other:?}"),
    }
    // Nothing was written outside the tempdir.
    assert!(!tmp.path().join("evil.txt").exists());
}

// ---------------------------------------------------------------------------
// listing, pruning, uninstall
// ---------------------------------------------------------------------------

#[test]
fn list_skips_and_prunes_stale_folder_packs() {
    let tmp = tempfile::tempdir().unwrap();
    let mgr = ResourcePackManager::open_at(tmp.path().join("data"));
    mgr.initialize().unwrap();

    let src = write_pack_dir(tmp.path(), "GonePack", MCMETA_26_3);
    mgr.import_folder(&src).unwrap();
    assert_eq!(mgr.list().unwrap().len(), 1);

    // The referenced folder disappears; list() skips it and prune removes it.
    std::fs::remove_dir_all(&src).unwrap();
    assert!(mgr.list().unwrap().is_empty());
    assert_eq!(mgr.prune_missing().unwrap(), 1);
    assert!(mgr.list().unwrap().is_empty());
}

#[test]
fn uninstall_zip_removes_extracted_copy_but_never_original() {
    let tmp = tempfile::tempdir().unwrap();
    let mgr = ResourcePackManager::open_at(tmp.path().join("data"));
    let zip_path = tmp.path().join("Del.zip");
    let entries = zip_entries("Del", MCMETA_26_3);
    let refs: Vec<(&str, &[u8])> = entries.iter().map(|(n, b)| (n.as_str(), b.as_slice())).collect();
    write_zip(&zip_path, &refs);

    let report = mgr.import_zip(&zip_path).unwrap();
    mgr.uninstall(&report.pack).unwrap();
    assert!(!report.pack.dir.exists(), "extracted copy removed");
    assert!(zip_path.is_file(), "user's original zip untouched");
    assert!(mgr.list().unwrap().is_empty());
}

// ---------------------------------------------------------------------------
// switching / settings persistence
// ---------------------------------------------------------------------------

#[test]
fn pack_switching_persists_across_instances() {
    let tmp = tempfile::tempdir().unwrap();
    let data = tmp.path().join("data");
    let mgr = ResourcePackManager::open_at(&data);
    mgr.initialize().unwrap();

    let a = write_pack_dir(tmp.path(), "Alpha", MCMETA_26_3);
    let b = write_pack_dir(tmp.path(), "Beta", MCMETA_1_21);
    let ra = mgr.import_folder(&a).unwrap().pack;
    let rb = mgr.import_folder(&b).unwrap().pack;

    let mut settings = mgr.settings();
    settings.active_pack = Some(ra.id.clone());
    settings.pack_format = ra.pack_format;
    mgr.save_settings(&settings).unwrap();

    // Switch.
    let mut settings = mgr.settings();
    settings.active_pack = Some(rb.id.clone());
    settings.pack_format = rb.pack_format;
    mgr.save_settings(&settings).unwrap();

    // A fresh manager instance sees the switch (persistence contract).
    let mgr2 = ResourcePackManager::open_at(&data);
    let settings = mgr2.settings();
    assert_eq!(settings.active_pack.as_deref(), Some(rb.id.as_str()));
    assert_eq!(settings.pack_format, Some(61));
    let listed = mgr2.list().unwrap();
    assert_eq!(listed.len(), 2);
    assert!(listed.iter().any(|p| p.id == rb.id));
}

// ---------------------------------------------------------------------------
// discovery + compilation integration
// ---------------------------------------------------------------------------

#[test]
fn pack_index_discovery_compiles_and_caches() {
    let tmp = tempfile::tempdir().unwrap();
    let mgr = ResourcePackManager::open_at(tmp.path().join("data"));
    let src = write_pack_dir(tmp.path(), "TinyPack", MCMETA_26_3);
    let report = mgr.import_folder(&src).unwrap();

    let index = mgr.pack_index(&report.pack).unwrap();
    assert_eq!(index.packs.len(), 1);
    assert!(
        index.get("minecraft", "blockstates/stone").is_some(),
        "discovery must index blockstates (namespace:path keys, extension stripped)"
    );
    assert!(index.get("minecraft", "textures/block/stone").is_some());

    // The manager's index feeds straight into the asset compiler. The
    // fixture intentionally lacks the ten required validation blocks, so a
    // full compile must fail with a clear, named error — this is the
    // incomplete-pack handling path.
    let err = match feathered_assets::compiler::compile_pack(&index, false) {
        Err(e) => e,
        Ok(_) => panic!("incomplete fixture pack must fail compilation"),
    };
    assert!(
        err.message.contains("required block"),
        "unexpected error: {err}"
    );
}

#[test]
fn scan_imports_packs_from_a_resourcepacks_folder() {
    let tmp = tempfile::tempdir().unwrap();
    let mgr = ResourcePackManager::open_at(tmp.path().join("data"));
    let rp = tmp.path().join("resourcepacks");
    std::fs::create_dir_all(&rp).unwrap();
    write_pack_dir(&rp, "FolderPack", MCMETA_26_3);
    let entries = zip_entries("ZippedPack", MCMETA_26_3);
    let refs: Vec<(&str, &[u8])> = entries.iter().map(|(n, b)| (n.as_str(), b.as_slice())).collect();
    write_zip(&rp.join("ZippedPack.zip"), &refs);

    let reports = mgr.import_all_folders_under(&rp).unwrap();
    assert_eq!(reports.len(), 2, "both folder and zip packs imported");
    assert!(mgr.list().unwrap().len() >= 2);
}

// ---------------------------------------------------------------------------
// zip-slip guard unit tests
// ---------------------------------------------------------------------------

#[test]
fn is_within_rejects_escapes() {
    assert!(is_within(Path::new(""), Path::new("a/b/c")));
    assert!(is_within(Path::new(""), Path::new("a/../b")));
    assert!(!is_within(Path::new(""), Path::new("../a")));
    assert!(!is_within(Path::new(""), Path::new("a/../../b")));
    assert!(!is_within(Path::new(""), Path::new("/abs")));
}
