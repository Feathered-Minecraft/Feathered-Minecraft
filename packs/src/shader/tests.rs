//! Shader-pack manager tests. Fixtures are synthetic OptiFine/Iris-style
//! trees built in tempdirs — no third-party shader code is used.

use super::*;
use std::io::Write;

fn write_optifine_pack(root: &Path, with_props: bool) -> PathBuf {
    let shaders = root.join("shaders");
    std::fs::create_dir_all(&shaders).unwrap();
    std::fs::write(shaders.join("terrain.vsh"), b"#version 150\nvoid main() {}\n").unwrap();
    std::fs::write(shaders.join("terrain.fsh"), b"#version 150\nvoid main() {}\n").unwrap();
    std::fs::write(shaders.join("composite.fsh"), b"#version 150\nvoid main() {}\n").unwrap();
    if with_props {
        std::fs::write(
            shaders.join("shaders.properties"),
            "profile.requires=OptiFine\nprofile.requires.Iris=1.6\n",
        )
        .unwrap();
    }
    root.to_path_buf()
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

#[test]
fn import_folder_detects_optifine_layout_and_engine_claims() {
    let tmp = tempfile::tempdir().unwrap();
    let mgr = ShaderPackManager::open_at(tmp.path().join("shaders"));
    let src = write_optifine_pack(&tmp.path().join("NobleStyleSHADERS"), true);

    let pack = mgr.import_folder(&src).unwrap();
    assert_eq!(pack.profile, ShaderLayout::OptifineStyle);
    assert_eq!(pack.engines.len(), 2, "both engine claims parsed");
    assert!(pack.engines.iter().any(|e| e.engine == "Iris" && e.min_version.as_deref() == Some("1.6")));
    // License summary must NOT claim the pack is GPL or otherwise licensed.
    let summary = pack.license_summary();
    assert!(summary.contains("Third-party"), "summary: {summary}");
    assert!(!summary.to_lowercase().contains("gpl"), "must not guess a license");
}

#[test]
fn import_rejects_folder_without_shaders_dir() {
    let tmp = tempfile::tempdir().unwrap();
    let mgr = ShaderPackManager::open_at(tmp.path().join("shaders"));
    let not_a_shader = tmp.path().join("notshader");
    std::fs::create_dir_all(not_a_shader.join("assets/minecraft")).unwrap();
    assert!(mgr.import_folder(&not_a_shader).is_err());
}

#[test]
fn import_zip_extracts_wrapped_shaders_dir() {
    let tmp = tempfile::tempdir().unwrap();
    let mgr = ShaderPackManager::open_at(tmp.path().join("shaders"));
    let zip_path = tmp.path().join("FancyShaders.zip");
    write_zip(
        &zip_path,
        &[
            ("FancyShaders/shaders/terrain.vsh", b"void main(){}".as_slice()),
            ("FancyShaders/shaders/terrain.fsh", b"void main(){}".as_slice()),
        ],
    );
    let pack = mgr.import_zip(&zip_path).unwrap();
    assert_eq!(pack.profile, ShaderLayout::OptifineStyle);
    assert!(pack.dir.join("shaders/terrain.fsh").is_file());
    assert!(zip_path.is_file(), "original archive untouched");
    assert_eq!(mgr.list().unwrap().len(), 1);
}

#[test]
fn import_zip_rejects_archive_without_shaders_dir() {
    let tmp = tempfile::tempdir().unwrap();
    let mgr = ShaderPackManager::open_at(tmp.path().join("shaders"));
    let zip_path = tmp.path().join("Random.zip");
    write_zip(&zip_path, &[("docs/readme.txt", b"hi")]);
    assert!(mgr.import_zip(&zip_path).is_err());
}

#[test]
fn license_file_is_preserved_not_relicensed() {
    let tmp = tempfile::tempdir().unwrap();
    let mgr = ShaderPackManager::open_at(tmp.path().join("shaders"));
    let src = tmp.path().join("LicensedPack");
    write_optifine_pack(&src, false);
    std::fs::write(src.join("LICENSE"), "Custom Shader License 1.0\nAll rights reserved.\n").unwrap();

    let pack = mgr.import_folder(&src).unwrap();
    let lic = pack.license_file().expect("license file found");
    let text = std::fs::read_to_string(&lic).unwrap();
    assert!(text.contains("Custom Shader License 1.0"), "verbatim preservation");
    assert!(pack.license_summary().contains("LICENSE"), "summary points at the pack's own file");
}

#[test]
fn enable_disable_persists_and_uninstall_clears_active() {
    let tmp = tempfile::tempdir().unwrap();
    let data = tmp.path().join("shaders");
    let mgr = ShaderPackManager::open_at(&data);
    let src = write_optifine_pack(&tmp.path().join("Pack A"), false);
    let pack = mgr.import_folder(&src).unwrap();

    assert!(mgr.active().is_none());
    assert!(mgr.enable(&pack).unwrap().is_none());
    assert_eq!(mgr.active().unwrap().id, pack.id);

    // Fresh instance sees the persisted enable.
    let mgr2 = ShaderPackManager::open_at(&data);
    assert_eq!(mgr2.active().unwrap().id, pack.id);

    mgr2.disable().unwrap();
    assert!(mgr2.active().is_none());

    // Re-enable then uninstall: active id must be cleared.
    mgr2.enable(&pack).unwrap();
    mgr2.uninstall(&pack).unwrap();
    assert!(mgr2.active().is_none());
    assert!(mgr2.list().unwrap().is_empty());
}

#[test]
fn glsl_passes_layout_detected() {
    let tmp = tempfile::tempdir().unwrap();
    let mgr = ShaderPackManager::open_at(tmp.path().join("shaders"));
    let src = tmp.path().join("GlslPack");
    std::fs::create_dir_all(src.join("shaders")).unwrap();
    std::fs::write(src.join("shaders/common.glsl"), b"").unwrap();
    std::fs::write(src.join("shaders/main.gsh"), b"").unwrap();
    let pack = mgr.import_folder(&src).unwrap();
    assert_eq!(pack.profile, ShaderLayout::GlslPasses);
}

#[test]
fn prune_missing_removes_gone_folder_packs() {
    let tmp = tempfile::tempdir().unwrap();
    let mgr = ShaderPackManager::open_at(tmp.path().join("shaders"));
    let src = write_optifine_pack(&tmp.path().join("Vanishing"), false);
    mgr.import_folder(&src).unwrap();
    assert_eq!(mgr.list().unwrap().len(), 1);
    std::fs::remove_dir_all(&src).unwrap();
    assert!(mgr.list().unwrap().is_empty());
    assert_eq!(mgr.prune_missing().unwrap(), 1);
}

#[test]
fn zip_slip_entries_are_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let mgr = ShaderPackManager::open_at(tmp.path().join("shaders"));
    let zip_path = tmp.path().join("Evil.zip");
    write_zip(
        &zip_path,
        &[
            ("shaders/terrain.fsh", b"x".as_slice()),
            ("../evil.cfg", b"gotcha".as_slice()),
        ],
    );
    assert!(mgr.import_zip(&zip_path).is_err());
    assert!(!tmp.path().join("evil.cfg").exists());
}
