//! feathered-packs — resource-pack and shader-pack management.
//!
//! Feathered does **not** bundle, download, or redistribute Minecraft assets.
//! Every visual asset comes from a pack the user supplies. This crate:
//!
//! * discovers packs already extracted into Feathered's data directory,
//! * imports ZIP archives and folders (with ZIP-slip protection),
//! * detects the pack format version from `pack.mcmeta`,
//! * remembers the active resource pack and shader pack across launches,
//! * treats third-party content as an external dependency: original files are
//!   never modified and each pack's own license/attribution is preserved.

pub mod resource;
pub mod shader;
pub mod shader_config;
pub mod settings;

pub use resource::{ImportReport, PackKind, ResourcePack, ResourcePackManager};
pub use shader::{ShaderPack, ShaderPackManager};
pub use shader_config::{ShaderPackConfig, translate_shader_config};
pub use settings::{PackSettings, Quality};

/// A reference to the source a pack was imported from. Kept for attribution
/// ("where did this file come from"); Feathered never claims ownership of
/// third-party content.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SourceInfo {
    /// Human-readable origin, e.g. a file name or a site the user named.
    pub origin: String,
    /// Optional license note captured at import time (free-form, verbatim).
    pub license_note: Option<String>,
}

/// Standard link table shown in the "Browse Resource Packs" UI section.
/// These are external sites — Feathered only opens a link; it never downloads
/// or rehosts third-party packs itself.
pub const BROWSE_LINKS: &[(&str, &str)] = &[
    (
        "CurseForge — Resource Packs",
        "https://www.curseforge.com/minecraft/search?class=resource-packs",
    ),
    (
        "Modrinth — Resource Packs",
        "https://modrinth.com/resourcepacks",
    ),
    (
        "Planet Minecraft — Texture Packs",
        "https://www.planetminecraft.com/resources/texture_packs/",
    ),
    (
        "CurseForge — Shaders",
        "https://www.curseforge.com/minecraft/search?class=shaders",
    ),
    (
        "Modrinth — Shaders",
        "https://modrinth.com/shaders",
    ),
];

/// Note attached to the default-template option in the Browse section.
pub const DEFAULT_TEMPLATE_NOTE: &str = "The 26.3 Default Template pack (pack_format 97) is an external \
dependency distributed by its own publisher. Feathered does not host it; its license and attribution \
belong to that project.";

