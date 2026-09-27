//! Menu system: the front-end state machine driving the title flow.
//!
//! Screens: Title → Worlds (list / create / delete) → Playing; Multiplayer
//! (server list UI wired for a future netplay phase — connecting reports
//! "not yet implemented"); Settings (live-applied); Profile (name + skin
//! library with Minecraft-format PNG support).
//!
//! The module owns *logic and layout*: it consumes clicks/typed text and
//! produces `MenuAction`s for the app plus tri-lists for the overlay
//! renderer. No GPU, no winit types — everything is testable headless.

use crate::font;
use crate::overlay::TriList;
use crate::profile::{Profile, ProfileStore, Skin};
use crate::settings::Settings;
use crate::ui::{self, Hover, Rect, BUTTON_GAP, BUTTON_H, BUTTON_W};
use std::path::{Path, PathBuf};

/// A world entry in the singleplayer list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorldEntry {
    /// Directory name (unique key).
    pub dir: String,
    /// Seed (shown in the subtitle).
    pub seed: u64,
    /// Metadata line (edits count / last played).
    pub info: String,
}

/// One configured multiplayer server (persisted; connect comes later).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ServerEntry {
    pub name: String,
    pub address: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Screen {
    Title,
    Worlds,
    CreateWorld,
    Multiplayer,
    AddServer,
    Settings,
    Profile,
    Skins,
}

/// What the app should do in response to the last interaction.
#[derive(Debug, Clone, PartialEq)]
pub enum MenuAction {
    None,
    /// Start playing the singleplayer world in this directory (creating
    /// the directory is the app's job).
    PlayWorld(PathBuf),
    DeleteWorld(PathBuf),
    /// Join the selected server (placeholder phase: app shows a toast).
    JoinServer(String),
    /// Quality changed live (app re-applies to the renderer).
    QualityChanged(String),
    Quit,
}

/// Layout constants for the menus.
pub struct Layout {
    pub w: f32,
    pub h: f32,
}

impl Layout {
    fn cx(&self) -> f32 {
        self.w / 2.0
    }
    /// Title-logo band height.
    fn logo_y(&self) -> f32 {
        self.h * 0.14
    }
}

/// Everything the menu draws/needs per frame (owned by the app).
pub struct MenuState {
    pub screen: Screen,
    /// Singleplayer worlds (scanned by the app from the worlds dir).
    pub worlds: Vec<WorldEntry>,
    /// Selected world index (usize::MAX = none).
    pub world_sel: usize,
    /// New-world name field.
    pub new_world_name: String,
    pub new_world_cursor: usize,
    pub new_world_seed_text: String,
    pub new_world_seed_cursor: usize,
    /// Which field has keyboard focus on CreateWorld.
    pub focus: Focus,
    /// Multiplayer server list.
    pub servers: Vec<ServerEntry>,
    pub server_sel: usize,
    pub new_server_name: String,
    pub new_server_name_cursor: usize,
    pub new_server_addr: String,
    pub new_server_addr_cursor: usize,
    pub server_focus: ServerFocus,
    /// Settings (edited in place; app applies + persists).
    pub settings: Settings,
    /// Profile (name + skin library).
    pub profile: Profile,
    pub profile_store: ProfileStore,
    pub profile_name_cursor: usize,
    pub profile_focus: ProfileFocus,
    /// Skin library file names (refreshed on screen entry).
    pub skins: Vec<String>,
    pub skin_sel: usize,
    /// Status line (bottom of screen; toasts).
    pub toast: String,
    /// Cursor position in surface pixels (hover feedback; synced by the app).
    pub cursor: (f32, f32),
    /// Worlds root (settings.json lives here; set at construction).
    settings_dir: PathBuf,
    /// Logo art aspect (width/height) when logo.png loaded; None = draw no
    /// textured quad (the renderer's fallback texture is transparent).
    pub logo_art: Option<f32>,
    /// Background art uploaded (background.png); gates the full-screen quad
    /// (without it the fallback texture is transparent and nothing draws).
    pub background: bool,
    /// Last computed widget rects (rebuilt each draw).
    hits: Hits,
    /// Cached skin preview palette.
    pub preview: Option<SkinPaletteCache>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    WorldName,
    WorldSeed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerFocus {
    Name,
    Address,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileFocus {
    Name,
    None,
}    /// Cached avatar palette (head/body/legs) for the profile screen.
    #[derive(Debug, Clone, Copy)]
pub struct SkinPaletteCache {
    pub head: [u8; 4],
    pub body: [u8; 4],
    pub legs: [u8; 4],
}

/// Widget rects captured during the last draw (hit-testing input).
#[derive(Debug, Default, Clone, Copy)]
struct Hits {
    // Title
    singleplayer: Option<Rect>,
    multiplayer: Option<Rect>,
    settings: Option<Rect>,
    profile: Option<Rect>,
    quit: Option<Rect>,
    // Worlds
    world_rows: [Option<Rect>; 6],
    world_play: Option<Rect>,
    world_create: Option<Rect>,
    world_delete: Option<Rect>,
    world_back: Option<Rect>,
    // Create
    create_go: Option<Rect>,
    create_back: Option<Rect>,
    create_name: Option<Rect>,
    create_seed: Option<Rect>,
    // Multiplayer
    server_rows: [Option<Rect>; 5],
    server_join: Option<Rect>,
    server_add: Option<Rect>,
    server_delete: Option<Rect>,
    server_back: Option<Rect>,
    // Add server
    add_go: Option<Rect>,
    add_back: Option<Rect>,
    add_name: Option<Rect>,
    add_addr: Option<Rect>,
    // Settings
    setting_rows: [Option<Rect>; 6],
    settings_back: Option<Rect>,
    // Profile
    profile_name: Option<Rect>,
    profile_back: Option<Rect>,
    profile_skins: Option<Rect>,
    // Skins
    skin_rows: [Option<Rect>; 6],
    skin_use: Option<Rect>,
    skin_back: Option<Rect>,
}

/// sRGB byte → linear byte (exact transfer curve), for authoring overlay
/// colors that must land as the authored byte on the renderer's sRGB
/// targets (the overlay shader's straight bytes are treated as linear).
fn lin(c: u8) -> u8 {
    static LUT: std::sync::OnceLock<[u8; 256]> = std::sync::OnceLock::new();
    let lut = LUT.get_or_init(|| {
        let mut table = [0u8; 256];
        for (i, slot) in table.iter_mut().enumerate() {
            let s = i as f64 / 255.0;
            let l = if s <= 0.04045 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            };
            *slot = (l * 255.0).round() as u8;
        }
        table
    });
    lut[c as usize]
}

/// Decode a whole RGBA color (leaves alpha untouched).
fn lin4(c: [u8; 4]) -> [u8; 4] {
    [lin(c[0]), lin(c[1]), lin(c[2]), c[3]]
}

/// Public linearize entry for sibling modules (ui.rs palettes).
pub fn lin_bytes(c: [u8; 4]) -> [u8; 4] {
    lin4(c)
}

/// Crop an RGBA image to its opaque content bounds (the logo PNG carries
/// transparent canvas padding; the mock's feather fills its box). Returns
/// the cropped pixels + dimensions, or the input untouched when fully
/// transparent/1×1.
pub fn crop_to_alpha(rgba: Vec<u8>, w: u32, h: u32) -> (Vec<u8>, u32, u32) {
    let mut min_x = w;
    let mut min_y = h;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    for y in 0..h {
        for x in 0..w {
            let a = rgba[((y * w + x) * 4 + 3) as usize];
            if a > 8 {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x + 1);
                max_y = max_y.max(y + 1);
            }
        }
    }
    if min_x >= max_x || min_y >= max_y {
        return (rgba, w, h);
    }
    let cw = max_x - min_x;
    let ch = max_y - min_y;
    let mut out = vec![0u8; (cw * ch * 4) as usize];
    for y in 0..ch {
        let src = ((min_y + y) * w + min_x) as usize * 4;
        let dst = (y * cw) as usize * 4;
        out[dst..dst + (cw * 4) as usize].copy_from_slice(&rgba[src..src + (cw * 4) as usize]);
    }
    (out, cw, ch)
}

impl MenuState {
    pub fn new(worlds_dir: &Path, settings: Settings, profile: Profile) -> MenuState {
        MenuState {
            screen: Screen::Title,
            worlds: Vec::new(),
            world_sel: usize::MAX,
            new_world_name: String::new(),
            new_world_cursor: 0,
            new_world_seed_text: String::new(),
            new_world_seed_cursor: 0,
            focus: Focus::WorldName,
            servers: Vec::new(),
            server_sel: usize::MAX,
            new_server_name: String::new(),
            new_server_name_cursor: 0,
            new_server_addr: String::new(),
            new_server_addr_cursor: 0,
            server_focus: ServerFocus::Name,
            settings,
            profile,
            profile_store: ProfileStore::new(worlds_dir.join("profile")),
            profile_name_cursor: 0,
            profile_focus: ProfileFocus::Name,
            skins: Vec::new(),
            skin_sel: usize::MAX,
            toast: String::new(),
            cursor: (0.0, 0.0),
            settings_dir: worlds_dir.to_path_buf(),
            logo_art: None,
            background: false,
            hits: Hits::default(),
            preview: None,
        }
    }

    /// Refresh the world list from disk (dirs with world.feathered).
    pub fn scan_worlds(&mut self, worlds_dir: &Path) {
        self.worlds.clear();
        if let Ok(rd) = std::fs::read_dir(worlds_dir) {
            let mut found: Vec<WorldEntry> = Vec::new();
            for e in rd.flatten() {
                let path = e.path();
                if !path.is_dir() {
                    continue;
                }
                if !path.join("world.feathered").exists() {
                    continue;
                }
                let dir = e.file_name().to_string_lossy().to_string();
                let (seed, info) = match feathered_world::save::load_from_dir(&path) {
                    Ok(save) => (
                        save.meta.seed,
                        format!(
                            "{} edit(s), {}",
                            save.edits.len(),
                            match save.meta.saved_at_unix {
                                Some(t) => {
                                    // Human-friendly enough without chrono.
                                    format!("saved {t}")
                                }
                                None => "not saved".into()
                            }
                        ),
                    ),
                    Err(_) => (0, "unreadable".into()),
                };
                found.push(WorldEntry { dir, seed, info });
            }
            found.sort_by(|a, b| a.dir.cmp(&b.dir));
            self.worlds = found;
        }
        if self.world_sel >= self.worlds.len() {
            self.world_sel = self.worlds.len().saturating_sub(1);
        }
    }

    /// Refresh the server list from disk.
    pub fn load_servers(&mut self, worlds_dir: &Path) {
        self.servers = std::fs::read(worlds_dir.join("servers.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        if self.server_sel >= self.servers.len() {
            self.server_sel = self.servers.len().saturating_sub(1);
        }
    }

    pub fn save_servers(&self, worlds_dir: &Path) {
        if let Ok(bytes) = serde_json::to_vec_pretty(&self.servers) {
            let _ = std::fs::write(worlds_dir.join("servers.json"), bytes);
        }
    }

    /// Refresh the skin library + preview palette.
    pub fn refresh_skins(&mut self) {
        self.skins = self.profile_store.list_skins();
        if self.skin_sel >= self.skins.len() {
            self.skin_sel = self.skins.len().saturating_sub(1);
        }
        self.preview = self.active_skin().map(|s| {
            let p = s.palette();
            SkinPaletteCache {
                head: p.head,
                body: p.body,
                legs: p.legs,
            }
        });
    }

    /// Parse the active skin (if any) for preview.
    pub fn active_skin(&self) -> Option<Skin> {
        let file = self.profile.skin.as_deref()?;
        let bytes = self.profile_store.read_skin(file).ok()?;
        Skin::from_png(&bytes).ok()
    }

    // ------------------------------------------------------------------
    // input
    // ------------------------------------------------------------------

    /// A click at UI-pixel `p`. Returns the resulting action.
    pub fn click(&mut self, p: (f32, f32), worlds_dir: &Path) -> MenuAction {
        let h = self.hits;
        match self.screen {
            Screen::Title => {
                if h.singleplayer.map(|r| ui::hit(r, p)).unwrap_or(false) {
                    self.scan_worlds(worlds_dir);
                    self.screen = Screen::Worlds;
                } else if h.multiplayer.map(|r| ui::hit(r, p)).unwrap_or(false) {
                    self.load_servers(worlds_dir);
                    self.screen = Screen::Multiplayer;
                } else if h.settings.map(|r| ui::hit(r, p)).unwrap_or(false) {
                    self.screen = Screen::Settings;
                } else if h.profile.map(|r| ui::hit(r, p)).unwrap_or(false) {
                    self.refresh_skins();
                    self.profile_name_cursor = self.profile.name.chars().count();
                    self.screen = Screen::Profile;
                } else if h.quit.map(|r| ui::hit(r, p)).unwrap_or(false) {
                    return MenuAction::Quit;
                }
            }
            Screen::Worlds => {
                for (i, slot) in h.world_rows.iter().enumerate() {
                    if slot.map(|r| ui::hit(r, p)).unwrap_or(false) {
                        if i < self.worlds.len() {
                            self.world_sel = i;
                        }
                        break;
                    }
                }
                if h.world_play.map(|r| ui::hit(r, p)).unwrap_or(false) {
                    if let Some(w) = self.worlds.get(self.world_sel) {
                        let path = worlds_dir.join(&w.dir);
                        return MenuAction::PlayWorld(path);
                    }
                }
                if h.world_create.map(|r| ui::hit(r, p)).unwrap_or(false) {
                    self.new_world_name.clear();
                    self.new_world_cursor = 0;
                    self.new_world_seed_text.clear();
                    self.new_world_seed_cursor = 0;
                    self.focus = Focus::WorldName;
                    self.screen = Screen::CreateWorld;
                }
                if h.world_delete.map(|r| ui::hit(r, p)).unwrap_or(false) {
                    if let Some(w) = self.worlds.get(self.world_sel) {
                        let path = worlds_dir.join(&w.dir);
                        return MenuAction::DeleteWorld(path);
                    }
                }
                if h.world_back.map(|r| ui::hit(r, p)).unwrap_or(false) {
                    self.screen = Screen::Title;
                }
            }
            Screen::CreateWorld => {
                if h.create_name.map(|r| ui::hit(r, p)).unwrap_or(false) {
                    self.focus = Focus::WorldName;
                }
                if h.create_seed.map(|r| ui::hit(r, p)).unwrap_or(false) {
                    self.focus = Focus::WorldSeed;
                }
                if h.create_go.map(|r| ui::hit(r, p)).unwrap_or(false) {
                    return self.create_world(worlds_dir);
                }
                if h.create_back.map(|r| ui::hit(r, p)).unwrap_or(false) {
                    self.screen = Screen::Worlds;
                }
            }
            Screen::Multiplayer => {
                for (i, slot) in h.server_rows.iter().enumerate() {
                    if slot.map(|r| ui::hit(r, p)).unwrap_or(false) {
                        if i < self.servers.len() {
                            self.server_sel = i;
                        }
                        break;
                    }
                }
                if h.server_join.map(|r| ui::hit(r, p)).unwrap_or(false) {
                    if let Some(s) = self.servers.get(self.server_sel) {
                        return MenuAction::JoinServer(s.address.clone());
                    }
                }
                if h.server_add.map(|r| ui::hit(r, p)).unwrap_or(false) {
                    self.new_server_name.clear();
                    self.new_server_name_cursor = 0;
                    self.new_server_addr.clear();
                    self.new_server_addr_cursor = 0;
                    self.server_focus = ServerFocus::Name;
                    self.screen = Screen::AddServer;
                }
                if h.server_delete.map(|r| ui::hit(r, p)).unwrap_or(false)
                    && self.server_sel < self.servers.len()
                {
                    self.servers.remove(self.server_sel);
                    if self.server_sel >= self.servers.len() {
                        self.server_sel = self.servers.len().saturating_sub(1);
                    }
                    self.save_servers(worlds_dir);
                }
                if h.server_back.map(|r| ui::hit(r, p)).unwrap_or(false) {
                    self.screen = Screen::Title;
                }
            }
            Screen::AddServer => {
                if h.add_name.map(|r| ui::hit(r, p)).unwrap_or(false) {
                    self.server_focus = ServerFocus::Name;
                }
                if h.add_addr.map(|r| ui::hit(r, p)).unwrap_or(false) {
                    self.server_focus = ServerFocus::Address;
                }
                if h.add_go.map(|r| ui::hit(r, p)).unwrap_or(false) {
                    return self.confirm_add_server(worlds_dir);
                }
                if h.add_back.map(|r| ui::hit(r, p)).unwrap_or(false) {
                    self.screen = Screen::Multiplayer;
                }
            }
            Screen::Settings => {
                // Rows carry arrow hit-rects; the app routes them via
                // `settings_click` (kept here for layout co-location).
                for (i, slot) in h.setting_rows.iter().enumerate() {
                    if let Some(r) = slot {
                        if ui::hit(*r, p) {
                            return self.setting_arrow(i, p);
                        }
                    }
                }
                if h.settings_back.map(|r| ui::hit(r, p)).unwrap_or(false) {
                    self.save_settings();
                    self.screen = Screen::Title;
                }
            }
            Screen::Profile => {
                if h.profile_name.map(|r| ui::hit(r, p)).unwrap_or(false) {
                    self.profile_focus = ProfileFocus::Name;
                }
                if h.profile_skins.map(|r| ui::hit(r, p)).unwrap_or(false) {
                    self.refresh_skins();
                    self.screen = Screen::Skins;
                }
                if h.profile_back.map(|r| ui::hit(r, p)).unwrap_or(false) {
                    self.save_profile();
                    self.screen = Screen::Title;
                }
            }
            Screen::Skins => {
                for (i, slot) in h.skin_rows.iter().enumerate() {
                    if slot.map(|r| ui::hit(r, p)).unwrap_or(false) {
                        if i < self.skins.len() {
                            self.skin_sel = i;
                        }
                        break;
                    }
                }
                if h.skin_use.map(|r| ui::hit(r, p)).unwrap_or(false) {
                    if let Some(s) = self.skins.get(self.skin_sel).cloned() {
                        self.profile.skin = Some(s);
                        self.save_profile();
                        self.refresh_skins();
                        self.toast = format!("Skin set: {}", self.profile.skin.as_deref().unwrap_or(""));
                    }
                }
                if h.skin_back.map(|r| ui::hit(r, p)).unwrap_or(false) {
                    self.screen = Screen::Profile;
                }
            }
        }
        MenuAction::None
    }

    /// Settings row click: left/right arrows or row toggle.
    fn setting_arrow(&mut self, row: usize, p: (f32, f32)) -> MenuAction {
        let left = [0.0; 4];
        let _ = left;
        // Arrow rects were captured per row during draw via hits; simplify:
        // clicking the left 40% decrements, right 40% increments, middle is
        // a no-op toggle for booleans. Rows are fixed order (see draw).
        let h = self.hits;
        if let Some(r) = h.setting_rows[row] {
            let frac = (p.0 - r[0]) / r[2];
            let forward = frac > 0.6;
            match row {
                0 => {
                    self.settings.cycle_quality(forward);
                    return MenuAction::QualityChanged(self.settings.quality.clone());
                }
                1 => {
                    crate::settings::Settings::adjust_f32(
                        &mut self.settings.sensitivity,
                        if forward { 0.1 } else { -0.1 },
                        crate::settings::SENS_MIN,
                        crate::settings::SENS_MAX,
                    );
                }
                2 => {
                    crate::settings::Settings::adjust_f32(
                        &mut self.settings.fov,
                        if forward { 5.0 } else { -5.0 },
                        crate::settings::FOV_MIN,
                        crate::settings::FOV_MAX,
                    );
                }
                3 => {
                    self.settings.view_distance += if forward { 1 } else { -1 };
                    self.settings.view_distance = self
                        .settings
                        .view_distance
                        .clamp(crate::settings::VIEW_MIN, crate::settings::VIEW_MAX);
                }
                4 => {
                    crate::settings::Settings::adjust_f32(
                        &mut self.settings.day_length,
                        if forward { 120.0 } else { -120.0 },
                        crate::settings::DAY_MIN,
                        crate::settings::DAY_MAX,
                    );
                }
                5 => self.settings.hud = !self.settings.hud,
                _ => {}
            }
        }
        MenuAction::None
    }

    /// Confirm the AddServer form (extracted so Enter routes through the
    /// same logic as the DONE button).
    fn confirm_add_server(&mut self, worlds_dir: &Path) -> MenuAction {
        if self.new_server_addr.trim().is_empty() {
            self.toast = "Enter an address first".into();
            return MenuAction::None;
        }
        self.servers.push(ServerEntry {
            name: if self.new_server_name.trim().is_empty() {
                self.new_server_addr.clone()
            } else {
                self.new_server_name.clone()
            },
            address: self.new_server_addr.trim().to_string(),
        });
        self.server_sel = self.servers.len() - 1;
        self.save_servers(worlds_dir);
        self.screen = Screen::Multiplayer;
        MenuAction::None
    }

    /// Persist the current settings to `<worlds>/settings.json`.
    pub fn save_settings(&self) {
        let path = Settings::path_for(&self.settings_dir);
        self.settings.save(&path);
    }

    /// Create the world from the CreateWorld fields.
    fn create_world(&mut self, worlds_dir: &Path) -> MenuAction {
        let name = self.new_world_name.trim().to_string();
        if name.is_empty() {
            self.toast = "Name your world first".into();
            return MenuAction::None;
        }
        let dir = worlds_dir.join(sanitize_world(&name));
        if dir.exists() {
            self.toast = "A world with that name exists".into();
            return MenuAction::None;
        }
        let seed: u64 = if self.new_world_seed_text.trim().is_empty() {
            // Time-based default seed (deterministic per creation moment).
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0)
        } else if let Ok(n) = self.new_world_seed_text.trim().parse::<u64>() {
            n
        } else {
            // Text seeds hash like vanilla-ish world creators.
            hash_text(self.new_world_seed_text.trim())
        };
        // Materialize the world dir with a minimal save so the world list
        // and the app's PlayWorld path agree.
        let meta = feathered_world::save::WorldMeta {
            seed,
            player: feathered_world::save::PlayerSave {
                pos: [0.5, 40.0, 0.5],
                yaw: 0.0,
                pitch: 0.0,
            },
            day_fraction: Some(0.02),
            saved_at_unix: None,
        };
        let save = feathered_world::save::WorldSave::new(meta);
        if let Err(e) = feathered_world::save::save_to_dir(&dir, &save) {
            self.toast = format!("Create failed: {e}");
            return MenuAction::None;
        }
        MenuAction::PlayWorld(dir)
    }

    /// Route typed text to the focused field. `ch` is a char.
    pub fn type_char(&mut self, ch: char) {
        match (self.screen, self.focus, self.server_focus, self.profile_focus) {
            (Screen::CreateWorld, Focus::WorldName, ..) => {
                if self.new_world_name.chars().count() < 32 {
                    self.new_world_name.push(ch);
                    self.new_world_cursor = self.new_world_name.chars().count();
                }
            }
            (Screen::CreateWorld, Focus::WorldSeed, ..) => {
                if self.new_world_seed_text.chars().count() < 20 {
                    self.new_world_seed_text.push(ch);
                    self.new_world_seed_cursor = self.new_world_seed_text.chars().count();
                }
            }
            (Screen::AddServer, _, ServerFocus::Name, _) => {
                self.new_server_name.push(ch);
                self.new_server_name_cursor = self.new_server_name.chars().count();
            }
            (Screen::AddServer, _, ServerFocus::Address, _) => {
                self.new_server_addr.push(ch);
                self.new_server_addr_cursor = self.new_server_addr.chars().count();
            }
            (Screen::Profile, .., ProfileFocus::Name) if self.profile_name_cursor < 16 => {
                self.profile.name.push(ch);
                self.profile_name_cursor = self.profile.name.chars().count();
            }
            _ => {}
        }
    }

    /// Backspace on the focused field.
    pub fn backspace(&mut self) {
        match (self.screen, self.focus, self.server_focus, self.profile_focus) {
            (Screen::CreateWorld, Focus::WorldName, ..) => {
                self.new_world_name.pop();
                self.new_world_cursor = self.new_world_name.chars().count();
            }
            (Screen::CreateWorld, Focus::WorldSeed, ..) => {
                self.new_world_seed_text.pop();
                self.new_world_seed_cursor = self.new_world_seed_text.chars().count();
            }
            (Screen::AddServer, _, ServerFocus::Name, _) => {
                self.new_server_name.pop();
                self.new_server_name_cursor = self.new_server_name.chars().count();
            }
            (Screen::AddServer, _, ServerFocus::Address, _) => {
                self.new_server_addr.pop();
                self.new_server_addr_cursor = self.new_server_addr.chars().count();
            }
            (Screen::Profile, .., ProfileFocus::Name) => {
                self.profile.name.pop();
                self.profile_name_cursor = self.profile.name.chars().count();
            }
            _ => {}
        }
    }

    fn save_profile(&mut self) {
        let p = self.profile.clone();
        if self.profile_store.save(&p) {
            self.toast = "Profile saved".into();
        } else {
            self.toast = "Profile save failed".into();
        }
    }

    /// Escape: back one screen (Worlds/CreateWorld → Title, Settings/Profile
    /// → Title, AddServer → Multiplayer, Skins → Profile, Title → nothing).
    pub fn escape(&mut self) {
        let from = self.screen;
        match from {
            Screen::AddServer => self.screen = Screen::Multiplayer,
            Screen::Skins => self.screen = Screen::Profile,
            Screen::Worlds
            | Screen::Multiplayer
            | Screen::Settings
            | Screen::Profile
            | Screen::CreateWorld => {
                if from == Screen::Settings {
                    self.save_settings();
                }
                if from == Screen::Profile {
                    self.save_profile();
                }
                self.screen = Screen::Title;
            }
            Screen::Title => {}
        }
    }

    /// Enter on a form screen: submit it (CreateWorld / AddServer).
    pub fn confirm(&mut self, worlds_dir: &Path) -> MenuAction {
        match self.screen {
            Screen::CreateWorld => self.create_world(worlds_dir),
            Screen::AddServer => self.confirm_add_server(worlds_dir),
            Screen::Worlds => {
                // Enter plays the selected world.
                if let Some(w) = self.worlds.get(self.world_sel) {
                    let path = worlds_dir.join(&w.dir);
                    return MenuAction::PlayWorld(path);
                }
                MenuAction::None
            }
            _ => MenuAction::None,
        }
    }

    // ------------------------------------------------------------------
    // drawing
    // ------------------------------------------------------------------

    /// Draw the current screen into overlay lists, recording hit rects.
    pub fn draw(&mut self, list: &mut TriList, w: f32, hgt: f32) {
        let layout = Layout { w, h: hgt };
        self.hits = Hits::default();
        let cx = layout.cx();
        match self.screen {
            Screen::Title => {
                // Full-screen background image (background.png) when the app
                // loaded one; the fallback texture is transparent so the sky
                // shows through until then.
                if self.background {
                    list.background_quad(
                        [0.0, 0.0],
                        [w, 0.0],
                        [w, hgt],
                        [0.0, hgt],
                        lin4([255, 255, 255, 255]),
                    );
                }
                // The fade: opaque on the left where the brand + menu live,
                // easing out to the right so the art reads. Held ~55% then
                // eased (the mock's panel stays dark under the whole menu).
                let steps = 24;
                let panel_w = w * 0.66;
                let step_w = panel_w / steps as f32;
                for i in 0..steps {
                    let t = i as f32 / steps as f32;
                    let a: u8 = if t < 0.55 {
                        252
                    } else {
                        (252.0 * (1.0 - (t - 0.55) / 0.45)) as u8
                    };
                    list.quad(
                        [i as f32 * step_w, 0.0],
                        [(i + 1) as f32 * step_w, 0.0],
                        [(i + 1) as f32 * step_w, hgt],
                        [i as f32 * step_w, hgt],
                        lin4([10, 12, 16, a]),
                    );
                }
                // Feather art (white on transparency) upper-left; the
                // transparent canvas padding shows the panel behind.
                if let Some(art) = self.logo_art {
                    let art_h = hgt * 0.32;
                    let art_w = art_h * art;
                    let (ax, ay) = (18.0, 40.0);
                    list.textured_quad(
                        [ax, ay],
                        [ax + art_w, ay],
                        [ax + art_w, ay + art_h],
                        [ax, ay + art_h],
                        lin4([250, 250, 252, 255]),
                    );
                }
                draw_branding(list, w, hgt);
                // Hover feedback: brighten the entry under the cursor.
                let hover = |r: [f32; 4]| {
                    if ui::hit(r, self.cursor) {
                        Hover::Hovered
                    } else {
                        Hover::Idle
                    }
                };
                // Icon menu (reference): box icon + label + underline rule.
                let menu_x = 20.0;
                let row_h = 52.0;
                let row_w = 236.0;
                let icon = 34.0;
                let y0 = hgt * 0.50;
                type IconFn = fn(f32, f32, f32, f32, &mut TriList, [u8; 4]);
                let rows: [(&str, &str, IconFn); 3] = [
                    ("singleplayer", "Play", draw_icon_play),
                    ("settings", "Settings", draw_icon_gear),
                    ("quit", "Quit", draw_icon_power),
                ];
                for (i, (key, label, icon_fn)) in rows.iter().enumerate() {
                    let y = y0 + i as f32 * (row_h + 10.0);
                    let r = [menu_x, y, row_w, row_h];
                    let (face, glyph) = if hover(r) == Hover::Hovered {
                        (lin4([34, 37, 44, 220]), lin4([250, 250, 250, 255]))
                    } else {
                        (lin4([22, 24, 28, 200]), lin4([225, 228, 232, 255]))
                    };
                    // Left-edge diamond node (the mock's dotted rail).
                    let dcx = menu_x - 12.0;
                    let dcy = y + row_h / 2.0;
                    let d = 4.0;
                    list.quad(
                        [dcx - d, dcy],
                        [dcx, dcy - d],
                        [dcx + d, dcy],
                        [dcx, dcy + d],
                        lin4([150, 155, 165, 220]),
                    );
                    // Icon box + glyph.
                    list.quad([r[0], r[1]], [r[0] + icon, r[1]], [r[0] + icon, r[1] + icon], [r[0], r[1] + icon], face);
                    icon_fn(r[0] + 7.0, r[1] + 7.0, icon - 14.0, icon - 14.0, list, glyph);
                    // Label + underline rule.
                    font::draw_text_shadow(list, label, r[0] + icon + 14.0, r[1] + (icon - 7.0 * 2.0) / 2.0, 2.0, glyph);
                    let ly = r[1] + row_h - 2.0;
                    list.quad(
                        [r[0], ly],
                        [r[0] + row_w, ly],
                        [r[0] + row_w, ly + 2.0],
                        [r[0], ly + 2.0],
                        if hover(r) == Hover::Hovered { lin4([235, 238, 242, 230]) } else { lin4([80, 84, 92, 190]) },
                    );
                    let slot = match *key {
                        "singleplayer" => &mut self.hits.singleplayer,
                        "settings" => &mut self.hits.settings,
                        _ => &mut self.hits.quit,
                    };
                    *slot = Some(r);
                }
                // Bottom-left version block.
                font::draw_text_shadow(list, "v1.0.0", 18.0, hgt - 44.0, 2.5, lin4([235, 235, 235, 255]));
                font::draw_text_shadow(list, "Feathered Minecraft", 18.0, hgt - 18.0, 1.5, lin4([150, 155, 165, 235]));
                // Bottom-right links: MULTIPLAYER · PROFILE (and the
                // not-affiliated note on the far right, mock's link row).
                let link_y = hgt - 20.0;
                let lx = w - 18.0 - font::text_width("NOT AFFILIATED WITH MOJANG", 1.5);
                font::draw_text_shadow(list, "NOT AFFILIATED WITH MOJANG", lx, link_y, 1.5, lin4([120, 126, 136, 210]));
                let profile_rect = [lx - 90.0, link_y - 4.0, 80.0, 18.0];
                self.hits.profile = Some(profile_rect);
                font::draw_text_shadow(list, "PROFILE", profile_rect[0] + 8.0, link_y, 1.5, lin4([225, 228, 235, 240]));
                let mp_rect = [profile_rect[0] - 150.0, link_y - 4.0, 140.0, 18.0];
                self.hits.multiplayer = Some(mp_rect);
                font::draw_text_shadow(list, "MULTIPLAYER", mp_rect[0] + 8.0, link_y, 1.5, lin4([225, 228, 235, 240]));
            }
            Screen::Worlds => {
                font::draw_text_shadow(list, "SELECT WORLD", cx - font::text_width("SELECT WORLD", 3.0) / 2.0, layout.logo_y(), 3.0, [235, 235, 235, 255]);
                // Up to 6 world rows.
                let row_h = 44.0;
                let row_w = 560.0;
                let rows = self.hits.world_rows.len();
                let first = 0usize; // no scrolling yet (6 visible)
                for i in 0..rows {
                    let y = layout.logo_y() + 60.0 + i as f32 * (row_h + 6.0);
                    let r = [cx - row_w / 2.0, y, row_w, row_h];
                    let selected = i == self.world_sel;
                    let entry = self.worlds.get(first + i);
                    let (label, sub, state) = match entry {
                        Some(e) => (
                            e.dir.clone(),
                            format!("seed {} · {}", e.seed, e.info),
                            if selected { Hover::Hovered } else { Hover::Idle },
                        ),
                        None => ("(empty slot)".into(), String::new(), Hover::Disabled),
                    };
                    ui::small_button(list, r, &label, state);
                    if !sub.is_empty() {
                        font::draw_text_shadow(list, &sub, r[0] + 10.0, r[1] + row_h - 16.0, 1.5, [170, 175, 185, 230]);
                    }
                    self.hits.world_rows[i] = Some(r);
                }
                let by = hgt - 3.0 * (BUTTON_H + BUTTON_GAP) - 14.0;
                let half = (BUTTON_W - BUTTON_GAP) / 2.0;
                let has_sel = self.world_sel < self.worlds.len();
                self.hits.world_play = Some(ui::button(list, cx - half / 2.0 - BUTTON_GAP / 2.0, by, half, "PLAY SELECTED", if has_sel { Hover::Idle } else { Hover::Disabled }));
                self.hits.world_create = Some(ui::button(list, cx + half / 2.0 + BUTTON_GAP / 2.0, by, half, "CREATE NEW WORLD", Hover::Idle));
                let by2 = by + BUTTON_H + BUTTON_GAP;
                self.hits.world_delete = Some(ui::button(list, cx - half / 2.0 - BUTTON_GAP / 2.0, by2, half, "DELETE", if has_sel { Hover::Idle } else { Hover::Disabled }));
                self.hits.world_back = Some(ui::button(list, cx + half / 2.0 + BUTTON_GAP / 2.0, by2, half, "BACK", Hover::Idle));
            }
            Screen::CreateWorld => {
                font::draw_text_shadow(list, "CREATE NEW WORLD", cx - font::text_width("CREATE NEW WORLD", 3.0) / 2.0, layout.logo_y(), 3.0, [235, 235, 235, 255]);
                let y0 = hgt / 2.0 - 80.0;
                self.hits.create_name = Some(ui::text_field(list, cx, y0, BUTTON_W, "WORLD NAME", &self.new_world_name, self.new_world_cursor, self.focus == Focus::WorldName));
                self.hits.create_seed = Some(ui::text_field(list, cx, y0 + 60.0, BUTTON_W, "SEED (BLANK = RANDOM, TEXT OK)", &self.new_world_seed_text, self.new_world_seed_cursor, self.focus == Focus::WorldSeed));
                let half = (BUTTON_W - BUTTON_GAP) / 2.0;
                let by = y0 + 140.0;
                self.hits.create_go = Some(ui::button(list, cx - half / 2.0 - BUTTON_GAP / 2.0, by, half, "CREATE", Hover::Idle));
                self.hits.create_back = Some(ui::button(list, cx + half / 2.0 + BUTTON_GAP / 2.0, by, half, "CANCEL", Hover::Idle));
            }
            Screen::Multiplayer => {
                font::draw_text_shadow(list, "MULTIPLAYER", cx - font::text_width("MULTIPLAYER", 3.0) / 2.0, layout.logo_y(), 3.0, [235, 235, 235, 255]);
                font::draw_text_shadow(list, "NETWORK PLAY IS NOT IMPLEMENTED YET — SERVERS SAVE FOR LATER", cx - font::text_width("NETWORK PLAY IS NOT IMPLEMENTED YET — SERVERS SAVE FOR LATER", 1.5) / 2.0, layout.logo_y() + 34.0, 1.5, [200, 160, 120, 230]);
                let row_h = 44.0;
                let row_w = 560.0;
                for i in 0..self.hits.server_rows.len() {
                    let y = layout.logo_y() + 70.0 + i as f32 * (row_h + 6.0);
                    let r = [cx - row_w / 2.0, y, row_w, row_h];
                    let selected = i == self.server_sel;
                    let (label, sub, state) = match self.servers.get(i) {
                        Some(s) => (
                            s.name.clone(),
                            s.address.clone(),
                            if selected { Hover::Hovered } else { Hover::Idle },
                        ),
                        None => ("(no server)".into(), String::new(), Hover::Disabled),
                    };
                    ui::small_button(list, r, &label, state);
                    if !sub.is_empty() {
                        font::draw_text_shadow(list, &sub, r[0] + 10.0, r[1] + row_h - 16.0, 1.5, [170, 175, 185, 230]);
                    }
                    self.hits.server_rows[i] = Some(r);
                }
                let by = hgt - 3.0 * (BUTTON_H + BUTTON_GAP) - 14.0;
                let third = (BUTTON_W - 2.0 * BUTTON_GAP) / 3.0;
                let has_sel = self.server_sel < self.servers.len();
                self.hits.server_join = Some(ui::button(list, cx - third, by, third, "JOIN", if has_sel { Hover::Idle } else { Hover::Disabled }));
                self.hits.server_add = Some(ui::button(list, cx, by, third, "ADD SERVER", Hover::Idle));
                self.hits.server_delete = Some(ui::button(list, cx + third, by, third, "DELETE", if has_sel { Hover::Idle } else { Hover::Disabled }));
                self.hits.server_back = Some(ui::button(list, cx, by + BUTTON_H + BUTTON_GAP, BUTTON_W, "BACK", Hover::Idle));
            }
            Screen::AddServer => {
                font::draw_text_shadow(list, "ADD SERVER", cx - font::text_width("ADD SERVER", 3.0) / 2.0, layout.logo_y(), 3.0, [235, 235, 235, 255]);
                let y0 = hgt / 2.0 - 80.0;
                self.hits.add_name = Some(ui::text_field(list, cx, y0, BUTTON_W, "SERVER NAME", &self.new_server_name, self.new_server_name_cursor, self.server_focus == ServerFocus::Name));
                self.hits.add_addr = Some(ui::text_field(list, cx, y0 + 60.0, BUTTON_W, "SERVER ADDRESS", &self.new_server_addr, self.new_server_addr_cursor, self.server_focus == ServerFocus::Address));
                let half = (BUTTON_W - BUTTON_GAP) / 2.0;
                let by = y0 + 140.0;
                self.hits.add_go = Some(ui::button(list, cx - half / 2.0 - BUTTON_GAP / 2.0, by, half, "DONE", Hover::Idle));
                self.hits.add_back = Some(ui::button(list, cx + half / 2.0 + BUTTON_GAP / 2.0, by, half, "CANCEL", Hover::Idle));
            }
            Screen::Settings => {
                font::draw_text_shadow(list, "OPTIONS", cx - font::text_width("OPTIONS", 3.0) / 2.0, layout.logo_y(), 3.0, [235, 235, 235, 255]);
                let row_w = 560.0;
                let row_h = 40.0;
                let labels = [
                    ("RENDER QUALITY", self.settings.quality.clone(), true),
                    ("MOUSE SENSITIVITY", format!("{:.1}x", self.settings.sensitivity), true),
                    ("FIELD OF VIEW", format!("{:.0}", self.settings.fov), true),
                    ("VIEW DISTANCE", format!("{} CHUNKS", self.settings.view_distance), true),
                    ("DAY LENGTH", format!("{} S", self.settings.day_length as u32), true),
                    ("HUD", if self.settings.hud { "SHOWN".into() } else { "HIDDEN".into() }, false),
                ];
                for (i, (label, value, arrows)) in labels.iter().enumerate() {
                    let y = layout.logo_y() + 60.0 + i as f32 * (row_h + 6.0);
                    let r = [cx - row_w / 2.0, y, row_w, row_h];
                    let (_, l, rt) = ui::slider_row(list, r, label, value, *arrows, Hover::Idle);
                    // Capture row + arrow rects (arrow areas flank the value).
                    self.hits.setting_rows[i] = Some(r);
                    if i == 0 {
                        let _ = (l, rt);
                    }
                }
                self.hits.settings_back = Some(ui::button(list, cx, hgt - BUTTON_H - 20.0, BUTTON_W, "DONE", Hover::Idle));
            }
            Screen::Profile => {
                font::draw_text_shadow(list, "PROFILE", cx - font::text_width("PROFILE", 3.0) / 2.0, layout.logo_y(), 3.0, [235, 235, 235, 255]);
                // Avatar preview from the palette (blocky front view).
                let ax = cx - 150.0;
                let ay = hgt / 2.0 - 110.0;
                if let Some(p) = self.preview {
                    draw_avatar(list, ax, ay, p);
                } else {
                    font::draw_text_shadow(list, "NO SKIN SET", ax, ay + 40.0, 1.5, [170, 175, 185, 230]);
                }
                self.hits.profile_name = Some(ui::text_field(list, cx + 90.0, hgt / 2.0 - 60.0, 320.0, "PLAYER NAME", &self.profile.name, self.profile_name_cursor, self.profile_focus == ProfileFocus::Name));
                self.hits.profile_skins = Some(ui::button(list, cx + 90.0, hgt / 2.0 + 10.0, 320.0, "CHOOSE SKIN...", Hover::Idle));
                font::draw_text_shadow(list, "DROP MINECRAFT SKIN PNGS (64X64 / 64X32) INTO THE SKINS FOLDER", cx - font::text_width("DROP MINECRAFT SKIN PNGS (64X64 / 64X32) INTO THE SKINS FOLDER", 1.2) / 2.0, hgt - 60.0, 1.2, [170, 175, 185, 220]);
                self.hits.profile_back = Some(ui::button(list, cx, hgt - BUTTON_H - 20.0, BUTTON_W, "SAVE + BACK", Hover::Idle));
                if !self.toast.is_empty() {
                    font::draw_text_shadow(list, &self.toast, cx - font::text_width(&self.toast, 1.5) / 2.0, hgt - 90.0, 1.5, [255, 220, 140, 240]);
                }
            }
            Screen::Skins => {
                font::draw_text_shadow(list, "CHOOSE SKIN", cx - font::text_width("CHOOSE SKIN", 3.0) / 2.0, layout.logo_y(), 3.0, [235, 235, 235, 255]);
                let row_h = 44.0;
                for i in 0..self.hits.skin_rows.len() {
                    let y = layout.logo_y() + 60.0 + i as f32 * (row_h + 6.0);
                    let r = [cx - 280.0, y, 560.0, row_h];
                    let selected = i == self.skin_sel;
                    let (label, state) = match self.skins.get(i) {
                        Some(name) => (
                            name.clone(),
                            if selected { Hover::Hovered } else { Hover::Idle },
                        ),
                        None => ("(no skins yet)".into(), Hover::Disabled),
                    };
                    ui::small_button(list, r, &label, state);
                    self.hits.skin_rows[i] = Some(r);
                }
                let by = hgt - 2.0 * (BUTTON_H + BUTTON_GAP) - 14.0;
                let half = (BUTTON_W - BUTTON_GAP) / 2.0;
                let has_sel = self.skin_sel < self.skins.len();
                self.hits.skin_use = Some(ui::button(list, cx - half / 2.0 - BUTTON_GAP / 2.0, by, half, "USE SKIN", if has_sel { Hover::Idle } else { Hover::Disabled }));
                self.hits.skin_back = Some(ui::button(list, cx + half / 2.0 + BUTTON_GAP / 2.0, by, half, "BACK", Hover::Idle));
            }
        }
    }
}

/// Blocky front-view avatar from a skin palette (head/body/legs colors).
fn draw_avatar(list: &mut TriList, x: f32, y: f32, p: SkinPaletteCache) {
    // Head 40x40, torso 40x60, legs 2x 20x50.
    let head = 40.0f32;
    let torso_w = 40.0f32;
    let torso_h = 56.0f32;
    let leg_h = 52.0f32;
    // Head.
    list.quad([x, y], [x + head, y], [x + head, y + head], [x, y + head], p.head);
    // Torso.
    let ty = y + head + 2.0;
    list.quad([x, ty], [x + torso_w, ty], [x + torso_w, ty + torso_h], [x, ty + torso_h], p.body);
    // Legs.
    let ly = ty + torso_h;
    let leg_w = torso_w / 2.0;
    list.quad([x, ly], [x + leg_w, ly], [x + leg_w, ly + leg_h], [x, ly + leg_h], p.legs);
    list.quad([x + leg_w, ly], [x + torso_w, ly], [x + torso_w, ly + leg_h], [x + leg_w, ly + leg_h], p.legs);
}

/// Title branding block (reference layout): FEATHERED wordmark with side
/// rules + letter-spaced MINECRAFT subtitle under the feather art.
fn draw_branding(list: &mut TriList, w: f32, h: f32) {
    const TEXT: &str = "FEATHERED";
    // Scale 7 ≈ 32% of a 1280-wide frame; left-aligned like the mock.
    let scale = 7.0f32;
    let x = 42.0f32;
    let y = h * 0.335;
    let tw = font::text_width(TEXT, scale);
    font::draw_text_shadow(list, TEXT, x, y, scale, lin4([240, 242, 246, 255]));
    // Side rules flanking MINECRAFT (letter-spaced feel via extra scale-1
    // gaps drawn between characters' widths).
    let sub = "M I N E C R A F T";
    let sub_scale = 2.0f32;
    let sub_w = font::text_width(sub, sub_scale);
    let sub_y = y + 7.0 * scale + 14.0;
    font::draw_text_shadow(list, sub, x, sub_y, sub_scale, lin4([190, 194, 202, 240]));
    let rule_y = sub_y + 9.0;
    list.quad([x, rule_y], [x + 26.0, rule_y], [x + 26.0, rule_y + 2.0], [x, rule_y + 2.0], lin4([120, 126, 136, 220]));
    list.quad([x + sub_w - 26.0, rule_y], [x + sub_w, rule_y], [x + sub_w, rule_y + 2.0], [x + sub_w - 26.0, rule_y + 2.0], lin4([120, 126, 136, 220]));
    let _ = (w, tw); // width reference for future responsive scaling
}

// --- title icon glyphs (24×24-ish vector boxes, drawn as flat quads) ---

fn draw_icon_play(x: f32, y: f32, w: f32, h: f32, list: &mut TriList, c: [u8; 4]) {
    // Triangle pointing right, centered in the box.
    let cx = x + w * 0.5;
    let cy = y + h * 0.5;
    let s = h * 0.32;
    list.quad([cx - s * 0.55, cy - s], [cx - s * 0.55, cy + s], [cx + s * 0.9, cy], [cx - s * 0.55, cy - s], c);
}

fn draw_icon_gear(x: f32, y: f32, w: f32, h: f32, list: &mut TriList, c: [u8; 4]) {
    // Ring + 4 notches (screen-space quad approximation of a gear).
    let cx = x + w * 0.5;
    let cy = y + h * 0.5;
    let r = h * 0.30;
    let t = h * 0.10;
    list.quad([cx - r, cy - t / 2.0], [cx + r, cy - t / 2.0], [cx + r, cy + t / 2.0], [cx - r, cy + t / 2.0], c);
    list.quad([cx - t / 2.0, cy - r], [cx + t / 2.0, cy - r], [cx + t / 2.0, cy + r], [cx - t / 2.0, cy + r], c);
    list.quad([cx - r * 0.72, cy - t / 2.0], [cx - r * 0.45, cy - t / 2.0], [cx - r * 0.45, cy + t / 2.0], [cx - r * 0.72, cy + t / 2.0], c);
    let _ = w;
}

fn draw_icon_power(x: f32, y: f32, w: f32, h: f32, list: &mut TriList, c: [u8; 4]) {
    // Power symbol: circle stroke (4 side strips) + vertical bar.
    let cx = x + w * 0.5;
    let cy = y + h * 0.55;
    let r = h * 0.26;
    let t = h * 0.09;
    list.quad([cx - r, cy - t / 2.0], [cx + r, cy - t / 2.0], [cx + r, cy + t / 2.0], [cx - r, cy + t / 2.0], c);
    list.quad([cx - r, cy], [cx - r + t, cy], [cx - r + t, cy + r], [cx - r, cy + r], c);
    list.quad([cx + r - t, cy], [cx + r, cy], [cx + r, cy + r], [cx + r - t, cy + r], c);
    list.quad([cx - t / 2.0, cy - r * 1.25], [cx + t / 2.0, cy - r * 1.25], [cx + t / 2.0, cy + r * 0.4], [cx - t / 2.0, cy + r * 0.4], c);
    let _ = w;
}

/// Sanitize a world name into a directory name.
fn sanitize_world(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-' || *c == ' ')
        .collect();
    let trimmed = cleaned.trim();
    if trimmed.is_empty() {
        "world".into()
    } else {
        trimmed.to_string()
    }
}

/// FNV-ish text hash for word seeds.
fn hash_text(s: &str) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100_0000_01B3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn menu() -> MenuState {
        let dir = std::env::temp_dir().join(format!("feathered-menu-{}", std::process::id()));
        MenuState::new(&dir, Settings::default(), Profile::default())
    }

    #[test]
    fn title_buttons_route_to_screens() {
        let dir = std::env::temp_dir().join(format!("feathered-menu-t-{}", std::process::id()));
        let mut m = menu();
        // Simulate a drawn title at 1280x720 by drawing first.
        let mut list = TriList::default();
        m.draw(&mut list, 1280.0, 720.0);
        let sp = m.hits.singleplayer.unwrap();
        assert_eq!(m.click(ui::center(sp), &dir), MenuAction::None);
        assert_eq!(m.screen, Screen::Worlds);

        m.screen = Screen::Title;
        let mp = m.hits.multiplayer.unwrap();
        m.click(ui::center(mp), &dir);
        assert_eq!(m.screen, Screen::Multiplayer);

        m.screen = Screen::Title;
        let q = m.hits.quit.unwrap();
        assert_eq!(m.click(ui::center(q), &dir), MenuAction::Quit);
    }

    #[test]
    fn create_world_flow_produces_play_action() {
        let dir = std::env::temp_dir().join(format!("feathered-menu-c-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut m = menu();
        m.screen = Screen::CreateWorld;
        for ch in "My World".chars() {
            m.type_char(ch);
        }
        m.focus = Focus::WorldSeed; // click the seed field, then type
        for ch in "12345".chars() {
            m.type_char(ch);
        }
        assert_eq!(m.new_world_name, "My World");
        assert_eq!(m.new_world_seed_text, "12345");
        let action = m.create_world(&dir);
        match action {
            MenuAction::PlayWorld(p) => {
                assert!(p.join("world.feathered").exists(), "world materialized");
                let save = feathered_world::save::load_from_dir(&p).unwrap();
                assert_eq!(save.meta.seed, 12345);
            }
            other => panic!("expected PlayWorld, got {other:?}"),
        }
        // Duplicate name is rejected (m2 must use the same worlds dir).
        let mut m2 = MenuState::new(&dir, Settings::default(), Profile::default());
        m2.screen = Screen::CreateWorld;
        m2.new_world_name = "My World".into();
        assert_eq!(m2.create_world(&dir), MenuAction::None);
        assert!(m2.toast.contains("exists"));
        // Empty name is rejected.
        let mut m3 = MenuState::new(&dir, Settings::default(), Profile::default());
        m3.screen = Screen::CreateWorld;
        assert_eq!(m3.create_world(&dir), MenuAction::None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn text_seeds_hash_deterministically() {
        let dir = std::env::temp_dir().join(format!("feathered-menu-h-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut a = menu();
        a.screen = Screen::CreateWorld;
        a.new_world_name = "HashWorld".into();
        a.new_world_seed_text = "hello".into();
        let pa = match a.create_world(&dir) {
            MenuAction::PlayWorld(p) => p,
            other => panic!("{other:?}"),
        };
        let mut b = menu();
        b.screen = Screen::CreateWorld;
        b.new_world_name = "HashWorld2".into();
        b.new_world_seed_text = "hello".into();
        let pb = match b.create_world(&dir) {
            MenuAction::PlayWorld(p) => p,
            other => panic!("{other:?}"),
        };
        let sa = feathered_world::save::load_from_dir(&pa).unwrap();
        let sb = feathered_world::save::load_from_dir(&pb).unwrap();
        assert_eq!(sa.meta.seed, sb.meta.seed, "same text seed → same seed");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn backspace_edits_fields() {
        let mut m = menu();
        m.screen = Screen::CreateWorld;
        for ch in "abc".chars() {
            m.type_char(ch);
        }
        m.backspace();
        assert_eq!(m.new_world_name, "ab");
        assert_eq!(m.new_world_cursor, 2);
    }

    #[test]
    fn settings_arrows_adjust_values() {
        let mut m = menu();
        m.screen = Screen::Settings;
        // Fake row rects for arrows (draw normally populates these).
        m.hits.setting_rows[0] = Some([100.0, 0.0, 400.0, 30.0]);
        m.hits.setting_rows[2] = Some([100.0, 0.0, 400.0, 30.0]);
        // Quality forward (right side of the row).
        assert_eq!(
            m.setting_arrow(0, (440.0, 15.0)),
            MenuAction::QualityChanged("High".into())
        );
        // FOV decrement (left side).
        let before = m.settings.fov;
        let _ = m.setting_arrow(2, (120.0, 15.0));
        assert!((m.settings.fov - (before - 5.0)).abs() < 1e-4);
    }

    #[test]
    fn sanitize_world_names() {
        assert_eq!(sanitize_world("My World!"), "My World");
        assert_eq!(sanitize_world("  "), "world");
        assert_eq!(sanitize_world("../evil"), "evil");
    }

    #[test]
    fn text_hash_is_stable() {
        assert_eq!(hash_text("abc"), hash_text("abc"));
        assert_ne!(hash_text("abc"), hash_text("abd"));
    }

    #[test]
    fn escape_walks_back_and_saves_settings() {
        let dir = std::env::temp_dir().join(format!("feathered-menu-esc-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut m = MenuState::new(&dir, Settings::default(), Profile::default());
        // Title escape is a no-op.
        m.escape();
        assert_eq!(m.screen, Screen::Title);
        // Settings → Title persists settings.json.
        m.screen = Screen::Settings;
        m.settings.fov = 90.0;
        m.escape();
        assert_eq!(m.screen, Screen::Title);
        let (loaded, _) = Settings::load(&Settings::path_for(&dir));
        assert!((loaded.fov - 90.0).abs() < 1e-4);
        // AddServer → Multiplayer, Skins → Profile.
        m.screen = Screen::AddServer;
        m.escape();
        assert_eq!(m.screen, Screen::Multiplayer);
        m.screen = Screen::Skins;
        m.escape();
        assert_eq!(m.screen, Screen::Profile);
    }

    #[test]
    fn enter_confirms_create_world_and_add_server() {
        let dir = std::env::temp_dir().join(format!("feathered-menu-enter-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut m = MenuState::new(&dir, Settings::default(), Profile::default());
        // CreateWorld: Enter submits the form.
        m.screen = Screen::CreateWorld;
        m.new_world_name = "Enter World".into();
        assert!(matches!(m.confirm(&dir), MenuAction::PlayWorld(_)));
        assert!(dir.join("Enter World").join("world.feathered").exists());
        // AddServer: Enter adds the entry.
        m.screen = Screen::AddServer;
        m.new_server_addr = "localhost:25565".into();
        m.confirm(&dir);
        assert_eq!(m.screen, Screen::Multiplayer);
        assert_eq!(m.servers.len(), 1);
        assert_eq!(m.servers[0].address, "localhost:25565");
        // Worlds: Enter plays the selected world.
        m.screen = Screen::Worlds;
        m.scan_worlds(&dir);
        m.world_sel = 0;
        match m.confirm(&dir) {
            MenuAction::PlayWorld(p) => assert!(p.ends_with("Enter World")),
            other => panic!("expected PlayWorld, got {other:?}"),
        }
    }
}
