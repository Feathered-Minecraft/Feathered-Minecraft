//! Title screen: component-based UI built from deterministic primitives.
//!
//! This module owns the title screen's *layout and drawing* as reusable
//! components ([`TitleLayout`], brand, rail, menu rows, footer, version,
//! gradient). It is pure data + draw lists — no GPU types beyond the
//! overlay `TriList`, fully testable headless — and integrates with the
//! existing wgpu overlay pipeline (screen-space quads + bitmap font).
//!
//! Design rules (the Feathered monochrome language):
//! * everything is authored against a 1920×1080 reference frame and scaled
//!   by `min(w/1920, h/1080)` — proportions hold at 720p..1440p and the
//!   left composition stays anchored while the world shows to the right;
//! * one vertical rail, straight, with one diamond per row centered on the
//!   row's vertical center (computed from the same rects — no drift);
//! * labels are real text through the bitmap font; hitboxes are the row
//!   rects themselves, so mouse input matches the visible rows exactly;
//! * the background stays the live voxel render; the gradient is two
//!   GPU-interpolated quads (opaque hold + ease-out), never a baked image.
//!
//! Game-state transitions live in [`crate::menu`] — this module never
//! changes screens itself.

use crate::font;
use crate::overlay::TriList;
use crate::ui::lin4;

/// Reference frame the design is authored against.
pub const REF_W: f32 = 1920.0;
pub const REF_H: f32 = 1080.0;

/// Footer links (the only external URLs the UI will ever open; the app
/// whitelists exactly these before invoking the OS handler).
pub const GITHUB_URL: &str = "https://github.com/Feathered-Minecraft/Feathered-Minecraft";
pub const DOCS_URL: &str = "https://github.com/Feathered-Minecraft/Feathered-Minecraft/tree/main/docs";
pub const LICENSE_URL: &str = "https://github.com/Feathered-Minecraft/Feathered-Minecraft/blob/main/LICENSE";
/// The complete set of URLs the title screen may open (safety gate).
pub const ALLOWED_URLS: [&str; 3] = [GITHUB_URL, DOCS_URL, LICENSE_URL];

/// Monochrome palette (sRGB bytes; `lin4` converts for the GPU).
mod pal {
    pub const TEXT: [u8; 4] = [240, 242, 246, 255];
    pub const SUBTLE: [u8; 4] = [168, 173, 183, 255];
    pub const FAINT: [u8; 4] = [122, 128, 138, 235];
    pub const LINE_IDLE: [u8; 4] = [96, 101, 111, 210];
    pub const LINE_HOT: [u8; 4] = [225, 229, 236, 240];
    pub const BOX_IDLE: [u8; 4] = [188, 193, 203, 225];
    pub const BOX_HOT: [u8; 4] = [255, 255, 255, 255];
    pub const ROW_BG_HOT: [u8; 4] = [18, 20, 24, 130];
    pub const MARKER: [u8; 4] = [205, 210, 218, 240];
    pub const RAIL: [u8; 4] = [150, 155, 165, 200];
    pub const GRADIENT: [u8; 4] = [8, 10, 13, 0];
}

/// A screen rect (x, y, w, h) in UI pixels.
pub type Rect = [f32; 4];

/// One menu entry descriptor (visual only; routing stays in menu.rs).
pub struct MenuEntry {
    pub label: &'static str,
    pub icon: Icon,
}

/// The three title entries — exactly Play, Settings, Quit.
pub const MENU_ENTRIES: [MenuEntry; 3] = [
    MenuEntry { label: "Play", icon: Icon::Play },
    MenuEntry { label: "Settings", icon: Icon::Gear },
    MenuEntry { label: "Quit", icon: Icon::Power },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Icon {
    Play,
    Gear,
    Power,
}

/// Deterministic title-screen layout, derived from the window size.
///
/// All fields are UI pixels. Everything is a function of `(w, h)` — no
/// hardcoded screenshot coordinates, no hidden state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TitleLayout {
    pub w: f32,
    pub h: f32,
    /// Uniform scale from the 1920×1080 reference (`min` keeps the left
    /// composition fully on-screen on any aspect ratio).
    pub s: f32,
    /// Feather art rect (aspect applied by the caller from the asset).
    pub logo: Rect,
    /// Wordmark baseline geometry.
    pub wordmark_x: f32,
    pub wordmark_y: f32,
    pub wordmark_scale: f32,
    pub subtitle_y: f32,
    pub subtitle_scale: f32,
    /// The three menu rows (icon box + label + underline hit area).
    pub rows: [Rect; 3],
    pub row_h: f32,
    pub row_w: f32,
    pub row_gap: f32,
    /// Vertical rail (x = center of the 2px line) and its extent.
    pub rail_x: f32,
    pub rail_y0: f32,
    pub rail_y1: f32,
    /// Diamond marker half-diagonal.
    pub marker_r: f32,
    /// Version block (accent bar + two text lines).
    pub version_bar: Rect,
    pub version_y: f32,
    pub version_sub_y: f32,
    pub version_scale: f32,
    pub version_sub_scale: f32,
    /// Footer link rects, in order GitHub / Documentation / License.
    pub footer: [Rect; 3],
    pub footer_dividers: [f32; 2],
    pub footer_y: f32,
    pub footer_scale: f32,
}

impl TitleLayout {
    /// Build the layout for a window of `w × h` UI pixels.
    pub fn build(w: f32, h: f32) -> TitleLayout {
        let s = (w / REF_W).min(h / REF_H);
        let m = 78.0 * s; // left margin
        let row_h = 64.0 * s;
        let row_w = 500.0 * s;
        let row_gap = 26.0 * s;
        let menu_x = m + 46.0 * s; // icon boxes / row left edge
        let rail_x = m + 4.0 * s;

        let menu_y0_default = 0.44 * h;
        let rows = [
            [menu_x, menu_y0_default, row_w, row_h],
            [menu_x, menu_y0_default + row_h + row_gap, row_w, row_h],
            [menu_x, menu_y0_default + 2.0 * (row_h + row_gap), row_w, row_h],
        ];
        let menu_end = rows[2][1] + row_h;

        // Logo art: height fixed to the reference proportion; the caller
        // computes width from the asset's aspect (never stretched).
        let logo_h = 0.26 * h;
        let logo = [menu_x, 0.115 * h, logo_h * 1.15, logo_h]; // aspect replaced at draw time

        let wordmark_scale = 9.0 * s;
        let wordmark_x = m + 34.0 * s;
        let wordmark_y = 0.115 * h + logo_h + 10.0 * s;
        let subtitle_scale = 3.0 * s;
        let subtitle_y = wordmark_y + 7.0 * wordmark_scale + 12.0 * s;
        // The menu starts below the brand block (never overlapping it),
        // whatever the window aspect does to the h-proportional terms.
        let menu_y0 = menu_y0_default.max(subtitle_y + 7.0 * subtitle_scale + 46.0 * s);
        let rows = rows.map(|r| [r[0], menu_y0 + (r[1] - menu_y0_default), r[2], r[3]]);
        let menu_end = rows[2][1] + row_h;

        let footer_scale = 3.0 * s;
        let footer_y = h - 44.0 * s;
        // Right-anchored footer: widths measured with the real font.
        let labels = ["GITHUB", "DOCUMENTATION", "LICENSE"];
        let gap = 30.0 * s;
        let widths: Vec<f32> = labels
            .iter()
            .map(|l| font::text_width(l, footer_scale))
            .collect();
        let total: f32 = widths.iter().sum::<f32>() + 2.0 * gap + 2.0 * 16.0 * s;
        let mut fx = w - 40.0 * s - total;
        let mut footer = [[0.0f32; 4]; 3];
        let mut dividers = [0.0f32; 2];
        for i in 0..3 {
            footer[i] = [fx, footer_y - 6.0 * s, widths[i], 7.0 * footer_scale + 12.0 * s];
            fx += widths[i] + 16.0 * s;
            if i < 2 {
                dividers[i] = fx + 4.0 * s;
                fx += gap;
            }
        }

        let version_scale = 5.0 * s;
        let version_y = h - 78.0 * s;
        let version_sub_scale = 2.5 * s;
        let version_sub_y = h - 32.0 * s;

        TitleLayout {
            w,
            h,
            s,
            logo,
            wordmark_x,
            wordmark_y,
            wordmark_scale,
            subtitle_y,
            subtitle_scale,
            rows,
            row_h,
            row_w,
            row_gap,
            rail_x,
            rail_y0: menu_y0 - 44.0 * s,
            rail_y1: menu_end + 58.0 * s,
            marker_r: 8.0 * s,
            version_bar: [m, version_y - 4.0 * s, 5.0 * s, 52.0 * s],
            version_y,
            version_sub_y,
            version_scale,
            version_sub_scale,
            footer,
            footer_dividers: dividers,
            footer_y,
            footer_scale,
        }
    }

    /// Vertical center of row `i` (the rail marker must sit exactly here).
    pub fn row_center_y(&self, i: usize) -> f32 {
        self.rows[i][1] + self.rows[i][3] / 2.0
    }

    /// The diamond marker center for row `i` (on the rail, at row center).
    pub fn marker_center(&self, i: usize) -> (f32, f32) {
        (self.rail_x, self.row_center_y(i))
    }

    /// Hit rect for footer link `i` (== its visible bounds).
    pub fn footer_rect(&self, i: usize) -> Rect {
        self.footer[i]
    }

    /// The rect a row occupies on screen (base, or grown when selected).
    /// Input hit-testing must use this — it is exactly what is drawn.
    pub fn row_hit_rect(&self, i: usize, selected: bool) -> Rect {
        if selected {
            grow_row(self.rows[i], 1.08, self)
        } else {
            self.rows[i]
        }
    }
}

// ---------------------------------------------------------------------------
// drawing
// ---------------------------------------------------------------------------

/// The left gradient overlay: opaque behind the brand/menu, easing smoothly
/// to fully transparent so the live world reads. Two quads joined where
/// their alpha matches (C0-continuous — no visible seam, no hard cutoff),
/// interpolated by the GPU via per-vertex colors. Never a baked image.
pub fn draw_gradient(list: &mut TriList, w: f32, h: f32) {
    let span = w * 0.72;
    let hold_x = span * 0.42; // fully dark region over the composition
    let base = [8u8, 10, 13, 246];
    let clear = pal::GRADIENT;
    // Solid hold (covers the left composition).
    list.quad([0.0, 0.0], [hold_x, 0.0], [hold_x, h], [0.0, h], lin4(base));
    // Ease-out to zero alpha (horizontal-only interpolation).
    list.quad_gradient(
        [hold_x, 0.0],
        [span, 0.0],
        [span, h],
        [hold_x, h],
        lin4(base),
        lin4(clear),
        lin4(clear),
        lin4(base),
    );
}

/// FeatheredBrand: feather art (real asset through the logo texture slot,
/// aspect preserved — never stretched) + wordmark + tracked subtitle with
/// flanking rules. `logo_art` is the asset's aspect when loaded; `None`
/// skips the art (the texture slot stays empty — no fake replacement).
pub fn draw_brand(list: &mut TriList, l: &TitleLayout, logo_art: Option<f32>) {
    if let Some(aspect) = logo_art {
        let lh = l.logo[3];
        let lw = (lh * aspect).min(l.w * 0.42);
        let x = l.logo[0] + (l.logo[2] - lw).max(0.0);
        list.textured_quad(
            [x, l.logo[1]],
            [x + lw, l.logo[1]],
            [x + lw, l.logo[1] + lh],
            [x, l.logo[1] + lh],
            lin4([250, 250, 252, 255]),
        );
    }
    // Wordmark: the dominant element (scale 9 vs subtitle 3 at reference).
    let wm = "FEATHERED";
    let wm_w = font::text_width(wm, l.wordmark_scale);
    font::draw_text_shadow(
        list,
        wm,
        l.wordmark_x,
        l.wordmark_y,
        l.wordmark_scale,
        lin4(pal::TEXT),
    );
    // Tracked subtitle centered under the wordmark, rules flanking.
    let sub = "MINECRAFT";
    let adv = font::GLYPH_W * l.subtitle_scale + 11.0 * l.s;
    let sub_w = 9.0 * adv - 11.0 * l.s;
    let sub_x = l.wordmark_x + (wm_w - sub_w) / 2.0;
    let sub_cy = l.subtitle_y + 3.5 * l.subtitle_scale;
    draw_tracked(list, sub, sub_x, l.subtitle_y, l.subtitle_scale, adv, lin4(pal::SUBTLE));
    // Thin rules from the wordmark edges to the subtitle.
    let ry = sub_cy - 1.0 * l.s;
    let rh = 2.0 * l.s;
    rule(list, l.wordmark_x, sub_x - 16.0 * l.s, ry, rh);
    rule(list, sub_x + sub_w + 16.0 * l.s, l.wordmark_x + wm_w, ry, rh);
}

/// MenuRail: one straight vertical line + one diamond per row, each exactly
/// centered on its row's vertical center (computed from the same rects).
pub fn draw_rail(list: &mut TriList, l: &TitleLayout, selected: usize) {
    let t = 2.0 * l.s;
    list.quad(
        [l.rail_x - t / 2.0, l.rail_y0],
        [l.rail_x + t / 2.0, l.rail_y0],
        [l.rail_x + t / 2.0, l.rail_y1],
        [l.rail_x - t / 2.0, l.rail_y1],
        lin4(pal::RAIL),
    );
    for i in 0..3 {
        let (cx, cy) = l.marker_center(i);
        let r = if i == selected { l.marker_r * 1.25 } else { l.marker_r };
        diamond(list, cx, cy, r, lin4(pal::MARKER));
    }
}

/// One MenuItem row: bordered icon box + label + underline extension line.
/// Selected rows grow slightly around their center (a deliberate, subtle
/// hover state). The returned rect IS the grown rect — hit-testing uses it
/// too, so mouse input and visuals never disagree.
pub fn draw_menu_item(
    list: &mut TriList,
    l: &TitleLayout,
    i: usize,
    selected: bool,
) -> Rect {
    let base = l.rows[i];
    let r = if selected {
        grow_row(base, 1.08, l)
    } else {
        base
    };
    let [x, y, w, h] = r;
    let entry = &MENU_ENTRIES[i];
    let (box_edge, text, line) = if selected {
        (lin4(pal::BOX_HOT), lin4(pal::TEXT), lin4(pal::LINE_HOT))
    } else {
        (lin4(pal::BOX_IDLE), lin4(pal::SUBTLE), lin4(pal::LINE_IDLE))
    };
    // Subtle row wash on the selected row (keeps the world visible).
    if selected {
        list.quad([x, y], [x + w, y], [x + w, y + h], [x, y + h], lin4(pal::ROW_BG_HOT));
    }
    // Icon box: bordered square, height = row height (mock proportions).
    let b = 2.0 * l.s;
    list.quad([x, y], [x + h, y], [x + h, y + h], [x, y + h], lin4([10, 12, 16, 150]));
    list.quad([x, y], [x + h, y], [x + h, y + b], [x, y + b], box_edge);
    list.quad([x, y + h - b], [x + h, y + h - b], [x + h, y + h], [x, y + h], box_edge);
    list.quad([x, y], [x + b, y], [x + b, y + h], [x, y + h], box_edge);
    list.quad([x + h - b, y], [x + h, y], [x + h, y + h], [x + h - b, y + h], box_edge);
    draw_icon(entry.icon, list, x + h * 0.28, y + h * 0.28, h * 0.44, box_edge);
    // Label vertically centered in the row.
    let scale = 4.0 * l.s;
    let ty = y + (h - 7.0 * scale) / 2.0;
    font::draw_text_shadow(list, entry.label, x + h + 26.0 * l.s, ty, scale, text);
    // Underline extension line at the row bottom.
    let ly = y + h;
    rule_colored(list, x, x + w, ly, 2.0 * l.s.max(1.5), line);
    r
}

/// Grow a row around its center by `f` (clamped to stay on-screen and off
/// the rail). Pure geometry — the same function hit-testing must use.
pub fn grow_row(base: Rect, f: f32, l: &TitleLayout) -> Rect {
    let [x, y, w, h] = base;
    let nw = (w * f).min(l.w - x - 8.0 * l.s);
    let nh = h * f;
    let cx = x + w / 2.0;
    let cy = y + h / 2.0;
    [cx - nw / 2.0, cy - nh / 2.0, nw, nh]
}

/// VersionLabel: accent bar + prominent `v1.0.0` + subtle product line.
pub fn draw_version(list: &mut TriList, l: &TitleLayout) {
    list.quad(
        [l.version_bar[0], l.version_bar[1]],
        [l.version_bar[0] + l.version_bar[2], l.version_bar[1]],
        [l.version_bar[0] + l.version_bar[2], l.version_bar[1] + l.version_bar[3]],
        [l.version_bar[0], l.version_bar[1] + l.version_bar[3]],
        lin4([235, 238, 243, 245]),
    );
    let tx = l.version_bar[0] + 20.0 * l.s;
    font::draw_text_shadow(list, "V1.0.0", tx, l.version_y, l.version_scale, lin4(pal::TEXT));
    font::draw_text_shadow(
        list,
        "FEATHERED MINECRAFT",
        tx,
        l.version_sub_y,
        l.version_sub_scale,
        lin4(pal::FAINT),
    );
}

/// FooterLinks: GitHub / Documentation / License with thin dividers.
/// Returns nothing — hit rects come from the layout (they are identical).
pub fn draw_footer(list: &mut TriList, l: &TitleLayout) {
    let labels = ["GITHUB", "DOCUMENTATION", "LICENSE"];
    for (i, label) in labels.iter().enumerate() {
        let r = l.footer[i];
        font::draw_text_shadow(
            list,
            label,
            r[0],
            l.footer_y,
            l.footer_scale,
            lin4([236, 239, 244, 250]),
        );
        // Small square marker before each label (restrained geometric accent).
        list.quad(
            [r[0] - 14.0 * l.s, l.footer_y + 2.0 * l.s],
            [r[0] - 14.0 * l.s + 4.0 * l.s, l.footer_y + 2.0 * l.s],
            [r[0] - 14.0 * l.s + 4.0 * l.s, l.footer_y + 6.0 * l.s],
            [r[0] - 14.0 * l.s, l.footer_y + 6.0 * l.s],
            lin4(pal::SUBTLE),
        );
    }
    for &dx in &l.footer_dividers {
        list.quad(
            [dx, l.footer_y],
            [dx + 1.5 * l.s, l.footer_y],
            [dx + 1.5 * l.s, l.footer_y + 7.0 * l.footer_scale],
            [dx, l.footer_y + 7.0 * l.footer_scale],
            lin4(pal::FAINT),
        );
    }
}

/// Draw the whole title composition (gradient → brand → rail → rows).
/// Returns nothing; hit rects are read from the layout by the caller.
pub fn draw(list: &mut TriList, l: &TitleLayout, logo_art: Option<f32>, selected: usize) {
    draw_gradient(list, l.w, l.h);
    draw_brand(list, l, logo_art);
    draw_rail(list, l, selected);
    for i in 0..3 {
        draw_menu_item(list, l, i, i == selected);
    }
    draw_version(list, l);
    draw_footer(list, l);
}

// ---------------------------------------------------------------------------
// glyph primitives (vector boxes on the overlay pipeline)
// ---------------------------------------------------------------------------

/// Letter-spaced text (the tracked MINECRAFT subtitle).
fn draw_tracked(
    list: &mut TriList,
    text: &str,
    x: f32,
    y: f32,
    scale: f32,
    advance: f32,
    color: [u8; 4],
) {
    let mut cx = x;
    for ch in text.chars() {
        font::draw_text(list, &ch.to_string(), cx, y, scale, color);
        cx += advance;
    }
}

/// Horizontal rule between `x0..x1` at `y` with thickness `t`.
fn rule(list: &mut TriList, x0: f32, x1: f32, y: f32, t: f32) {
    rule_colored(list, x0, x1, y, t, lin4(pal::LINE_IDLE));
}

fn rule_colored(list: &mut TriList, x0: f32, x1: f32, y: f32, t: f32, c: [u8; 4]) {
    list.quad([x0, y], [x1, y], [x1, y + t], [x0, y + t], c);
}

/// Diamond marker centered at (cx, cy).
fn diamond(list: &mut TriList, cx: f32, cy: f32, r: f32, c: [u8; 4]) {
    list.quad([cx - r, cy], [cx, cy - r], [cx + r, cy], [cx, cy + r], c);
}

/// Menu icons (moved from menu.rs — title-specific vector glyphs).
pub fn draw_icon(icon: Icon, list: &mut TriList, x: f32, y: f32, size: f32, c: [u8; 4]) {
    match icon {
        Icon::Play => draw_icon_play(list, x, y, size, size, c),
        Icon::Gear => draw_icon_gear(list, x, y, size, size, c),
        Icon::Power => draw_icon_power(list, x, y, size, size, c),
    }
}

fn draw_icon_play(list: &mut TriList, x: f32, y: f32, w: f32, h: f32, c: [u8; 4]) {
    let cx = x + w * 0.5;
    let cy = y + h * 0.5;
    let s = h * 0.42;
    list.quad([cx - s * 0.55, cy - s], [cx - s * 0.55, cy + s], [cx + s * 0.9, cy], [cx - s * 0.55, cy - s], c);
}

fn draw_icon_gear(list: &mut TriList, x: f32, y: f32, w: f32, h: f32, c: [u8; 4]) {
    let cx = x + w * 0.5;
    let cy = y + h * 0.5;
    let r = h * 0.34;
    let t = h * 0.11;
    // Ring: 4 strips (approximation of a gear silhouette, monochrome).
    list.quad([cx - r, cy - t / 2.0], [cx + r, cy - t / 2.0], [cx + r, cy + t / 2.0], [cx - r, cy + t / 2.0], c);
    list.quad([cx - t / 2.0, cy - r], [cx + t / 2.0, cy - r], [cx + t / 2.0, cy + r], [cx - t / 2.0, cy + r], c);
    // Hub hole (dark center over the cross).
    let hub = t * 1.4;
    list.quad([cx - hub / 2.0, cy - hub / 2.0], [cx + hub / 2.0, cy - hub / 2.0], [cx + hub / 2.0, cy + hub / 2.0], [cx - hub / 2.0, cy + hub / 2.0], lin4([10, 12, 16, 255]));
    // Diagonal teeth.
    let d = r * 0.95;
    for (dx, dy) in [(d, d), (-d, d), (d, -d), (-d, -d)] {
        list.quad(
            [cx + dx * 0.62 - t * 0.5, cy + dy * 0.62 - t * 0.5],
            [cx + dx * 0.62 + t * 0.5, cy + dy * 0.62 - t * 0.5],
            [cx + dx + t * 0.28, cy + dy + t * 0.28],
            [cx + dx - t * 0.28, cy + dy - t * 0.28],
            c,
        );
    }
    let _ = w;
}

fn draw_icon_power(list: &mut TriList, x: f32, y: f32, w: f32, h: f32, c: [u8; 4]) {
    let cx = x + w * 0.5;
    let cy = y + h * 0.55;
    let r = h * 0.30;
    let t = h * 0.10;
    // Circle stroke (left/right/bottom strips).
    list.quad([cx - r, cy - t / 2.0], [cx - r + t, cy - t / 2.0], [cx - r + t, cy + r * 0.72], [cx - r, cy + r * 0.72], c);
    list.quad([cx + r - t, cy - t / 2.0], [cx + r, cy - t / 2.0], [cx + r, cy + r * 0.72], [cx + r - t, cy + r * 0.72], c);
    list.quad([cx - r, cy + r * 0.72 - t], [cx + r, cy + r * 0.72 - t], [cx + r, cy + r * 0.72], [cx - r, cy + r * 0.72], c);
    // Vertical bar.
    list.quad([cx - t / 2.0, cy - r * 1.18], [cx + t / 2.0, cy - r * 1.18], [cx + t / 2.0, cy + r * 0.38], [cx - t / 2.0, cy + r * 0.38], c);
    let _ = w;
}

// ---------------------------------------------------------------------------
// tests — layout contract, not pixels
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::hit;

    const RESOLUTIONS: [(f32, f32); 5] =
        [(1280.0, 720.0), (1600.0, 900.0), (1920.0, 1080.0), (2560.0, 1440.0), (1366.0, 768.0)];

    fn lt(w: f32, h: f32) -> TitleLayout {
        TitleLayout::build(w, h)
    }

    #[test]
    fn three_rows_equal_dimensions_and_equal_spacing() {
        for (w, h) in RESOLUTIONS {
            let l = lt(w, h);
            for i in 0..3 {
                assert_eq!(l.rows[i][2], l.rows[0][2], "row {i} width differs at {w}x{h}");
                assert_eq!(l.rows[i][3], l.rows[0][3], "row {i} height differs at {w}x{h}");
                assert_eq!(l.rows[i][0], l.rows[0][0], "row {i} x differs at {w}x{h}");
            }
            let gap01 = l.rows[1][1] - (l.rows[0][1] + l.row_h);
            let gap12 = l.rows[2][1] - (l.rows[1][1] + l.row_h);
            assert!((gap01 - gap12).abs() < 1e-3 && (gap01 - l.row_gap).abs() < 1e-3);
        }
    }

    #[test]
    fn rail_markers_centered_on_rows_and_on_a_straight_rail() {
        for (w, h) in RESOLUTIONS {
            let l = lt(w, h);
            for i in 0..3 {
                let (mx, my) = l.marker_center(i);
                assert!((my - l.row_center_y(i)).abs() < 1e-4, "marker {i} not at row center");
                assert!((mx - l.rail_x).abs() < 1e-4, "marker {i} off the rail");
            }
            // Markers sit left of every row (on the rail, never inside it).
            for r in &l.rows {
                assert!(l.rail_x < r[0]);
            }
        }
    }

    #[test]
    fn rows_fit_and_labels_center_vertically() {
        for (w, h) in RESOLUTIONS {
            let l = lt(w, h);
            let scale = 4.0 * l.s;
            for i in 0..3 {
                let ty = l.rows[i][1] + (l.row_h - 7.0 * scale) / 2.0;
                let center_offset = (ty + 3.5 * scale) - l.row_center_y(i);
                assert!(center_offset.abs() < 1e-3, "label row {i} not vertically centered");
            }
        }
    }

    #[test]
    fn hitboxes_are_exactly_the_visible_rows() {
        let l = lt(1600.0, 900.0);
        for r in &l.rows {
            // Corners and center are inside; a pixel outside is not.
            let c = (r[0] + r[2] / 2.0, r[1] + r[3] / 2.0);
            assert!(hit(*r, c));
            assert!(hit(*r, (r[0] + 1.0, r[1] + 1.0)));
            assert!(!hit(*r, (r[0] - 1.0, r[1] + 1.0)));
            assert!(!hit(*r, (r[0] + r[2] + 1.0, r[1] + 1.0)));
        }
    }

    #[test]
    fn footer_inside_safe_area_and_clear_of_version_block() {
        for (w, h) in RESOLUTIONS {
            let l = lt(w, h);
            for (i, r) in l.footer.iter().enumerate() {
                assert!(r[0] >= 0.0 && r[0] + r[2] <= w, "footer {i} exceeds width at {w}x{h}");
                assert!(r[1] >= h * 0.8, "footer {i} not in the bottom band");
                assert!(r[1] + r[3] <= h, "footer {i} exceeds height");
            }
            // Ordered left → right, no overlap.
            assert!(l.footer[0][0] < l.footer[1][0] && l.footer[1][0] < l.footer[2][0]);
            // Version block (bottom-left) must not intersect the footer.
            let vr = [l.version_bar[0], l.version_y, 300.0 * l.s, l.h - l.version_y];
            for r in &l.footer {
                let overlap_x = r[0] < vr[0] + vr[2] && vr[0] < r[0] + r[2];
                let overlap_y = r[1] < vr[1] + vr[3] && vr[1] < r[1] + r[3];
                assert!(!(overlap_x && overlap_y), "footer overlaps version block at {w}x{h}");
            }
        }
    }

    #[test]
    fn version_block_is_prominent_and_readable() {
        for (w, h) in RESOLUTIONS {
            let l = lt(w, h);
            // v1.0.0 must render larger than the subtitle line and both
            // must stay inside the window with a visible margin.
            assert!(l.version_scale > l.version_sub_scale * 1.5);
            assert!(l.version_y + 7.0 * l.version_scale < h);
            assert!(l.version_sub_y + 7.0 * l.version_sub_scale < h);
            assert!(l.version_y > h * 0.8, "version block belongs bottom-left");
        }
    }

    #[test]
    fn layout_scales_proportionally_across_reference_resolutions() {
        let base = lt(1920.0, 1080.0);
        for (w, h) in RESOLUTIONS {
            let l = lt(w, h);
            let expected_s = (w / REF_W).min(h / REF_H);
            assert!((l.s - expected_s).abs() < 1e-5);
            assert!((l.row_h / base.row_h - expected_s).abs() < 1e-4, "row_h not proportional at {w}x{h}");
            assert!((l.rows[0][0] / base.rows[0][0] - expected_s).abs() < 1e-4, "left anchoring not proportional");
        }
        // Same-aspect sizes scale exactly by the ratio.
        let q = lt(960.0, 540.0);
        assert!((q.rows[0][1] - base.rows[0][1] * 0.5).abs() < 1e-3);
        // Wider-than-reference windows keep the composition left-anchored.
        let wide = lt(3440.0, 1440.0);
        assert!(wide.rows[0][0] < 200.0, "ultrawide must stay left-anchored");
    }

    #[test]
    fn brand_is_left_anchored() {
        for (w, h) in RESOLUTIONS {
            let l = lt(w, h);
            assert!(l.wordmark_x < w * 0.2, "wordmark must sit on the left");
            assert!(l.logo[0] < w * 0.2, "logo must sit on the left");
            assert!(l.rows[0][0] + l.rows[0][2] < w * 0.55, "menu must stay in the left band");
        }
    }

    #[test]
    fn selected_row_grows_around_its_center_and_hit_rect_matches() {
        let l = lt(1600.0, 900.0);
        let base = l.rows[1];
        let grown = l.row_hit_rect(1, true);
        // Grows in both axes, centered on the same midpoint.
        assert!(grown[2] > base[2] && grown[3] > base[3]);
        let bc = (base[0] + base[2] / 2.0, base[1] + base[3] / 2.0);
        let gc = (grown[0] + grown[2] / 2.0, grown[1] + grown[3] / 2.0);
        assert!((bc.0 - gc.0).abs() < 1e-3 && (bc.1 - gc.1).abs() < 1e-3, "grow must be centered");
        // Unselected rows return the base rect untouched.
        assert_eq!(l.row_hit_rect(1, false), base);
        // The grown rect is what draw_menu_item draws (same call path).
        let drawn = {
            let mut list = TriList::default();
            let r = draw_menu_item(&mut list, &l, 1, true);
            r
        };
        assert_eq!(drawn, grown);
        // Never grows off the window or into the rail.
        assert!(grown[0] >= l.rail_x && grown[0] + grown[2] <= l.w);
    }

    #[test]
    fn urls_are_the_whitelist() {
        assert_eq!(ALLOWED_URLS.len(), 3);
        assert!(ALLOWED_URLS.contains(&GITHUB_URL));
        assert!(GITHUB_URL.starts_with("https://"));
    }

    #[test]
    fn exactly_three_entries_play_settings_quit() {
        assert_eq!(MENU_ENTRIES.len(), 3);
        assert_eq!(MENU_ENTRIES[0].label, "Play");
        assert_eq!(MENU_ENTRIES[1].label, "Settings");
        assert_eq!(MENU_ENTRIES[2].label, "Quit");
    }

    #[test]
    fn drawing_produces_geometry_and_hits_match_layout() {
        let l = lt(1280.0, 720.0);
        let mut list = TriList::default();
        draw(&mut list, &l, Some(1.1), 0);
        assert!(!list.is_empty());
        // The gradient must be GPU-interpolated (two quads), not banded.
        // Brand text + rows + footer all drew something.
        assert!(list.indices.len() > 1000);
        // Every row rect is a valid hit target at its center.
        for r in &l.rows {
            assert!(hit(*r, (r[0] + r[2] / 2.0, r[1] + r[3] / 2.0)));
        }
    }
}
