//! Menu widget primitives: layout + draw lists for the title/menus.
//!
//! Everything is authored in UI pixels with the same overlay pipeline used
//! by the HUD (screen-space tri-lists + the microfont), so menus need no
//! new GPU plumbing. Every widget is a plain rect the app can hit-test —
//! the module is pure data + drawing, fully testable without a window.

use crate::font;
use crate::overlay::TriList;

/// Linearize an RGBA color for the renderer's sRGB targets: straight bytes
/// in the overlay vertex are treated as linear on write, so authored UI
/// colors must be sRGB-decoded first to land as designed.
pub fn lin4(c: [u8; 4]) -> [u8; 4] {
    crate::menu::lin_bytes(c)
}

/// A screen rect (x, y, w, h) in UI pixels.
pub type Rect = [f32; 4];

/// A widget's visual state for one frame (drives its palette).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hover {
    /// Pointer is over the widget and it can take a click.
    Hovered,
    Idle,
    /// Non-interactive/disabled (e.g. multiplayer entry without a selection).
    Disabled,
}

/// Style constants shared by all menus (vanilla-ish proportions).
pub const BUTTON_W: f32 = 400.0;
pub const BUTTON_H: f32 = 40.0;
pub const BUTTON_GAP: f32 = 8.0;

/// Center point of a rect (for synthetic clicks in tests).
pub fn center(r: Rect) -> (f32, f32) {
    (r[0] + r[2] / 2.0, r[1] + r[3] / 2.0)
}

/// Is `p` inside `r`?
pub fn hit(r: Rect, p: (f32, f32)) -> bool {
    p.0 >= r[0] && p.0 < r[0] + r[2] && p.1 >= r[1] && p.1 < r[1] + r[3]
}

/// Draw one menu button. Returns its rect (for hit-testing).
pub fn button(
    list: &mut TriList,
    cx: f32,
    y: f32,
    w: f32,
    label: &str,
    state: Hover,
) -> Rect {
    let x = cx - w / 2.0;
    let r = [x, y, w, BUTTON_H];
    let (face, edge, text) = match state {
        Hover::Hovered => (lin4([70, 70, 78, 235]), lin4([190, 195, 210, 255]), lin4([255, 240, 160, 255])),
        Hover::Idle => (lin4([46, 46, 52, 225]), lin4([110, 112, 124, 255]), lin4([235, 235, 235, 255])),
        Hover::Disabled => (lin4([34, 34, 38, 190]), lin4([70, 70, 78, 160]), lin4([120, 120, 120, 190])),
    };
    // Face + border strips (crisp 2px, like the hotbar).
    list.quad([x, y], [x + w, y], [x + w, y + BUTTON_H], [x, y + BUTTON_H], face);
    const B: f32 = 2.0;
    list.quad([x, y], [x + w, y], [x + w, y + B], [x, y + B], edge);
    list.quad([x, y + BUTTON_H - B], [x + w, y + BUTTON_H - B], [x + w, y + BUTTON_H], [x, y + BUTTON_H], edge);
    list.quad([x, y], [x + B, y], [x + B, y + BUTTON_H], [x, y + BUTTON_H], edge);
    list.quad([x + w - B, y], [x + w, y], [x + w, y + BUTTON_H], [x + w - B, y + BUTTON_H], edge);
    // Centered label.
    font::draw_text_shadow(list, label, cx - font::text_width(label, 2.0) / 2.0, y + (BUTTON_H - 7.0 * 2.0) / 2.0, 2.0, text);
    r
}

/// Small variant (world rows, list entries).
pub fn small_button(
    list: &mut TriList,
    r: Rect,
    label: &str,
    state: Hover,
) {
    let [x, y, w, h] = r;
    let (face, edge, text) = match state {
        Hover::Hovered => (lin4([70, 70, 78, 235]), lin4([190, 195, 210, 255]), lin4([255, 240, 160, 255])),
        Hover::Idle => (lin4([46, 46, 52, 225]), lin4([110, 112, 124, 255]), lin4([235, 235, 235, 255])),
        Hover::Disabled => (lin4([34, 34, 38, 190]), lin4([70, 70, 78, 160]), lin4([120, 120, 120, 190])),
    };
    list.quad([x, y], [x + w, y], [x + w, y + h], [x, y + h], face);
    list.quad([x, y], [x + w, y], [x + w, y + 2.0], [x, y + 2.0], edge);
    // Left-aligned label, vertically centered.
    font::draw_text_shadow(list, label, x + 10.0, y + (h - 7.0 * 1.5) / 2.0, 1.5, text);
}

/// A single-line text field. `cursor` is the caret position (in chars);
/// draws a blinking caret only when `focused` (the caller gates blinking).
/// Returns the field rect.
#[allow(clippy::too_many_arguments)] // (cx, y, w, label, value, cursor, focused) + list
pub fn text_field(
    list: &mut TriList,
    cx: f32,
    y: f32,
    w: f32,
    label: &str,
    value: &str,
    cursor: usize,
    focused: bool,
) -> Rect {
    const H: f32 = 34.0;
    let x = cx - w / 2.0;
    let r = [x, y, w, H];
    let edge = if focused { [220, 225, 240, 255] } else { [110, 112, 124, 255] };
    list.quad([x, y], [x + w, y], [x + w, y + H], [x, y + H], [22, 22, 26, 235]);
    const B: f32 = 2.0;
    list.quad([x, y], [x + w, y], [x + w, y + B], [x, y + B], edge);
    list.quad([x, y + H - B], [x + w, y + H - B], [x + w, y + H], [x, y + H], edge);
    list.quad([x, y], [x + B, y], [x + B, y + H], [x, y + H], edge);
    list.quad([x + w - B, y], [x + w, y], [x + w, y + H], [x + w - B, y + H], edge);
    // Small caption above the box.
    font::draw_text_shadow(list, label, x + 2.0, y - 12.0, 1.5, [200, 200, 210, 230]);
    // Value (left-aligned inside).
    let text_scale = 1.5;
    font::draw_text(list, value, x + 8.0, y + (H - 7.0 * text_scale) / 2.0, text_scale, [240, 240, 240, 255]);
    // Caret: a 2px bar after the cursor-th character when focused.
    if focused {
        let before = &value[..char_byte_index(value, cursor)];
        let tx = x + 8.0 + font::text_width(before, text_scale);
        let ty = y + (H - 7.0 * text_scale) / 2.0;
        list.quad([tx, ty], [tx + 2.0, ty], [tx + 2.0, ty + 7.0 * text_scale], [tx, ty + 7.0 * text_scale], [240, 240, 240, 230]);
    }
    r
}

/// A horizontal list row with left label and right value (+ arrows when
/// adjustable). Returns the row rect and the two arrow rects (None when
/// not adjustable).
pub fn slider_row(
    list: &mut TriList,
    r: Rect,
    label: &str,
    value: &str,
    arrows: bool,
    state: Hover,
) -> (Rect, Option<Rect>, Option<Rect>) {
    let [x, y, w, h] = r;
    let edge = match state {
        Hover::Hovered => lin4([190, 195, 210, 255]),
        _ => [110, 112, 124, 255],
    };
    list.quad([x, y], [x + w, y], [x + w, y + h], [x, y + h], [30, 30, 36, 225]);
    list.quad([x, y], [x + w, y], [x + w, y + 2.0], [x, y + 2.0], edge);
    font::draw_text_shadow(list, label, x + 10.0, y + (h - 7.0 * 1.5) / 2.0, 1.5, [235, 235, 235, 255]);
    let vw = font::text_width(value, 1.5);
    let arrow_w = 22.0f32;
    let mut left = None;
    let mut right = None;
    if arrows {
        font::draw_text(list, "<", x + w - arrow_w * 2.0 - vw - 12.0, y + (h - 7.0 * 1.5) / 2.0, 1.5, [200, 205, 220, 255]);
        font::draw_text(list, ">", x + w - arrow_w - 6.0, y + (h - 7.0 * 1.5) / 2.0, 1.5, [200, 205, 220, 255]);
        left = Some([x + w - arrow_w * 2.0 - vw - 12.0, y, arrow_w, h]);
        right = Some([x + w - arrow_w - 6.0, y, arrow_w, h]);
    } else {
        font::draw_text_shadow(list, value, x + w - vw - 10.0, y + (h - 7.0 * 1.5) / 2.0, 1.5, [235, 235, 235, 255]);
    }
    (r, left, right)
}

/// Vertically-centered stack of `n` full-width buttons starting at `y`.
pub fn stack_y(n: usize, h: f32) -> f32 {
    h / 2.0 - (n as f32 * (BUTTON_H + BUTTON_GAP)) / 2.0
}

/// Byte index of character `idx` in `s` (clamped).
pub fn char_byte_index(s: &str, idx: usize) -> usize {
    s.char_indices()
        .nth(idx)
        .map(|(b, _)| b)
        .unwrap_or(s.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_testing_is_exact() {
        let r = [10.0, 20.0, 30.0, 40.0];
        assert!(hit(r, (10.0, 20.0)), "top-left inclusive");
        assert!(hit(r, (39.9, 59.9)));
        assert!(!hit(r, (40.0, 30.0)), "right edge exclusive");
        assert!(!hit(r, (20.0, 60.0)), "bottom edge exclusive");
        assert!(!hit(r, (5.0, 5.0)));
    }

    #[test]
    fn button_draws_and_returns_rect() {
        let mut list = TriList::default();
        let r = button(&mut list, 640.0, 100.0, 400.0, "SINGLEPLAYER", Hover::Idle);
        assert_eq!(r, [440.0, 100.0, 400.0, BUTTON_H]);
        assert!(!list.is_empty());
        assert!(hit(r, (640.0, 120.0)));
    }

    #[test]
    fn text_field_caret_moves_with_cursor() {
        let caret_x = |l: &TriList| {
            // The caret bar is uniquely colored [240,240,240,230]; take its
            // leftmost vertex x.
            l.vertices
                .iter()
                .filter(|v| v.color == [240u8, 240, 240, 230])
                .map(|v| v.pos[0])
                .fold(f32::MAX, f32::min)
        };
        let mut a = TriList::default();
        text_field(&mut a, 640.0, 100.0, 400.0, "NAME", "ab", 0, true);
        let mut b = TriList::default();
        text_field(&mut b, 640.0, 100.0, 400.0, "NAME", "ab", 2, true);
        let (xa, xb) = (caret_x(&a), caret_x(&b));
        assert!(xb > xa, "caret advances with the cursor ({xa} → {xb})");
        // Unfocused: no caret at all.
        let mut c = TriList::default();
        text_field(&mut c, 640.0, 100.0, 400.0, "NAME", "ab", 2, false);
        assert!(caret_x(&c) == f32::MAX, "no caret when unfocused");
    }

    #[test]
    fn char_index_handles_multibyte() {
        assert_eq!(char_byte_index("héllo", 2), 3);
        assert_eq!(char_byte_index("héllo", 99), 6, "clamped to len");
        assert_eq!(char_byte_index("", 0), 0);
    }

    #[test]
    fn slider_row_reports_arrows_only_when_adjustable() {
        let mut list = TriList::default();
        let (r, l, rt) = slider_row(&mut list, [0.0, 0.0, 400.0, 30.0], "QUALITY", "HIGH", true, Hover::Idle);
        assert!(l.is_some() && rt.is_some());
        assert!(hit(r, (5.0, 5.0)));
        let (_, l2, r2) = slider_row(&mut list, [0.0, 0.0, 400.0, 30.0], "NAME", "Steve", false, Hover::Idle);
        assert!(l2.is_none() && r2.is_none());
    }
}
