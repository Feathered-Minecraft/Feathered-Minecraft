//! A tiny 5×7 bitmap font for HUD text (F3 screen, hotbar labels, toasts).
//!
//! Classic LCD-style glyphs, 7 rows of 5 bits each (bit 4 = leftmost
//! column). Lowercase input renders as uppercase; unknown chars fall back
//! to a hollow box. Drawn as colored pixel quads into the overlay tri-list
//! — a full F3 screen is a few hundred quads, far below any budget.

use crate::overlay::TriList;

/// One glyph: 7 rows, 5 used bits per row (leftmost = bit 4).
const FONT: &[(char, [u8; 7])] = &[
    (' ', [0b00000; 7]),
    (
        'A',
        [
            0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
    ),
    (
        'B',
        [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110,
        ],
    ),
    (
        'C',
        [
            0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110,
        ],
    ),
    (
        'D',
        [
            0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110,
        ],
    ),
    (
        'E',
        [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111,
        ],
    ),
    (
        'F',
        [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
    ),
    (
        'G',
        [
            0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111,
        ],
    ),
    (
        'H',
        [
            0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
    ),
    (
        'I',
        [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b11111,
        ],
    ),
    (
        'J',
        [
            0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100,
        ],
    ),
    (
        'K',
        [
            0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001,
        ],
    ),
    (
        'L',
        [
            0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
        ],
    ),
    (
        'M',
        [
            0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001,
        ],
    ),
    (
        'N',
        [
            0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001,
        ],
    ),
    (
        'O',
        [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
    ),
    (
        'P',
        [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
    ),
    (
        'Q',
        [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101,
        ],
    ),
    (
        'R',
        [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
        ],
    ),
    (
        'S',
        [
            0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
    ),
    (
        'T',
        [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
    ),
    (
        'U',
        [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
    ),
    (
        'V',
        [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100,
        ],
    ),
    (
        'W',
        [
            0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010,
        ],
    ),
    (
        'X',
        [
            0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001,
        ],
    ),
    (
        'Y',
        [
            0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
    ),
    (
        'Z',
        [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111,
        ],
    ),
    (
        '0',
        [
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ],
    ),
    (
        '1',
        [
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
    ),
    (
        '2',
        [
            0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111,
        ],
    ),
    (
        '3',
        [
            0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
    ),
    (
        '4',
        [
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ],
    ),
    (
        '5',
        [
            0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
        ],
    ),
    (
        '6',
        [
            0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
        ],
    ),
    (
        '7',
        [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ],
    ),
    (
        '8',
        [
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ],
    ),
    (
        '9',
        [
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100,
        ],
    ),
    ('.', [0, 0, 0, 0, 0, 0b00100, 0b00100]),
    (',', [0, 0, 0, 0, 0, 0b00100, 0b01000]),
    (':', [0, 0b00100, 0b00100, 0, 0b00100, 0b00100, 0]),
    (
        '/',
        [
            0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000,
        ],
    ),
    (
        '(',
        [
            0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010,
        ],
    ),
    (
        ')',
        [
            0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000,
        ],
    ),
    ('-', [0, 0, 0, 0b01110, 0, 0, 0]),
    ('+', [0, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0]),
    (
        '%',
        [
            0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011,
        ],
    ),
    (
        '!',
        [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0, 0b00100],
    ),
    (
        '?',
        [0b01110, 0b10001, 0b00001, 0b00110, 0b00100, 0, 0b00100],
    ),
    (
        '>',
        [
            0b01000, 0b00100, 0b00010, 0b00001, 0b00010, 0b00100, 0b01000,
        ],
    ),
    (
        '<',
        [
            0b00010, 0b00100, 0b01000, 0b10000, 0b01000, 0b00100, 0b00010,
        ],
    ),
    ('=', [0, 0, 0b11111, 0, 0b11111, 0, 0]),
    (
        '*',
        [
            0b00000, 0b10101, 0b01110, 0b11111, 0b01110, 0b10101, 0b00000,
        ],
    ),
    (
        '#',
        [
            0b01010, 0b01010, 0b11111, 0b01010, 0b11111, 0b01010, 0b01010,
        ],
    ),
    ('_', [0, 0, 0, 0, 0, 0, 0b11111]),
    (
        '[',
        [
            0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110,
        ],
    ),
    (
        ']',
        [
            0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110,
        ],
    ),
    ('°', [0b01100, 0b10010, 0b10010, 0b01100, 0, 0, 0]),
];

const UNKNOWN: [u8; 7] = [
    0b01110, 0b10001, 0b10101, 0b10101, 0b10101, 0b10001, 0b01110,
];

fn glyph(ch: char) -> &'static [u8; 7] {
    let up = ch.to_ascii_uppercase();
    FONT.iter()
        .find(|(c, _)| *c == up)
        .map(|(_, rows)| rows)
        .unwrap_or(&UNKNOWN)
}

/// Glyph cell size in units (5 wide + 1 spacing; 7 tall).
pub const GLYPH_W: f32 = 6.0;
pub const GLYPH_H: f32 = 7.0;

/// Draw one string. Every "on" pixel becomes a `scale×scale` quad, merged
/// horizontally into runs to cut the quad count roughly 4× on text rows.
pub fn draw_text(list: &mut TriList, text: &str, x: f32, y: f32, scale: f32, color: [u8; 4]) {
    let mut cx = x;
    for ch in text.chars() {
        let rows = glyph(ch);
        for (ry, &row) in rows.iter().enumerate() {
            let py = y + ry as f32 * scale;
            // Horizontal run merging within the row.
            let mut run_start: Option<u32> = None;
            for bx in 0..=5u32 {
                let on = bx < 5 && (row >> (4 - bx)) & 1 == 1;
                if on && run_start.is_none() {
                    run_start = Some(bx);
                }
                if !on && run_start.is_some() {
                    let s = run_start.take().unwrap();
                    let px = cx + s as f32 * scale;
                    let pw = (bx - s) as f32 * scale;
                    list.quad(
                        [px, py],
                        [px + pw, py],
                        [px + pw, py + scale],
                        [px, py + scale],
                        color,
                    );
                }
            }
        }
        cx += GLYPH_W * scale;
    }
}

/// Text width in px (`n_chars * GLYPH_W - 1` spacing unit, scaled).
pub fn text_width(text: &str, scale: f32) -> f32 {
    (text.chars().count().max(1) as f32 * GLYPH_W - 1.0) * scale
}

/// Draw text with a 1px (× scale) black shadow for readability on any sky.
pub fn draw_text_shadow(
    list: &mut TriList,
    text: &str,
    x: f32,
    y: f32,
    scale: f32,
    color: [u8; 4],
) {
    draw_text(list, text, x + scale, y + scale, scale, [0, 0, 0, 160]);
    draw_text(list, text, x, y, scale, color);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_glyph_has_seven_rows_of_five_bits() {
        for (ch, rows) in FONT {
            assert_eq!(rows.len(), 7, "glyph {ch} must have 7 rows");
            for (i, &r) in rows.iter().enumerate() {
                assert!(r & 0b1110_0000 == 0, "glyph {ch} row {i} uses bits above 4");
            }
        }
    }

    #[test]
    fn known_and_unknown_chars_resolve() {
        assert_eq!(glyph('A')[0], 0b01110);
        assert_eq!(glyph('a')[0], 0b01110, "lowercase folds to uppercase");
        assert_eq!(glyph('~'), &UNKNOWN, "unknown chars get the hollow box");
        assert_eq!(glyph('3')[0], 0b11110);
    }

    #[test]
    fn text_draws_quads_and_reports_width() {
        let mut list = TriList::default();
        draw_text(&mut list, "HI", 0.0, 0.0, 1.0, [255, 255, 255, 255]);
        assert!(!list.is_empty());
        // H row 0: columns 0,4 on → 2 runs; I row 0: 1 run → 3 quads in row 0.
        // Total quad count equals indices/6.
        let quads = list.indices.len() / 6;
        assert!(quads > 4, "both glyphs must draw (quads={quads})");
        assert_eq!(text_width("HI", 1.0), 11.0);
        assert_eq!(text_width("HI", 2.0), 22.0);
        // A space advances without drawing: 'A A' draws exactly as many
        // quads as 'AA' (both are two A glyphs).
        let mut a = TriList::default();
        draw_text(&mut a, "A A", 0.0, 0.0, 1.0, [255, 255, 255, 255]);
        let mut b = TriList::default();
        draw_text(&mut b, "AA", 0.0, 0.0, 1.0, [255, 255, 255, 255]);
        assert_eq!(a.indices.len(), b.indices.len(), "space draws nothing");
        let mut one = TriList::default();
        draw_text(&mut one, "A", 0.0, 0.0, 1.0, [255, 255, 255, 255]);
        assert_eq!(a.indices.len(), one.indices.len() * 2, "two As = 2× one A");
    }

    #[test]
    fn shadow_draws_double_quads() {
        let mut plain = TriList::default();
        draw_text(&mut plain, "FPS", 0.0, 0.0, 1.0, [255, 255, 255, 255]);
        let mut shadowed = TriList::default();
        draw_text_shadow(&mut shadowed, "FPS", 0.0, 0.0, 1.0, [255, 255, 255, 255]);
        assert_eq!(shadowed.indices.len(), plain.indices.len() * 2);
    }
}
