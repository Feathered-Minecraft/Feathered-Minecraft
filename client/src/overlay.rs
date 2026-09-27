//! In-game HUD and world-space overlays.
//!
//! Three drawing systems share one triangle-list vertex format
//! (`HudVertex`, 32 bytes) but render at different stages:
//!
//! * **Screen-space** (crosshair, hotbar, text, break progress): positioned
//!   in UI pixels, projected by `ScreenUniform` (surface w/h), drawn with a
//!   triangle-list pipeline into the final surface. No depth test, alpha
//!   blended, drawn last.
//! * **World-space outlines** (targeted block, placement preview): 12 edges
//!   of a unit cube expanded by a small epsilon, positioned in world blocks,
//!   drawn with the same vertex format but the scene's view-projection
//!   uniform, depth-tested against the world so outlines hide behind hills.
//!
//! All geometry is CPU-built each frame into plain `Vec`s (hundreds of
//! quads at most — far cheaper than an instancing/graphic-UI framework at
//! this scale) and uploaded per frame. The shaders are engine/renderer
//! `overlay.wgsl`; this module is pure mesh/uniform authoring plus the
//! draw-list structs the renderer consumes.

/// Vertex for overlay triangles: position is either UI pixels (screen
/// pass) or world blocks (outline pass) — the two pipelines differ only in
/// the uniform they bind. Color is straight RGBA8; `px` carries UVs into
/// the overlay art texture on the screen pass (x < 0 = flat, untinted by
/// any texture) and is unused on the outline pass. `uv` is reserved
/// padding so screen and outline vertices share one format. 32 bytes
/// (12 pos + 4 color + 8 px + 8 uv).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HudVertex {
    pub pos: [f32; 3], // z = 0 (screen) / unused (outline)
    pub color: [u8; 4],
    pub px: [f32; 2],  // screen pass: UV (x < 0 = flat quad)
    pub uv: [f32; 2],  // reserved
}

const _: () = assert!(std::mem::size_of::<HudVertex>() == 32);

/// Projection uniforms for the screen-space pass (UI pixels → NDC).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScreenUniform {
    pub size: [f32; 2],
    _pad: [f32; 2],
}

/// View-projection for the world-space outline pass (mat4 as storage order).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OutlineUniform {
    pub view_proj: [[f32; 4]; 4],
}

/// One triangle list ready for upload (screen or outline space — the
/// renderer knows which pipeline each list belongs to).
#[derive(Debug, Default)]
pub struct TriList {
    pub vertices: Vec<HudVertex>,
    pub indices: Vec<u32>,
}

/// The two lists one frame of HUD/overlay draws (client-side authoring).
#[derive(Debug, Default)]
pub struct HudLists {
    pub screen: TriList,
    pub world: TriList,
}

impl TriList {
    /// Push one 2D quad (a=bottom-left … d=top-left, CCW, screen space).
    pub fn quad(&mut self, a: [f32; 2], b: [f32; 2], c: [f32; 2], d: [f32; 2], color: [u8; 4]) {
        // a=bottom-left, b=bottom-right, c=top-right, d=top-left (CCW).
        // Flat quads carry the UV sentinel (x < 0) so the fragment shader
        // shades them by color alone (px was pixel padding pre-textures).
        let base = self.vertices.len() as u32;
        let z = 0.0;
        for p in [(a), (b), (c), (d)] {
            self.vertices.push(HudVertex {
                pos: [p[0], p[1], z],
                color,
                px: [-1.0, 0.0],
                uv: [0.0; 2],
            });
        }
        self.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    /// True when nothing has been authored into this list.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Push one texture-mapped quad sampling `tex` (full 0..1 UVs; the
    /// texture is set separately via the renderer's `set_logo_texture`):
    /// a=bottom-left … d=top-left, CCW, UI pixels. The tint color multiplies
    /// the sampled texels (straight alpha × straight alpha).
    pub fn textured_quad(
        &mut self,
        a: [f32; 2],
        b: [f32; 2],
        c: [f32; 2],
        d: [f32; 2],
        color: [u8; 4],
    ) {
        let base = self.vertices.len() as u32;
        let z = 0.0;
        let uvs = [
            [0.0, 1.0], // a: bottom-left (v flipped: UV 0 = top row)
            [1.0, 1.0], // b: bottom-right
            [1.0, 0.0], // c: top-right
            [0.0, 0.0], // d: top-left
        ];
        for (p, uv) in [(a, uvs[0]), (b, uvs[1]), (c, uvs[2]), (d, uvs[3])] {
            self.vertices.push(HudVertex {
                pos: [p[0], p[1], z],
                color,
                px: uv,
                uv: [0.0; 2],
            });
        }
        self.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

// ---------------------------------------------------------------------------
// draw-list authoring (pure functions — unit-testable without a GPU)
// ---------------------------------------------------------------------------

/// Build the crosshair: two thin centered rectangles with a 1px gap, plus a
/// subtle black outline behind for visibility on any background.
pub fn build_crosshair(list: &mut TriList, w: f32, h: f32) {
    let (cx, cy) = (w / 2.0, h / 2.0);
    let t = 2.0f32; // stroke thickness
    let arm = 9.0f32; // arm length from center
    let gap = 3.0f32;
    let color = [240u8, 240, 240, 230];
    let outline = [0u8, 0, 0, 120];
    // (offset from center along the arm axis, perpendicular half-extent)
    let bars = [
        // horizontal bar: extends ±arm in x, ±t/2 in y
        (cx - arm, cx + arm, cy - t / 2.0, cy + t / 2.0),
        // vertical bar
        (cx - t / 2.0, cx + t / 2.0, cy - arm, cy + arm),
    ];
    for (x0, x1, y0, y1) in bars {
        // 1px outline behind (expanded).
        list.quad(
            [x0 - 1.0, y0 - 1.0],
            [x1 + 1.0, y0 - 1.0],
            [x1 + 1.0, y1 + 1.0],
            [x0 - 1.0, y1 + 1.0],
            outline,
        );
        list.quad([x0, y0], [x1, y0], [x1, y1], [x0, y1], color);
    }
    let _ = gap; // reserved: dot-in-gap style variants
}

/// Build the hotbar: 9 slots with the selected one highlighted. Returns the
/// slot rects so glyph/text pass can center labels inside them.
pub fn build_hotbar(list: &mut TriList, w: f32, h: f32, selected: usize) -> Vec<[f32; 4]> {
    const SLOTS: usize = 9;
    const SLOT: f32 = 46.0;
    const BORDER: f32 = 2.0;
    let total = SLOTS as f32 * SLOT;
    let x0 = (w - total) / 2.0;
    let y0 = h - SLOT - 14.0;
    let mut rects = Vec::with_capacity(SLOTS);
    for i in 0..SLOTS {
        let sx = x0 + i as f32 * SLOT;
        let selected = i == selected;
        // Slot background (dark), then border (bright when selected).
        list.quad(
            [sx, y0],
            [sx + SLOT, y0],
            [sx + SLOT, y0 + SLOT],
            [sx, y0 + SLOT],
            [16, 16, 16, 165],
        );
        let bc = if selected {
            [250u8, 250, 250, 235]
        } else {
            [110u8, 110, 110, 200]
        };
        // 4 border strips.
        list.quad([sx, y0], [sx + SLOT, y0], [sx + SLOT, y0 + BORDER], [sx, y0 + BORDER], bc);
        list.quad(
            [sx, y0 + SLOT - BORDER],
            [sx + SLOT, y0 + SLOT - BORDER],
            [sx + SLOT, y0 + SLOT],
            [sx, y0 + SLOT],
            bc,
        );
        list.quad([sx, y0], [sx + BORDER, y0], [sx + BORDER, y0 + SLOT], [sx, y0 + SLOT], bc);
        list.quad(
            [sx + SLOT - BORDER, y0],
            [sx + SLOT, y0],
            [sx + SLOT, y0 + SLOT],
            [sx + SLOT - BORDER, y0 + SLOT],
            bc,
        );
        rects.push([sx, y0, sx + SLOT, y0 + SLOT]);
    }
    rects
}

/// Build a horizontal progress bar (block breaking). `progress` in 0..1;
/// hidden entirely when progress is 0 or negative.
pub fn build_progress(list: &mut TriList, w: f32, h: f32, progress: f32) {
    // Hidden entirely when not breaking (progress ≤ 0) or out of range.
    if progress <= 0.0 || progress > 1.0 {
        return;
    }
    let bw = 220.0f32;
    let bh = 8.0f32;
    let x0 = (w - bw) / 2.0;
    let y0 = h / 2.0 + 26.0; // just below the crosshair
    // Track.
    list.quad(
        [x0 - 1.0, y0 - 1.0],
        [x0 + bw + 1.0, y0 - 1.0],
        [x0 + bw + 1.0, y0 + bh + 1.0],
        [x0 - 1.0, y0 + bh + 1.0],
        [0, 0, 0, 140],
    );
    list.quad([x0, y0], [x0 + bw, y0], [x0 + bw, y0 + bh], [x0, y0 + bh], [30, 30, 30, 190]);
    // Fill.
    let fw = bw * progress.clamp(0.0, 1.0);
    let shade = (60.0 + 160.0 * progress) as u8;
    list.quad([x0, y0], [x0 + fw, y0], [x0 + fw, y0 + bh], [x0, y0 + bh], [shade, shade, shade, 220]);
}

/// A small positioned text run (already-layouted; see `layout_text`).
pub struct TextRun {
    pub glyphs: Vec<(char, [f32; 2])>, // (char, top-left px)
    pub color: [u8; 4],
}

/// Layout a string left-to-right with the 5×7 microfont metrics.
pub fn layout_text(s: &str, x: f32, y: f32, scale: f32) -> Vec<(char, [f32; 2])> {
    let mut out = Vec::new();
    let mut cx = x;
    for ch in s.chars() {
        out.push((ch, [cx, y]));
        // 6 units advance (5 glyph + 1 space), scaled.
        cx += 6.0 * scale;
    }
    out
}

/// Center a text run inside a slot rect (used for hotbar slot numbers).
pub fn centered_text(s: &str, rect: [f32; 4], scale: f32) -> Vec<(char, [f32; 2])> {
    let width = s.chars().count() as f32 * 6.0 * scale;
    let height = 7.0 * scale;
    let cx = (rect[0] + rect[2]) / 2.0;
    let cy = (rect[1] + rect[3]) / 2.0;
    layout_text(s, cx - width / 2.0, cy - height / 2.0, scale)
}

/// Build the world-space outline mesh for one block cell: the 12 edges of
/// the cube [x, x+1)³ expanded by `expand`. Every edge is drawn as a hollow
/// rectangular beam (4 thin quads) so the outline has real thickness and
/// reads correctly from any viewing angle while staying depth-tested
/// against the world.
pub fn build_block_outline(list: &mut TriList, x: i64, y: i64, z: i64, expand: f32, color: [u8; 4]) {
    let (x0, y0, z0) = (x as f32 - expand, y as f32 - expand, z as f32 - expand);
    let (x1, y1, z1) = (x as f32 + 1.0 + expand, y as f32 + 1.0 + expand, z as f32 + 1.0 + expand);
    let t = expand * 2.0; // beam half-thickness around the edge line

    // One axis-aligned beam from a to b: 4 quads forming a hollow tube of
    // half-thickness `t` around the segment. Exactly one axis differs.
    fn beam(
        list: &mut TriList,
        a: (f32, f32, f32),
        b: (f32, f32, f32),
        t: f32,
        color: [u8; 4],
    ) {
        let (ax, ay, az) = a;
        let (bx, by, bz) = b;
        if ax != bx {
            // Along X: cross-section in YZ.
            list.quad3([ax, ay - t, az - t], [bx, ay - t, az - t], [bx, ay - t, az + t], [ax, ay - t, az + t], color);
            list.quad3([ax, ay + t, az - t], [bx, ay + t, az - t], [bx, ay + t, az + t], [ax, ay + t, az + t], color);
            list.quad3([ax, ay - t, az - t], [bx, ay - t, az - t], [bx, ay + t, az - t], [ax, ay + t, az - t], color);
            list.quad3([ax, ay - t, az + t], [bx, ay - t, az + t], [bx, ay + t, az + t], [ax, ay + t, az + t], color);
        } else if ay != by {
            // Along Y: cross-section in XZ.
            list.quad3([ax - t, ay, az - t], [bx + t, ay, az - t], [bx + t, by, az - t], [ax - t, by, az - t], color);
            list.quad3([ax - t, ay, az + t], [bx + t, ay, az + t], [bx + t, by, az + t], [ax - t, by, az + t], color);
            list.quad3([ax - t, ay, az - t], [ax - t, ay, az + t], [ax - t, by, az + t], [ax - t, by, az - t], color);
            list.quad3([bx + t, ay, az - t], [bx + t, ay, az + t], [bx + t, by, az + t], [bx + t, by, az - t], color);
        } else {
            // Along Z: cross-section in XY.
            list.quad3([ax - t, ay - t, az], [bx + t, ay - t, az], [bx + t, ay - t, bz], [ax - t, ay - t, bz], color);
            list.quad3([ax - t, ay + t, az], [bx + t, ay + t, az], [bx + t, ay + t, bz], [ax - t, ay + t, bz], color);
            list.quad3([ax - t, ay - t, az], [ax - t, ay + t, az], [ax - t, ay + t, bz], [ax - t, ay - t, bz], color);
            list.quad3([bx + t, ay - t, az], [bx + t, ay + t, az], [bx + t, ay + t, bz], [bx + t, ay - t, bz], color);
        }
    }

    // 12 edges: bottom 4, top 4, verticals 4.
    beam(list, (x0, y0, z0), (x1, y0, z0), t, color);
    beam(list, (x1, y0, z0), (x1, y0, z1), t, color);
    beam(list, (x1, y0, z1), (x0, y0, z1), t, color);
    beam(list, (x0, y0, z1), (x0, y0, z0), t, color);
    beam(list, (x0, y1, z0), (x1, y1, z0), t, color);
    beam(list, (x1, y1, z0), (x1, y1, z1), t, color);
    beam(list, (x1, y1, z1), (x0, y1, z1), t, color);
    beam(list, (x0, y1, z1), (x0, y1, z0), t, color);
    beam(list, (x0, y0, z0), (x0, y1, z0), t, color);
    beam(list, (x1, y0, z0), (x1, y1, z0), t, color);
    beam(list, (x1, y0, z1), (x1, y1, z1), t, color);
    beam(list, (x0, y0, z1), (x0, y1, z1), t, color);
}

impl TriList {
    /// World-space quad (3D positions) — used by the outline builder and
    /// the menu's skyline panorama (flat-UV sentinel, color-shaded).
    pub fn quad3(
        &mut self,
        a: [f32; 3],
        b: [f32; 3],
        c: [f32; 3],
        d: [f32; 3],
        color: [u8; 4],
    ) {
        let base = self.vertices.len() as u32;
        for p in [a, b, c, d] {
            self.vertices.push(HudVertex {
                pos: p,
                color,
                px: [-1.0, 0.0], // outlines never sample
                uv: [0.0; 2],
            });
        }
        self.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

// ---------------------------------------------------------------------------
// break particles (CPU state, meshed every frame into the outline pass)
// ---------------------------------------------------------------------------

/// One break particle: a tiny textured cube corner flying with gravity.
/// Colors come from the broken block's average sprite color (client-side).
#[derive(Debug, Clone, Copy)]
pub struct Particle {
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub life: f32,
    pub ttl: f32,
    pub color: [u8; 4],
}

impl Particle {
    fn step(&mut self, dt: f32) {
        self.vel[1] -= 18.0 * dt; // gravity (blocks/s²) — snappier than the player's
        self.pos = [
            self.pos[0] + self.vel[0] * dt,
            self.pos[1] + self.vel[1] * dt,
            self.pos[2] + self.vel[2] * dt,
        ];
        self.life += dt;
    }
}

/// Deterministic burst: seed derived from the block position so tests can
/// assert the burst shape without RNG.
pub fn spawn_burst(out: &mut Vec<Particle>, x: i64, y: i64, z: i64, color: [u8; 4], seed: u64) {
    const COUNT: usize = 12;
    for i in 0..COUNT {
        let h = pos_hash(seed ^ (i as u64));
        let dir = [
            ((h & 0xFF) as f32 / 255.0) * 2.0 - 1.0,
            ((h >> 8) & 0xFF) as f32 / 255.0,
            ((h >> 16) as f32 / 255.0) * 2.0 - 1.0,
        ];
        let speed = 1.5 + ((h >> 24) & 0xFF) as f32 / 255.0 * 1.5;
        out.push(Particle {
            pos: [x as f32 + 0.5 + dir[0] * 0.3, y as f32 + 0.5, z as f32 + 0.5 + dir[2] * 0.3],
            vel: [dir[0] * speed, dir[1] * speed + 1.0, dir[2] * speed],
            life: 0.0,
            ttl: 0.55 + ((h >> 32) & 0xFF) as f32 / 255.0 * 0.25,
            color,
        });
    }
}

fn pos_hash(mut h: u64) -> u64 {
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 31;
    h
}

/// Advance all particles; drop expired.
pub fn step_particles(ps: &mut Vec<Particle>, dt: f32) {
    for p in ps.iter_mut() {
        p.step(dt);
    }
    ps.retain(|p| p.life < p.ttl);
}

/// Mesh particles as small camera-independent colored cubes (billboarding
/// is not worth a second pipeline; a 0.08 block cube reads fine).
pub fn build_particles(list: &mut TriList, ps: &[Particle]) {
    const S: f32 = 0.045; // half-size
    for p in ps {
        let [x, y, z] = p.pos;
        let fade = (1.0 - p.life / p.ttl).clamp(0.35, 1.0);
        let mut c = p.color;
        c[3] = (c[3] as f32 * fade) as u8;
        // 3 cross quads (X, Y, Z planes) per particle — 12 tris total,
        // reads as a small blocky chunk from any angle.
        list.quad3(
            [x - S, y - S, z - S],
            [x + S, y - S, z - S],
            [x + S, y + S, z - S],
            [x - S, y + S, z - S],
            c,
        );
        list.quad3(
            [x - S, y - S, z + S],
            [x + S, y - S, z + S],
            [x + S, y + S, z + S],
            [x - S, y + S, z + S],
            c,
        );
        list.quad3(
            [x - S, y - S, z - S],
            [x - S, y - S, z + S],
            [x - S, y + S, z + S],
            [x - S, y + S, z - S],
            c,
        );
        list.quad3(
            [x + S, y - S, z - S],
            [x + S, y - S, z + S],
            [x + S, y + S, z + S],
            [x + S, y + S, z - S],
            c,
        );
        list.quad3(
            [x - S, y + S, z - S],
            [x + S, y + S, z - S],
            [x + S, y + S, z + S],
            [x - S, y + S, z + S],
            c,
        );
        list.quad3(
            [x - S, y - S, z - S],
            [x + S, y - S, z - S],
            [x + S, y - S, z + S],
            [x - S, y - S, z + S],
            c,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crosshair_is_centered_and_bounded() {
        let mut list = TriList::default();
        build_crosshair(&mut list, 1280.0, 720.0);
        assert!(!list.is_empty());
        // All vertices within a small box around the center.
        for v in &list.vertices {
            assert!((v.pos[0] - 640.0).abs() <= 12.0);
            assert!((v.pos[1] - 360.0).abs() <= 12.0);
        }
        // Triangle-list indexing: every quad is 6 indices over 4 verts.
        assert_eq!(list.indices.len() % 6, 0);
        assert_eq!(list.vertices.len() * 3, list.indices.len() * 2);
    }

    #[test]
    fn hotbar_has_nine_slots_and_highlights_selection() {
        let mut list = TriList::default();
        let rects = build_hotbar(&mut list, 1280.0, 720.0, 3);
        assert_eq!(rects.len(), 9);
        // Slots are contiguous, centered horizontally, near the bottom.
        assert!((rects[0][0] + rects[8][2]) / 2.0 - 640.0 < 1.0);
        assert!(rects[0][3] > 720.0 - 80.0, "hotbar near the bottom edge");
        // Selected slot has brighter border: count white-ish border pixels.
        let bright = list
            .vertices
            .iter()
            .filter(|v| v.color[0] > 200 && v.color[3] > 200)
            .count();
        assert!(bright > 0, "selected slot must be highlighted");
    }

    #[test]
    fn progress_bar_grows_and_hides() {
        let mut empty = TriList::default();
        build_progress(&mut empty, 1280.0, 720.0, 0.0);
        assert!(empty.is_empty(), "no progress → no bar");

        let mut a = TriList::default();
        build_progress(&mut a, 1280.0, 720.0, 0.25);
        let mut b = TriList::default();
        build_progress(&mut b, 1280.0, 720.0, 0.75);
        assert!(!a.is_empty() && !b.is_empty());
        // Fill width grows with progress. The fill quads are the only ones
        // drawn with alpha 220 (track 190, outline 140).
        let fill_max_x = |l: &TriList| l
            .vertices
            .iter()
            .filter(|v| v.color[3] == 220)
            .map(|v| v.pos[0])
            .fold(0.0f32, f32::max);
        assert!(fill_max_x(&b) > fill_max_x(&a), "fill grows with progress");
    }

    #[test]
    fn text_layout_is_predictable() {
        let g = layout_text("hi", 10.0, 20.0, 2.0);
        assert_eq!(g.len(), 2);
        assert_eq!(g[0], ('h', [10.0, 20.0]));
        assert_eq!(g[1], ('i', [22.0, 20.0]), "6 units × scale 2 advance");
        // Centering lands mid-slot.
        let c = centered_text("123", [0.0, 0.0, 60.0, 40.0], 2.0);
        let width = 3.0 * 6.0 * 2.0;
        let start = c[0].1[0];
        assert!((start - (30.0 - width / 2.0)).abs() < 0.01);
    }

    #[test]
    fn outline_is_a_bounded_beam_cage() {
        let mut list = TriList::default();
        build_block_outline(&mut list, -3, 40, 7, 0.002, [10, 10, 10, 255]);
        assert!(!list.is_empty());
        // All vertices hug the unit cube expanded slightly (12 edges × 4
        // sides × 4 verts).
        for v in &list.vertices {
            assert!(v.pos[0] >= -3.0 - 0.01 && v.pos[0] <= -2.0 + 0.01);
            assert!(v.pos[1] >= 40.0 - 0.01 && v.pos[1] <= 41.0 + 0.01);
            assert!(v.pos[2] >= 7.0 - 0.01 && v.pos[2] <= 8.0 + 0.01);
        }
        assert_eq!(list.vertices.len(), 12 * 4 * 4);
    }

    #[test]
    fn particles_spawn_deterministically_and_expire() {
        let mut ps = Vec::new();
        spawn_burst(&mut ps, 5, 60, -2, [120, 90, 60, 255], 0xB10C);
        assert_eq!(ps.len(), 12);
        // Deterministic: a second burst is identical.
        let mut ps2 = Vec::new();
        spawn_burst(&mut ps2, 5, 60, -2, [120, 90, 60, 255], 0xB10C);
        assert_eq!(ps.len(), ps2.len());
        for (a, b) in ps.iter().zip(ps2.iter()) {
            assert_eq!(a.pos, b.pos);
            assert_eq!(a.vel, b.vel);
        }
        // Step past ttl: all expire.
        for _ in 0..120 {
            step_particles(&mut ps, 1.0 / 60.0);
        }
        assert!(ps.is_empty());
    }

    #[test]
    fn particles_fall_with_gravity() {
        let mut ps = vec![Particle {
            pos: [0.0; 3],
            vel: [0.0, 0.0, 0.0],
            life: 0.0,
            ttl: 10.0,
            color: [0, 0, 0, 255],
        }];
        step_particles(&mut ps, 0.5);
        assert!(ps[0].vel[1] < 0.0, "gravity pulls down");
        assert!(ps[0].pos[1] < 0.0);
    }
}
