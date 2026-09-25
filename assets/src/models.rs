//! Model compiler: resolves parent chains, texture variables and element
//! geometry into `FaceQuad` prototypes ready for the runtime chunk mesher.

use serde::{Deserialize, Serialize};

use crate::error::{AssetError, AssetResult};
use crate::models_raw::{RawElement, RawModel, RawTextureValue};
use crate::sprites::SpriteRef;
use std::collections::HashMap;

/// Interned sprite reference. The compiler assigns these; the atlas and the
/// runtime mesher translate them to UV rects without touching strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpriteId(pub u32);

/// Bidirectional sprite <-> id interning.
#[derive(Debug, Default)]
pub struct SpriteInterner {
    ids: HashMap<SpriteRef, u32>,
    names: Vec<SpriteRef>,
}

impl SpriteInterner {
    pub fn intern(&mut self, name: SpriteRef) -> SpriteId {
        if let Some(&id) = self.ids.get(&name) {
            return SpriteId(id);
        }
        let id = self.names.len() as u32;
        self.names.push(name.clone());
        self.ids.insert(name, id);
        SpriteId(id)
    }

    pub fn name(&self, id: SpriteId) -> &SpriteRef {
        &self.names[id.0 as usize]
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn names(&self) -> &[SpriteRef] {
        &self.names
    }
}

/// Interned model reference (into the compiled model table).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId(pub u32);

/// A resolved texture reference: a concrete sprite plus layering info.
#[derive(Debug, Clone)]
pub struct ResolvedSprite {
    pub name: SpriteRef,
    pub force_translucent: bool,
}

/// A baked quad in block-local coordinates (0..16) with sprite-space UVs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FaceQuad {
    /// Corner positions: [min_x, min_y, min_z, max_x, max_y, max_z] in 0..16 units.
    /// The axis-aligned bounding box of `corners` (kept for occlusion
    /// classification and quick size queries — geometry itself is `corners`).
    pub bounds: [f32; 6],
    /// The four quad corners in CCW-from-outside winding, block-local 0..16.
    /// Baked (not an AABB) so rotated elements keep their diagonal geometry:
    /// a 45° cross plane is a real slanted quad, not a box around one.
    pub corners: [[f32; 3]; 4],
    /// Which of the 6 face directions this quad belongs to (post-rotation).
    pub dir: Direction,
    /// Interned sprite this quad samples.
    pub sprite: SpriteId,
    /// UV rect in the sprite's 0..16 space: [u0, v0, u1, v1].
    pub uv: [f32; 4],
    /// Tint index: None = untinted, Some(i) = biome tint slot i.
    pub tint: Option<i32>,
    /// Cull direction (face hidden when the neighbor in this dir occludes).
    pub cull: Option<Direction>,
    /// Forced translucent blending (redstone dust sprites).
    pub force_translucent: bool,
    /// Element light emission 0..15.
    pub light_emission: u8,
    /// Shade direction override (directional brightness), if present.
    pub shade_override: Option<Direction>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Direction {
    Down,
    Up,
    North,
    South,
    West,
    East,
}

impl Direction {
    pub const ALL: [Direction; 6] = [
        Direction::Down,
        Direction::Up,
        Direction::North,
        Direction::South,
        Direction::West,
        Direction::East,
    ];

    pub fn from_name(s: &str) -> Option<Direction> {
        Some(match s {
            "down" => Direction::Down,
            "up" => Direction::Up,
            "north" => Direction::North,
            "south" => Direction::South,
            "west" => Direction::West,
            "east" => Direction::East,
            _ => return None,
        })
    }
}

/// A compiled model prototype.
#[derive(Debug, Clone)]
pub struct CompiledModel {
    pub quads: Vec<FaceQuad>,
    pub particle: SpriteRef,
    pub ambient_occlusion: bool,
    /// `gui_light`: `side` (default) or `front`.
    pub gui_light: GuiLight,
    /// Display context transforms, keyed by context name (raw, resolved later).
    pub display: HashMap<String, crate::models_raw::RawDisplay>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GuiLight {
    Side,
    Front,
}

/// Parsed but unresolved models, keyed by `namespace:path`.
pub type RawModels = HashMap<(String, String), RawModel>;

/// Load every JSON under `models/`.
pub fn load_raw(index: &crate::pack::PackIndex) -> AssetResult<RawModels> {
    let mut out = HashMap::new();
    for ((ns, path), file) in &index.files {
        // Index keys are extension-stripped; JSON-ness was decided at discovery.
        if !path.starts_with("models/") {
            continue;
        }
        let text = std::fs::read_to_string(file).map_err(|e| AssetError {
            path: file.display().to_string(),
            message: format!("cannot read: {e}"),
        })?;
        let model = RawModel::parse(&text).map_err(|e| AssetError {
            path: file.display().to_string(),
            message: format!("model JSON invalid: {e}"),
        })?;
        let key = (
            ns.clone(),
            path.strip_prefix("models/").unwrap_or(path).to_string(),
        );
        out.insert(key, model);
    }
    Ok(out)
}

fn normalize_ref(s: &str) -> (String, String) {
    match s.split_once(':') {
        Some((ns, p)) => (ns.to_string(), p.to_string()),
        None => ("minecraft".to_string(), s.to_string()),
    }
}

/// Compile one model by flattening its parent chain.
///
/// Resolution rules (matching the observed pack semantics):
/// * `textures`: child wins per-key; keys from all ancestors merge.
/// * `elements`: the *nearest* definition wins wholesale (no merge).
/// * `ambientocclusion` / `gui_light` / `display`: nearest wins.
/// * Parent links that hit `builtin/generated` (item extrusion) stop the chain;
///   Phase 1 renders block geometry only.
pub fn compile(
    name: &(String, String),
    raws: &RawModels,
    interner: &mut SpriteInterner,
) -> AssetResult<CompiledModel> {
    let mut chain: Vec<&RawModel> = Vec::new();
    let mut cur = Some(name.clone());
    let mut depth = 0;
    while let Some(n) = cur {
        depth += 1;
        if depth > 32 {
            return crate::error::err(&format!("{}:{}", name.0, name.1), "parent chain too deep (cycle?)");
        }
        let model = raws.get(&n).ok_or_else(|| AssetError {
            path: format!("{}:{}", n.0, n.1),
            message: "model not found in pack".into(),
        })?;
        chain.push(model);
        let next = model.parent.as_ref().map(|p| normalize_ref(p));
        if let Some((_, p)) = &next {
            if p == "builtin/generated" || p == "builtin/entity" {
                break;
            }
        }
        cur = next;
    }
    if chain.is_empty() {
        return crate::error::err(&format!("{}:{}", name.0, name.1), "empty parent chain");
    }

    // Merge textures: ancestors first, child overrides.
    let mut textures: HashMap<String, ResolvedSprite> = HashMap::new();
    for model in chain.iter().rev() {
        for (k, v) in &model.textures {
            let val = match v {
                RawTextureValue::Str(s) => RawTextureValue::Str(s.clone()),
                RawTextureValue::Sprite { sprite, force_translucent } => RawTextureValue::Sprite {
                    sprite: sprite.clone(),
                    force_translucent: *force_translucent,
                },
            };
            let sprite_name = normalize_ref(val.sprite_ref());
            textures.insert(
                k.clone(),
                ResolvedSprite {
                    name: sprite_name,
                    force_translucent: val.force_translucent(),
                },
            );
        }
    }

    // Nearest-wins scalar fields.
    let ambient_occlusion = chain.iter().find_map(|m| m.ambientocclusion).unwrap_or(true);
    let gui_light = chain
        .iter()
        .find_map(|m| m.gui_light.clone())
        .map(|g| if g == "front" { GuiLight::Front } else { GuiLight::Side })
        .unwrap_or(GuiLight::Side);
    let mut display = HashMap::new();
    for model in &chain {
        for (k, v) in &model.display {
            display.entry(k.clone()).or_insert_with(|| v.clone());
        }
    }
    let particle = resolve_particle(&textures);

    // Elements: nearest definition.
    let elements_owner = chain.iter().find(|m| m.elements.is_some());
    let mut quads = Vec::new();
    if let Some(model) = elements_owner {
        for el in model.elements.as_ref().unwrap() {
            bake_element(el, &textures, interner, &mut quads);
        }
    }

    Ok(CompiledModel { quads, particle, ambient_occlusion, gui_light, display })
}

/// Resolve a `#var` chain against the merged texture map.
/// A var whose chain never resolves to a concrete sprite returns None
/// (e.g. `#all` on a model that never defines it); callers skip those faces.
fn resolve_var(var: &str, textures: &HashMap<String, ResolvedSprite>) -> Option<ResolvedSprite> {
    let mut cur = var.to_string();
    for _ in 0..12 {
        let Some(rest) = cur.strip_prefix('#') else {
            let r = normalize_ref(&cur);
            return Some(ResolvedSprite { name: r, force_translucent: false });
        };
        let resolved = textures.get(rest)?;
        if resolved.name.1.starts_with('#') {
            cur = resolved.name.1.clone();
            continue;
        }
        return Some(resolved.clone());
    }
    None
}

/// Same as `resolve_var` but for particle sprites: falls back to the first
/// face sprite when the particle var never resolves to a concrete name.
fn resolve_particle(
    textures: &HashMap<String, ResolvedSprite>,
) -> SpriteRef {
    if let Some(p) = textures.get("particle") {
        if !p.name.1.starts_with('#') {
            return p.name.clone();
        }
    }
    // Fall back: any concrete sprite in the map (faces usually cover it).
    textures
        .values()
        .find(|r| !r.name.1.starts_with('#'))
        .map(|r| r.name.clone())
        .unwrap_or_else(|| ("minecraft".into(), "missingno".into()))
}

fn bake_element(
    el: &RawElement,
    textures: &HashMap<String, ResolvedSprite>,
    interner: &mut SpriteInterner,
    out: &mut Vec<FaceQuad>,
) {
    let [fx0, fy0, fz0] = el.from;
    let [fx1, fy1, fz1] = el.to;
    let rot = el.rotation.clone();

    for (face_name, face) in &el.faces {
        let Some(dir0) = Direction::from_name(face_name) else { continue };
        let Some(sprite) = resolve_var(&face.texture, textures) else { continue };
        let sprite_id = interner.intern(sprite.name);

        // Default UVs: full sprite, oriented per face.
        let uv = face.uv.unwrap_or(default_uv(dir0));

        // Build the quad in block space, then rotate if needed.
        let mut quad = FaceQuad {
            bounds: [fx0, fy0, fz0, fx1, fy1, fz1],
            corners: face_corners(dir0, [fx0, fy0, fz0, fx1, fy1, fz1]),
            dir: dir0,
            sprite: sprite_id,
            uv,
            tint: face.tintindex,
            cull: face.cullface.as_deref().and_then(Direction::from_name),
            force_translucent: sprite.force_translucent,
            light_emission: el.light_emission.unwrap_or(0),
            shade_override: el
                .shade_direction_override
                .as_deref()
                .and_then(Direction::from_name),
        };

        if let Some(r) = &rot {
            rotate_quad(&mut quad, r);
        }

        out.push(quad);
    }
}

fn default_uv(_dir: Direction) -> [f32; 4] {
    // Vanilla defaults: full 0..16 face square. Deepslate-style rotated
    // defaults (north/south for Y-axis) are not needed by our models list.
    [0.0, 0.0, 16.0, 16.0]
}

/// The four corners of a face plane in CCW-from-outside winding (block space).
/// Winding verified by unit test: the (c1-c0)×(c2-c0) normal points outward.
pub fn face_corners(dir: Direction, b: [f32; 6]) -> [[f32; 3]; 4] {
    let [x0, y0, z0, x1, y1, z1] = b;
    match dir {
        Direction::Up => [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]],
        Direction::Down => [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],
        Direction::North => [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]],
        Direction::South => [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
        Direction::West => [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]],
        Direction::East => [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]],
    }
}

/// Apply a blockstate variant rotation to a quad: vanilla order is X then Y,
/// about the block center. Ground truth from this pack: `furnace` facing=east
/// is y=90 (north face must land on east); `barrel` facing=east is x=90,y=90
/// (up face must land on east after the full chain).
pub fn apply_variant_rotation(quad: &mut FaceQuad, rot: &crate::blockstates::VariantRot) {
    let center = [8.0, 8.0, 8.0];
    if rot.x != 0.0 {
        rotate_face_quad(quad, "x", rot.x, center, false);
    }
    if rot.y != 0.0 {
        rotate_face_quad(quad, "y", rot.y, center, false);
    }
}

/// Rotate a quad's bounds, corners and direction by the element rotation.
fn rotate_quad(quad: &mut FaceQuad, r: &crate::models_raw::RawRotation) {
    // Classic single-axis form or 26.x arbitrary euler form.
    let axes: Vec<(&str, f32)> = if let (Some(axis), Some(angle)) = (r.axis.as_deref(), r.angle) {
        vec![(axis, angle)]
    } else {
        let mut v = Vec::new();
        if let Some(x) = r.x {
            v.push(("x", x));
        }
        if let Some(y) = r.y {
            v.push(("y", y));
        }
        if let Some(z) = r.z {
            v.push(("z", z));
        }
        v
    };
    if axes.is_empty() {
        return;
    }
    let origin = r.origin;

    for (axis, angle_deg) in axes {
        rotate_face_quad(quad, axis, angle_deg, origin, r.rescale);
    }
}

/// One-axis rotation of a quad about `origin`. `rescale=true` applies the
/// vanilla 1/cos(θ) stretch (element rotations only; variant rotations pass
/// false). Direction sign matches `rotate_direction` below (verified against
/// the pack's furnace/barrel ground truth).
fn rotate_face_quad(
    quad: &mut FaceQuad,
    axis: &str,
    angle_deg: f32,
    origin: [f32; 3],
    rescale_enabled: bool,
) {
    let (sin, cos) = angle_deg.to_radians().sin_cos();
    let rescale = if rescale_enabled {
        1.0 / angle_deg.abs().to_radians().cos()
    } else {
        1.0
    };

    // Point rotation, inverse right-hand sense to match `rotate_direction`
    // below (both derived from the same furnace/barrel ground truth).
    let rot_pt = |p: [f32; 3]| -> [f32; 3] {
        let rel = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
        let mut q = match axis {
            "x" => [
                rel[0],
                rel[1] * cos + rel[2] * sin,
                -rel[1] * sin + rel[2] * cos,
            ],
            "y" => [
                rel[0] * cos - rel[2] * sin,
                rel[1],
                rel[0] * sin + rel[2] * cos,
            ],
            "z" => [
                rel[0] * cos + rel[1] * sin,
                -rel[0] * sin + rel[1] * cos,
                rel[2],
            ],
            _ => rel,
        };
        if rescale_enabled {
            // Rescale on the two axes perpendicular to the rotation axis
            // (vanilla applies it to the pair that moves; y-axis spins don't
            // need it since vanilla never combines rescale with y).
            match axis {
                "x" => {
                    q[1] *= rescale;
                    q[2] *= rescale;
                }
                "z" => {
                    q[0] *= rescale;
                    q[1] *= rescale;
                }
                _ => {}
            }
        }
        [q[0] + origin[0], q[1] + origin[1], q[2] + origin[2]]
    };

    // Rotate the four baked corners; bounds become their AABB.
    let rotated: Vec<[f32; 3]> = quad.corners.iter().map(|c| rot_pt(*c)).collect();
    quad.corners = [rotated[0], rotated[1], rotated[2], rotated[3]];
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for c in &rotated {
        for i in 0..3 {
            mn[i] = mn[i].min(c[i]);
            mx[i] = mx[i].max(c[i]);
        }
    }
    quad.bounds = [mn[0], mn[1], mn[2], mx[0], mx[1], mx[2]];

    // Rotate the face direction and cull.
    quad.dir = rotate_direction(quad.dir, axis, angle_deg);
    quad.cull = quad.cull.map(|c| rotate_direction(c, axis, angle_deg));
}

/// Rotate a direction by angle degrees around the given axis (right-handed,
/// matching the point-rotation math above).
pub fn rotate_direction(d: Direction, axis: &str, angle: f32) -> Direction {
    use Direction::*;
    // Vanilla rotates by the signed angle about the axis; quarter steps only.
    let q = ((angle.round() as i64).rem_euclid(360) / 90) as u32; // 0..3
    let mut d = d;
    for _ in 0..q {
        d = match axis {
            // Ground truth (this pack): furnace facing=east is y=90, so
            // +90° y sends North→East (and Down/Up are unchanged).
            "y" => match d {
                North => East,
                East => South,
                South => West,
                West => North,
                other => other,
            },
            // Ground truth: barrel facing=north is x=90 (top lands on north),
            // so +90° x sends Up→North (inverse right-hand sense).
            "x" => match d {
                Up => North,
                North => Down,
                Down => South,
                South => Up,
                other => other,
            },
            // Consistent inverse-right-hand mapping for z (same convention).
            "z" => match d {
                Up => East,
                East => Down,
                Down => West,
                West => Up,
                other => other,
            },
            _ => d,
        };
    }
    d
}
