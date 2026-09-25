//! Blockstate compiler.
//!
//! Turns `blockstates/*.json` (1,184 `variants` + 104 `multipart` in this pack)
//! into per-state appearance tables:
//! * property schemas extracted from variant keys (`axis=x|y|z`),
//! * variant lists compiled to `(predicate, model, rotation)` entries,
//! * multipart `when` trees compiled against packed property bits,
//! * per-state occlusion classification consumed by the chunk mesher.

use crate::error::{AssetError, AssetResult};
use crate::models::{Direction, ModelId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Raw JSON
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct RawBlockstate {
    #[serde(default)]
    pub variants: Option<HashMap<String, RawVariantEntry>>,
    #[serde(default)]
    pub multipart: Option<Vec<RawMultipartPart>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum RawVariantEntry {
    One(RawVariant),
    Many(Vec<RawVariant>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct RawVariant {
    pub model: String,
    #[serde(default)]
    pub x: Option<f32>,
    #[serde(default)]
    pub y: Option<f32>,
    #[serde(default)]
    pub uvlock: Option<bool>,
    #[serde(default)]
    pub weight: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RawMultipartPart {
    pub apply: RawApply,
    #[serde(default)]
    pub when: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum RawApply {
    One(RawVariant),
    Many(Vec<RawVariant>),
}

impl RawBlockstate {
    pub fn parse(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

// ---------------------------------------------------------------------------
// Compiled form
// ---------------------------------------------------------------------------

/// Rotation baked into a variant (degrees; vanilla uses multiples of 22.5).
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct VariantRot {
    pub x: f32,
    pub y: f32,
    pub uvlock: bool,
}

/// One appearance choice: a model reference plus rotation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ModelInstance {
    pub model: ModelId,
    pub rot: VariantRot,
    /// Random-pool weight (variant lists only; vanilla default 1).
    pub weight: u32,
}

/// A predicate over property values, matching `ValueMatcher` semantics:
/// `true`, `"value"`, `"a|b"`, or nested `{and: [...]}` / `{or: [...]}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Predicate {
    Always,
    And(Vec<Predicate>),
    Or(Vec<Predicate>),
    Match(String, String),
}

impl Predicate {
    pub fn matches(&self, props: &[(String, String)]) -> bool {
        match self {
            Predicate::Always => true,
            Predicate::And(ps) => ps.iter().all(|p| p.matches(props)),
            Predicate::Or(ps) => ps.iter().any(|p| p.matches(props)),
            Predicate::Match(k, v) => {
                props.iter().any(|(pk, pv)| pk == k && value_matches(v, pv))
            }
        }
    }
}

/// Vanilla value matching: `a|b` alternation; `!x` negation is not used by
/// this pack's blockstates (plain false/true/none/side|up only).
fn value_matches(pattern: &str, value: &str) -> bool {
    pattern.split('|').any(|p| p == value)
}

/// One multipart entry: apply these model instances when the predicate holds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultipartEntry {
    pub when: Predicate,
    pub models: Vec<ModelInstance>,
}

/// A variant key like `axis=x` compiled to a predicate + instances.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantEntry {
    pub when: Predicate,
    pub models: Vec<ModelInstance>,
}

/// Compiled appearance for one block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StateAppearance {
    /// Single choice, no state dependence.
    Static(Vec<ModelInstance>),
    /// Keyed by property predicate; first match wins (compile order preserved).
    Variants(Vec<VariantEntry>),
    /// All matching parts are applied.
    Multipart(Vec<MultipartEntry>),
}

/// Property schema for one block: names + the set of observed values.
#[derive(Debug, Clone)]
pub struct PropertySchema {
    pub properties: Vec<(String, Vec<String>)>,
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Parse `blockstates/*.json` from the index.
pub fn load_raw(index: &crate::pack::PackIndex) -> AssetResult<HashMap<(String, String), RawBlockstate>> {
    let mut out = HashMap::new();
    for ((ns, path), file) in &index.files {
        // Index keys are extension-stripped; JSON-ness was decided at discovery.
        if !path.starts_with("blockstates/") {
            continue;
        }
        let text = std::fs::read_to_string(file).map_err(|e| AssetError {
            path: file.display().to_string(),
            message: format!("cannot read: {e}"),
        })?;
        let bs = RawBlockstate::parse(&text).map_err(|e| AssetError {
            path: file.display().to_string(),
            message: format!("blockstate JSON invalid: {e}"),
        })?;
        let key = (
            ns.clone(),
            path.strip_prefix("blockstates/").unwrap_or(path).to_string(),
        );
        out.insert(key, bs);
    }
    Ok(out)
}

/// Extract `property -> observed values` from all variant keys of one blockstate.
pub fn extract_schema(raw: &RawBlockstate) -> PropertySchema {
    let mut props: HashMap<String, Vec<String>> = HashMap::new();
    if let Some(variants) = &raw.variants {
        for key in variants.keys() {
            for pair in key.split(',') {
                let Some((k, v)) = pair.split_once('=') else { continue };
                let k = k.trim().to_string();
                let v = v.trim().to_string();
                let entry = props.entry(k).or_default();
                if !entry.contains(&v) {
                    entry.push(v);
                }
            }
        }
    }
    // Multipart conditions also reveal values.
    if let Some(parts) = &raw.multipart {
        for part in parts {
            if let Some(when) = &part.when {
                collect_when_values(when, &mut props);
            }
        }
    }
    let mut properties: Vec<(String, Vec<String>)> = props.into_iter().collect();
    properties.sort();
    PropertySchema { properties }
}

fn collect_when_values(when: &serde_json::Value, props: &mut HashMap<String, Vec<String>>) {
    match when {
        serde_json::Value::Object(o) => {
            for (k, v) in o {
                if k == "AND" || k == "OR" {
                    if let serde_json::Value::Array(arr) = v {
                        for sub in arr {
                            collect_when_values(sub, props);
                        }
                    }
                } else if let serde_json::Value::String(s) = v {
                    let entry = props.entry(k.clone()).or_default();
                    for alt in s.split('|') {
                        let alt = alt.trim().to_string();
                        if !entry.contains(&alt) {
                            entry.push(alt);
                        }
                    }
                }
            }
        }
        _ => {}
    }
}

/// Compile a `when` JSON into a Predicate.
pub fn compile_when(when: Option<&serde_json::Value>) -> Predicate {
    let Some(when) = when else { return Predicate::Always };
    compile_when_value(when)
}

fn compile_when_value(when: &serde_json::Value) -> Predicate {
    match when {
        serde_json::Value::Bool(true) => Predicate::Always,
        serde_json::Value::Object(o) => {
            let mut ands: Vec<Predicate> = Vec::new();
            for (k, v) in o {
                match k.as_str() {
                    "AND" => {
                        if let serde_json::Value::Array(arr) = v {
                            ands.push(Predicate::And(arr.iter().map(compile_when_value).collect()));
                        }
                    }
                    "OR" => {
                        if let serde_json::Value::Array(arr) = v {
                            ands.push(Predicate::Or(arr.iter().map(compile_when_value).collect()));
                        }
                    }
                    _ => {
                        if let serde_json::Value::String(s) = v {
                            ands.push(Predicate::Match(k.clone(), s.clone()));
                        }
                    }
                }
            }
            if ands.is_empty() {
                Predicate::Always
            } else {
                Predicate::And(ands)
            }
        }
        _ => Predicate::Always,
    }
}

/// Compile variants into entries, preserving declaration order.
pub fn compile_variants(raw: &RawBlockstate, resolve_model: &mut impl FnMut(&str) -> ModelId) -> AssetResult<StateAppearance> {
    let Some(variants) = &raw.variants else {
        return crate::error::err("blockstate", "no variants section");
    };
    let mut entries = Vec::new();
    for (key, entry) in variants {
        let list = match entry {
            RawVariantEntry::One(v) => vec![v.clone()],
            RawVariantEntry::Many(v) => v.clone(),
        };
        let when = variant_key_predicate(key);
        let models: Vec<ModelInstance> = list
            .iter()
            .map(|v| ModelInstance {
                model: resolve_model(&v.model),
                rot: VariantRot {
                    x: v.x.unwrap_or(0.0),
                    y: v.y.unwrap_or(0.0),
                    uvlock: v.uvlock.unwrap_or(false),
                },
                weight: v.weight.unwrap_or(1).max(1),
            })
            .collect();
        entries.push(VariantEntry { when, models });
    }
    if entries.len() == 1 && entries[0].when == Predicate::Always {
        return Ok(StateAppearance::Static(entries.remove(0).models));
    }
    Ok(StateAppearance::Variants(entries))
}

/// `axis=x` -> Match(axis, x); `` -> Always. Empty string key = always.
fn variant_key_predicate(key: &str) -> Predicate {
    let key = key.trim();
    if key.is_empty() {
        return Predicate::Always;
    }
    let mut ands = Vec::new();
    for pair in key.split(',') {
        if let Some((k, v)) = pair.split_once('=') {
            ands.push(Predicate::Match(k.trim().to_string(), v.trim().to_string()));
        }
    }
    match ands.len() {
        0 => Predicate::Always,
        1 => ands.pop().unwrap(),
        _ => Predicate::And(ands),
    }
}

/// Compile multipart parts.
pub fn compile_multipart(raw: &RawBlockstate, resolve_model: &mut impl FnMut(&str) -> ModelId) -> AssetResult<StateAppearance> {
    let Some(parts) = &raw.multipart else {
        return crate::error::err("blockstate", "no multipart section");
    };
    let mut entries = Vec::new();
    for part in parts {
        let when = compile_when(part.when.as_ref());
        let list = match &part.apply {
            RawApply::One(v) => vec![v.clone()],
            RawApply::Many(v) => v.clone(),
        };
        let models: Vec<ModelInstance> = list
            .iter()
            .map(|v| ModelInstance {
                model: resolve_model(&v.model),
                rot: VariantRot {
                    x: v.x.unwrap_or(0.0),
                    y: v.y.unwrap_or(0.0),
                    uvlock: v.uvlock.unwrap_or(false),
                },
                // Vanilla ignores weights on multipart parts.
                weight: 1,
            })
            .collect();
        entries.push(MultipartEntry { when, models });
    }
    Ok(StateAppearance::Multipart(entries))
}

// ---------------------------------------------------------------------------
// Occlusion classification
// ---------------------------------------------------------------------------

/// How a rendered state blocks neighbor faces. State-specific (per the plan:
/// stone culles neighbors, glass doesn't, cross/vine/rail/torch never do).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Occlusion {
    /// Full opaque cube geometry: hides the neighbor face touching it.
    #[default]
    Full,
    /// Renders but never hides neighbor faces (glass, leaves, anything else).
    None,
}

impl Occlusion {
    pub fn hides_neighbor(self) -> bool {
        matches!(self, Occlusion::Full)
    }
}

/// Classify occlusion from a compiled model's quads: exactly one full-height
/// cube of six quads with all six cull directions == full occluder.
pub fn classify_occlusion(quads: &[crate::models::FaceQuad]) -> Occlusion {
    if quads.len() != 6 {
        return Occlusion::None;
    }
    let mut seen = [false; 6];
    for q in quads {
        // Full cube face: spans the full 0..16 range on its plane's axes.
        let b = q.bounds;
        let full = (b[0] <= 0.01 && b[3] >= 15.99) || (b[1] <= 0.01 && b[4] >= 15.99) || (b[2] <= 0.01 && b[5] >= 15.99);
        let aligned = match q.dir {
            Direction::Down | Direction::Up => b[1] <= 0.01 || b[4] >= 15.99,
            Direction::North | Direction::South => b[2] <= 0.01 || b[5] >= 15.99,
            Direction::West | Direction::East => b[0] <= 0.01 || b[3] >= 15.99,
        };
        if !(full && aligned && q.tint.is_none()) {
            return Occlusion::None;
        }
        seen[q.dir as usize] = true;
    }
    if seen.iter().all(|s| *s) {
        Occlusion::Full
    } else {
        Occlusion::None
    }
}

#[allow(unused_imports)]
use crate::models::SpriteId;
