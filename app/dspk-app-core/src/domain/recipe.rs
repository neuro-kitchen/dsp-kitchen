//! Recipes: a chain of ops saved under a name, reused as one node. A recipe is defined once (in
//! the session, `recipes`, and in the library as `<name>.recipe.json`) and used by any number of
//! recipe nodes, which link to the definition: editing the recipe changes every use, while a use
//! may override some of its values (`overrides`: `"<op>.<key>"` → value).
//!
//! Its ports are typed: a recipe takes a signal and gives a signal, events or a sorting; one that
//! gives a sorting (or is labelled `sorter`) is listed by the Sorting workspace. It carries labels
//! (`filter`, `emg`, `sorter`, …), a version bumped by every edit of the definition, and the dspk
//! version that wrote it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use dspk_pipeline::Node;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::session::OpDoc;

/// The recipe file version this app reads and writes (`dspk_recipe` field).
pub const RECIPE_VERSION: u32 = 1;
/// File name suffix of a recipe in the library.
pub const RECIPE_SUFFIX: &str = ".recipe.json";

/// What flows through a port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PortKind {
    /// Samples × channels.
    #[default]
    Signal,
    /// Times (and channels) of events: spikes, triggers.
    Events,
    /// Units and their spikes.
    Sorting,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RecipeDoc {
    /// Version of the file format; must equal [`RECIPE_VERSION`].
    pub dspk_recipe: u32,
    /// Unique in a session and in the library.
    pub name: String,
    /// Bumped by every edit of the definition.
    #[serde(default = "one")]
    pub version: u32,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// What it is (`filter`, `reference`, `emg`, `sorter`, …): search, grouping, and where it is
    /// offered.
    #[serde(default)]
    pub labels: Vec<String>,
    #[serde(default)]
    pub input: PortKind,
    #[serde(default)]
    pub output: PortKind,
    /// The chain, in order.
    pub ops: Vec<OpDoc>,
    /// Values shown on a use (`"<op>.<key>"`); empty: every value.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exposed: Vec<String>,
    /// The dspk version that wrote it (provenance).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub dspk: String,
}

fn one() -> u32 {
    1
}

#[derive(Debug, thiserror::Error)]
pub enum RecipeError {
    #[error("recipe {0}: {1}")]
    Parse(String, String),
    #[error("recipe {0}: version {1} is not supported (this app reads version {RECIPE_VERSION})")]
    Version(String, u32),
    #[error("{0}")]
    Io(String),
}

impl RecipeDoc {
    pub fn new(name: impl Into<String>, labels: Vec<String>, ops: Vec<OpDoc>) -> Self {
        Self {
            dspk_recipe: RECIPE_VERSION,
            name: name.into(),
            version: 1,
            description: String::new(),
            labels,
            input: PortKind::Signal,
            output: PortKind::Signal,
            ops,
            exposed: Vec::new(),
            dspk: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Whether the Sorting workspace lists it.
    pub fn is_sorter(&self) -> bool {
        self.output == PortKind::Sorting || self.labels.iter().any(|l| l == "sorter")
    }

    /// Its ops with a use's `overrides` applied (`"<op>.<key>"` → value); an override that does
    /// not fit its op is left out.
    pub fn ops_with(&self, overrides: &BTreeMap<String, Value>) -> Vec<OpDoc> {
        self.ops
            .iter()
            .map(|op| {
                let prefix = format!("{}.", op.id);
                let mine: Vec<(&str, &Value)> = overrides.iter().filter_map(|(k, v)| k.strip_prefix(&prefix).map(|key| (key, v))).collect();
                if mine.is_empty() {
                    return op.clone();
                }
                let mut value = serde_json::to_value(&op.node).unwrap_or(Value::Null);
                for (key, v) in mine {
                    value[key] = v.clone();
                }
                let node = serde_json::from_value::<Node>(value).unwrap_or_else(|_| op.node.clone());
                OpDoc { node, ..op.clone() }
            })
            .collect()
    }

    /// The overrides that turn op `op`'s values into `node`'s.
    pub fn overrides_for(&self, op: &str, node: &Node) -> BTreeMap<String, Value> {
        let Some(def) = self.ops.iter().find(|o| o.id == op) else { return BTreeMap::new() };
        let (a, b) = (serde_json::to_value(&def.node).unwrap_or(Value::Null), serde_json::to_value(node).unwrap_or(Value::Null));
        let mut out = BTreeMap::new();
        if let (Value::Object(a), Value::Object(b)) = (a, b) {
            for (k, v) in b {
                if a.get(&k) != Some(&v) {
                    out.insert(format!("{op}.{k}"), v);
                }
            }
        }
        out
    }

    pub fn from_json(text: &str) -> Result<Self, RecipeError> {
        let r: Self = serde_json::from_str(text).map_err(|e| RecipeError::Parse("?".into(), e.to_string()))?;
        if r.dspk_recipe != RECIPE_VERSION {
            return Err(RecipeError::Version(r.name, r.dspk_recipe));
        }
        Ok(r)
    }

    pub fn load(path: &Path) -> Result<Self, RecipeError> {
        let text = std::fs::read_to_string(path).map_err(|e| RecipeError::Io(format!("{}: {e}", path.display())))?;
        Self::from_json(&text).map_err(|e| match e {
            RecipeError::Parse(_, m) => RecipeError::Parse(path.display().to_string(), m),
            other => other,
        })
    }

    /// Writes it as `<dir>/<name>.recipe.json`.
    pub fn save_in(&self, dir: &Path) -> std::io::Result<PathBuf> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join(format!("{}{RECIPE_SUFFIX}", file_stem(&self.name)));
        std::fs::write(&path, serde_json::to_string_pretty(self).expect("a recipe always serializes"))?;
        Ok(path)
    }

    /// JSON Schema of a recipe file.
    pub fn json_schema() -> String {
        serde_json::to_string_pretty(&schemars::schema_for!(RecipeDoc)).expect("a schema always serializes")
    }
}

/// `Spike band (CMR)` → `spike-band-cmr`
pub fn file_stem(name: &str) -> String {
    let s: String = name.chars().map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { '-' }).collect();
    s.split('-').filter(|p| !p.is_empty()).collect::<Vec<_>>().join("-")
}

/// The recipes in `dir` (`*.recipe.json`), by name; unreadable files are reported and left out.
pub fn library(dir: &Path) -> (Vec<RecipeDoc>, Vec<String>) {
    let mut found = Vec::new();
    let mut errors = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.file_name().is_some_and(|n| n.to_string_lossy().ends_with(RECIPE_SUFFIX)) {
                match RecipeDoc::load(&p) {
                    Ok(r) => found.push(r),
                    Err(e) => errors.push(e.to_string()),
                }
            }
        }
    }
    found.sort_by(|a, b| a.name.cmp(&b.name));
    (found, errors)
}

/// Recipes that come with the app.
pub fn builtin() -> Vec<RecipeDoc> {
    let op = |id: &str, node: Value| OpDoc { id: id.into(), enabled: true, node: serde_json::from_value(node).expect("built-in recipe ops are valid") };
    let mut spike = RecipeDoc::new(
        "Spike band",
        vec!["filter".into(), "reference".into(), "ephys".into()],
        vec![op("bandpass", json!({"op": "iir", "band": "bandpass", "cutoff": [300.0, 6000.0]})), op("cmr", json!({"op": "cmr"}))],
    );
    spike.description = "Band-pass 300–6000 Hz then common median reference: what sorters expect.".into();
    let mut lfp = RecipeDoc::new(
        "LFP",
        vec!["filter".into(), "ephys".into()],
        vec![op("lowpass", json!({"op": "iir", "band": "lowpass", "cutoff": [300.0], "order": 4})), op("car", json!({"op": "car"}))],
    );
    lfp.description = "Low-pass 300 Hz then common average reference.".into();
    let mut emg = RecipeDoc::new(
        "EMG envelope",
        vec!["filter".into(), "emg".into()],
        vec![
            op("bandpass", json!({"op": "iir", "band": "bandpass", "cutoff": [20.0, 450.0], "order": 4})),
            op("energy", json!({"op": "teager_kaiser"})),
            op("smooth", json!({"op": "gaussian_smooth", "sigma_samples": 20.0})),
        ],
    );
    emg.description = "Band-pass 20–450 Hz, Teager–Kaiser energy, Gaussian smoothing.".into();
    let mut mains = RecipeDoc::new("Mains removal (60 Hz)", vec!["filter".into()], vec![op("notch60", json!({"op": "notch", "freq": 60.0, "q": 30.0})), op("notch120", json!({"op": "notch", "freq": 120.0, "q": 30.0}))]);
    mains.description = "Notches at 60 Hz and its first harmonic.".into();
    let mut rectified = RecipeDoc::new(
        "EMG envelope (rectified)",
        vec!["filter".into(), "emg".into()],
        vec![
            op("bandpass", json!({"op": "iir", "band": "bandpass", "cutoff": [20.0, 450.0], "order": 4})),
            op("rectify", json!({"op": "rectify"})),
            op("lowpass", json!({"op": "iir", "band": "lowpass", "cutoff": [6.0], "order": 4})),
        ],
    );
    rectified.description = "Band-pass 20–450 Hz, full-wave rectification, low-pass 6 Hz: the linear envelope.".into();
    let mut rms = RecipeDoc::new(
        "RMS envelope",
        vec!["filter".into(), "emg".into()],
        vec![
            op("square", json!({"op": "square"})),
            op("lowpass", json!({"op": "iir", "band": "lowpass", "cutoff": [10.0], "order": 4})),
            op("sqrt", json!({"op": "sqrt"})),
        ],
    );
    rms.description = "Square, low-pass 10 Hz (a moving mean of the power), square root: the moving RMS.".into();
    let mut onsets = RecipeDoc::new(
        "EMG onsets",
        vec!["events".into(), "emg".into()],
        vec![
            op("bandpass", json!({"op": "iir", "band": "bandpass", "cutoff": [20.0, 450.0], "order": 4})),
            op("rectify", json!({"op": "rectify"})),
            op("lowpass", json!({"op": "iir", "band": "lowpass", "cutoff": [6.0], "order": 4})),
            op("crossings", json!({"op": "crossings", "level": 50.0, "release": 25.0, "min_duration_ms": 30.0, "refractory_ms": 80.0})),
        ],
    );
    onsets.description = "The linear envelope, then its crossings of a level: contractions as events (set the level with the ruler on the traces; a Write saves them).".into();
    vec![spike, lfp, emg, rectified, rms, mains, onsets]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overrides_apply_and_round_trip() {
        let r = builtin().remove(0);
        let mut over = BTreeMap::new();
        over.insert("bandpass.cutoff".to_string(), json!([500.0, 5000.0]));
        let ops = r.ops_with(&over);
        assert_eq!(serde_json::to_value(&ops[0].node).unwrap()["cutoff"], json!([500.0, 5000.0]));
        assert_eq!(r.overrides_for("bandpass", &ops[0].node), over);
        assert_eq!(RecipeDoc::from_json(&serde_json::to_string(&r).unwrap()).unwrap(), r);
        assert!(!r.is_sorter());
        assert!(RecipeDoc::json_schema().contains("dspk_recipe"));
        assert_eq!(file_stem("Mains removal (60 Hz)"), "mains-removal-60-hz");
    }

    #[test]
    fn test_builtin_ops_are_view_ops() {
        for r in builtin() {
            for op in &r.ops {
                assert!(op.node.stage().is_ok(), "{}: {}", r.name, op.id);
            }
        }
    }
}
