//! Structural checks of an NWB-Zarr store: what pynwb needs to open it and DANDI expects.
//!
//! Not a replacement for pynwb's validator or nwbinspector (which check the full schema and
//! best practices); this catches broken references, mismatched lengths and missing required
//! fields without needing Python.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::Value;
use zarrs::array::Array;
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableStorageTraits;

use crate::error::{Error, Result};
use crate::metadata::Issue;

/// Every node of the store: path (`/acquisition/HDEMG`) → its `zarr.json`.
struct Store {
    root: PathBuf,
    nodes: BTreeMap<String, Value>,
    store: Arc<FilesystemStore>,
}

impl Store {
    fn open(root: &Path) -> Result<Self> {
        let mut nodes = BTreeMap::new();
        walk(root, root, &mut nodes)?;
        if nodes.is_empty() {
            return Err(Error::format("nwb-zarr", format!("{} is not a Zarr v3 store", root.display())));
        }
        let store = Arc::new(FilesystemStore::new(root).map_err(|e| Error::format("nwb-zarr", e.to_string()))?);
        Ok(Self { root: root.to_path_buf(), nodes, store })
    }

    fn attr(&self, path: &str, key: &str) -> Option<&Value> {
        self.nodes.get(path)?.get("attributes")?.get(key)
    }

    fn kind(&self, path: &str) -> Option<&str> {
        self.attr(path, "neurodata_type").and_then(Value::as_str)
    }

    fn is_array(&self, path: &str) -> bool {
        self.nodes.get(path).and_then(|n| n.get("node_type")).and_then(Value::as_str) == Some("array")
    }

    fn shape(&self, path: &str) -> Option<Vec<u64>> {
        self.nodes.get(path)?.get("shape")?.as_array()?.iter().map(Value::as_u64).collect()
    }

    fn children<'a>(&'a self, path: &'a str) -> impl Iterator<Item = &'a str> + 'a {
        let prefix = if path == "/" { "/".to_string() } else { format!("{path}/") };
        self.nodes.keys().filter_map(move |k| {
            let rest = k.strip_prefix(&prefix)?;
            (!rest.is_empty() && !rest.contains('/')).then_some(k.as_str())
        })
    }

    fn read<T: zarrs::array::ElementOwned>(&self, path: &str) -> Option<Vec<T>> {
        let store: Arc<dyn ReadableStorageTraits> = self.store.clone();
        let a = Array::open(store, path).ok()?;
        a.retrieve_array_subset::<Vec<T>>(&a.subset_all()).ok()
    }
}

fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, Value>) -> Result<()> {
    let meta = dir.join("zarr.json");
    if let Ok(text) = std::fs::read_to_string(&meta) {
        let v: Value = serde_json::from_str(&text).map_err(|e| Error::format("nwb-zarr", format!("{}: {e}", meta.display())))?;
        let rel = dir.strip_prefix(root).unwrap_or(dir).to_string_lossy().replace('\\', "/");
        let is_array = v.get("node_type").and_then(Value::as_str) == Some("array");
        out.insert(format!("/{rel}").replace("//", "/"), v);
        if is_array {
            return Ok(()); // chunks only below arrays
        }
    }
    for e in std::fs::read_dir(dir).map_err(|e| Error::io(dir, e))?.flatten() {
        if e.file_type().is_ok_and(|t| t.is_dir()) {
            walk(root, &e.path(), out)?;
        }
    }
    Ok(())
}

pub fn validate(path: &Path) -> Result<Vec<Issue>> {
    let s = Store::open(path)?;
    let mut issues = Vec::new();
    let mut err = |m: String| issues.push(Issue::error(m));

    // Root
    if s.kind("/") != Some("NWBFile") {
        err("root is not an NWBFile".into());
    }
    match s.attr("/", ".specloc").and_then(Value::as_str) {
        Some(loc) if s.nodes.contains_key(&format!("/{loc}")) => {}
        _ => issues.push(Issue::warning("no cached specifications (/specifications)")),
    }
    let mut err = |m: String| issues.push(Issue::error(m));
    for required in ["/session_description", "/identifier", "/session_start_time", "/timestamps_reference_time", "/file_create_date"] {
        if !s.is_array(required) {
            err(format!("{required} is missing"));
        }
    }
    if let Some(v) = s.read::<String>("/session_description").and_then(|v| v.into_iter().next()) {
        if v.trim().is_empty() {
            err("/session_description is empty".into());
        }
    }
    if let Some(t) = s.read::<String>("/session_start_time").and_then(|v| v.into_iter().next()) {
        let time = t.split_once('T').map_or("", |(_, t)| t);
        if !(time.ends_with('Z') || time.contains('+') || time.contains('-')) {
            err(format!("/session_start_time {t:?} has no time zone"));
        }
    }

    // Electrodes table
    let table = "/general/extracellular_ephys/electrodes";
    let electrode_rows = s.shape(&format!("{table}/id")).map(|v| v[0]);
    if s.nodes.contains_key(table) {
        check_table(&s, table, &mut issues);
        if let Some(groups) = s.read::<String>(&format!("{table}/group")) {
            for g in groups.iter().collect::<std::collections::BTreeSet<_>>() {
                if s.kind(g) != Some("ElectrodeGroup") {
                    issues.push(Issue::error(format!("electrodes/group points to {g}, which is not an ElectrodeGroup")));
                }
            }
        }
    }
    for g in s.children("/general/extracellular_ephys").filter(|p| s.kind(p) == Some("ElectrodeGroup")) {
        let links = s.attr(g, "_LINKS").and_then(Value::as_array).cloned().unwrap_or_default();
        for l in links {
            let target = l.get("path").and_then(Value::as_str).unwrap_or("");
            if s.kind(target) != Some("Device") {
                issues.push(Issue::error(format!("{g}: link {:?} → {target} is not a Device", l.get("name"))));
            }
        }
    }

    // Time series
    for p in s.children("/acquisition").chain(s.children("/stimulus/presentation")).collect::<Vec<_>>() {
        let kind = s.kind(p).unwrap_or("");
        if !matches!(kind, "TimeSeries" | "ElectricalSeries") {
            continue;
        }
        let data = format!("{p}/data");
        let Some(shape) = s.shape(&data) else {
            issues.push(Issue::error(format!("{p}: no data")));
            continue;
        };
        if s.attr(&data, "unit").is_none() {
            issues.push(Issue::error(format!("{p}/data has no unit")));
        }
        let rows = shape.first().copied().unwrap_or(0);
        let starting = format!("{p}/starting_time");
        let stamps = format!("{p}/timestamps");
        if s.is_array(&starting) {
            if s.attr(&starting, "rate").and_then(Value::as_f64).is_none_or(|r| r <= 0.0) {
                issues.push(Issue::error(format!("{p}: starting_time without a positive rate")));
            }
        } else if let Some(ts) = s.shape(&stamps) {
            if ts.first() != Some(&rows) {
                issues.push(Issue::error(format!("{p}: {} timestamps for {rows} samples", ts.first().unwrap_or(&0))));
            }
        } else {
            issues.push(Issue::error(format!("{p}: neither starting_time nor timestamps")));
        }
        if kind == "ElectricalSeries" {
            let region = format!("{p}/electrodes");
            let target = s.attr(&region, "table").and_then(|t| t.get("_REFERENCE")).and_then(|r| r.get("path")).and_then(Value::as_str);
            if target != Some(table) {
                issues.push(Issue::error(format!("{p}/electrodes does not reference {table}")));
            }
            match (s.read::<i64>(&region), electrode_rows) {
                (Some(idx), Some(n)) => {
                    if idx.iter().any(|&i| i < 0 || i as u64 >= n) {
                        issues.push(Issue::error(format!("{p}/electrodes has rows outside the electrodes table (0..{n})")));
                    }
                    if shape.get(1).copied().unwrap_or(1) != idx.len() as u64 {
                        issues.push(Issue::error(format!("{p}: {} channels but {} electrodes", shape.get(1).unwrap_or(&1), idx.len())));
                    }
                }
                _ => issues.push(Issue::error(format!("{p}/electrodes is missing or unreadable"))),
            }
        }
    }

    // Tables
    for p in s.children("/intervals").chain(s.children("/analysis")).collect::<Vec<_>>() {
        if matches!(s.kind(p), Some("TimeIntervals" | "DynamicTable")) {
            check_table(&s, p, &mut issues);
        }
        if s.kind(p) == Some("TimeIntervals") {
            if let (Some(a), Some(b)) = (s.read::<f64>(&format!("{p}/start_time")), s.read::<f64>(&format!("{p}/stop_time"))) {
                if a.iter().zip(&b).any(|(x, y)| y < x) {
                    issues.push(Issue::error(format!("{p}: an interval stops before it starts")));
                }
            }
        }
    }
    let _ = &s.root;
    Ok(issues)
}

/// Every listed column exists and has as many rows as `id`.
fn check_table(s: &Store, table: &str, issues: &mut Vec<Issue>) {
    let Some(rows) = s.shape(&format!("{table}/id")).map(|v| v[0]) else {
        issues.push(Issue::error(format!("{table}: no id column")));
        return;
    };
    let cols = s.attr(table, "colnames").and_then(Value::as_array).cloned().unwrap_or_default();
    for c in cols.iter().filter_map(Value::as_str) {
        match s.shape(&format!("{table}/{c}")) {
            Some(shape) if shape.first() == Some(&rows) => {}
            Some(shape) => issues.push(Issue::error(format!("{table}/{c}: {} rows, table has {rows}", shape.first().unwrap_or(&0)))),
            None => issues.push(Issue::error(format!("{table}: column {c} listed but missing"))),
        }
    }
}
