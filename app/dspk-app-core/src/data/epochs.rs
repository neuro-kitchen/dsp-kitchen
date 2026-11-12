//! A recording's epochs, saved after every change of them and read when the recording opens: the
//! `epochs` group of `<recording>.dspk.zarr` ([`crate::data::derived`]): arrays `start_s` and
//! `end_s` (seconds from the recording's start, `float64`); ids, names, types and notes as
//! attributes in the same order (`dspk_epochs`: format version).
//!
//! Earlier versions wrote `<recording>.epochs.zarr` next to the recording (arrays `/start_us`,
//! `/end_us`; ids, names, types and notes in the root attributes): still read when there is no
//! Parquet table yet, and left in place.

use std::path::{Path, PathBuf};

use dspk_core::{DspError, DspResult};
use dspk_io::container::zarr::{has_array, open_rw_store, read_array, read_node_attributes, read_node_json, write_array_f64, write_array_u64, write_group};
use serde_json::{json, Map, Value};

use crate::domain::epoch::EpochDoc;

/// Format version written in the root attributes.
pub const EPOCHS_VERSION: u32 = 1;

/// Saves `recording`'s epochs (all of them) as the `epochs` group of its derived store, replacing
/// what is there. Returns the store.
pub fn save(recording: &Path, epochs: &[EpochDoc]) -> DspResult<PathBuf> {
    let dir = crate::data::derived::ensure(recording, &[])?;
    let group = dir.join("epochs");
    if group.exists() {
        std::fs::remove_dir_all(&group).map_err(|e| DspError::Io(format!("{}: {e}", crate::domain::paths::short_path(&group))))?;
    }
    let store = open_rw_store(&dir)?;
    let list = |f: fn(&EpochDoc) -> String| Value::Array(epochs.iter().map(|e| json!(f(e))).collect());
    let mut attrs = Map::new();
    attrs.insert("dspk_epochs".into(), json!(EPOCHS_VERSION));
    attrs.insert("ids".into(), list(|e| e.id.clone()));
    attrs.insert("names".into(), list(|e| e.name.clone()));
    attrs.insert("kinds".into(), list(|e| e.kind.clone()));
    attrs.insert("notes".into(), list(|e| e.notes.clone()));
    write_group(&store, &dir, "/epochs", attrs)?;
    let n = [epochs.len()];
    write_array_f64(&store, &dir, "/epochs/start_s", &epochs.iter().map(|e| e.start_sec).collect::<Vec<_>>(), &n, &["epochs"], Map::new())?;
    write_array_f64(&store, &dir, "/epochs/end_s", &epochs.iter().map(|e| e.end_sec).collect::<Vec<_>>(), &n, &["epochs"], Map::new())?;
    Ok(dir)
}

/// `recording`'s saved epochs: the derived store's `epochs` group, else the older
/// `.epochs.zarr`; none when neither exists.
pub fn load(recording: &Path) -> DspResult<Vec<EpochDoc>> {
    let dir = crate::data::derived::derived_store(recording);
    if !has_array(&dir, "/epochs/start_s") {
        return read_zarr(&legacy_path(recording), recording);
    }
    let attrs = read_node_attributes(&dir, "/epochs").unwrap_or_default();
    let text = |key: &str, i: usize| attrs.get(key).and_then(|v| v.get(i)).and_then(Value::as_str).unwrap_or_default().to_string();
    let (start, end) = (read_array::<f64>(&dir, "/epochs/start_s")?.data, read_array::<f64>(&dir, "/epochs/end_s")?.data);
    Ok(start
        .iter()
        .zip(&end)
        .enumerate()
        .map(|(i, (&a, &b))| EpochDoc {
            id: text("ids", i),
            recording: recording.to_path_buf(),
            name: text("names", i),
            kind: Some(text("kinds", i)).filter(|k| !k.is_empty()).unwrap_or_else(|| "other".into()),
            start_sec: a,
            end_sec: b,
            notes: text("notes", i),
        })
        .collect())
}

/// `rec.bin` → `rec.epochs.zarr` next to it (where earlier versions saved epochs).
pub fn legacy_path(recording: &Path) -> PathBuf {
    recording.with_file_name(format!("{}.epochs.zarr", crate::data::derived::stem(recording)))
}

/// Writes `epochs` as the older Zarr store at `path` (replacing what is there); tests and
/// migration checks only.
pub fn write_zarr(path: &Path, epochs: &[EpochDoc]) -> DspResult<()> {
    if path.exists() {
        std::fs::remove_dir_all(path).map_err(|e| DspError::Io(e.to_string()))?;
    }
    let store = open_rw_store(path)?;
    let list = |f: fn(&EpochDoc) -> String| Value::Array(epochs.iter().map(|e| json!(f(e))).collect());
    let mut root = Map::new();
    root.insert("dspk_epochs".into(), json!(EPOCHS_VERSION));
    root.insert("ids".into(), list(|e| e.id.clone()));
    root.insert("names".into(), list(|e| e.name.clone()));
    root.insert("kinds".into(), list(|e| e.kind.clone()));
    root.insert("notes".into(), list(|e| e.notes.clone()));
    write_group(&store, path, "/", root)?;
    let us = |s: f64| (s * 1e6).round().max(0.0) as u64;
    let n = epochs.len();
    write_array_u64(&store, path, "/start_us", &epochs.iter().map(|e| us(e.start_sec)).collect::<Vec<_>>(), &[n], &["epochs"], Map::new())?;
    write_array_u64(&store, path, "/end_us", &epochs.iter().map(|e| us(e.end_sec)).collect::<Vec<_>>(), &[n], &["epochs"], Map::new())?;
    Ok(())
}

/// The epochs of the older Zarr store at `path`, marking `recording`; none when there is no store.
pub fn read_zarr(path: &Path, recording: &Path) -> DspResult<Vec<EpochDoc>> {
    if !has_array(path, "/start_us") {
        return Ok(Vec::new());
    }
    let root = read_node_json(path, "/").unwrap_or(Value::Null);
    let attrs = root.get("attributes").unwrap_or(&root);
    let text = |key: &str, i: usize| attrs.get(key).and_then(|v| v.get(i)).and_then(Value::as_str).unwrap_or_default().to_string();
    let (start, end) = (read_array::<u64>(path, "/start_us")?.data, read_array::<u64>(path, "/end_us")?.data);
    Ok(start
        .iter()
        .zip(&end)
        .enumerate()
        .map(|(i, (&a, &b))| EpochDoc {
            id: text("ids", i),
            recording: recording.to_path_buf(),
            name: text("names", i),
            kind: Some(text("kinds", i)).filter(|k| !k.is_empty()).unwrap_or_else(|| "other".into()),
            start_sec: a as f64 / 1e6,
            end_sec: b as f64 / 1e6,
            notes: text("notes", i),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epochs_round_trip_and_legacy() {
        let dir = std::env::temp_dir().join(format!("dspk_epochs_{}", std::process::id()));
        let rec = dir.join("rec.bin");
        std::fs::create_dir_all(&dir).unwrap();
        let e = EpochDoc { id: "e1".into(), recording: rec.clone(), name: "tone".into(), kind: "stimulus".into(), start_sec: 1.25, end_sec: 2.5, notes: "60 dB".into() };
        // An older store is read while there is no table
        assert_eq!(legacy_path(&rec), dir.join("rec.epochs.zarr"));
        write_zarr(&legacy_path(&rec), std::slice::from_ref(&e)).unwrap();
        assert_eq!(load(&rec).unwrap(), vec![e.clone()]);
        // Saved in the derived store, which then wins; saving again replaces the group
        let mut f = e.clone();
        f.name = "tone 2".into();
        assert_eq!(save(&rec, &[e.clone(), f.clone()]).unwrap(), dir.join("rec.dspk.zarr"));
        assert_eq!(save(&rec, std::slice::from_ref(&f)).unwrap(), dir.join("rec.dspk.zarr"));
        assert_eq!(load(&rec).unwrap(), vec![f]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
