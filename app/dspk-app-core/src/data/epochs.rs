//! A recording's epochs next to it, `<recording>.epochs.zarr`: arrays `/start_us` and `/end_us`
//! (microseconds from the recording's start, `uint64`) and, in the root attributes, the epochs'
//! ids, names, types and notes in the same order (`dspk_epochs`: format version). Written after
//! every change of the recording's epochs; read when the recording opens.

use std::path::{Path, PathBuf};

use dspk_core::{DspError, DspResult};
use dspk_io::container::zarr::{has_array, open_rw_store, read_array, read_node_json, write_array_u64, write_group};
use serde_json::{json, Map, Value};

use crate::domain::epoch::EpochDoc;

/// Format version written in the root attributes.
pub const EPOCHS_VERSION: u32 = 1;

/// `rec.bin` → `rec.epochs.zarr` next to it.
pub fn epochs_path(recording: &Path) -> PathBuf {
    let name = recording.file_name().map_or_else(|| "recording".into(), |n| n.to_string_lossy().into_owned());
    let stem = name.trim_end_matches('/').trim_end_matches(".zarr").trim_end_matches(".nwb").trim_end_matches(".bin").trim_end_matches(".cbin");
    recording.with_file_name(format!("{stem}.epochs.zarr"))
}

/// Writes `epochs` (all of one recording) into `path` (replacing what is there).
pub fn write(path: &Path, epochs: &[EpochDoc]) -> DspResult<()> {
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

/// The epochs stored at `path`, marking `recording`; none when there is no store.
pub fn read(path: &Path, recording: &Path) -> DspResult<Vec<EpochDoc>> {
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
    fn test_epochs_round_trip() {
        let dir = std::env::temp_dir().join(format!("dspk_epochs_{}", std::process::id()));
        let rec = dir.join("rec.bin");
        std::fs::create_dir_all(&dir).unwrap();
        let path = epochs_path(&rec);
        assert_eq!(path, dir.join("rec.epochs.zarr"));
        let e = EpochDoc { id: "e1".into(), recording: rec.clone(), name: "tone".into(), kind: "stimulus".into(), start_sec: 1.25, end_sec: 2.5, notes: "60 dB".into() };
        write(&path, std::slice::from_ref(&e)).unwrap();
        assert_eq!(read(&path, &rec).unwrap(), vec![e]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
