//! A recording's epochs, saved after every change of them and read when the recording opens:
//! `<recording>.dspk/epochs.parquet` ([`crate::data::derived`]), one row per epoch (`id`, `name`,
//! `kind`, `start_s`, `end_s`, `notes`; seconds from the recording's start).
//!
//! Earlier versions wrote `<recording>.epochs.zarr` next to the recording (arrays `/start_us`,
//! `/end_us`; ids, names, types and notes in the root attributes): still read when there is no
//! Parquet table yet, and left in place.

use std::path::{Path, PathBuf};

use dspk_core::{DspError, DspResult};
use dspk_io::container::zarr::{has_array, open_rw_store, read_array, read_node_json, write_array_u64, write_group};
use serde_json::{json, Map, Value};

use crate::domain::epoch::EpochDoc;

/// Format version written in the root attributes.
pub const EPOCHS_VERSION: u32 = 1;

/// Saves `recording`'s epochs (all of them) in its derived folder, replacing what is there.
pub fn save(recording: &Path, epochs: &[EpochDoc]) -> DspResult<PathBuf> {
    use polars::prelude::*;
    let path = crate::data::derived::epochs_file(recording);
    let io = |e: &dyn std::fmt::Display| DspError::Io(format!("{}: {e}", path.display()));
    let col = |f: fn(&EpochDoc) -> String| epochs.iter().map(f).collect::<Vec<String>>();
    let mut df = df!(
        "id" => col(|e| e.id.clone()),
        "name" => col(|e| e.name.clone()),
        "kind" => col(|e| e.kind.clone()),
        "start_s" => epochs.iter().map(|e| e.start_sec).collect::<Vec<f64>>(),
        "end_s" => epochs.iter().map(|e| e.end_sec).collect::<Vec<f64>>(),
        "notes" => col(|e| e.notes.clone()),
    )
    .map_err(|e| io(&e))?;
    std::fs::create_dir_all(path.parent().expect("a derived folder")).map_err(|e| io(&e))?;
    let file = std::fs::File::create(&path).map_err(|e| io(&e))?;
    ParquetWriter::new(file).finish(&mut df).map_err(|e| io(&e))?;
    Ok(path)
}

/// `recording`'s saved epochs: the Parquet table, else the older `.epochs.zarr`; none when
/// neither exists.
pub fn load(recording: &Path) -> DspResult<Vec<EpochDoc>> {
    use polars::prelude::*;
    let path = crate::data::derived::epochs_file(recording);
    if !path.exists() {
        return read_zarr(&legacy_path(recording), recording);
    }
    let io = |e: &dyn std::fmt::Display| DspError::Io(format!("{}: {e}", path.display()));
    let file = std::fs::File::open(&path).map_err(|e| io(&e))?;
    let df = ParquetReader::new(file).finish().map_err(|e| io(&e))?;
    let text = |name: &str| -> DspResult<Vec<String>> {
        Ok(df.column(name).and_then(|c| c.str().cloned()).map_err(|e| io(&e))?.iter().map(|v| v.unwrap_or_default().to_string()).collect())
    };
    let secs = |name: &str| -> DspResult<Vec<f64>> { Ok(df.column(name).and_then(|c| c.f64().cloned()).map_err(|e| io(&e))?.iter().map(|v| v.unwrap_or(0.0)).collect()) };
    let (id, name, kind, notes) = (text("id")?, text("name")?, text("kind")?, text("notes")?);
    let (start, end) = (secs("start_s")?, secs("end_s")?);
    Ok((0..df.height())
        .map(|i| EpochDoc {
            id: id[i].clone(),
            recording: recording.to_path_buf(),
            name: name[i].clone(),
            kind: Some(kind[i].clone()).filter(|k| !k.is_empty()).unwrap_or_else(|| "other".into()),
            start_sec: start[i],
            end_sec: end[i],
            notes: notes[i].clone(),
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
        // Saved as Parquet in the derived folder, which then wins
        let mut f = e.clone();
        f.name = "tone 2".into();
        assert_eq!(save(&rec, std::slice::from_ref(&f)).unwrap(), dir.join("rec.dspk/epochs.parquet"));
        assert_eq!(load(&rec).unwrap(), vec![f]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
