//! A recording's session metadata ([`crate::session_meta`]) as filled in, kept as the derived
//! store's root attribute `nwb_session`; what an NWB-Zarr file says itself, and writing it into
//! one (an NWB export: the fields it lacks, never over what it says).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use dspk_core::{DspError, DspResult};
use dspk_io::container::zarr::{open_rw_store, open_store, read_all, read_node_json, write_array_str, write_group};
use serde_json::{json, Map};

use crate::session_meta::{SessionMetadata, FIELDS};

/// The root attribute that holds it.
pub const ATTRIBUTE: &str = "nwb_session";

/// What was filled in for `recording` (none yet: empty).
pub fn read(recording: &Path) -> SessionMetadata {
    let dir = super::derived_store(recording);
    read_node_json(&dir, "/").and_then(|n| n.get("attributes")?.get(ATTRIBUTE).cloned()).and_then(|v| serde_json::from_value(v).ok()).unwrap_or_default()
}

/// Keeps `meta` for `recording` (the store's other attributes stay); returns the store.
pub fn write(recording: &Path, meta: &SessionMetadata) -> DspResult<PathBuf> {
    let dir = super::ensure(recording, &[])?;
    let mut attrs: Map<String, serde_json::Value> = read_node_json(&dir, "/").and_then(|n| n.get("attributes")?.as_object().cloned()).unwrap_or_default();
    attrs.insert(ATTRIBUTE.into(), serde_json::to_value(meta).map_err(|e| DspError::Io(e.to_string()))?);
    let store = open_rw_store(&dir)?;
    write_group(&store, &dir, "/", attrs)?;
    Ok(dir)
}

/// What an NWB-Zarr file (`*.nwb.zarr`) says of its session and subject (fields it lacks: empty).
pub fn from_nwb_zarr(dir: &Path) -> SessionMetadata {
    let mut out = SessionMetadata::default();
    let Ok(store) = open_store(dir) else { return out };
    let store: Arc<dyn zarrs::storage::ReadableStorageTraits> = store;
    for f in FIELDS {
        if let Some(values) = read_all::<String>(&store, &format!("/{}", f.nwb)) {
            let line = values.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect::<Vec<_>>().join("; ");
            if !line.is_empty() {
                out.set(f.key, &line);
            }
        }
    }
    out
}

/// Writes `meta`'s fields that the NWB-Zarr store `dir` lacks (the store and its groups are made
/// when missing). Returns how many fields were written.
pub fn write_nwb_zarr(dir: &Path, meta: &SessionMetadata) -> DspResult<usize> {
    let store = open_rw_store(dir)?;
    let group = |path: &str, kind: &str| -> DspResult<()> {
        if read_node_json(dir, path).is_none() {
            let mut attrs = Map::new();
            if !kind.is_empty() {
                attrs.insert("neurodata_type".into(), json!(kind));
                attrs.insert("namespace".into(), json!("core"));
            }
            write_group(&store, dir, path, attrs)?;
        }
        Ok(())
    };
    group("/", "NWBFile")?;
    let mut written = 0;
    for f in FIELDS {
        let path = format!("/{}", f.nwb);
        if read_node_json(dir, &path).is_some() {
            continue;
        }
        let list: Vec<String> = match f.key {
            "experimenter" => meta.experimenter.clone(),
            "keywords" => meta.keywords.clone(),
            k => Some(meta.get(k)).filter(|v| !v.is_empty()).into_iter().collect(),
        };
        if list.is_empty() {
            continue;
        }
        if f.nwb.starts_with("general/") {
            group("/general", "")?;
        }
        if f.nwb.starts_with("general/subject/") {
            group("/general/subject", "Subject")?;
        }
        let values: Vec<&str> = list.iter().map(String::as_str).collect();
        // Lists are 1-D (experimenter, keywords); the rest scalars, as NWB writes them
        let shape: Vec<usize> = if matches!(f.key, "experimenter" | "keywords") { vec![values.len()] } else { Vec::new() };
        write_array_str(&store, dir, &path, &values, &shape, Map::new())?;
        written += 1;
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What is filled in is kept beside the store's other attributes and read back.
    #[test]
    fn filled_in_fields_are_kept_and_read_back() {
        let dir = std::env::temp_dir().join(format!("dspk_session_meta_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let rec = dir.join("rec.ap.bin");
        assert_eq!(read(&rec), SessionMetadata::default());
        let mut meta = SessionMetadata::default();
        meta.set("experimenter", "Ada");
        meta.set("subject.species", "Mus musculus");
        write(&rec, &meta).unwrap();
        assert_eq!(read(&rec), meta);
        let root = read_node_json(&super::super::derived_store(&rec), "/").unwrap();
        assert!(root["attributes"]["dspk_derived"].is_number(), "the store's own attributes stay");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The NWB-Zarr example of the datasets folder names its session (when it is there).
    #[test]
    #[ignore = "needs .datasets/nwb/15-25-33_meps.nwb.zarr"]
    fn an_nwb_zarr_file_says_its_session() {
        let nwb = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.datasets/nwb/15-25-33_meps.nwb.zarr");
        let meta = from_nwb_zarr(&nwb);
        assert!(!meta.session_description.is_empty() && !meta.identifier.is_empty(), "{meta:?}");
    }

    /// The fields an NWB file lacks are written and read back; what it says already stays.
    #[test]
    fn missing_fields_are_written_and_read_back() {
        let dir = std::env::temp_dir().join(format!("dspk_nwb_session_{}.nwb.zarr", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut first = SessionMetadata::default();
        first.set("session_description", "from the file");
        assert_eq!(write_nwb_zarr(&dir, &first).unwrap(), 1);
        let mut filled = SessionMetadata::default();
        filled.set("session_description", "typed later");
        filled.set("identifier", "m1-2026-11-17");
        filled.set("experimenter", "Ada; Grace");
        filled.set("subject.species", "Mus musculus");
        assert_eq!(write_nwb_zarr(&dir, &filled).unwrap(), 3, "the description was there: kept");
        let back = from_nwb_zarr(&dir);
        assert_eq!(back.session_description, "from the file");
        assert_eq!((back.identifier.as_str(), back.experimenter.clone(), back.subject.species.as_str()), ("m1-2026-11-17", vec!["Ada".to_string(), "Grace".to_string()], "Mus musculus"));
        assert_eq!(read_node_json(&dir, "/general/subject").unwrap()["attributes"]["neurodata_type"], "Subject");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
