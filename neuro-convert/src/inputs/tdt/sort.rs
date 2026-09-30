//! Offline spike-sort results: `sort/<sort id>/<store>.SortResult` (TDT OpenSorter).
//!
//! Layout: 1024 bytes of per-channel flags (non-zero = channel sorted), then one `u8` sort code
//! per TSQ record, indexed by the record's position among named records ([`Record::seq`]).
//!
//! [`Record::seq`]: super::tsq::Record::seq

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};


const CHANNEL_MAP_BYTES: usize = 1024;

/// Sort ids under `sort/`, each with the stores it has results for.
pub fn available(block_dir: &Path) -> BTreeMap<String, BTreeMap<String, PathBuf>> {
    let mut out = BTreeMap::new();
    let Ok(ids) = std::fs::read_dir(block_dir.join("sort")) else { return out };
    for id in ids.flatten().filter(|e| e.path().is_dir()) {
        let stores: BTreeMap<String, PathBuf> = std::fs::read_dir(id.path())
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "SortResult"))
            .filter_map(|p| Some((p.file_name()?.to_string_lossy().split('.').next()?.to_string(), p)))
            .collect();
        if !stores.is_empty() {
            out.insert(id.file_name().to_string_lossy().into_owned(), stores);
        }
    }
    out
}

/// Sort codes of a `.SortResult` file, indexed by [`Record::seq`](super::tsq::Record::seq).
pub fn load(file: &Path) -> std::io::Result<Vec<u8>> {
    let bytes = std::fs::read(file)?;
    Ok(bytes.get(CHANNEL_MAP_BYTES..).unwrap_or(&[]).to_vec())
}
