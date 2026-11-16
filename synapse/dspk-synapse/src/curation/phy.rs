//! Phy folders around the working format. A `.sorting.zarr` is what curation works on; a Phy /
//! Kilosort folder comes in by [`import_phy`] (a store written next to it, which remembers the
//! folder) and a curated sorting goes back by [`export_phy`], which writes only what phy's own
//! save writes (`spike_clusters.npy`, `cluster_group.tsv`, `cluster_info.tsv`), so the folder
//! stays valid in phy.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use dspk_core::error::{DspError, DspResult};

use super::sorting::{start_sec, SortingData};
use crate::io::phy::{ClusterId, PhyFolder};

/// In a store imported from a Phy folder: where it came from (`{"phy_folder": "<path>"}`).
pub const PHY_ORIGIN_FILE: &str = "dspk-phy.json";

/// The store a Phy folder is imported into: `<folder>.sorting.zarr` next to it.
pub fn import_path(folder: &Path) -> PathBuf {
    let name = folder.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "phy".into());
    folder.with_file_name(format!("{name}.sorting.zarr"))
}

/// The Phy folder the store at `store` was imported from, when it still exists.
pub fn phy_origin(store: &Path) -> Option<PathBuf> {
    let text = std::fs::read_to_string(store.join(PHY_ORIGIN_FILE)).ok()?;
    let folder = PathBuf::from(serde_json::from_str::<serde_json::Value>(&text).ok()?.get("phy_folder")?.as_str()?);
    PhyFolder::is_phy_folder(&folder).then_some(folder)
}

/// Imports the Phy / Kilosort folder `folder` as a `.sorting.zarr` at [`import_path`] and returns
/// it. A store already there is kept (its curation log with it), not written again.
///
/// # Errors
///
/// Fails when `folder` is not a Phy folder, or when the store cannot be written.
pub fn import_phy(folder: &Path) -> DspResult<PathBuf> {
    if !PhyFolder::is_phy_folder(folder) {
        return Err(DspError::Io(format!("{} is not a Phy folder (no spike_times.npy)", folder.display())));
    }
    let store = import_path(folder);
    if !store.exists() {
        let sorting = crate::storage::load_phy_folder(folder)?;
        crate::storage::save_sorting_zarr(&sorting, &store)?;
    }
    let folder = std::fs::canonicalize(folder).unwrap_or_else(|_| folder.to_path_buf());
    std::fs::write(store.join(PHY_ORIGIN_FILE), serde_json::json!({ "phy_folder": folder }).to_string()).map_err(|e| DspError::Io(format!("{}: {e}", store.display())))?;
    Ok(store)
}

/// Writes `curated` (the sorting of the store `store`, with its curation applied) into the Phy
/// folder `dir`. A folder that is not a Phy folder yet first gets the whole sorting as the sorter
/// left it; then the curation is written over it: the cluster of every spike, the labels and the
/// cluster table. Files replaced keep their original as `<name>.bak`. Returns the clusters written.
///
/// # Errors
///
/// Fails when the folder's spikes are not this sorting's (another count or other times), or when
/// a file cannot be written.
pub fn export_phy(store: &Path, curated: &SortingData, dir: &Path) -> DspResult<usize> {
    if !PhyFolder::is_phy_folder(dir) {
        crate::storage::save_phy_folder(&crate::storage::load_sorting_zarr(store)?, dir)?;
    }
    let folder = PhyFolder::read(dir)?;
    if folder.spike_times.len() != curated.times.len() {
        return Err(DspError::Io(format!("{} holds {} spikes, the sorting {}: not the same sorting", dir.display(), folder.spike_times.len(), curated.times.len())));
    }
    // The sorting's spikes are in time order (ties in the folder's order); the folder's own
    // order is kept in the file
    let mut order: Vec<usize> = (0..folder.spike_times.len()).collect();
    order.sort_by_key(|&i| folder.spike_times[i]);
    let offset = start_sec(store).map_or(0, |s| (s * curated.sample_rate).round().max(0.0) as u64);
    let mut clusters: Vec<ClusterId> = folder.spike_clusters.clone();
    clusters.resize(folder.spike_times.len(), 0);
    for (k, &i) in order.iter().enumerate() {
        if folder.spike_times[i] + offset != curated.times[k] {
            return Err(DspError::Io(format!("{}: spike {i} is at sample {}, the sorting has {}: not the same sorting", dir.display(), folder.spike_times[i], curated.times[k].saturating_sub(offset))));
        }
        // A spike of a cluster the store does not list keeps the folder's cluster
        if let Some(u) = curated.unit_of[k] {
            clusters[i] = curated.units[u as usize].info.unit_id as ClusterId;
        }
    }
    let mut groups = BTreeMap::new();
    let mut rows = BTreeMap::new();
    for u in curated.alive() {
        let info = &curated.units[u].info;
        let n = curated.by_unit.get(u).map_or(info.num_spikes, Vec::len);
        if n == 0 {
            continue;
        }
        let id = info.unit_id as ClusterId;
        let label = info.quality_label.trim();
        if !label.is_empty() && label != "unsorted" {
            groups.insert(id, label.to_string());
        }
        let mut row = BTreeMap::new();
        row.insert("group".to_string(), label.to_string());
        row.insert("n_spikes".to_string(), n.to_string());
        row.insert("ch".to_string(), info.primary_channel.to_string());
        rows.insert(id, row);
    }
    let columns: Vec<String> = ["cluster_id", "group", "n_spikes", "ch"].map(String::from).to_vec();
    PhyFolder::write_curation(dir, &clusters, &groups, &columns, &rows)?;
    Ok(rows.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curation::{apply, CurationOp};

    fn scratch(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("dspk-curation-phy-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// A Phy folder of 3 clusters on 4 channels, 9 spikes.
    fn phy_folder(dir: &Path) {
        let folder = PhyFolder {
            spike_times: vec![10, 20, 30, 40, 50, 60, 70, 80, 90],
            spike_clusters: vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
            spike_templates: vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
            amplitudes: vec![1.0; 9],
            ..PhyFolder::default()
        };
        let mut folder = folder;
        folder.params.sample_rate = 30000.0;
        folder.params.n_channels_dat = 4;
        folder.fill_channel_defaults();
        folder.write(dir).unwrap();
    }

    #[test]
    fn a_phy_folder_is_imported_once_and_curation_goes_back_into_it() {
        let root = scratch("round");
        let dir = root.join("ks");
        phy_folder(&dir);
        let store = import_phy(&dir).unwrap();
        assert_eq!(store, root.join("ks.sorting.zarr"));
        assert_eq!(phy_origin(&store), Some(std::fs::canonicalize(&dir).unwrap()));
        // Importing again keeps the store (and what was saved in it)
        std::fs::write(store.join("marker"), "kept").unwrap();
        assert_eq!(import_phy(&dir).unwrap(), store);
        assert!(store.join("marker").exists());

        let base = SortingData::load(&store).unwrap();
        assert_eq!((base.times.len(), base.units.len()), (9, 3));
        let ops = vec![CurationOp::Label { units: vec![2], label: "noise".into() }, CurationOp::Merge { units: vec![0, 1], into: 3 }];
        let curated = apply(&base, &ops);
        assert_eq!(export_phy(&store, &curated, &dir).unwrap(), 2, "clusters 2 and the merged 3");

        let back = PhyFolder::read(&dir).unwrap();
        assert_eq!(back.spike_clusters, vec![3, 3, 2, 3, 3, 2, 3, 3, 2]);
        assert_eq!(back.tables.group.get(&2).map(String::as_str), Some("noise"));
        assert!(!back.tables.group.contains_key(&3), "a merged cluster is unsorted, as in phy");
        assert!(dir.join("spike_clusters.npy.bak").exists(), "the sorter's clusters are kept");
    }

    #[test]
    fn exporting_to_a_new_folder_writes_the_whole_sorting_and_another_sorting_is_refused() {
        let root = scratch("new");
        let dir = root.join("ks");
        phy_folder(&dir);
        let store = import_phy(&dir).unwrap();
        let base = SortingData::load(&store).unwrap();
        let curated = apply(&base, &[CurationOp::Label { units: vec![0], label: "good".into() }]);
        let out = root.join("exported");
        assert_eq!(export_phy(&store, &curated, &out).unwrap(), 3);
        let back = PhyFolder::read(&out).unwrap();
        assert_eq!(back.spike_times, vec![10, 20, 30, 40, 50, 60, 70, 80, 90]);
        assert_eq!(back.tables.group.get(&0).map(String::as_str), Some("good"));

        // A folder of another sorting: nothing is written
        let other = root.join("other");
        let mut f = PhyFolder::read(&dir).unwrap();
        f.spike_times[0] = 11;
        f.write(&other).unwrap();
        assert!(export_phy(&store, &curated, &other).is_err());
        assert!(!other.join("spike_clusters.npy.bak").exists());
    }
}
