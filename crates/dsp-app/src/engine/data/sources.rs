//! The signals of the open file (e.g. HD-EMG, EMG, temperature of an NWB store). Listed from
//! metadata when the file opens; each is opened the first time a view shows it, then cached.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use dsp_io::{CachedRecording, SourceEntry, SourceKind};

use super::dataset::Dataset;

/// Decoded chunks kept per opened source (memory bound): views, hover readouts and repeated
/// frames over the same region decode each chunk once.
const CHUNK_CACHE_BYTES: usize = 128 << 20;

/// Recordings at least this large automatically build their persistent min/max cache file in the
/// background when opened.
const AUTO_LOD_BYTES: u64 = 64 << 20;

pub struct SourceSet {
    path: Option<PathBuf>,
    entries: Vec<SourceEntry>,
    default: usize,
    opened: Mutex<Vec<Option<Arc<Dataset>>>>,
    /// The user asked for min/max caches: sources opened later build theirs too.
    build_caches: AtomicBool,
}

impl SourceSet {
    /// Lists `path`'s sources and opens the default one (the largest electrical signal).
    pub fn open(path: &Path) -> Result<Self> {
        let entries = dsp_io::sources(path).with_context(|| format!("Failed to open {}", path.display()))?;
        let default_id = dsp_io::default_source(&entries).map(|e| e.id.clone()).unwrap_or_default();
        let default = entries.iter().position(|e| e.id == default_id).unwrap_or(0);
        let set = Self { path: Some(path.to_path_buf()), opened: Mutex::new(vec![None; entries.len()]), entries, default, build_caches: AtomicBool::new(false) };
        set.load(default)?;
        Ok(set)
    }

    /// A single in-memory or procedural recording.
    pub fn single(dataset: Dataset) -> Self {
        let entry = SourceEntry {
            id: dsp_io::sources::MAIN.into(),
            name: dataset.name.clone(),
            kind: SourceKind::Electrical,
            channels: dataset.total_channels,
            samples: dataset.total_samples as u64,
            sample_rate: dataset.sample_rate,
            start_time_sec: dataset.start_time_sec,
            unit: dataset.unit.clone(),
        };
        Self { path: None, entries: vec![entry], default: 0, opened: Mutex::new(vec![Some(Arc::new(dataset))]), build_caches: AtomicBool::new(false) }
    }

    pub fn entries(&self) -> &[SourceEntry] {
        &self.entries
    }

    pub fn default_entry(&self) -> &SourceEntry {
        &self.entries[self.default]
    }

    /// Index of source `id`; the default for an empty or unknown id.
    pub fn index_of(&self, id: &str) -> usize {
        self.entries.iter().position(|e| e.id == id).unwrap_or(self.default)
    }

    pub fn entry(&self, id: &str) -> &SourceEntry {
        &self.entries[self.index_of(id)]
    }

    pub fn default_dataset(&self) -> Arc<Dataset> {
        self.load(self.default).expect("default source opened in SourceSet::open")
    }

    /// Source `id` (the default when empty, unknown or unreadable).
    pub fn get(&self, id: &str) -> Arc<Dataset> {
        let i = self.index_of(id);
        match self.load(i) {
            Ok(ds) => ds,
            Err(e) => {
                tracing::warn!("{e:#}");
                self.default_dataset()
            }
        }
    }

    fn load(&self, i: usize) -> Result<Arc<Dataset>> {
        if let Some(ds) = &self.opened.lock().unwrap()[i] {
            return Ok(ds.clone());
        }
        let path = self.path.as_deref().context("in-memory source set has one source")?;
        let rec = dsp_io::open_source(path, &self.entries[i].id).with_context(|| format!("Failed to open source {}", self.entries[i].name))?;
        let large = rec.info().data_bytes() >= AUTO_LOD_BYTES;
        let mut ds = Dataset::new(Arc::from(CachedRecording::wrap(rec, CHUNK_CACHE_BYTES)));
        ds.unit = self.entries[i].unit.clone();
        ds.open_lod(path, &self.entries[i].id);
        if self.build_caches.load(Ordering::Relaxed) {
            ds.build_lod(self.recording(i));
        } else if large {
            // Large recordings get a cache file for next time, written after the summary
            ds.cache_after_summary(self.recording(i));
        }
        let ds = Arc::new(ds);
        self.opened.lock().unwrap()[i] = Some(ds.clone());
        Ok(ds)
    }

    /// Builds the min/max cache of every source (user request): the opened ones now, the others
    /// when first shown.
    pub fn build_caches(&self) {
        self.build_caches.store(true, Ordering::Relaxed);
        let opened = self.opened.lock().unwrap().clone();
        for (i, ds) in opened.iter().enumerate() {
            if let Some(ds) = ds {
                ds.build_lod(self.recording(i));
            }
        }
    }

    /// The file and source id of source `i` (`None` for in-memory sets).
    fn recording(&self, i: usize) -> Option<(PathBuf, String)> {
        self.path.clone().map(|p| (p, self.entries[i].id.clone()))
    }

    /// Latest end time over all sources (the shared timeline's length).
    pub fn extent_sec(&self) -> f64 {
        self.entries.iter().map(|e| e.start_time_sec + e.duration_sec()).fold(0.0, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_source_set() {
        let set = SourceSet::single(Dataset::generate_synthetic(4, 1000.0, 2.0));
        assert_eq!(set.entries().len(), 1);
        assert_eq!(set.get("").total_channels, 4);
        assert_eq!(set.get("unknown").total_channels, 4, "unknown ids fall back to the default");
        assert!((set.extent_sec() - 2.0).abs() < 1e-9);
    }
}
