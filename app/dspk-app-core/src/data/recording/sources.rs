//! The signals of the open file (e.g. HD-EMG, EMG, temperature of an NWB store), listed by the
//! plugin that reads the format (else dspk-io's generic formats) from metadata when the file opens; each is opened the first time a view shows it, then cached.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use dspk_core::SignalUnit;
use dspk_io::{CachedRecording, SourceEntry, SourceKind};

use super::dataset::Dataset;

/// Decoded chunks kept per opened source (memory bound): views, hover readouts and repeated
/// frames over the same region decode each chunk once.
const CHUNK_CACHE_BYTES: usize = 128 << 20;

pub struct SourceSet {
    path: Option<PathBuf>,
    entries: Vec<SourceEntry>,
    default: usize,
    opened: Mutex<Vec<Option<Arc<Dataset>>>>,
}

impl SourceSet {
    /// Lists `path`'s sources and opens the default one (the largest electrical signal). A
    /// `dspk://` path is a recording on a server: one source, drawn from the server's pyramid and
    /// read there.
    pub fn open(path: &Path) -> Result<Self> {
        // A recording on a server: the desktop app's own connection, or the page's
        if crate::data::remote::parse_uri(path).is_some() {
            let signal = crate::data::remote::open_uri(path)?;
            // Every source the server's file holds (an NWB store's series); each opened the first
            // time a view shows it, beside the others on the connection
            let listed = crate::data::remote::sources_uri(&signal, path).unwrap_or_else(|e| {
                tracing::warn!("the server did not list the file's sources: {e:#}");
                Vec::new()
            });
            let source: Arc<dyn dspk_core::RecordingSource> = signal.source();
            let first = Arc::new(Dataset::new(signal, None).with_source(source));
            if listed.len() <= 1 {
                let mut set = Self::single(Arc::try_unwrap(first).unwrap_or_else(|_| unreachable!("just made")));
                set.path = Some(path.to_path_buf());
                return Ok(set);
            }
            let default = dspk_io::default_source(&listed).and_then(|d| listed.iter().position(|e| e.id == d.id)).unwrap_or(0);
            let mut opened = vec![None; listed.len()];
            opened[default] = Some(first);
            // The others opened now, here (a worker: opening waits on the server), not when a view
            // first shows one: that is the page's main thread, which may not wait (switching
            // sources froze the page)
            for (i, e) in listed.iter().enumerate() {
                if i == default {
                    continue;
                }
                match crate::data::remote::open_uri_source(path, Some(&e.id)) {
                    Ok(signal) => {
                        let source: Arc<dyn dspk_core::RecordingSource> = signal.source();
                        opened[i] = Some(Arc::new(Dataset::new(signal, None).with_source(source)));
                    }
                    Err(err) => tracing::warn!("source {} of the server's file: {err:#}", e.name),
                }
            }
            return Ok(Self { path: Some(path.to_path_buf()), entries: listed, default, opened: Mutex::new(opened) });
        }
        let entries = crate::plugin::sources(path).with_context(|| format!("Failed to open {}", dspk_session::paths::short_path(&path)))?;
        let default_id = dspk_io::default_source(&entries).map(|e| e.id.clone()).unwrap_or_default();
        let default = entries.iter().position(|e| e.id == default_id).unwrap_or(0);
        let set = Self { path: Some(path.to_path_buf()), opened: Mutex::new(vec![None; entries.len()]), entries, default };
        set.load(default)?;
        Ok(set)
    }

    /// A single in-memory or procedural recording.
    pub fn single(dataset: Dataset) -> Self {
        let info = dataset.signal().info();
        let entry = SourceEntry {
            id: dspk_io::sources::MAIN.into(),
            name: dataset.name.clone(),
            kind: SourceKind::Electrical,
            channels: dataset.total_channels,
            samples: dataset.total_samples as u64,
            sample_rate: info.sample_rate,
            start_time: info.start_time,
            format: info.format,
            unit: info.channels.first().map_or(SignalUnit::Dimensionless, |c| c.unit.clone()),
        };
        Self { path: None, entries: vec![entry], default: 0, opened: Mutex::new(vec![Some(Arc::new(dataset))]) }
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
        let entry = &self.entries[i];
        // A source of a file on a server: opened there, drawn from its pyramid
        if crate::data::remote::parse_uri(path).is_some() {
            let signal = crate::data::remote::open_uri_source(path, Some(&entry.id)).with_context(|| format!("Failed to open source {} on the server", entry.name))?;
            let source: Arc<dyn dspk_core::RecordingSource> = signal.source();
            let ds = Arc::new(Dataset::new(signal, None).with_source(source));
            self.opened.lock().unwrap()[i] = Some(ds.clone());
            return Ok(ds);
        }
        let rec = crate::plugin::open_source(path, &entry.id).with_context(|| format!("Failed to open source {}", entry.name))?;
        let source: Arc<dyn dspk_core::RecordingSource> = Arc::from(CachedRecording::wrap(rec, CHUNK_CACHE_BYTES));
        let signal = dspk_view::LocalSignal::open(source.clone(), Some((path, &entry.id))).with_context(|| format!("Failed to prepare {} for viewing", entry.name))?;
        let layout = crate::plugin::layout(path, &entry.id);
        let ds = Arc::new(Dataset::new(Arc::new(signal), layout).with_source(source));
        self.opened.lock().unwrap()[i] = Some(ds.clone());
        Ok(ds)
    }

    /// Latest end time over all sources (the shared timeline's length).
    pub fn extent_sec(&self) -> f64 {
        self.entries.iter().map(|e| e.start_time.as_seconds_f64() + e.duration_sec()).fold(0.0, f64::max)
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
