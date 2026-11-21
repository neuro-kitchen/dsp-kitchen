//! Curation served: what a server computes for the Curation of a page it serves (the page has no
//! files of its own; they are the server's). Calls by method, bodies and replies as JSON (but
//! `read`, raw bytes):
//!
//! - `files` [`FilesCall`] → [`SortingFiles`]: a `.sorting.zarr`'s files and their sizes;
//! - `read` [`ReadCall`] → bytes: a piece of one of them (a page reads the sorting in pieces, under
//!   its connection's message size, and computes the spike-train views itself);
//! - `views` [`ViewsCall`] → [`Views`]: the views read from the recording (waveforms, features),
//!   computed here on the server's own recording rather than sending its windows to the page;
//! - `split` [`SplitCall`] → spike times (`Vec<u64>`): the lead unit's spikes inside a lasso,
//!   found here (a features lasso projects every spike of the unit, read from the recording as the
//!   views read it: one pass on the server, not a round trip per second of signal from the page);
//! - `save` [`SaveCall`]: the curation log written into the sorting, on the server.
//!
//! The page sends its curation edits with each `views` call, so units are numbered as on the page
//! (merges and splits add units), saved or not. Paths are under the server's data root, resolved
//! by the server (`resolve`); this module knows no transport.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use dspk_core::{DspError, DspResult, RecordingSource};
use serde::{Deserialize, Serialize};

use super::sorting::SortingData;
use super::views::{from_signal, ViewsRequest};
use super::waveforms::Windows;
use super::{apply, Curation, CurationOp};

/// The plugin's id, as clients name it.
pub const PLUGIN: &str = "synapse";
/// The largest piece `read` gives (under a browser connection's 1 MiB messages).
pub const READ_MAX: u64 = 512 << 10;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilesCall {
    /// The `.sorting.zarr`, under the data root.
    pub path: String,
}

/// A sorting's files: path inside the store, size in bytes.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SortingFiles {
    pub files: Vec<(String, u64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadCall {
    pub path: String,
    /// A file inside the store (as [`SortingFiles`] names it).
    pub file: String,
    pub offset: u64,
    pub len: u64,
}

/// A lasso to split the lead unit by: on the amplitude view (time s, amplitude) or on the
/// features view (PC1, PC2 of `basis`, which the page's views were computed with).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitCall {
    pub path: String,
    /// The page's curation edits, saved or not (units numbered as on the page).
    pub ops: Vec<CurationOp>,
    /// The unit split (its index after `ops`).
    pub unit: usize,
    pub polygon: Vec<(f32, f32)>,
    /// The features view's projection; `None`: an amplitude lasso.
    #[serde(default)]
    pub features: Option<super::views::FeatureBasis>,
    /// As [`ViewsCall::processing`]: the signal read as the views read it.
    #[serde(default)]
    pub processing: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewsCall {
    pub path: String,
    /// The page's curation edits, saved or not.
    pub ops: Vec<CurationOp>,
    pub request: ViewsRequest,
    /// The page's view processing (pipeline nodes in JSON, in order): waveforms and features
    /// from the signal as the page shows it. Applied by the server before the call reaches the
    /// service (which reads whatever recording it is given); empty: as stored.
    #[serde(default)]
    pub processing: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaveCall {
    pub path: String,
    pub ops: Vec<CurationOp>,
    /// When and where each edit was made (one per edit); the server adds who (the client's name).
    #[serde(default)]
    pub meta: Vec<super::EditMeta>,
    /// Edits undone and not replaced, as [`Curation::undone`] (the last undone last).
    #[serde(default)]
    pub undone: Vec<(CurationOp, super::EditMeta)>,
}

/// Who calls: the client's name as the server knows it, and whether it may write (paired with
/// control); saving is refused to a client that may only read.
#[derive(Debug, Clone, Copy)]
pub struct Caller<'a> {
    pub name: &'a str,
    pub may_write: bool,
}

/// The server's side: sortings read once, and the open recording's windows read once.
#[derive(Default)]
pub struct Service {
    sortings: Mutex<HashMap<PathBuf, Arc<SortingData>>>,
    /// The windows of the last recording asked about, by its address.
    windows: Mutex<Option<((usize, Option<std::ops::Range<u64>>), Arc<Windows>)>>,
}

fn bad(e: impl std::fmt::Display) -> DspError {
    DspError::InvalidConfig(format!("curation call: {e}"))
}

impl Service {
    /// Answers `method` with `body`; `resolve` checks a path under the data root, `recording` is
    /// the one the client has open (as stored).
    pub fn call(&self, method: &str, body: &[u8], resolve: &dyn Fn(&str) -> DspResult<PathBuf>, recording: Option<&Arc<dyn RecordingSource>>, caller: Caller) -> DspResult<Vec<u8>> {
        let json = |v: &dyn erased::Json| v.to_json();
        match method {
            "files" => {
                let call: FilesCall = serde_json::from_slice(body).map_err(bad)?;
                json(&files_of(&resolve(&call.path)?)?)
            }
            "read" => {
                let call: ReadCall = serde_json::from_slice(body).map_err(bad)?;
                read_piece(&resolve(&call.path)?, &call.file, call.offset, call.len)
            }
            "views" => {
                let call: ViewsCall = serde_json::from_slice(body).map_err(bad)?;
                let path = resolve(&call.path)?;
                let recording = recording.ok_or_else(|| bad("no recording is open on the server"))?;
                let sorting = apply(self.sorting(&path)?.as_ref(), &call.ops);
                let windows = self.windows(recording, sorting.span());
                json(&from_signal(&sorting, &windows, &call.request, || false))
            }
            "split" => {
                let call: SplitCall = serde_json::from_slice(body).map_err(bad)?;
                let path = resolve(&call.path)?;
                let sorting = apply(self.sorting(&path)?.as_ref(), &call.ops);
                if call.unit >= sorting.units.len() {
                    return Err(bad(format!("no unit {} in the sorting as curated", call.unit)));
                }
                let spikes = match &call.features {
                    None => {
                        if sorting.amplitudes.is_empty() {
                            return Err(bad("this sorting stores no spike amplitudes: split it on Features"));
                        }
                        super::split::spikes_in_polygon(&sorting, call.unit, &call.polygon)
                    }
                    Some(f) => {
                        let recording = recording.ok_or_else(|| bad("no recording is open on the server"))?;
                        let total = recording.info().samples;
                        super::split::spikes_in_feature_polygon(&sorting, call.unit, f.before, f.after, &f.basis, &call.polygon, total, |range, out| recording.read(&[f.channel], range, out))?
                    }
                };
                json(&spikes)
            }
            "save" => {
                if !caller.may_write {
                    return Err(bad(format!("{} may only read (paired without --control): the curation is not saved on the server", caller.name)));
                }
                let call: SaveCall = serde_json::from_slice(body).map_err(bad)?;
                let path = resolve(&call.path)?;
                // Who made each edit: the client that sends them, unless the edit says
                let by = |mut m: super::EditMeta| {
                    m.by.get_or_insert_with(|| caller.name.to_string());
                    m
                };
                let mut meta: Vec<_> = call.meta.into_iter().map(by).collect();
                meta.resize(call.ops.len(), by(super::EditMeta::default()));
                let undone = call.undone.into_iter().map(|(op, m)| (op, by(m))).collect();
                let mut log = Curation { ops: call.ops, meta, undone, saved: 0 };
                log.save(&path)?;
                Ok(Vec::new())
            }
            other => Err(bad(format!("no method {other:?}"))),
        }
    }

    /// The sorting at `path`, read once.
    fn sorting(&self, path: &Path) -> DspResult<Arc<SortingData>> {
        if let Some(s) = self.sortings.lock().unwrap_or_else(|e| e.into_inner()).get(path) {
            return Ok(s.clone());
        }
        let s = Arc::new(SortingData::load(path)?);
        self.sortings.lock().unwrap_or_else(|e| e.into_inner()).insert(path.to_path_buf(), s.clone());
        Ok(s)
    }

    /// The windows of `recording` over `span` (the sorting's), read once (until another recording
    /// or sorting is asked about).
    fn windows(&self, recording: &Arc<dyn RecordingSource>, span: Option<std::ops::Range<u64>>) -> Arc<Windows> {
        let key = (Arc::as_ptr(recording) as *const () as usize, span.clone());
        let mut cache = self.windows.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((k, w)) = cache.as_ref() {
            if *k == key {
                return w.clone();
            }
        }
        let w = Arc::new(Windows::read_with(recording.info(), span, |channels, samples, out| recording.read(channels, samples, out)));
        *cache = Some((key, w.clone()));
        w
    }
}

/// Every file of the store at `dir`, by its path inside it (with `/`), and its size.
pub fn files_of(dir: &Path) -> DspResult<SortingFiles> {
    let mut files = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).map_err(|e| DspError::Io(format!("{}: {e}", d.display())))? {
            let entry = entry.map_err(|e| DspError::Io(e.to_string()))?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if let Ok(rel) = path.strip_prefix(dir) {
                let key = rel.components().map(|c| c.as_os_str().to_string_lossy()).collect::<Vec<_>>().join("/");
                files.push((key, entry.metadata().map(|m| m.len()).unwrap_or(0)));
            }
        }
    }
    files.sort();
    Ok(SortingFiles { files })
}

/// `len` bytes (at most [`READ_MAX`]) of `file` inside the store `dir`, from `offset`.
pub fn read_piece(dir: &Path, file: &str, offset: u64, len: u64) -> DspResult<Vec<u8>> {
    use std::io::{Read as _, Seek as _, SeekFrom};
    // Inside the store only
    if file.split('/').any(|part| part == ".." || part.is_empty()) {
        return Err(bad(format!("{file:?} is not a file of the store")));
    }
    let path = dir.join(file);
    let mut f = std::fs::File::open(&path).map_err(|e| DspError::Io(format!("{}: {e}", path.display())))?;
    f.seek(SeekFrom::Start(offset)).map_err(|e| DspError::Io(e.to_string()))?;
    let mut out = Vec::new();
    f.take(len.min(READ_MAX)).read_to_end(&mut out).map_err(|e| DspError::Io(e.to_string()))?;
    Ok(out)
}

mod erased {
    /// A reply as JSON (one helper for every reply type).
    pub trait Json {
        fn to_json(&self) -> dspk_core::DspResult<Vec<u8>>;
    }

    impl<T: serde::Serialize> Json for T {
        fn to_json(&self) -> dspk_core::DspResult<Vec<u8>> {
            serde_json::to_vec(self).map_err(|e| dspk_core::DspError::Io(format!("curation reply: {e}")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sorting's files are listed, and read back in pieces as they are on disk; a path out of
    /// the store is refused.
    #[test]
    fn a_stores_files_list_and_read_in_pieces() {
        let dir = std::env::temp_dir().join(format!("dspk_curation_service_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("spikes/times/c")).unwrap();
        std::fs::write(dir.join("zarr.json"), "{}").unwrap();
        let big: Vec<u8> = (0..(READ_MAX as usize + 100)).map(|i| (i % 251) as u8).collect();
        std::fs::write(dir.join("spikes/times/c/0"), &big).unwrap();
        let files = files_of(&dir).unwrap();
        assert_eq!(files.files, vec![("spikes/times/c/0".to_string(), big.len() as u64), ("zarr.json".to_string(), 2)]);
        let first = read_piece(&dir, "spikes/times/c/0", 0, u64::MAX).unwrap();
        assert_eq!(first.len() as u64, READ_MAX, "at most READ_MAX at a time");
        let rest = read_piece(&dir, "spikes/times/c/0", READ_MAX, u64::MAX).unwrap();
        assert_eq!([first, rest].concat(), big);
        assert!(read_piece(&dir, "../outside", 0, 10).is_err());
        let service = Service::default();
        let resolve = |p: &str| -> DspResult<PathBuf> { Ok(dir.join(p)) };
        let listed: SortingFiles = serde_json::from_slice(&service.call("files", br#"{"path": ""}"#, &resolve, None, Caller { name: "test", may_write: true }).unwrap()).unwrap();
        assert_eq!(listed.files.len(), 2);
        assert!(service.call("views", br#"{"path": "", "ops": [], "request": {}}"#, &resolve, None, Caller { name: "test", may_write: true }).is_err(), "a malformed call is refused");
        assert!(service.call("nothing", b"{}", &resolve, None, Caller { name: "test", may_write: true }).is_err());
        // Saving: refused to a client that may only read; else each edit says who sent it
        let save = br#"{"path": "", "ops": [{"op": "label", "units": [1], "label": "good"}], "meta": [{"at": "2026-11-17T10:00:00Z", "via": "page"}]}"#;
        let refused = service.call("save", save, &resolve, None, Caller { name: "alice", may_write: false }).unwrap_err();
        assert!(refused.to_string().contains("alice may only read"), "{refused}");
        service.call("save", save, &resolve, None, Caller { name: "lab", may_write: true }).unwrap();
        let saved = super::super::Curation::load(&dir).unwrap();
        assert_eq!((saved.meta[0].by.as_deref(), saved.meta[0].via.as_deref()), (Some("lab"), Some("page")));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
