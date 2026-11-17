//! Curation served: what a server computes for the Curation of a page it serves (the page has no
//! files of its own; they are the server's). Calls by method, bodies and replies as JSON (but
//! `read`, raw bytes):
//!
//! - `files` [`FilesCall`] → [`SortingFiles`]: a `.sorting.zarr`'s files and their sizes;
//! - `read` [`ReadCall`] → bytes: a piece of one of them (a page reads the sorting in pieces, under
//!   its connection's message size, and computes the spike-train views itself);
//! - `views` [`ViewsCall`] → [`Views`]: the views read from the recording (waveforms, features),
//!   computed here on the server's own recording rather than sending its windows to the page;
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewsCall {
    pub path: String,
    /// The page's curation edits, saved or not.
    pub ops: Vec<CurationOp>,
    pub request: ViewsRequest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaveCall {
    pub path: String,
    pub ops: Vec<CurationOp>,
}

/// The server's side: sortings read once, and the open recording's windows read once.
#[derive(Default)]
pub struct Service {
    sortings: Mutex<HashMap<PathBuf, Arc<SortingData>>>,
    /// The windows of the last recording asked about, by its address.
    windows: Mutex<Option<(usize, Arc<Windows>)>>,
}

fn bad(e: impl std::fmt::Display) -> DspError {
    DspError::InvalidConfig(format!("curation call: {e}"))
}

impl Service {
    /// Answers `method` with `body`; `resolve` checks a path under the data root, `recording` is
    /// the one the client has open (as stored).
    pub fn call(&self, method: &str, body: &[u8], resolve: &dyn Fn(&str) -> DspResult<PathBuf>, recording: Option<&Arc<dyn RecordingSource>>) -> DspResult<Vec<u8>> {
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
                let windows = self.windows(recording);
                json(&from_signal(&sorting, &windows, &call.request, || false))
            }
            "save" => {
                let call: SaveCall = serde_json::from_slice(body).map_err(bad)?;
                let path = resolve(&call.path)?;
                let mut log = Curation { ops: call.ops, ..Default::default() };
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

    /// The windows of `recording`, read once (until another recording is asked about).
    fn windows(&self, recording: &Arc<dyn RecordingSource>) -> Arc<Windows> {
        let key = Arc::as_ptr(recording) as *const () as usize;
        let mut cache = self.windows.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((k, w)) = cache.as_ref() {
            if *k == key {
                return w.clone();
            }
        }
        let w = Arc::new(Windows::read_with(recording.info(), |channels, samples, out| recording.read(channels, samples, out)));
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
        let listed: SortingFiles = serde_json::from_slice(&service.call("files", br#"{"path": ""}"#, &resolve, None).unwrap()).unwrap();
        assert_eq!(listed.files.len(), 2);
        assert!(service.call("views", br#"{"path": "", "ops": [], "request": {}}"#, &resolve, None).is_err(), "a malformed call is refused");
        assert!(service.call("nothing", b"{}", &resolve, None).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
