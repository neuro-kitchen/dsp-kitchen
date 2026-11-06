//! The files a server serves: everything under one data root, and nothing else.
//!
//! A client names a path relative to the root. [`DataRoot::resolve`] refuses absolute paths,
//! `..`, empty components and hidden ones (any component starting with `.`: the server's
//! authority lives in `.dspk-auth` under the root), then resolves links and refuses a path that
//! ends up outside the root. Listings leave out hidden entries and links that lead out.

use std::path::{Component, Path, PathBuf};
use std::time::UNIX_EPOCH;

use crate::error::{StreamError, StreamResult};
use crate::protocol::wire;

/// File extensions of recordings dspk opens (raw, SpikeGLX, NWB, mtscomp).
pub const RECORDING_EXTENSIONS: [&str; 6] = ["bin", "cbin", "dat", "raw", "nwb", "h5"];
/// Most entries one listing returns (a directory of more is cut, with the first ones).
pub const MAX_LIST_ENTRIES: usize = 10_000;

/// The data root of a server.
#[derive(Debug, Clone)]
pub struct DataRoot {
    /// Canonical (links resolved), so prefixes compare.
    root: PathBuf,
}

fn refused(path: &str, why: &str) -> StreamError {
    StreamError::Protocol(format!("{path:?}: {why}"))
}

impl DataRoot {
    /// The data root at `dir`, which must exist.
    pub fn new(dir: &Path) -> StreamResult<Self> {
        let root = dir.canonicalize().map_err(|e| StreamError::Protocol(format!("data root {}: {e}", dir.display())))?;
        if !root.is_dir() {
            return Err(StreamError::Protocol(format!("data root {} is not a directory", root.display())));
        }
        Ok(Self { root })
    }

    pub fn path(&self) -> &Path {
        &self.root
    }

    /// The existing file or directory `relative` names under the root ("" is the root).
    pub fn resolve(&self, relative: &str) -> StreamResult<PathBuf> {
        let rel = Path::new(relative);
        if rel.is_absolute() || relative.starts_with('/') || relative.starts_with('\\') {
            return Err(refused(relative, "paths are relative to the data root"));
        }
        for component in rel.components() {
            match component {
                Component::Normal(name) => {
                    let name = name.to_string_lossy();
                    if name.starts_with('.') {
                        return Err(refused(relative, "hidden files are not served"));
                    }
                }
                Component::CurDir => {}
                _ => return Err(refused(relative, "only plain names (no '..', no drive or root)")),
            }
        }
        let full = self.root.join(rel).canonicalize().map_err(|e| refused(relative, &e.to_string()))?;
        if !full.starts_with(&self.root) {
            return Err(refused(relative, "outside the data root"));
        }
        Ok(full)
    }

    /// A path for something new under the root: its folder exists under the root, its name is
    /// plain and not hidden, and nothing is there yet (never overwritten).
    pub fn resolve_new(&self, relative: &str) -> StreamResult<PathBuf> {
        let rel = Path::new(relative);
        let name = rel.file_name().map(|n| n.to_string_lossy().into_owned()).ok_or_else(|| refused(relative, "no file name"))?;
        if name.starts_with('.') || name.contains(['/', '\\']) {
            return Err(refused(relative, "not a plain name"));
        }
        let parent = rel.parent().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
        let dir = self.resolve(&parent)?;
        if !dir.is_dir() {
            return Err(refused(relative, "its folder is not a directory"));
        }
        let full = dir.join(&name);
        if full.symlink_metadata().is_ok() {
            return Err(refused(relative, "exists already (never overwritten)"));
        }
        Ok(full)
    }

    /// `path` (under the root) relative to the root, with `/` separators.
    pub fn relative(&self, path: &Path) -> String {
        path.strip_prefix(&self.root).map(|p| p.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect::<Vec<_>>().join("/")).unwrap_or_default()
    }

    /// The entries of directory `relative`: hidden entries and links out of the root left out,
    /// directories first, then by name; at most [`MAX_LIST_ENTRIES`].
    pub fn list(&self, relative: &str) -> StreamResult<wire::FileList> {
        let dir = self.resolve(relative)?;
        if !dir.is_dir() {
            return Err(refused(relative, "not a directory"));
        }
        let mut entries = Vec::new();
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            // Follow links, but only to places under the root
            let Ok(target) = entry.path().canonicalize() else { continue };
            if !target.starts_with(&self.root) {
                continue;
            }
            let Ok(meta) = std::fs::metadata(&target) else { continue };
            let modified_unix = meta.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs());
            let kind = kind_of(&target, meta.is_dir());
            entries.push(wire::FileEntry { name, kind: kind as i32, size: if meta.is_dir() { 0 } else { meta.len() }, modified_unix, directory: meta.is_dir() });
            if entries.len() >= MAX_LIST_ENTRIES {
                break;
            }
        }
        entries.sort_by(|a, b| (b.kind == wire::FileKind::Directory as i32).cmp(&(a.kind == wire::FileKind::Directory as i32)).then(a.name.cmp(&b.name)));
        Ok(wire::FileList { dir: self.relative(&dir), entries })
    }
}

/// What `path` is: a sorting (`*.sorting.zarr`), a recording (Zarr store or a recording
/// extension), a directory, or something else.
pub fn kind_of(path: &Path, is_dir: bool) -> wire::FileKind {
    let name = path.file_name().map(|n| n.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
    if name.ends_with(".sorting.zarr") {
        return wire::FileKind::Sorting;
    }
    if name.ends_with(".zarr") || name.ends_with(".nwb.zarr") {
        return wire::FileKind::Recording;
    }
    if is_dir {
        return wire::FileKind::Directory;
    }
    let ext = path.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
    if RECORDING_EXTENSIONS.contains(&ext.as_str()) { wire::FileKind::Recording } else { wire::FileKind::Other }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_plain_paths_under_the_root() {
        let base = std::env::temp_dir().join(format!("dspk-stream-files-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let root = base.join("data");
        std::fs::create_dir_all(root.join("mouse1/run.sorting.zarr")).unwrap();
        std::fs::create_dir_all(root.join(".dspk-auth")).unwrap();
        std::fs::write(root.join(".dspk-auth/ca.key.pem"), "secret").unwrap();
        std::fs::write(root.join("mouse1/rec.ap.bin"), [0u8; 8]).unwrap();
        std::fs::write(root.join("mouse1/rec.ap.meta"), "x").unwrap();
        std::fs::write(base.join("outside.bin"), [0u8; 8]).unwrap();
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(base.join("outside.bin"), root.join("link-out.bin")).unwrap();
            std::os::unix::fs::symlink(root.join("mouse1/rec.ap.bin"), root.join("link-in.bin")).unwrap();
        }
        let data = DataRoot::new(&root).unwrap();

        assert!(data.resolve("mouse1/rec.ap.bin").is_ok());
        assert!(data.resolve("").is_ok(), "the root itself");
        for bad in ["/etc/passwd", "../outside.bin", "mouse1/../../outside.bin", ".dspk-auth/ca.key.pem", "mouse1/.hidden", "missing.bin"] {
            assert!(data.resolve(bad).is_err(), "{bad} must be refused");
        }
        #[cfg(unix)]
        {
            assert!(data.resolve("link-out.bin").is_err(), "a link out of the root");
            assert!(data.resolve("link-in.bin").is_ok(), "a link within the root");
        }

        let top = data.list("").unwrap();
        let names: Vec<&str> = top.entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"mouse1") && !names.contains(&".dspk-auth") && !names.contains(&"link-out.bin"), "{names:?}");
        assert_eq!(top.entries[0].name, "mouse1", "directories first");
        let inner = data.list("mouse1").unwrap();
        let kind = |n: &str| inner.entries.iter().find(|e| e.name == n).map(|e| e.kind);
        assert_eq!(kind("rec.ap.bin"), Some(wire::FileKind::Recording as i32));
        assert_eq!(kind("run.sorting.zarr"), Some(wire::FileKind::Sorting as i32));
        assert_eq!(kind("rec.ap.meta"), Some(wire::FileKind::Other as i32));
        assert_eq!(inner.dir, "mouse1");
        assert!(data.resolve_new("mouse1/new.sorting.zarr").is_ok());
        for bad in ["mouse1/rec.ap.bin", "mouse1/.hidden", "../new", "missing/new", ".dspk-auth/x", "/tmp/new"] {
            assert!(data.resolve_new(bad).is_err(), "{bad} must be refused as a new path");
        }
        let _ = std::fs::remove_dir_all(&base);
    }
}
