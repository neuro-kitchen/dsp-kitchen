//! Locating the files of one TDT block.

use std::path::{Path, PathBuf};

/// Files of one block. A block folder holds `<tank>_<block>.{tsq,tev,Tbk,Tdx,tin,tnt}`, plus
/// optional Synapse sidecars (`Notes.txt`, `StoresListing.txt`), per-channel `.sev` files
/// and exported tables (`*.csv`).
#[derive(Debug, Clone, Default)]
pub struct BlockFiles {
    pub dir: PathBuf,
    /// Block name (folder name).
    pub name: String,
    pub tsq: PathBuf,
    pub tev: Option<PathBuf>,
    pub tbk: Option<PathBuf>,
    pub tdx: Option<PathBuf>,
    pub tin: Option<PathBuf>,
    pub tnt: Option<PathBuf>,
    pub notes: Option<PathBuf>,
    pub stores_listing: Option<PathBuf>,
    pub sev: Vec<PathBuf>,
    pub csv: Vec<PathBuf>,
}

impl BlockFiles {
    /// Finds the block for a block folder or any of its files; `None` without a `.tsq`.
    pub fn find(path: &Path) -> Option<Self> {
        let dir = if path.is_dir() { path.to_path_buf() } else { path.parent()?.to_path_buf() };
        let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir).ok()?.filter_map(|e| e.ok().map(|e| e.path())).collect();
        entries.sort();
        let ext_is = |p: &Path, ext: &str| p.extension().is_some_and(|e| e.eq_ignore_ascii_case(ext));
        // macOS resource forks (`._name`) sit next to real files on shared drives
        let real = |p: &&PathBuf| !p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("._"));

        let tsq = entries.iter().filter(real).find(|p| ext_is(p, "tsq"))?.clone();
        let sibling = |ext: &str| {
            let candidate = entries.iter().filter(real).find(|p| ext_is(p, ext) && p.file_stem() == tsq.file_stem());
            candidate.cloned()
        };
        let named = |name: &str| entries.iter().find(|p| p.file_name().is_some_and(|n| n.eq_ignore_ascii_case(name))).cloned();

        Some(Self {
            name: dir.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned()),
            tev: sibling("tev"),
            tbk: sibling("tbk"),
            tdx: sibling("tdx"),
            tin: sibling("tin"),
            tnt: sibling("tnt"),
            notes: named("Notes.txt"),
            stores_listing: named("StoresListing.txt"),
            sev: entries.iter().filter(real).filter(|p| ext_is(p, "sev")).cloned().collect(),
            csv: entries.iter().filter(real).filter(|p| ext_is(p, "csv")).cloned().collect(),
            tsq,
            dir,
        })
    }

    /// Every file that exists, for provenance.
    pub fn all(&self) -> Vec<&Path> {
        let mut v: Vec<&Path> = vec![&self.tsq];
        for p in [&self.tev, &self.tbk, &self.tdx, &self.tin, &self.tnt, &self.notes, &self.stores_listing].into_iter().flatten() {
            v.push(p);
        }
        v.extend(self.sev.iter().map(PathBuf::as_path));
        v.extend(self.csv.iter().map(PathBuf::as_path));
        v
    }
}
