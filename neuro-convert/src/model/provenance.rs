use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Where a session came from, carried into outputs and conversion reports.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    /// Input format name (`tdt`, `spikeglx`, …).
    pub format: String,
    /// Detected format version (e.g. `Synapse 53575`).
    pub version: Option<String>,
    pub files: Vec<SourceFile>,
    /// `neuro-convert <version>`.
    pub reader: String,
    /// Recoverable problems met while reading (damaged index, missing files, …).
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceFile {
    pub path: PathBuf,
    pub bytes: u64,
}

impl Provenance {
    pub fn new(format: &str) -> Self {
        Self { format: format.into(), reader: format!("neuro-convert {}", env!("CARGO_PKG_VERSION")), ..Default::default() }
    }

    pub fn add_file(&mut self, path: impl Into<PathBuf>) {
        let path = path.into();
        let bytes = std::fs::metadata(&path).map_or(0, |m| m.len());
        self.files.push(SourceFile { path, bytes });
    }
}
