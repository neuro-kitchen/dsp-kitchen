//! Input formats known to this build, and how a path is matched to one.

use std::path::Path;

use crate::error::{Error, Result};
use crate::model::Session;
use crate::options::OpenOptions;

/// A reader for one family of files.
pub trait InputFormat: Sync {
    /// Short id used on the command line (`tdt`).
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    /// Human-readable list of the versions / variants this reader handles.
    fn versions(&self) -> &'static [&'static str];
    /// Whether `path` looks like this format.
    fn detect(&self, path: &Path) -> Option<Detection>;
    fn open(&self, path: &Path, options: &OpenOptions) -> Result<Session>;
}

/// A reader's claim on a path.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub format: &'static str,
    pub version: Option<String>,
    /// 0..1; the highest claim wins.
    pub confidence: f32,
}

/// Every input format compiled into this build.
pub fn inputs() -> Vec<&'static dyn InputFormat> {
    vec![
        #[cfg(feature = "tdt")]
        &crate::inputs::tdt::Tdt,
    ]
}

/// Readers claiming `path`, best first.
pub fn detect(path: &Path) -> Vec<Detection> {
    let mut found: Vec<Detection> = inputs().iter().filter_map(|f| f.detect(path)).collect();
    found.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    found
}

/// Opens `path` with the reader that claims it most confidently.
pub fn open(path: &Path, options: &OpenOptions) -> Result<Session> {
    let best = detect(path).into_iter().next().ok_or_else(|| Error::UnknownFormat(path.to_path_buf()))?;
    let format = inputs().into_iter().find(|f| f.name() == best.format).expect("detected by a registered format");
    format.open(path, options)
}
