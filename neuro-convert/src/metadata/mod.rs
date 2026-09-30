//! User-supplied metadata: loaded from YAML, merged over what the input read, and checked by
//! each output for the fields it requires.

pub mod file;

pub use file::{ElectrodeGroupSpec, ImpedanceSpec, ItemSpec, MetadataFile, StreamSpec, StreamType};

/// Severity of a problem found while planning an output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    /// The output cannot be written.
    Error,
    /// Written, but incomplete for archives such as DANDI.
    Warning,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Issue {
    pub level: Level,
    pub message: String,
}

impl Issue {
    pub fn error(message: impl Into<String>) -> Self {
        Self { level: Level::Error, message: message.into() }
    }

    pub fn warning(message: impl Into<String>) -> Self {
        Self { level: Level::Warning, message: message.into() }
    }
}
