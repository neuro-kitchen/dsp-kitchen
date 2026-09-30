use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{path}: {source}")]
    Io { path: PathBuf, source: std::io::Error },

    /// A file that claims to be `format` but does not follow it.
    #[error("{format}: {message}")]
    Format { format: &'static str, message: String },

    #[error("unsupported: {0}")]
    Unsupported(String),

    #[error("no reader recognizes {0}")]
    UnknownFormat(PathBuf),

    #[error("channel {channel} is outside 0..{total}")]
    Channel { channel: usize, total: usize },

    #[error("sample range {start}..{end} is outside 0..{total}")]
    SampleRange { start: u64, end: u64, total: u64 },

    #[error("output buffer holds {actual} values, the request needs {expected}")]
    BufferSize { expected: usize, actual: usize },
}

impl Error {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io { path: path.into(), source }
    }

    pub fn format(format: &'static str, message: impl Into<String>) -> Self {
        Error::Format { format, message: message.into() }
    }
}
