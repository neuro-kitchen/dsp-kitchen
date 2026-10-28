//! Format-independent entry points: the [`Format`] trait, detection and opening, source listing,
//! and chunk caching.

mod cached;
mod format;
mod merged;
mod open;
pub mod sources;

pub use cached::CachedRecording;
pub use format::Format;
pub use merged::MergedRecording;
pub use open::{detect, detect_in, open, open_source, sources};
pub use sources::{default_source, single_source, SourceEntry, SourceKind, MAIN};

