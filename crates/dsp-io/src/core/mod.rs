//! Format-independent entry points: the [`Format`] trait, detection and opening, source listing,
//! and chunk caching.

mod cached;
mod format;
mod open;
pub mod sources;

pub use cached::CachedRecording;
pub use format::Format;
pub use open::{detect, open, open_source, sources};
pub use sources::{default_source, single_source, SourceEntry, SourceKind, MAIN};

#[cfg(all(test, feature = "neuro"))]
pub(crate) use open::tests;
