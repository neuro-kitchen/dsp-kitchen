//! Format-independent entry points: the [`Format`] trait, detection and opening, source listing,
//! chunk caching, and double-buffered out-of-core window reading.

mod cached;
mod format;
mod open;
mod prefetch;
pub mod sources;

pub use cached::CachedRecording;
pub use format::Format;
pub use open::{detect, open, open_source, sources};
pub use prefetch::PrefetchReader;
pub use sources::{default_source, single_source, SourceEntry, SourceKind, MAIN};

#[cfg(all(test, feature = "neuro"))]
pub(crate) use open::tests;
