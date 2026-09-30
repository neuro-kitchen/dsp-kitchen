//! Neutral data model: inputs map into a [`Session`], outputs map out of it.

pub mod events;
pub mod memory;
pub mod metadata;
pub mod probe;
pub mod provenance;
pub mod recording;
pub mod sample;
pub mod session;
pub mod snippets;
pub mod table;

pub use events::EventSeries;
pub use memory::MemoryRecording;
pub use metadata::{Device, SessionMetadata, Subject};
pub use probe::{Contact, ProbeGeometry};
pub use provenance::{Provenance, SourceFile};
pub use recording::{check_read, ChannelInfo, Recording, RecordingInfo, SignalKind};
pub use sample::{MemoryOrder, SampleType};
pub use session::Session;
pub use snippets::SnippetSeries;
pub use table::Table;
