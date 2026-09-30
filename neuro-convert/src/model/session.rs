use std::sync::Arc;

use super::events::EventSeries;
use super::metadata::SessionMetadata;
use super::probe::ProbeGeometry;
use super::provenance::Provenance;
use super::recording::Recording;
use super::snippets::SnippetSeries;
use super::table::Table;

/// Everything read from one recording session: what inputs produce and outputs consume.
#[derive(Default)]
pub struct Session {
    pub metadata: SessionMetadata,
    pub recordings: Vec<Arc<dyn Recording>>,
    pub events: Vec<EventSeries>,
    pub snippets: Vec<SnippetSeries>,
    pub probes: Vec<ProbeGeometry>,
    pub tables: Vec<Table>,
    pub provenance: Provenance,
}

impl Session {
    pub fn recording(&self, name: &str) -> Option<&Arc<dyn Recording>> {
        self.recordings.iter().find(|r| r.info().name == name)
    }

    pub fn event_series(&self, name: &str) -> Option<&EventSeries> {
        self.events.iter().find(|e| e.name == name)
    }

    /// Longest recording duration in seconds.
    pub fn duration(&self) -> f64 {
        self.recordings.iter().map(|r| r.info().start_time + r.info().duration()).fold(0.0, f64::max)
    }
}
