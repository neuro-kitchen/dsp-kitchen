//! The signals ("sources") a file offers, listed cheaply and opened on demand.
//!
//! A multi-signal file (e.g. an NWB store with several series, a SpikeGLX run with several
//! streams) lists one entry per signal; plain recordings are one source, [`MAIN`]. Listing reads
//! only metadata.

use std::path::Path;

use dsp_core::{DspError, DspResult, RationalTime, RecordingSource, SampleFormat, SampleRate, SignalUnit};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    /// Electric potential (values in a volt unit).
    Electrical,
    /// Anything else (pressure, temperature, analog inputs, …).
    Other,
}

impl SourceKind {
    /// [`SourceKind::Electrical`] for volt units, else [`SourceKind::Other`].
    pub fn from_unit(unit: &SignalUnit) -> Self {
        match unit {
            SignalUnit::Volt | SignalUnit::Millivolt | SignalUnit::Microvolt => SourceKind::Electrical,
            _ => SourceKind::Other,
        }
    }
}

/// One openable signal of a file.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceEntry {
    /// Stable id within the file (e.g. NWB series path, SpikeGLX file name, [`MAIN`]).
    pub id: String,
    pub name: String,
    pub kind: SourceKind,
    pub channels: usize,
    pub samples: u64,
    pub sample_rate: SampleRate,
    pub start_time: RationalTime,
    /// Stored sample type.
    pub format: SampleFormat,
    /// Unit of the values a read returns.
    pub unit: SignalUnit,
}

impl SourceEntry {
    pub fn duration_sec(&self) -> f64 {
        self.samples as f64 / self.sample_rate.rate_hz()
    }

    /// e.g. `HDEMG · 32 ch · 24.4 kHz · µV`
    pub fn summary(&self) -> String {
        let hz = self.sample_rate.rate_hz();
        let rate = if hz >= 1000.0 { format!("{:.1} kHz", hz / 1000.0) } else { format!("{hz:.1} Hz") };
        format!("{} · {} ch · {rate} · {}", self.name, self.channels, self.unit.symbol())
    }
}

/// The id of the only source of a single-recording file.
pub const MAIN: &str = "main";

/// The [`MAIN`] entry describing an opened single-recording file.
pub fn single_source(rec: &dyn RecordingSource) -> SourceEntry {
    let info = rec.info();
    let unit = info.channels.first().map(|c| c.unit.clone()).unwrap_or_default();
    SourceEntry {
        id: MAIN.into(),
        name: info.name.clone(),
        kind: SourceKind::from_unit(&unit),
        channels: info.channel_count(),
        samples: info.samples,
        sample_rate: info.sample_rate,
        start_time: info.start_time,
        format: info.format,
        unit,
    }
}

/// `Ok` when `id` is [`MAIN`], for single-recording formats.
pub fn require_main(path: &Path, id: &str) -> DspResult<()> {
    if id == MAIN {
        Ok(())
    } else {
        Err(DspError::InvalidConfig(format!("{} has no source {id}", path.display())))
    }
}

/// The source to show first: the largest electrical source, else the largest one.
pub fn default_source(list: &[SourceEntry]) -> Option<&SourceEntry> {
    // Most data first; equal sizes: the higher sample rate (e.g. SpikeGLX ap over lf)
    let size = |s: &&SourceEntry| (s.samples * s.channels as u64, s.sample_rate.as_ratio());
    list.iter().filter(|s| s.kind == SourceKind::Electrical).max_by_key(size).or_else(|| list.iter().max_by_key(size))
}
