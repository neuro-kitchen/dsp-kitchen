//! [`Format`] for NWB files stored as Zarr v3: one source per continuous `/acquisition` series.

use std::path::Path;

use dsp_core::{DspResult, RationalTime, RecordingSource, SampleRate, SignalUnit};

use super::acquisition::{is_nwb_zarr, list_series, NwbZarrRecording, SeriesEntry};
use crate::core::sources::{SourceEntry, SourceKind};
use crate::core::Format;

pub struct Nwb;

/// Unit of the values read from `series`: electrical and volt series are scaled to µV.
fn read_unit(series: &SeriesEntry) -> SignalUnit {
    if series.neurodata_type == "ElectricalSeries" || series.unit == "volts" {
        return SignalUnit::Microvolt;
    }
    match series.unit.as_str() {
        "a.u." | "" => SignalUnit::Dimensionless,
        other => SignalUnit::Other(other.to_string()),
    }
}

impl Format for Nwb {
    fn name(&self) -> &'static str {
        "nwb"
    }

    fn detect(&self, path: &Path) -> bool {
        is_nwb_zarr(path)
    }

    fn sources(&self, path: &Path) -> DspResult<Vec<SourceEntry>> {
        list_series(path)
            .into_iter()
            .map(|s| {
                let unit = read_unit(&s);
                Ok(SourceEntry {
                    name: s.path.rsplit('/').next().unwrap_or(&s.path).to_string(),
                    kind: if s.neurodata_type == "ElectricalSeries" { SourceKind::Electrical } else { SourceKind::from_unit(&unit) },
                    channels: s.channels,
                    samples: s.samples,
                    sample_rate: SampleRate::new(s.rate)?,
                    start_time: RationalTime::from_seconds_f64(s.start_time)?,
                    format: s.format,
                    unit,
                    id: s.path,
                })
            })
            .collect()
    }

    fn open(&self, path: &Path, id: &str) -> DspResult<Box<dyn RecordingSource>> {
        Ok(Box::new(NwbZarrRecording::open_series(path, id)?))
    }

    /// The largest `ElectricalSeries`, else the largest series.
    fn open_default(&self, path: &Path) -> DspResult<Box<dyn RecordingSource>> {
        Ok(Box::new(NwbZarrRecording::open(path)?))
    }
}
