//! Event sets of a recording: `<recording>.dspk/events/<name>.parquet` ([`crate::data::derived`]),
//! the table a `crossings` pipeline writes (`dspk_pipeline::events_frame`: `channel`,
//! `channel_name`, `onset_sample`, `offset_sample`, `peak_sample`, `peak_value`, `onset_s`,
//! `offset_s`, `duration_s`). The app reads the columns it draws and aligns to.

use std::path::Path;

use dspk_core::{DspError, DspResult};

/// One event, in seconds from the recording's start.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Event {
    pub channel: u32,
    pub onset_s: f64,
    /// `None`: the run ended inside the event.
    pub offset_s: Option<f64>,
    pub peak_value: f32,
}

/// The events of the table at `path`, by onset.
pub fn read(path: &Path) -> DspResult<Vec<Event>> {
    use polars::prelude::*;
    let io = |e: &dyn std::fmt::Display| DspError::Io(format!("{}: {e}", path.display()));
    let file = std::fs::File::open(path).map_err(|e| io(&e))?;
    let df = ParquetReader::new(file).finish().map_err(|e| io(&e))?;
    let channel: Vec<u32> = df.column("channel").and_then(|c| c.u32().cloned()).map_err(|e| io(&e))?.iter().map(|v| v.unwrap_or(0)).collect();
    let onset: Vec<f64> = df.column("onset_s").and_then(|c| c.f64().cloned()).map_err(|e| io(&e))?.iter().map(|v| v.unwrap_or(0.0)).collect();
    let offset: Vec<Option<f64>> = df.column("offset_s").and_then(|c| c.f64().cloned()).map_err(|e| io(&e))?.iter().collect();
    let peak: Vec<f32> = df.column("peak_value").and_then(|c| c.f32().cloned()).map_err(|e| io(&e))?.iter().map(|v| v.unwrap_or(0.0)).collect();
    let mut out: Vec<Event> = (0..df.height()).map(|i| Event { channel: channel[i], onset_s: onset[i], offset_s: offset[i], peak_value: peak[i] }).collect();
    out.sort_by(|a, b| a.onset_s.total_cmp(&b.onset_s));
    Ok(out)
}

/// Writes `events` (a selection of a set) as a table at `path`, in the pipeline's columns
/// (samples at `sample_rate`; channel names from `names`).
pub fn write(path: &Path, events: &[Event], names: &[String], sample_rate: f64) -> DspResult<()> {
    let io = |e: &dyn std::fmt::Display| DspError::Io(format!("{}: {e}", path.display()));
    let to_sample = |s: f64| (s * sample_rate).round().max(0.0) as u64;
    let rows: Vec<dspk_ops::crossings::CrossingEvent> = events
        .iter()
        .map(|e| dspk_ops::crossings::CrossingEvent {
            channel: e.channel as usize,
            onset: to_sample(e.onset_s),
            offset: e.offset_s.map(to_sample),
            peak_sample: to_sample(e.onset_s),
            peak_value: e.peak_value,
        })
        .collect();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| io(&e))?;
    }
    dspk_pipeline::write_events(path, &rows, names, sample_rate).map_err(|e| io(&e))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_round_trip_through_parquet() {
        let dir = std::env::temp_dir().join(format!("dspk_events_{}", std::process::id()));
        let path = dir.join("rec.dspk/events/biceps.parquet");
        let ev = vec![Event { channel: 1, onset_s: 0.5, offset_s: Some(0.75), peak_value: 3.0 }, Event { channel: 0, onset_s: 0.25, offset_s: None, peak_value: -2.0 }];
        write(&path, &ev, &["a".into(), "b".into()], 1000.0).unwrap();
        let back = read(&path).unwrap();
        assert_eq!(back, vec![ev[1], ev[0]], "by onset");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
