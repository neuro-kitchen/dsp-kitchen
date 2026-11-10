//! Event sets of a recording: the groups `events/<name>` of `<recording>.dspk.zarr`
//! ([`crate::data::derived`]), in the layout a `crossings` pipeline writes to a `.zarr` path
//! (`dspk_pipeline::write_events_zarr`: arrays `channel`, `onset_sample`, `offset_sample`,
//! `peak_sample`, `peak_value`; the sample rate and channel names as attributes). The app reads
//! the events in seconds, to draw them and align to them.

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

/// The events of the group at `dir`, by onset.
pub fn read(dir: &Path) -> DspResult<Vec<Event>> {
    let (events, rate, _) = dspk_pipeline::read_events_zarr(dir).map_err(|e| DspError::Io(e.to_string()))?;
    let secs = |s: u64| s as f64 / rate;
    let mut out: Vec<Event> =
        events.iter().map(|e| Event { channel: e.channel as u32, onset_s: secs(e.onset), offset_s: e.offset.map(secs), peak_value: e.peak_value }).collect();
    out.sort_by(|a, b| a.onset_s.total_cmp(&b.onset_s));
    Ok(out)
}

/// Writes `events` (e.g. a selection of a set) as event set `name` of `recording`, at
/// `sample_rate` with channel names `names`. Returns the group.
pub fn write(recording: &Path, name: &str, events: &[Event], names: &[String], sample_rate: f64) -> DspResult<std::path::PathBuf> {
    crate::data::derived::ensure(recording, &["/events"])?;
    let group = crate::data::derived::events_group(recording, name);
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
    dspk_pipeline::write_events_zarr(&group, &rows, names, sample_rate).map_err(|e| DspError::Io(e.to_string()))?;
    Ok(group)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_round_trip_through_the_derived_store() {
        let dir = std::env::temp_dir().join(format!("dspk_events_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let rec = dir.join("rec.bin");
        let ev = vec![Event { channel: 1, onset_s: 0.5, offset_s: Some(0.75), peak_value: 3.0 }, Event { channel: 0, onset_s: 0.25, offset_s: None, peak_value: -2.0 }];
        let group = write(&rec, "biceps onsets", &ev, &["a".into(), "b".into()], 1000.0).unwrap();
        assert_eq!(group, dir.join("rec.dspk.zarr/events/biceps_onsets"));
        assert_eq!(read(&group).unwrap(), vec![ev[1], ev[0]], "by onset");
        // A second set beside it, and the epochs group, share the store
        write(&rec, "triceps", &ev[..1], &["a".into(), "b".into()], 1000.0).unwrap();
        assert_eq!(read(&group).unwrap().len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
