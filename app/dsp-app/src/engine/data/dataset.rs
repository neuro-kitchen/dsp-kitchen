//! The open recording as the UI sees it: a [`SignalBackend`] (today a [`LocalSignal`]: the
//! recording and its min/max pyramid in this process) plus the fields every frame reads. The
//! backend answers views and builds the pyramid; the app only asks and draws.

#[cfg(test)]
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use dsp_core::{DspResult, RecordingSource};
use dsp_io::neuro::probe::{SensorLayout, SensorSite};
use dsp_io::{SyntheticParams, SyntheticRecording};
use dsp_view::envelope::mean_range;
use dsp_view::{Envelope, LocalSignal, SignalBackend, View};

/// Channels per spiking unit of a procedural recording (each unit spreads over a few channels).
const PROCEDURAL_CHANNELS_PER_UNIT: usize = 4;
/// Most spiking units of a procedural recording.
const PROCEDURAL_MAX_UNITS: usize = 64;

/// A recording's backend plus the summary fields the UI reads every frame.
pub struct Dataset {
    signal: Arc<dyn SignalBackend>,
    pub total_channels: usize,
    pub total_samples: usize,
    /// Sample rate in Hz, for display and pixel arithmetic (the exact rate is in the backend's
    /// info).
    pub sample_rate: f64,
    pub name: String,
    /// Session time of sample 0, seconds (sources of one file can start at different times).
    pub start_time_sec: f64,
    /// Probe geometry of the recording, when its format stores one.
    pub probe: Option<SensorLayout>,
}

impl Dataset {
    /// The UI handle of `signal`, with the probe geometry of its file if known.
    pub fn new(signal: Arc<dyn SignalBackend>, probe: Option<SensorLayout>) -> Self {
        let info = signal.info();
        Self {
            total_channels: info.channel_count(),
            total_samples: info.samples as usize,
            sample_rate: info.sample_rate_hz(),
            name: info.name.clone(),
            start_time_sec: info.start_time.as_seconds_f64(),
            probe,
            signal,
        }
    }

    /// `source` viewed in this process ([`LocalSignal`]), with the pyramid of source `id` of the
    /// file `path` when given.
    pub fn local(source: Arc<dyn RecordingSource>, recording: Option<(&Path, &str)>) -> DspResult<Self> {
        Ok(Self::new(Arc::new(LocalSignal::open(source, recording)?), None))
    }

    /// The backend views ask.
    pub fn signal(&self) -> &Arc<dyn SignalBackend> {
        &self.signal
    }

    /// Physical sensor site of `channel` when the recording has a probe layout.
    pub fn site(&self, channel: usize) -> Option<&SensorSite> {
        self.probe.as_ref()?.get_site(channel).ok()
    }

    /// Distinct shank ids in ascending order when the recording has a probe layout.
    pub fn shanks(&self) -> Vec<usize> {
        let Some(probe) = &self.probe else { return Vec::new() };
        let mut shanks: Vec<usize> = probe.contacts.iter().map(|s| s.shank_id).collect();
        shanks.sort_unstable();
        shanks.dedup();
        shanks
    }

    /// Procedural recording of any length (noise, hum, drifting units), computed on demand.
    pub fn procedural(channels: usize, sample_rate: f64, duration_sec: f64) -> Result<Self> {
        let units = (channels / PROCEDURAL_CHANNELS_PER_UNIT).clamp(1, PROCEDURAL_MAX_UNITS);
        let rec = SyntheticRecording::new(SyntheticParams { channels, sample_rate_hz: sample_rate, duration_sec, units, ..Default::default() })?;
        Ok(Self::local(Arc::new(rec), None)?)
    }

    /// Wraps an in-memory channel-major buffer.
    #[cfg(test)]
    pub fn from_samples(name: &str, data: Vec<f32>, total_channels: usize, sample_rate: f64) -> Self {
        Self::local(Arc::new(dsp_core::MemoryRecording::new(name, data, total_channels, sample_rate).expect("valid shape")), None).expect("memory pyramid")
    }

    #[cfg(test)]
    pub fn total_duration_sec(&self) -> f64 {
        if self.sample_rate > 0.0 {
            self.total_samples as f64 / self.sample_rate
        } else {
            0.0
        }
    }

    /// One sample in µV (0 when out of range or unreadable).
    #[cfg(test)]
    pub fn sample(&self, channel: usize, sample: usize) -> f32 {
        let mut v = [0.0f32];
        let s = sample as u64;
        match self.signal.read(&[channel], s..s + 1, &mut v) {
            Ok(()) => v[0],
            Err(_) => 0.0,
        }
    }

    /// Every channel over `samples`, channel-major.
    #[cfg(test)]
    pub fn read_all(&self, samples: Range<usize>) -> DspResult<Vec<f32>> {
        let channels: Vec<usize> = (0..self.total_channels).collect();
        let mut out = vec![0.0f32; channels.len() * samples.len()];
        self.signal.read(&channels, samples.start as u64..samples.end as u64, &mut out)?;
        Ok(out)
    }

    /// Synthetic 32-channel-style signal (hum, noise, a spike every ~300 ms per channel) held in
    /// memory; tests rely on its exact content.
    #[cfg(test)]
    pub fn generate_synthetic(total_channels: usize, sample_rate: f64, duration_sec: f64) -> Self {
        let total_samples = (sample_rate * duration_sec) as usize;
        let mut raw_data = vec![0.0f32; total_channels * total_samples];

        let dt = 1.0 / sample_rate;
        let omega_60 = 2.0 * std::f64::consts::PI * 60.0;
        for ch in 0..total_channels {
            let ch_offset = ch * total_samples;
            let phase = (ch as f64 * 0.1).fract() * 2.0 * std::f64::consts::PI;
            for s in 0..total_samples {
                let t = s as f64 * dt;
                let hum = (25.0 * (omega_60 * t + phase).sin()) as f32;
                let noise = (((s * 37 + ch * 101) % 1000) as f32 / 1000.0 - 0.5) * 20.0;
                raw_data[ch_offset + s] = hum + noise;
            }

            // Inject action potentials every ~300ms
            let spike_interval = (sample_rate * 0.3).max(1.0) as usize;
            for sp in 1..(total_samples / spike_interval) {
                let center = sp * spike_interval + (ch * 17) % 500;
                if center + 40 < total_samples {
                    for i in 0..40 {
                        let t_rel = (i as f32 - 15.0) / 6.0;
                        let shape = -90.0 * (-0.5 * t_rel * t_rel).exp();
                        raw_data[ch_offset + center + i] += shape;
                    }
                }
            }
        }

        let rec = dsp_core::MemoryRecording::new(format!("synthetic_{total_channels}ch"), raw_data, total_channels.max(1), sample_rate)
            .expect("valid synthetic shape");
        Self::local(Arc::new(rec), None).expect("memory pyramid")
    }
}

/// Activity of the whole recording in `width` columns, for the timeline overview: the mean
/// min-to-max range over channels per column, scaled so the busiest column is 1. Columns whose
/// pyramid is not built yet are NaN.
pub fn activity(ds: &Dataset, width: usize) -> Vec<f32> {
    let total = ds.total_samples as u64;
    if width == 0 || ds.total_channels == 0 || total == 0 {
        return Vec::new();
    }
    let view = View { channels: (0..ds.total_channels).collect(), start: 0, end: total, width };
    let mut env = Envelope::Samples(Vec::new());
    if let Err(e) = ds.signal.view(&view, &mut env) {
        tracing::warn!("timeline activity unavailable: {e}");
        return Vec::new();
    }
    // A recording shorter than the overview is a few samples per channel: no activity strip
    let Envelope::Columns { values, .. } = env else { return Vec::new() };
    let mut out = mean_range(&values, ds.total_channels, width);
    let max = out.iter().copied().filter(|v| v.is_finite()).fold(0.0f32, f32::max);
    if max > 0.0 {
        out.iter_mut().for_each(|v| *v /= max);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opens_raw_file_through_dsp_io() {
        let dir = std::env::temp_dir().join(format!("dsp_app_ds_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rec.bin");
        let values: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, bytes).unwrap();
        std::fs::write(path.with_extension("meta"), r#"{"channels": 3, "sample_rate_hz": 1000.0}"#).unwrap();

        let source: Arc<dyn RecordingSource> = Arc::from(dsp_io::open(&path).unwrap());
        let ds = Dataset::local(source, None).unwrap();
        assert_eq!(ds.total_samples, 4);
        assert_eq!(ds.sample(1, 2), 6.0);
        assert_eq!(ds.read_all(1..3).unwrap(), vec![1.0, 2.0, 5.0, 6.0, 9.0, 10.0]);
        // Out of range reads are zero, not a panic
        assert_eq!(ds.sample(7, 0), 0.0);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_synthetic_dataset_generation() {
        let ds = Dataset::generate_synthetic(8, 10_000.0, 1.0);
        assert_eq!(ds.total_channels, 8);
        assert_eq!(ds.total_samples, 10_000);
        assert!((ds.total_duration_sec() - 1.0).abs() < 1e-6);

        let long = Dataset::procedural(384, 30_000.0, 4.0 * 3600.0).unwrap();
        assert_eq!(long.total_samples, 432_000_000);
    }

    /// Time to build a whole recording's pyramid in the background. Run with:
    /// `DSP_APP_BENCH_FILE=<recording> cargo test -p dsp-app --release bench_pyramid -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_pyramid() {
        let Some(path) = std::env::var_os("DSP_APP_BENCH_FILE") else { return };
        let sources = crate::engine::data::SourceSet::open(Path::new(&path)).unwrap();
        let ds = sources.default_dataset();
        let (tx, rx) = std::sync::mpsc::channel();
        ds.signal().watch(Arc::new(move |p: dsp_view::Progress| {
            if p.done >= p.total {
                let _ = tx.send(());
            }
        }));
        let t0 = std::time::Instant::now();
        ds.signal().focus(0);
        rx.recv().unwrap();
        let mb = ds.total_samples as f64 * ds.total_channels as f64 * 2.0 / 1e6;
        let secs = t0.elapsed().as_secs_f64();
        println!("{} ch × {} samples ({mb:.0} MB int16): {secs:.2} s ({:.0} MB/s)", ds.total_channels, ds.total_samples, mb / secs);
    }
}
