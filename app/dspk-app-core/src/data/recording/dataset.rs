//! The open recording as the UI sees it: a [`SignalBackend`] (today a [`LocalSignal`]: the
//! recording and its min/max pyramid in this process) plus the fields every frame reads. The
//! backend answers views and builds the pyramid; the app only asks and draws.

#[cfg(test)]
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use dspk_core::{DspResult, RecordingSource};
use dspk_view::envelope::mean_range;
use dspk_view::{Envelope, LocalSignal, SignalBackend, View};

use crate::plugin::ChannelLayout;


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
    /// Channel groups (e.g. a probe's shanks) and labels, when a plugin knows them.
    pub layout: Option<ChannelLayout>,
    /// The recording itself (stored-value reads for the processed preview), when local.
    pub source: Option<Arc<dyn RecordingSource>>,
}

impl Dataset {
    /// The UI handle of `signal`, with its channel layout if known.
    pub fn new(signal: Arc<dyn SignalBackend>, layout: Option<ChannelLayout>) -> Self {
        let info = signal.info();
        Self {
            total_channels: info.channel_count(),
            total_samples: info.samples as usize,
            sample_rate: info.sample_rate_hz(),
            name: info.name.clone(),
            start_time_sec: info.start_time.as_seconds_f64(),
            layout,
            source: None,
            signal,
        }
    }

    /// The same, keeping `source` for stored-value reads (the processed preview).
    pub fn with_source(mut self, source: Arc<dyn RecordingSource>) -> Self {
        self.source = Some(source);
        self
    }

    /// `source` viewed in this process ([`LocalSignal`]), with the pyramid of source `id` of the
    /// file `path` when given.
    pub fn local(source: Arc<dyn RecordingSource>, recording: Option<(&Path, &str)>) -> DspResult<Self> {
        Ok(Self::new(Arc::new(LocalSignal::open(source.clone(), recording)?), None).with_source(source))
    }

    /// The backend views ask.
    pub fn signal(&self) -> &Arc<dyn SignalBackend> {
        &self.signal
    }

    /// The label of `channel` from the layout (e.g. its site), if any.
    pub fn channel_label(&self, channel: usize) -> Option<&str> {
        self.layout.as_ref()?.label(channel)
    }

    /// A recording computed on demand by a plugin (e.g. electrophysiology: noise, hum, drifting
    /// units); an error when no plugin makes one.
    pub fn procedural(channels: usize, sample_rate: f64, duration_sec: f64) -> Result<Self> {
        let source = crate::plugin::synthetic(channels, sample_rate, duration_sec).ok_or_else(|| anyhow::anyhow!("no plugin makes synthetic recordings"))??;
        Ok(Self::local(source, None)?)
    }

    /// Wraps an in-memory channel-major buffer.
    #[cfg(test)]
    pub fn from_samples(name: &str, data: Vec<f32>, total_channels: usize, sample_rate: f64) -> Self {
        Self::local(Arc::new(dspk_core::MemoryRecording::new(name, data, total_channels, sample_rate).expect("valid shape")), None).expect("memory pyramid")
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

        let rec = dspk_core::MemoryRecording::new(format!("synthetic_{total_channels}ch"), raw_data, total_channels.max(1), sample_rate)
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
        let dir = std::env::temp_dir().join(format!("dspk_app_ds_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rec.bin");
        let values: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, bytes).unwrap();
        std::fs::write(path.with_extension("meta"), r#"{"channels": 3, "sample_rate_hz": 1000.0}"#).unwrap();

        let source: Arc<dyn RecordingSource> = Arc::from(dspk_io::open(&path).unwrap());
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

    }

    /// Time to build a whole recording's pyramid in the background. Run with:
    /// `DSP_APP_BENCH_FILE=<recording> cargo test -p dspk-app --release bench_pyramid -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_pyramid() {
        let Some(path) = std::env::var_os("DSP_APP_BENCH_FILE") else { return };
        let sources = crate::data::recording::SourceSet::open(Path::new(&path)).unwrap();
        let ds = sources.default_dataset();
        let (tx, rx) = std::sync::mpsc::channel();
        ds.signal().watch(Arc::new(move |p: dspk_view::Progress| {
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
