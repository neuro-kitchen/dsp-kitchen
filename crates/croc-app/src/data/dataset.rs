//! The open recording: any `dsp-io` format behind [`RecordingSource`], read in chunks.

use std::ops::Range;
use std::sync::Arc;

use anyhow::Result;
use dsp_core::{DspResult, RecordingInfo, RecordingSource};
use dsp_io::{SyntheticParams, SyntheticRecording};

/// A recording plus the summary fields the UI reads every frame.
pub struct Dataset {
    source: Arc<dyn RecordingSource>,
    pub total_channels: usize,
    pub total_samples: usize,
    pub sample_rate: f64,
    pub name: String,
    /// Session time of sample 0 (sources of one file can start at different times).
    pub start_time_sec: f64,
    /// Unit of the values reads return (`µV` for electrical recordings).
    pub unit: String,
}

impl Dataset {
    pub fn new(source: Arc<dyn RecordingSource>) -> Self {
        let info = source.info();
        Self {
            total_channels: info.channel_count(),
            total_samples: info.samples as usize,
            sample_rate: info.sample_rate_hz(),
            name: info.name.clone(),
            start_time_sec: info.start_time_sec,
            unit: info.metadata.get("unit").map_or_else(|| "µV".into(), |u| u.replace("uV", "µV")),
            source,
        }
    }

    /// Opens any format `dsp-io` detects (SpikeGLX, IBL `.cbin`, raw binary + JSON sidecar, Zarr).
    #[cfg(test)]
    pub fn open(path: &std::path::Path) -> Result<Self> {
        use anyhow::Context;
        let source = dsp_io::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
        Ok(Self::new(Arc::from(source)))
    }

    /// Procedural recording of any length (noise, hum, drifting units), computed on demand.
    pub fn procedural(channels: usize, sample_rate: f64, duration_sec: f64) -> Result<Self> {
        let units = (channels / 4).clamp(1, 64);
        let rec = SyntheticRecording::new(SyntheticParams { channels, sample_rate_hz: sample_rate, duration_sec, units, ..Default::default() })?;
        Ok(Self::new(Arc::new(rec)))
    }

    /// Wraps an in-memory channel-major buffer.
    #[cfg(test)]
    pub fn from_samples(name: &str, data: Vec<f32>, total_channels: usize, sample_rate: f64) -> Self {
        Self::new(Arc::new(dsp_core::MemoryRecording::new(name, data, total_channels, sample_rate).expect("valid shape")))
    }

    pub fn total_duration_sec(&self) -> f64 {
        if self.sample_rate > 0.0 {
            self.total_samples as f64 / self.sample_rate
        } else {
            0.0
        }
    }

    /// One sample in µV (0 when out of range or unreadable).
    pub fn sample(&self, channel: usize, sample: usize) -> f32 {
        let mut v = [0.0f32];
        let s = sample as u64;
        match self.source.read(&[channel], s..s + 1, &mut v) {
            Ok(()) => v[0],
            Err(_) => 0.0,
        }
    }

    /// Every channel over `samples`, channel-major. Callers bound the range (whole-recording
    /// passes belong in chunked background jobs).
    pub fn read_all(&self, samples: Range<usize>) -> DspResult<Vec<f32>> {
        let channels: Vec<usize> = (0..self.total_channels).collect();
        let mut out = vec![0.0f32; channels.len() * samples.len()];
        self.source.read(&channels, samples.start as u64..samples.end as u64, &mut out)?;
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
        Self::new(Arc::new(rec))
    }
}

/// Views and workers hold `Arc<Dataset>` as a plain `RecordingSource`.
impl RecordingSource for Dataset {
    fn info(&self) -> &RecordingInfo {
        self.source.info()
    }

    fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> DspResult<()> {
        self.source.read(channels, samples, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opens_raw_file_through_dsp_io() {
        let dir = std::env::temp_dir().join(format!("croc_ds_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rec.bin");
        let values: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, bytes).unwrap();
        std::fs::write(path.with_extension("meta"), r#"{"channels": 3, "sample_rate_hz": 1000.0}"#).unwrap();

        let ds = Dataset::open(&path).unwrap();
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
}
