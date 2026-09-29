//! Dataset loading and synthetic electrophysiology signal generation model.

use std::fs::File;
use std::path::{Path, PathBuf};
use anyhow::{Context, Result};
use memmap2::Mmap;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
struct SidecarMetadata {
    channels: Option<usize>,
    samples: Option<usize>,
    sample_rate_hz: Option<f64>,
}

/// Multi-channel continuous recording dataset.
pub struct Dataset {
    pub raw_data: Vec<f32>,
    pub total_channels: usize,
    pub total_samples: usize,
    pub sample_rate: f64,
    pub name: String,
}

impl Dataset {
    pub fn total_duration_sec(&self) -> f64 {
        if self.sample_rate > 0.0 {
            self.total_samples as f64 / self.sample_rate
        } else {
            0.0
        }
    }

    /// Loads dataset from a binary file (and optional `.meta` JSON sidecar) or generates
    /// a synthetic 32-channel electrophysiology signal if no file is available.
    pub fn load_or_synthetic(
        file_arg: Option<PathBuf>,
        channels_arg: Option<usize>,
        sample_rate_arg: Option<f64>,
    ) -> Result<Self> {
        let default_bin = PathBuf::from("playground/data/mearec_32ch_10s.bin");
        let target_file = file_arg.or_else(|| {
            if default_bin.exists() {
                Some(default_bin)
            } else {
                None
            }
        });

        match target_file {
            Some(path) => Self::load_from_file(&path, channels_arg, sample_rate_arg),
            None => {
                let ch = channels_arg.unwrap_or(32);
                let sr = sample_rate_arg.unwrap_or(30_000.0);
                Ok(Self::generate_synthetic(ch, sr, 5.0))
            }
        }
    }

    pub fn load_from_file(
        path: &Path,
        channels_override: Option<usize>,
        sample_rate_override: Option<f64>,
    ) -> Result<Self> {
        let meta_path = path.with_extension("meta");
        let mut detected_ch = channels_override;
        let mut detected_s = None;
        let mut detected_sr = sample_rate_override;

        if meta_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&meta_path) {
                if let Ok(meta) = serde_json::from_str::<SidecarMetadata>(&content) {
                    if detected_ch.is_none() { detected_ch = meta.channels; }
                    if detected_s.is_none() { detected_s = meta.samples; }
                    if detected_sr.is_none() { detected_sr = meta.sample_rate_hz; }
                }
            }
        }

        let total_channels = detected_ch.unwrap_or(32);
        let sample_rate = detected_sr.unwrap_or(32_000.0);

        let file = File::open(path)
            .with_context(|| format!("Failed to open dataset: {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };

        let total_floats = mmap.len() / std::mem::size_of::<f32>();
        let total_samples = detected_s.unwrap_or(total_floats / total_channels);

        let slice = unsafe {
            std::slice::from_raw_parts(mmap.as_ptr() as *const f32, total_channels * total_samples)
        };

        let name = path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "dataset.bin".to_string());

        Ok(Self {
            raw_data: slice.to_vec(),
            total_channels,
            total_samples,
            sample_rate,
            name,
        })
    }

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

        Self {
            raw_data,
            total_channels,
            total_samples,
            sample_rate,
            name: "synthetic_32ch.bin".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_dataset_generation() {
        let ds = Dataset::generate_synthetic(8, 10_000.0, 1.0);
        assert_eq!(ds.total_channels, 8);
        assert_eq!(ds.total_samples, 10_000);
        assert_eq!(ds.raw_data.len(), 80_000);
        assert!((ds.total_duration_sec() - 1.0).abs() < 1e-6);
    }
}
