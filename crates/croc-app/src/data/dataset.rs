//! Dataset loading and synthetic electrophysiology signal generation model.

use std::fs::File;
use std::path::{Path, PathBuf};
use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use serde::Deserialize;

use super::source::SignalSource;

#[derive(Deserialize, Debug)]
struct SidecarMetadata {
    channels: Option<usize>,
    samples: Option<usize>,
    sample_rate_hz: Option<f64>,
}

/// Backing storage: memory-mapped file (zero-copy) or an owned buffer (synthetic data).
enum Storage {
    Mapped(Mmap),
    Owned(Vec<f32>),
}

/// Multi-channel continuous recording dataset (channel-major: `[ch][sample]`).
pub struct Dataset {
    storage: Storage,
    pub total_channels: usize,
    pub total_samples: usize,
    pub sample_rate: f64,
    pub name: String,
}

impl Dataset {
    /// All samples, channel-major. For mapped files this reads the OS page cache directly.
    pub fn data(&self) -> &[f32] {
        let len = self.total_channels * self.total_samples;
        match &self.storage {
            Storage::Owned(v) => &v[..len],
            // SAFETY: `load_from_file` verified the map holds at least `len` f32s, and mmap
            // bases are page-aligned so the pointer is aligned for f32.
            Storage::Mapped(m) => unsafe { std::slice::from_raw_parts(m.as_ptr() as *const f32, len) },
        }
    }

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

        if total_channels == 0 {
            bail!("Channel count must be at least 1");
        }
        let total_floats = mmap.len() / std::mem::size_of::<f32>();
        let total_samples = detected_s.unwrap_or(total_floats / total_channels);
        if total_channels * total_samples > total_floats {
            bail!(
                "{} holds {} f32 values but {} channels x {} samples were requested",
                path.display(),
                total_floats,
                total_channels,
                total_samples
            );
        }

        let name = path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "dataset.bin".to_string());

        Ok(Self {
            storage: Storage::Mapped(mmap),
            total_channels,
            total_samples,
            sample_rate,
            name,
        })
    }

    /// Wraps an in-memory channel-major buffer.
    #[cfg(test)]
    pub fn from_samples(name: &str, data: Vec<f32>, total_channels: usize, sample_rate: f64) -> Self {
        let total_samples = if total_channels == 0 { 0 } else { data.len() / total_channels };
        Self {
            storage: Storage::Owned(data),
            total_channels,
            total_samples,
            sample_rate,
            name: name.to_string(),
        }
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
            storage: Storage::Owned(raw_data),
            total_channels,
            total_samples,
            sample_rate,
            name: format!("synthetic_{total_channels}ch"),
        }
    }
}

impl SignalSource for Dataset {
    fn channels(&self) -> usize {
        self.total_channels
    }
    fn samples(&self) -> usize {
        self.total_samples
    }
    fn sample_rate(&self) -> f64 {
        self.sample_rate
    }
    fn channel(&self, ch: usize) -> &[f32] {
        let n = self.total_samples;
        &self.data()[ch * n..(ch + 1) * n]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mapped_file_zero_copy_and_size_check() {
        let dir = std::env::temp_dir().join(format!("croc_ds_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rec.bin");
        let values: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, bytes).unwrap();

        let ds = Dataset::load_from_file(&path, Some(3), Some(1000.0)).unwrap();
        assert_eq!(ds.total_samples, 4);
        assert_eq!(ds.channel(1), &[4.0, 5.0, 6.0, 7.0]);

        // Asking for more channels than the file holds is an error, not an out-of-bounds read
        std::fs::write(path.with_extension("meta"), r#"{"samples": 100}"#).unwrap();
        assert!(Dataset::load_from_file(&path, Some(3), None).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_synthetic_dataset_generation() {
        let ds = Dataset::generate_synthetic(8, 10_000.0, 1.0);
        assert_eq!(ds.total_channels, 8);
        assert_eq!(ds.total_samples, 10_000);
        assert_eq!(ds.data().len(), 80_000);
        assert_eq!(ds.channel(7).len(), 10_000);
        assert!((ds.total_duration_sec() - 1.0).abs() < 1e-6);
    }
}
