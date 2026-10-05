//! Kilosort4 universal template matcher (`kilosort4/universal-templates-v1` / `wTEMP.npy`).
//!
//! Convolves multi-channel electrophysiology chunks with L2-normalized universal templates
//! on the active [`ComputeTarget`] using [`crate::runtime::TemplateFilterTask`] and extracts
//! refractory-enforced local peaks.

use std::path::{Path, PathBuf};

use dsp_core::{ComputeTarget, DspError, DspResult};
use dsp_synapse::{SpikeDetector, SpikeEvent, estimate_noise_std};

use crate::hub::NpyTensorF32;
use crate::runtime::{
    TemplateFilterTask, default_compute_target, pull_model, run_on_target,
};

pub const KILOSORT4_TEMPLATES_MODEL_ID: &str = "kilosort4/universal-templates-v1";

/// Pretrained Kilosort4 universal template filter (`wTEMP.npy`).
#[derive(Debug, Clone)]
pub struct Kilosort4TemplateMatcher {
    /// Row-major `[num_templates, window_len]` L2-normalized universal templates.
    templates: Vec<f32>,
    num_templates: usize,
    window_len: usize,
    /// Center sample offset within `0..window_len`, derived dynamically from the primary template trough.
    center_offset: usize,
    pub threshold_sigma: f32,
    pub refractory_samples: usize,
    source_path: PathBuf,
    target: ComputeTarget,
}

impl Kilosort4TemplateMatcher {
    /// Pulls and verifies `kilosort4/universal-templates-v1` from [`crate::hub::ModelHub`] and binds to `target`.
    pub fn from_hub(
        threshold_sigma: f32,
        refractory_samples: usize,
        target: Option<ComputeTarget>,
    ) -> DspResult<Self> {
        let (_manifest, path) = pull_model(KILOSORT4_TEMPLATES_MODEL_ID)?;
        Self::from_npy(path, threshold_sigma, refractory_samples, target)
    }

    /// Loads Kilosort4 universal templates directly from a `.npy` file on disk.
    pub fn from_npy(
        path: impl AsRef<Path>,
        threshold_sigma: f32,
        refractory_samples: usize,
        target: Option<ComputeTarget>,
    ) -> DspResult<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let npy = NpyTensorF32::from_file(&path_buf)
            .map_err(|e| DspError::Model(e.to_string()))?;
        Self::from_tensor(npy, path_buf, threshold_sigma, refractory_samples, target)
    }

    fn from_tensor(
        npy: NpyTensorF32,
        source_path: PathBuf,
        threshold_sigma: f32,
        refractory_samples: usize,
        target: Option<ComputeTarget>,
    ) -> DspResult<Self> {
        if npy.shape.len() != 2 || npy.shape[0] == 0 || npy.shape[1] == 0 {
            return Err(DspError::Model(format!(
                "Kilosort4TemplateMatcher expects non-empty 2D [num_templates, window_len] .npy, got {:?}",
                npy.shape
            )));
        }

        let compute_target = match target {
            Some(t) => t.checked().map_err(|e| DspError::Model(e.to_string()))?,
            None => default_compute_target()?,
        };

        let num_templates = npy.shape[0];
        let window_len = npy.shape[1];
        let mut templates = npy.data;

        for row in templates.chunks_exact_mut(window_len) {
            let peak_val = row
                .iter()
                .copied()
                .max_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);
            let sign = if peak_val > 0.0 { -1.0f32 } else { 1.0f32 };
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 1e-8 {
                let scale = sign / norm;
                for v in row.iter_mut() {
                    *v *= scale;
                }
            }
        }

        // Derive the canonical trough center dynamically from the minimum (trough) sample
        // in the primary universal template.
        let center_offset = templates[..window_len]
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(window_len / 2);

        Ok(Self {
            templates,
            num_templates,
            window_len,
            center_offset,
            threshold_sigma,
            refractory_samples: refractory_samples.max(1),
            source_path,
            target: compute_target,
        })
    }

    pub fn num_templates(&self) -> usize {
        self.num_templates
    }

    pub fn window_len(&self) -> usize {
        self.window_len
    }

    pub fn center_offset(&self) -> usize {
        self.center_offset
    }

    pub fn templates(&self) -> &[f32] {
        &self.templates
    }

    pub fn source_path(&self) -> &Path {
        &self.source_path
    }

    pub fn target(&self) -> ComputeTarget {
        self.target
    }

    /// Computes the `[channels, samples]` universal template matched-filter energy envelope
    /// on the active [`ComputeTarget`].
    pub fn filter_energy(
        &self,
        data: &[f32],
        channels: usize,
        samples: usize,
    ) -> DspResult<Vec<f32>> {
        if data.len() != channels * samples {
            return Err(DspError::ShapeMismatch {
                expected: vec![channels, samples],
                actual: vec![data.len()],
            });
        }

        run_on_target(
            self.target,
            TemplateFilterTask {
                trace: data,
                channels,
                samples,
                templates: &self.templates,
                num_templates: self.num_templates,
                window_len: self.window_len,
                center_offset: self.center_offset,
            },
        )
    }
}

impl SpikeDetector for Kilosort4TemplateMatcher {
    fn detect(
        &self,
        data: &[f32],
        channels: usize,
        samples: usize,
        _sample_rate_hz: f64,
    ) -> DspResult<Vec<SpikeEvent>> {
        if data.len() != channels * samples {
            return Err(DspError::ShapeMismatch {
                expected: vec![channels, samples],
                actual: vec![data.len()],
            });
        }
        if samples < self.window_len || channels == 0 {
            return Ok(Vec::new());
        }

        let energy = self.filter_energy(data, channels, samples)?;
        let post_span = self.window_len - self.center_offset;
        let start_s = self.center_offset.max(1);
        let end_s = (samples - post_span).min(samples.saturating_sub(1));

        let mut events = Vec::new();
        for ch in 0..channels {
            let ch_trace = &data[ch * samples..(ch + 1) * samples];
            let ch_energy = &energy[ch * samples..(ch + 1) * samples];
            let sigma = estimate_noise_std(ch_trace).max(1e-6);
            let min_energy = self.threshold_sigma * sigma;

            let mut s = start_s;
            let mut last_spike: Option<usize> = None;
            while s < end_s {
                let e = ch_energy[s];
                if e >= min_energy && e >= ch_energy[s - 1] && e >= ch_energy[s + 1] {
                    let within_refrac = last_spike
                        .map(|prev| s.saturating_sub(prev) < self.refractory_samples)
                        .unwrap_or(false);

                    if !within_refrac {
                        events.push(SpikeEvent {
                            channel_id: ch,
                            sample_index: s as u64,
                            peak_amplitude_uv: ch_trace[s],
                        });
                        last_spike = Some(s);
                    } else if let Some(last_ev) = events.last_mut() {
                        let prev_s = last_ev.sample_index as usize;
                        if e > ch_energy[prev_s] {
                            last_ev.sample_index = s as u64;
                            last_ev.peak_amplitude_uv = ch_trace[s];
                            last_spike = Some(s);
                        }
                    }
                }
                s += 1;
            }
        }

        events.sort_by_key(|ev| (ev.sample_index, ev.channel_id));
        Ok(events)
    }
}
