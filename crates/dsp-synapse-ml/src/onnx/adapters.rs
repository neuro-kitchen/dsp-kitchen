//! Polymorphic `dsp_synapse::traits` adapters backed by [`OnnxGraphRunner`].
//!
//! Enables any external `.onnx` model (from Kilosort4, DARTsort, CEBRA, Bombcell/UnitMatch,
//! or custom PyTorch exports) to be dropped directly into the `dsp-synapse` spike-sorting
//! pipeline as a [`SpikeDetector`], [`WaveformDenoiser`], [`FeatureEmbedder`],
//! [`PeakLocalizer`], or automated unit curator.

use dsp_core::{DspError, DspResult, SensorLayout};
use dsp_synapse::{
    FeatureEmbedder, PeakLocalizer, SnippetBatch, SpikeDetector, SpikeEvent, UnitQualityLabel,
    WaveformDenoiser,
};
use serde::{Deserialize, Serialize};
use crate::backend::{Tensor2D, Tensor3D, snippet_batch_to_tensor, tensor_to_snippet_batch};
use crate::curation::{UnitCurationPrediction, UnitQualityFeatures};
use super::ir_runner::OnnxGraphRunner;

/// An ONNX run failure as a [`DspError`], with its full context chain.
fn model_error(stage: &str, e: anyhow::Error) -> DspError {
    DspError::Model(format!("{stage}: {e:#}"))
}

/// Tensor memory layout expected by an external `.onnx` model operating on multi-channel snippets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OnnxSnippetLayout {
    /// `[batch, channels, samples]` (`[N, K, T]`) — standard PyTorch `Conv1d` / `SnippetBatch` layout.
    ChannelsFirstNkt,
    /// `[batch, samples, channels]` (`[N, T, K]`) — temporal-first layout used by Transformer / CEBRA / Kilosort4 temporal blocks.
    TimeFirstNtk,
}

/// Input waveform amplitude normalization policy applied before feeding a `SnippetBatch` into an ONNX model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OnnxNormalization {
    /// Raw microvolt ($\mu\text{V}$) values unchanged.
    RawMicrovolts,
    /// Per-snippet, per-channel zero-mean unit-variance ($z$-score) normalization.
    ZScorePerChannel,
    /// Scales each snippet by its peak absolute trough amplitude so values lie in $[-1, 1]$.
    PeakAbsNormalized,
}

/// Prepares a `[N, K, T]` `Tensor3D` from a `SnippetBatch` with the requested normalization and layout,
/// returning `(prepared_tensor, per_spike_scale_factors)` so denoisers can invert scaling if needed.
pub fn prepare_snippet_tensor(
    batch: &SnippetBatch,
    layout: OnnxSnippetLayout,
    norm: OnnxNormalization,
    device: crate::backend::SynapseMlDevice,
) -> (Tensor3D, Vec<f32>) {
    let [n, k, t] = batch.shape();
    let mut nkt_data = batch.data.clone();
    let mut scales = vec![1.0f32; n.max(1)];

    match norm {
        OnnxNormalization::RawMicrovolts => {}
        OnnxNormalization::ZScorePerChannel => {
            for s in 0..n {
                for ch in 0..k {
                    let off = (s * k + ch) * t;
                    let slice = &mut nkt_data[off..off + t];
                    if t > 0 {
                        let mean = slice.iter().sum::<f32>() / (t as f32);
                        let var = slice
                            .iter()
                            .map(|&v| {
                                let d = v - mean;
                                d * d
                            })
                            .sum::<f32>()
                            / (t as f32);
                        let inv_std = 1.0 / (var + 1e-5).sqrt();
                        for v in slice.iter_mut() {
                            *v = (*v - mean) * inv_std;
                        }
                    }
                }
            }
        }
        OnnxNormalization::PeakAbsNormalized => {
            let kt = k * t;
            for s in 0..n {
                let slice = &mut nkt_data[s * kt..(s + 1) * kt];
                let max_abs = slice
                    .iter()
                    .fold(0.0f32, |acc, &v| acc.max(v.abs()))
                    .max(1e-5);
                scales[s] = max_abs;
                let inv = 1.0 / max_abs;
                for v in slice.iter_mut() {
                    *v *= inv;
                }
            }
        }
    }

    match layout {
        OnnxSnippetLayout::ChannelsFirstNkt => {
            (Tensor3D::from_floats(nkt_data, [n, k, t], device), scales)
        }
        OnnxSnippetLayout::TimeFirstNtk => {
            let mut ntk_data = vec![0.0f32; n * t * k];
            for s in 0..n {
                for ch in 0..k {
                    for ti in 0..t {
                        ntk_data[(s * t + ti) * k + ch] = nkt_data[(s * k + ch) * t + ti];
                    }
                }
            }
            (Tensor3D::from_floats(ntk_data, [n, t, k], device), scales)
        }
    }
}

/// Stage 1 [`SpikeDetector`] adapter powered by [`OnnxGraphRunner`].
///
/// Slides a window of `window_samples` along each channel (or multi-channel group) and invokes
/// an ONNX classifier returning either a 1-column spike probability `[N, 1]` (post-sigmoid)
/// or a 2+/3-column logit/probability vector where `spike_class_index` is the spike class.
#[derive(Debug, Clone)]
pub struct OnnxSpikeDetector {
    pub runner: OnnxGraphRunner,
    pub window_samples: usize,
    pub stride_samples: usize,
    pub probability_threshold: f32,
    pub refractory_samples: usize,
    pub spike_class_index: usize,
    pub apply_softmax: bool,
}

impl OnnxSpikeDetector {
    pub fn new(runner: OnnxGraphRunner, window_samples: usize) -> Self {
        Self {
            runner,
            window_samples: window_samples.max(8),
            stride_samples: (window_samples / 2).max(1),
            probability_threshold: 0.5,
            refractory_samples: 15,
            spike_class_index: 0,
            apply_softmax: true,
        }
    }
}

impl SpikeDetector for OnnxSpikeDetector {
    fn detect(
        &self,
        data: &[f32],
        channels: usize,
        samples: usize,
        _sample_rate_hz: f64,
    ) -> DspResult<Vec<SpikeEvent>> {
        if channels == 0 || samples < self.window_samples {
            return Ok(Vec::new());
        }
        let w = self.window_samples;
        let stride = self.stride_samples.max(1);
        let num_win_per_ch = (samples - w) / stride + 1;
        let total_windows = channels * num_win_per_ch;

        let mut batch_data = Vec::with_capacity(total_windows * w);
        let mut meta = Vec::with_capacity(total_windows);

        for ch in 0..channels {
            let ch_slice = &data[ch * samples..(ch + 1) * samples];
            for wi in 0..num_win_per_ch {
                let start = wi * stride;
                let window = &ch_slice[start..start + w];
                let mut min_val = f32::INFINITY;
                let mut min_idx = 0usize;
                for (i, &v) in window.iter().enumerate() {
                    if v < min_val {
                        min_val = v;
                        min_idx = i;
                    }
                }
                batch_data.extend_from_slice(window);
                meta.push((ch, (start + min_idx) as u64, min_val));
            }
        }

        let input_3d = Tensor3D::from_floats(batch_data, [total_windows, 1, w], self.runner.device);
        let out_2d = self.runner.run_3d_to_2d(&input_3d).map_err(|e| model_error("OnnxSpikeDetector", e))?;
        if out_2d.shape[0] != total_windows {
            return Err(DspError::Model(format!(
                "OnnxSpikeDetector: {} output rows for {total_windows} windows",
                out_2d.shape[0]
            )));
        }

        let probs = if out_2d.shape[1] > 1 && self.apply_softmax {
            out_2d.softmax()
        } else if out_2d.shape[1] == 1 {
            out_2d.sigmoid()
        } else {
            out_2d
        };

        let cols = probs.shape[1];
        let cls_idx = self.spike_class_index.min(cols.saturating_sub(1));
        let mut events = Vec::new();
        let mut last_sample_per_ch = vec![None::<u64>; channels];

        for (i, &(ch, sample_idx, peak_uv)) in meta.iter().enumerate() {
            let p_spike = probs.data[i * cols + cls_idx];
            if p_spike >= self.probability_threshold {
                if let Some(prev) = last_sample_per_ch[ch]
                    && sample_idx.saturating_sub(prev) < self.refractory_samples as u64
                {
                    continue;
                }
                last_sample_per_ch[ch] = Some(sample_idx);
                events.push(SpikeEvent {
                    channel_id: ch,
                    sample_index: sample_idx,
                    peak_amplitude_uv: peak_uv,
                });
            }
        }

        events.sort_by_key(|e| (e.sample_index, e.channel_id));
        Ok(events)
    }
}

/// Stage 2 [`WaveformDenoiser`] adapter powered by [`OnnxGraphRunner`].
///
/// Supports both `[N, K, T] -> [N, K, T]` spatio-temporal ONNX denoisers (e.g., DARTsort transformer
/// denoiser, 1D UNet) and `[N * K, 1, T] -> [N * K, 1, T]` single-channel ONNX denoisers.
#[derive(Debug, Clone)]
pub struct OnnxWaveformDenoiser {
    pub runner: OnnxGraphRunner,
    pub layout: OnnxSnippetLayout,
    pub normalization: OnnxNormalization,
    pub restore_scale: bool,
}

impl OnnxWaveformDenoiser {
    pub fn new(runner: OnnxGraphRunner) -> Self {
        Self {
            runner,
            layout: OnnxSnippetLayout::ChannelsFirstNkt,
            normalization: OnnxNormalization::RawMicrovolts,
            restore_scale: true,
        }
    }

    pub fn with_layout(mut self, layout: OnnxSnippetLayout) -> Self {
        self.layout = layout;
        self
    }

    pub fn with_normalization(mut self, normalization: OnnxNormalization) -> Self {
        self.normalization = normalization;
        self
    }
}

impl WaveformDenoiser for OnnxWaveformDenoiser {
    fn denoise(&self, batch: &SnippetBatch) -> DspResult<SnippetBatch> {
        if batch.num_spikes == 0 {
            return Ok(batch.clone());
        }
        let [n, k, t] = batch.shape();
        let (in_tensor, scales) =
            prepare_snippet_tensor(batch, self.layout, self.normalization, self.runner.device);

        let raw_out = self.runner.run_3d(&in_tensor).map_err(|e| model_error("OnnxWaveformDenoiser", e))?;
        if raw_out.data.len() != n * k * t {
            return Err(DspError::Model(format!(
                "OnnxWaveformDenoiser: output shape {:?} does not match the input snippets {:?}",
                raw_out.shape,
                [n, k, t]
            )));
        }

        // Convert back to [N, K, T] if the ONNX graph operated in [N, T, K]
        let mut nkt_out = match self.layout {
            OnnxSnippetLayout::ChannelsFirstNkt => raw_out.data,
            OnnxSnippetLayout::TimeFirstNtk => {
                let mut buf = vec![0.0f32; n * k * t];
                for s in 0..n {
                    for ti in 0..t {
                        for ch in 0..k {
                            buf[(s * k + ch) * t + ti] = raw_out.data[(s * t + ti) * k + ch];
                        }
                    }
                }
                buf
            }
        };

        if self.restore_scale && self.normalization == OnnxNormalization::PeakAbsNormalized {
            let kt = k * t;
            for s in 0..n {
                let scale = scales[s];
                for v in &mut nkt_out[s * kt..(s + 1) * kt] {
                    *v *= scale;
                }
            }
        }

        let out_3d = Tensor3D::from_floats(nkt_out, [n, k, t], self.runner.device);
        Ok(tensor_to_snippet_batch(out_3d, batch))
    }
}

/// Stage 3 [`FeatureEmbedder`] adapter powered by [`OnnxGraphRunner`].
///
/// Runs an ONNX encoder (`[N, K, T] -> [N, D]` or `[N, T, K] -> [N, D]`) such as DARTsort VAE,
/// CEBRA contrastive encoder, or Kilosort4 template SVD/PC projector, with optional $L_2$ hypersphere normalization.
#[derive(Debug, Clone)]
pub struct OnnxFeatureEmbedder {
    pub runner: OnnxGraphRunner,
    pub embedding_dim: usize,
    pub layout: OnnxSnippetLayout,
    pub normalization: OnnxNormalization,
    pub l2_normalize_output: bool,
}

impl OnnxFeatureEmbedder {
    pub fn new(runner: OnnxGraphRunner, embedding_dim: usize) -> Self {
        Self {
            runner,
            embedding_dim,
            layout: OnnxSnippetLayout::ChannelsFirstNkt,
            normalization: OnnxNormalization::RawMicrovolts,
            l2_normalize_output: false,
        }
    }

    pub fn with_layout(mut self, layout: OnnxSnippetLayout) -> Self {
        self.layout = layout;
        self
    }

    pub fn with_normalization(mut self, normalization: OnnxNormalization) -> Self {
        self.normalization = normalization;
        self
    }

    pub fn with_l2_normalize(mut self, l2_normalize: bool) -> Self {
        self.l2_normalize_output = l2_normalize;
        self
    }
}

impl FeatureEmbedder for OnnxFeatureEmbedder {
    fn embed(&self, batch: &SnippetBatch) -> DspResult<(Vec<f32>, usize)> {
        if batch.num_spikes == 0 {
            return Ok((Vec::new(), self.embedding_dim));
        }
        let (in_tensor, _scales) =
            prepare_snippet_tensor(batch, self.layout, self.normalization, self.runner.device);

        let out_2d = self.runner.run_3d_to_2d(&in_tensor).map_err(|e| model_error("OnnxFeatureEmbedder", e))?;
        let actual_dim = out_2d.shape[1];
        let final_2d = if self.l2_normalize_output {
            out_2d.l2_normalize(1e-6)
        } else {
            out_2d
        };
        Ok((final_2d.into_vec(), actual_dim))
    }
}

/// Stage 4 [`PeakLocalizer`] adapter powered by [`OnnxGraphRunner`].
///
/// Predicts 3D physical coordinates `[dx, dy, z]` relative to the primary electrode's
/// `(x, y)` coordinates on the [`SensorLayout`].
#[derive(Debug, Clone)]
pub struct OnnxPeakLocalizer {
    pub runner: OnnxGraphRunner,
    pub layout: OnnxSnippetLayout,
    pub normalization: OnnxNormalization,
}

impl OnnxPeakLocalizer {
    pub fn new(runner: OnnxGraphRunner) -> Self {
        Self {
            runner,
            layout: OnnxSnippetLayout::ChannelsFirstNkt,
            normalization: OnnxNormalization::PeakAbsNormalized,
        }
    }
}

impl PeakLocalizer for OnnxPeakLocalizer {
    fn localize(&self, batch: &SnippetBatch, sensor_layout: &SensorLayout) -> DspResult<Vec<[f32; 3]>> {
        if batch.num_spikes == 0 {
            return Ok(Vec::new());
        }
        let (in_tensor, _scales) =
            prepare_snippet_tensor(batch, self.layout, self.normalization, self.runner.device);
        let raw_2d = match self.runner.run_3d_to_2d(&in_tensor) {
            Ok(t) => t,
            Err(e3) => {
                // Fallback if the ONNX localizer expects a pre-flattened 2D input [N, K * T]
                let flat = snippet_batch_to_tensor(batch, self.runner.device).flatten_channels_time();
                self.runner.run_2d(&flat).map_err(|e2| {
                    DspError::Model(format!("OnnxPeakLocalizer: [N, K, T] input failed ({e3:#}); [N, K·T] input failed ({e2:#})"))
                })?
            }
        };

        let cols = raw_2d.shape[1];
        if cols < 3 || raw_2d.shape[0] != batch.num_spikes {
            return Err(DspError::Model(format!(
                "OnnxPeakLocalizer: output {:?}, expected [{}, ≥3] ([x, y, z] per spike)",
                raw_2d.shape, batch.num_spikes
            )));
        }

        let mut coords = Vec::with_capacity(batch.num_spikes);
        for s in 0..batch.num_spikes {
            let primary_ch = batch.primary_channels[s];
            let (base_x, base_y) = sensor_layout
                .get_site(primary_ch)
                .map(|site| (site.position.x_um, site.position.y_um))
                .unwrap_or((0.0, 0.0));
            let dx = raw_2d.data[s * cols];
            let dy = raw_2d.data[s * cols + 1];
            let z = raw_2d.data[s * cols + 2].abs().max(1.0);
            coords.push([base_x + dx, base_y + dy, z]);
        }
        Ok(coords)
    }
}

/// Automated single-unit quality curator powered by [`OnnxGraphRunner`]
/// (compatible with Bombcell / UnitMatch / Allen IBL quality metric classifiers).
#[derive(Debug, Clone)]
pub struct OnnxUnitCurator {
    pub runner: OnnxGraphRunner,
}

impl OnnxUnitCurator {
    pub fn new(runner: OnnxGraphRunner) -> Self {
        Self { runner }
    }

    /// Evaluates `[N, 8]` normalized [`UnitQualityFeatures`] through the ONNX classifier
    /// and returns `[P(SUA), P(MUA), P(Noise)]` predictions.
    pub fn classify_units(&self, units: &[UnitQualityFeatures]) -> DspResult<Vec<UnitCurationPrediction>> {
        let n = units.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let mut flat = Vec::with_capacity(n * 8);
        for u in units {
            flat.extend_from_slice(&u.to_normalized_array());
        }
        let input = Tensor2D::from_floats(flat, [n, 8], self.runner.device);
        let logits = self.runner.run_2d(&input).map_err(|e| model_error("OnnxUnitCurator", e))?;
        let probs = logits.softmax();
        let cols = probs.shape[1];
        if cols < 3 || probs.shape[0] != n {
            return Err(DspError::Model(format!(
                "OnnxUnitCurator: output {:?}, expected [{n}, 3] classes [SUA, MUA, Noise]",
                probs.shape
            )));
        }

        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let p_sua = probs.data[i * cols];
            let p_mua = probs.data[i * cols + 1];
            let p_noise = probs.data[i * cols + 2];
            let label = if p_sua >= p_mua && p_sua >= p_noise {
                UnitQualityLabel::SingleUnit
            } else if p_mua >= p_noise {
                UnitQualityLabel::MultiUnit
            } else {
                UnitQualityLabel::Noise
            };
            out.push(UnitCurationPrediction {
                label,
                p_single_unit: p_sua,
                p_multi_unit: p_mua,
                p_noise,
            });
        }
        Ok(out)
    }
}
