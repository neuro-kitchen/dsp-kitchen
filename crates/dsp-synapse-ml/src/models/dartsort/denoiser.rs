//! DARTsort pretrained single-channel waveform denoiser (`dartsort/singlechan-denoiser-v1`).

use std::path::Path;

use dsp_core::{ComputeTarget, DspError, DspResult};
use dsp_synapse::{SnippetBatch, WaveformDenoiser};

use crate::hub::{SafetensorsMap, TensorIoSpec};
use crate::runtime::{
    burn_conv1d, burn_linear_2d, default_compute_target, pull_model, validate_tensor_port,
};

pub const DARTSORT_DENOISER_MODEL_ID: &str = "dartsort/singlechan-denoiser-v1";

/// Pretrained DARTsort single-channel waveform denoiser loaded from `.safetensors`.
pub struct DartsortWaveformDenoiser {
    weights: SafetensorsMap,
    io_spec: Option<TensorIoSpec>,
    target: ComputeTarget,
}

impl DartsortWaveformDenoiser {
    /// Pulls and initializes `dartsort/singlechan-denoiser-v1` from [`crate::hub::ModelHub`].
    pub fn from_hub(target: Option<ComputeTarget>) -> DspResult<Self> {
        let (manifest, path) = pull_model(DARTSORT_DENOISER_MODEL_ID)?;
        let compute_target = match target {
            Some(t) => t.checked().map_err(|e| DspError::Model(e.to_string()))?,
            None => default_compute_target()?,
        };
        let weights = SafetensorsMap::from_file(&path)
            .map_err(|e| DspError::Model(e.to_string()))?;
        Ok(Self {
            weights,
            io_spec: Some(manifest.io_spec),
            target: compute_target,
        })
    }

    /// Loads a DARTsort `.safetensors` denoiser file from disk.
    pub fn from_safetensors_file(
        path: impl AsRef<Path>,
        target: Option<ComputeTarget>,
    ) -> DspResult<Self> {
        let compute_target = match target {
            Some(t) => t.checked().map_err(|e| DspError::Model(e.to_string()))?,
            None => default_compute_target()?,
        };
        let weights = SafetensorsMap::from_file(path)
            .map_err(|e| DspError::Model(e.to_string()))?;
        Ok(Self {
            weights,
            io_spec: None,
            target: compute_target,
        })
    }

    pub fn target(&self) -> ComputeTarget {
        self.target
    }
}

impl WaveformDenoiser for DartsortWaveformDenoiser {
    fn denoise(&self, batch: &SnippetBatch) -> DspResult<SnippetBatch> {
        let [n, k, t] = batch.shape();
        if n == 0 || k == 0 || t == 0 {
            return Ok(batch.clone());
        }

        let single_chan_batches = n * k;
        if let Some(io) = &self.io_spec {
            if let Some(in_port) = io.inputs.first() {
                validate_tensor_port(in_port, &[single_chan_batches, 1, t])?;
            }
        }

        let (conv_w_shape, conv_w) = self
            .weights
            .get_raw("conv1.weight")
            .map_err(|e| DspError::Model(e.to_string()))?;
        if conv_w_shape.len() != 3 {
            return Err(DspError::Model(format!(
                "conv1.weight expected 3D shape, got {:?}",
                conv_w_shape
            )));
        }
        let out_ch = conv_w_shape[0];
        let kernel_size = conv_w_shape[2];
        let conv_b = self.weights.get_raw("conv1.bias").ok().map(|(_, d)| d);
        let pad = kernel_size / 2;

        let (mut hidden, out_len) = burn_conv1d(
            self.target,
            &batch.data,
            single_chan_batches,
            1,
            t,
            conv_w,
            out_ch,
            kernel_size,
            conv_b,
            1,
            pad,
            1,
            1,
        )?;
        for v in &mut hidden {
            *v = v.max(0.0);
        }

        let (lin_w_shape, lin_w) = self
            .weights
            .get_raw("fc.weight")
            .map_err(|e| DspError::Model(e.to_string()))?;
        if lin_w_shape.len() != 2 {
            return Err(DspError::Model(format!(
                "fc.weight expected 2D shape, got {:?}",
                lin_w_shape
            )));
        }
        let lin_b = self.weights.get_raw("fc.bias").ok().map(|(_, d)| d);
        let denoised = burn_linear_2d(
            self.target,
            &hidden,
            single_chan_batches,
            out_ch * out_len,
            lin_w,
            lin_w_shape[0],
            lin_b,
        )?;

        Ok(SnippetBatch {
            data: denoised,
            num_spikes: n,
            num_channels: k,
            num_samples: t,
            primary_channels: batch.primary_channels.clone(),
            center_samples: batch.center_samples.clone(),
            subsample_offsets: batch.subsample_offsets.clone(),
            channel_ids: batch.channel_ids.clone(),
        })
    }
}
