//! DARTsort pretrained single-channel waveform denoiser (`dartsort/singlechan-denoiser-v1`).

use std::path::Path;
use std::sync::OnceLock;

use burn_tensor::activation::relu;
use burn_tensor::{Device, Shape, Tensor, TensorData};
use dspk_core::{ComputeTarget, DspError, DspResult};
use dspk_synapse::{SnippetBatch, WaveformDenoiser};

use crate::manifest::{SafetensorsMap, TensorIoSpec};
use crate::runtime::burn_engine::{burn_device, PreparedConv1d, PreparedLinear2d};
use crate::runtime::validate_tensor_port;

pub const DARTSORT_DENOISER_MODEL_ID: &str = "dartsort/singlechan-denoiser-v1";

struct PreparedDenoiser {
    device: Device,
    conv1: PreparedConv1d,
    fc: PreparedLinear2d,
}

/// Pretrained DARTsort single-channel waveform denoiser loaded from `.safetensors`.
pub struct DartsortWaveformDenoiser {
    weights: SafetensorsMap,
    io_spec: Option<TensorIoSpec>,
    target: ComputeTarget,
    prepared: OnceLock<PreparedDenoiser>,
}

impl DartsortWaveformDenoiser {
    /// Pulls and initializes `dartsort/singlechan-denoiser-v1` from [`crate::runtime::pull_model`].
    #[cfg(feature = "hub")]
    pub fn from_hub(target: ComputeTarget) -> DspResult<Self> {
        let (manifest, path) = crate::runtime::pull_model(DARTSORT_DENOISER_MODEL_ID)?;
        let compute_target = target.checked().map_err(|e| DspError::ComputeError(e.to_string()))?;
        let weights = SafetensorsMap::from_file(&path)
            .map_err(|e| DspError::InvalidConfig(e.to_string()))?;
        Ok(Self {
            weights,
            io_spec: manifest.io_spec,
            target: compute_target,
            prepared: OnceLock::new(),
        })
    }

    /// Loads a DARTsort `.safetensors` denoiser file from disk.
    pub fn from_safetensors_file(
        path: impl AsRef<Path>,
        target: ComputeTarget,
    ) -> DspResult<Self> {
        let compute_target = target.checked().map_err(|e| DspError::ComputeError(e.to_string()))?;
        let weights = SafetensorsMap::from_file(path)
            .map_err(|e| DspError::InvalidConfig(e.to_string()))?;
        Ok(Self {
            weights,
            io_spec: None,
            target: compute_target,
            prepared: OnceLock::new(),
        })
    }

    pub fn target(&self) -> ComputeTarget {
        self.target
    }

    fn prepared(&self) -> DspResult<&PreparedDenoiser> {
        if let Some(p) = self.prepared.get() {
            return Ok(p);
        }
        let dev = burn_device(self.target)?;
        let (conv_w_shape, conv_w) = self
            .weights
            .get_raw("conv1.weight")
            .map_err(|e| DspError::InvalidConfig(e.to_string()))?;
        if conv_w_shape.len() != 3 {
            return Err(DspError::InvalidConfig(format!(
                "conv1.weight expected 3D shape, got {:?}",
                conv_w_shape
            )));
        }
        let out_ch = conv_w_shape[0];
        let in_per_group = conv_w_shape[1];
        let kernel_size = conv_w_shape[2];
        let conv_b = self.weights.get_raw("conv1.bias").ok().map(|(_, d)| d);
        let pad = kernel_size / 2;
        let conv1 = PreparedConv1d::new(&dev, conv_w, out_ch, in_per_group, kernel_size, conv_b, 1, pad, 1, 1)?;

        let (lin_w_shape, lin_w) = self
            .weights
            .get_raw("fc.weight")
            .map_err(|e| DspError::InvalidConfig(e.to_string()))?;
        if lin_w_shape.len() != 2 {
            return Err(DspError::InvalidConfig(format!(
                "fc.weight expected 2D shape, got {:?}",
                lin_w_shape
            )));
        }
        let lin_b = self.weights.get_raw("fc.bias").ok().map(|(_, d)| d);
        let fc = PreparedLinear2d::new(&dev, lin_w, lin_w_shape[0], lin_w_shape[1], lin_b)?;

        let _ = self.prepared.set(PreparedDenoiser { device: dev, conv1, fc });
        Ok(self.prepared.get().expect("initialized prepared denoiser"))
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

        let prepared = self.prepared()?;
        let x = Tensor::<3>::from_data(
            TensorData::new(batch.data.clone(), Shape::new([single_chan_batches, 1, t])),
            &prepared.device,
        );
        // Keep intermediate activations in VRAM across conv1d -> relu -> flatten -> linear
        let hidden = relu(prepared.conv1.forward(x));
        let [_, out_ch, out_len] = hidden.dims();
        let flat = hidden.reshape([single_chan_batches, out_ch * out_len]);
        let denoised = prepared
            .fc
            .forward(flat)
            .into_data()
            .try_to_vec::<f32>()
            .expect("f32 tensor output");

        Ok(SnippetBatch {
            data: denoised,
            num_spikes: n,
            num_channels: k,
            num_samples: t,
            // Denoising keeps the alignment
            peak_index: batch.peak_index,
            primary_channels: batch.primary_channels.clone(),
            center_samples: batch.center_samples.clone(),
            subsample_offsets: batch.subsample_offsets.clone(),
            channel_ids: batch.channel_ids.clone(),
        })
    }
}
