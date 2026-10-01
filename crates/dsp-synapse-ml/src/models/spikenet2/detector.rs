//! SpikeNet2 pretrained ONNX waveform classifier (`spikenet2/ieeg-detector-v1`).

use std::path::Path;

use dsp_core::{ComputeTarget, DspError, DspResult};

use crate::hub::TensorIoSpec;
use crate::runtime::{
    OnnxRuntimeSession, RuntimeTensor, default_compute_target, pull_model, validate_tensor_port,
};

pub const SPIKENET2_IED_MODEL_ID: &str = "spikenet2/ieeg-detector-v1";

/// Pretrained SpikeNet2 ONNX waveform detector/classifier.
pub struct SpikeNet2Detector {
    session: OnnxRuntimeSession,
    io_spec: Option<TensorIoSpec>,
    pub probability_threshold: f32,
}

impl SpikeNet2Detector {
    /// Pulls and initializes `spikenet2/ieeg-detector-v1` from [`crate::hub::ModelHub`].
    pub fn from_hub(
        probability_threshold: f32,
        target: Option<ComputeTarget>,
    ) -> DspResult<Self> {
        let (manifest, path) = pull_model(SPIKENET2_IED_MODEL_ID)?;
        let compute_target = match target {
            Some(t) => t.checked().map_err(|e| DspError::Model(e.to_string()))?,
            None => default_compute_target()?,
        };
        let session = OnnxRuntimeSession::from_file(&path, compute_target)?;
        Ok(Self {
            session,
            io_spec: Some(manifest.io_spec),
            probability_threshold,
        })
    }

    /// Loads a SpikeNet2 `.onnx` model from disk.
    pub fn from_onnx_file(
        path: impl AsRef<Path>,
        probability_threshold: f32,
        target: Option<ComputeTarget>,
    ) -> DspResult<Self> {
        let compute_target = match target {
            Some(t) => t.checked().map_err(|e| DspError::Model(e.to_string()))?,
            None => default_compute_target()?,
        };
        let session = OnnxRuntimeSession::from_file(path, compute_target)?;
        Ok(Self {
            session,
            io_spec: None,
            probability_threshold,
        })
    }

    /// Evaluates `[batch, channels, samples]` windows and returns the positive-class spike
    /// probability for each window in `0..batch`.
    pub fn predict_probabilities(
        &self,
        windows: &[f32],
        batch: usize,
        channels: usize,
        samples: usize,
    ) -> DspResult<Vec<f32>> {
        let shape = vec![batch, channels, samples];
        if let Some(io) = &self.io_spec {
            if let Some(in_port) = io.inputs.first() {
                validate_tensor_port(in_port, &shape)?;
            }
        }

        let input = RuntimeTensor::new(windows.to_vec(), shape)?;
        let output = self.session.run(input)?;

        if let Some(io) = &self.io_spec {
            if let Some(out_port) = io.outputs.first() {
                validate_tensor_port(out_port, &output.shape)?;
            }
        }

        let cols = *output.shape.last().unwrap_or(&1).max(&1);
        let mut probs = Vec::with_capacity(batch);
        for row in output.data.chunks_exact(cols) {
            if cols == 1 {
                probs.push(row[0]);
            } else {
                // Positive class is last column
                probs.push(*row.last().unwrap_or(&0.0));
            }
        }
        Ok(probs)
    }
}
