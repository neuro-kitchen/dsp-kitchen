//! DARTsort pretrained spatiotemporal VAE embedder (`dartsort/spatiotemporal-vae-v1`).

use std::path::Path;

use dsp_core::{ComputeTarget, DspError, DspResult};
use dsp_synapse::{FeatureEmbedder, SnippetBatch};

use crate::hub::TensorIoSpec;
use crate::runtime::{
    OnnxRuntimeSession, RuntimeTensor, validate_tensor_port,
};

pub const DARTSORT_VAE_MODEL_ID: &str = "dartsort/spatiotemporal-vae-v1";

/// Pretrained DARTsort spatiotemporal VAE feature embedder executing via ONNX runtime.
pub struct DartsortVaeEmbedder {
    session: OnnxRuntimeSession,
    io_spec: Option<TensorIoSpec>,
}

impl DartsortVaeEmbedder {
    /// Pulls and initializes `dartsort/spatiotemporal-vae-v1` from [`crate::hub::ModelHub`].
    #[cfg(feature = "hub")]
    pub fn from_hub(target: ComputeTarget) -> DspResult<Self> {
        let (manifest, path) = crate::runtime::pull_model(DARTSORT_VAE_MODEL_ID)?;
        let compute_target = target.checked().map_err(|e| DspError::ComputeError(e.to_string()))?;
        let session = OnnxRuntimeSession::from_file(&path, compute_target)?;
        Ok(Self {
            session,
            io_spec: manifest.io_spec,
        })
    }

    /// Loads a DARTsort VAE `.onnx` model from disk.
    pub fn from_onnx_file(
        path: impl AsRef<Path>,
        target: ComputeTarget,
    ) -> DspResult<Self> {
        let compute_target = target.checked().map_err(|e| DspError::ComputeError(e.to_string()))?;
        let session = OnnxRuntimeSession::from_file(path, compute_target)?;
        Ok(Self {
            session,
            io_spec: None,
        })
    }
}

impl FeatureEmbedder for DartsortVaeEmbedder {
    fn embed(&self, batch: &SnippetBatch) -> DspResult<(Vec<f32>, usize)> {
        let [n, k, t] = batch.shape();
        let in_shape = vec![n, k, t];
        if let Some(io) = &self.io_spec {
            if let Some(in_port) = io.inputs.first() {
                validate_tensor_port(in_port, &in_shape)?;
            }
        }

        if n == 0 {
            let latent_dim = self
                .io_spec
                .as_ref()
                .and_then(|io| io.outputs.first())
                .and_then(|p| p.shape.last().copied())
                .unwrap_or(0)
                .max(0) as usize;
            return Ok((Vec::new(), latent_dim));
        }

        let input = RuntimeTensor::new(batch.data.clone(), in_shape)?;
        let output = self.session.run(input)?;

        if let Some(io) = &self.io_spec {
            if let Some(out_port) = io.outputs.first() {
                validate_tensor_port(out_port, &output.shape)?;
            }
        }

        let latent_dim = *output.shape.last().unwrap_or(&0);
        Ok((output.data, latent_dim))
    }
}
