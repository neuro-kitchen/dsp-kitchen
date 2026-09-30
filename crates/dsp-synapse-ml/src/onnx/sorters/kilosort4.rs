//! **Kilosort4** external sorter profile (`kilosort.spikedetect` / `kilosort.clustering_qr`).
//!
//! Bridges Kilosort4's learned temporal PCA / SVD basis (`wPCA` `[n_pcs, nt]`) and
//! spatio-temporal template projection ONNX / `.safetensors` models into `dsp-synapse-ml`.

use std::path::Path;
use anyhow::Result;
use crate::backend::SynapseMlDevice;
use crate::hub::{PyTorchRemapRule, PyTorchWeightAdapter};
use crate::onnx::adapters::{
    OnnxFeatureEmbedder, OnnxNormalization, OnnxSnippetLayout, OnnxSpikeDetector,
};
use crate::onnx::ir_runner::OnnxGraphRunner;
use super::{ExternalSorterFamily, ExternalSorterProfile};

/// Pre-configured I/O and weight-remapping profile for **Kilosort4**.
#[derive(Debug, Clone)]
pub struct Kilosort4Profile {
    pub profile: ExternalSorterProfile,
    /// Number of temporal principal components per channel (`n_pcs`, typically `3` or `6` in KS4).
    pub num_temporal_pcs: usize,
}

impl Kilosort4Profile {
    /// Standard Kilosort4 Neuropixels 1.0 / 2.0 configuration (`nt = 61` samples, `n_pcs = 6`).
    pub fn neuropixels_default(num_channels: usize, num_samples: usize) -> Self {
        let num_temporal_pcs = 6;
        let embedding_dim = num_channels * num_temporal_pcs;
        Self {
            profile: ExternalSorterProfile {
                family: ExternalSorterFamily::Kilosort4,
                name: "kilosort4-template-pc-projector".to_string(),
                num_channels,
                num_samples,
                embedding_dim,
                snippet_layout: OnnxSnippetLayout::ChannelsFirstNkt,
                normalization: OnnxNormalization::ZScorePerChannel,
                l2_normalize_embeddings: false,
            },
            num_temporal_pcs,
        }
    }

    /// Returns a [`PyTorchWeightAdapter`] configured to remap Kilosort4 PyTorch `.safetensors`
    /// checkpoints (`wPCA`, `wTEMP`, `encoder.fc.*`) into `dsp-synapse-ml` native backbones.
    pub fn weight_adapter() -> PyTorchWeightAdapter {
        PyTorchWeightAdapter::new()
            .with_passthrough(true)
            .add_rules(PyTorchRemapRule::linear_pair(
                "wPCA_proj",
                "conv_ae.fc_bottleneck",
            ))
            .add_rules(PyTorchRemapRule::conv1d_or_norm_pair(
                "temporal_filter.conv1",
                "conv_ae.enc1",
            ))
            .add_rules(PyTorchRemapRule::conv1d_or_norm_pair(
                "temporal_filter.conv2",
                "conv_ae.enc2",
            ))
    }

    /// Loads a Kilosort4 feature-projection `.onnx` model from disk as an [`OnnxFeatureEmbedder`].
    pub fn load_onnx_embedder(
        &self,
        path: impl AsRef<Path>,
        device: SynapseMlDevice,
    ) -> Result<OnnxFeatureEmbedder> {
        let runner = OnnxGraphRunner::from_file(path, device)?;
        Ok(self.wrap_embedder(runner))
    }

    /// Wraps an [`OnnxGraphRunner`] with Kilosort4's layout and normalization settings.
    pub fn wrap_embedder(&self, runner: OnnxGraphRunner) -> OnnxFeatureEmbedder {
        OnnxFeatureEmbedder::new(runner, self.profile.embedding_dim)
            .with_layout(self.profile.snippet_layout)
            .with_normalization(self.profile.normalization)
            .with_l2_normalize(self.profile.l2_normalize_embeddings)
    }

    /// Wraps a Kilosort4 template-matching / learned filter `.onnx` graph as a Stage 1 [`OnnxSpikeDetector`].
    pub fn wrap_detector(&self, runner: OnnxGraphRunner) -> OnnxSpikeDetector {
        let mut det = OnnxSpikeDetector::new(runner, self.profile.num_samples);
        det.probability_threshold = 0.50;
        det
    }
}
