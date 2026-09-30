//! **CEBRA** external sorter / neural latent embedding profile (Schneider, Lee, Mathis 2023).
//!
//! Bridges CEBRA (`cebra-time` / `cebra-behavior` / `offset10-model`) contrastive learning
//! encoders into `dsp-synapse-ml`, projecting multi-channel spike snippets onto the unit
//! hypersphere $\mathbb{S}^{D-1}$ (`||z||_2 = 1.0`).

use std::path::Path;
use anyhow::Result;
use crate::backend::SynapseMlDevice;
use crate::hub::{PyTorchRemapRule, PyTorchWeightAdapter};
use crate::onnx::adapters::{OnnxFeatureEmbedder, OnnxNormalization, OnnxSnippetLayout};
use crate::onnx::ir_runner::OnnxGraphRunner;
use super::{ExternalSorterFamily, ExternalSorterProfile};

/// Pre-configured I/O and weight-remapping profile for **CEBRA** contrastive waveform/population encoders.
#[derive(Debug, Clone)]
pub struct CebraProfile {
    pub profile: ExternalSorterProfile,
}

impl CebraProfile {
    /// Creates a CEBRA contrastive hypersphere embedding profile (`||z||_2 = 1.0`).
    pub fn new(num_channels: usize, num_samples: usize, output_dim: usize) -> Self {
        Self {
            profile: ExternalSorterProfile {
                family: ExternalSorterFamily::Cebra,
                name: "cebra-contrastive-hypersphere".to_string(),
                num_channels,
                num_samples,
                embedding_dim: output_dim,
                snippet_layout: OnnxSnippetLayout::ChannelsFirstNkt,
                normalization: OnnxNormalization::ZScorePerChannel,
                l2_normalize_embeddings: true,
            },
        }
    }

    /// Returns a [`PyTorchWeightAdapter`] mapping PyTorch CEBRA `net.*` layers into
    /// `ContrastiveWaveformEmbedder` (`simclr.*`) in `dsp-synapse-ml`.
    pub fn weight_adapter() -> PyTorchWeightAdapter {
        PyTorchWeightAdapter::new()
            .with_passthrough(true)
            .add_rules(PyTorchRemapRule::conv1d_or_norm_pair(
                "net.0",
                "simclr.stem",
            ))
            .add_rules(PyTorchRemapRule::conv1d_or_norm_pair(
                "net.1.block.0",
                "simclr.res1.conv1",
            ))
            .add_rules(PyTorchRemapRule::conv1d_or_norm_pair(
                "net.2.block.0",
                "simclr.res2.conv1",
            ))
            .add_rules(PyTorchRemapRule::linear_pair(
                "net.head.0",
                "simclr.proj_head.hidden.0.linear",
            ))
            .add_rules(PyTorchRemapRule::linear_pair(
                "net.head.2",
                "simclr.proj_head.out_layer",
            ))
    }

    /// Wraps an [`OnnxGraphRunner`] as a Stage 3 [`OnnxFeatureEmbedder`] with $L_2$ unit-sphere normalization.
    pub fn wrap_embedder(&self, runner: OnnxGraphRunner) -> OnnxFeatureEmbedder {
        OnnxFeatureEmbedder::new(runner, self.profile.embedding_dim)
            .with_layout(self.profile.snippet_layout)
            .with_normalization(self.profile.normalization)
            .with_l2_normalize(true)
    }

    /// Loads a CEBRA `.onnx` model from disk.
    pub fn load_onnx_embedder(
        &self,
        path: impl AsRef<Path>,
        device: SynapseMlDevice,
    ) -> Result<OnnxFeatureEmbedder> {
        let runner = OnnxGraphRunner::from_file(path, device)?;
        Ok(self.wrap_embedder(runner))
    }
}
