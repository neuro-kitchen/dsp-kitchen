//! **DARTsort** external sorter profile (Boussard, Varol, Windolf, Paninski et al.).
//!
//! Bridges DARTsort's pretrained **Single-Channel / Spatiotemporal Transformer Waveform Denoiser**,
//! **Variational Autoencoder (VAE)** latent embedder, and **Point-Source / Dipole Localizer**
//! `.onnx` and `.safetensors` checkpoints into `dsp-synapse-ml`.

use std::path::Path;
use anyhow::Result;
use crate::backend::SynapseMlDevice;
use crate::hub::{PyTorchRemapRule, PyTorchWeightAdapter};
use crate::onnx::adapters::{
    OnnxFeatureEmbedder, OnnxNormalization, OnnxPeakLocalizer, OnnxSnippetLayout,
    OnnxWaveformDenoiser,
};
use crate::onnx::ir_runner::OnnxGraphRunner;
use super::{ExternalSorterFamily, ExternalSorterProfile};

/// Pre-configured I/O and weight-remapping profile for **DARTsort**.
#[derive(Debug, Clone)]
pub struct DartsortProfile {
    pub profile: ExternalSorterProfile,
}

impl DartsortProfile {
    /// Standard DARTsort multi-channel neighborhood configuration (`K` channels, `T` samples, `D` VAE latent dim).
    pub fn new(num_channels: usize, num_samples: usize, latent_dim: usize) -> Self {
        Self {
            profile: ExternalSorterProfile {
                family: ExternalSorterFamily::Dartsort,
                name: "dartsort-pretrained-featurizer".to_string(),
                num_channels,
                num_samples,
                embedding_dim: latent_dim,
                snippet_layout: OnnxSnippetLayout::ChannelsFirstNkt,
                normalization: OnnxNormalization::PeakAbsNormalized,
                l2_normalize_embeddings: false,
            },
        }
    }

    /// Returns a [`PyTorchWeightAdapter`] that maps DARTsort's PyTorch `WaveformVAE` /
    /// `SingleChannelWaveformDenoiser` `state_dict` keys into `DartsortVaeEmbedder` and
    /// `SingleChannelDenoiser` in `dsp-synapse-ml`.
    pub fn weight_adapter() -> PyTorchWeightAdapter {
        PyTorchWeightAdapter::new()
            .with_passthrough(true)
            // DARTsort single-channel conv denoiser remap
            .add_rules(PyTorchRemapRule::conv1d_or_norm_pair(
                "denoiser.conv1.0",
                "sc_denoiser.conv1",
            ))
            .add_rules(PyTorchRemapRule::batch_norm_1d(
                "denoiser.conv1.1",
                "sc_denoiser.bn1",
            ))
            .add_rules(PyTorchRemapRule::conv1d_or_norm_pair(
                "denoiser.conv2.0",
                "sc_denoiser.conv2",
            ))
            .add_rules(PyTorchRemapRule::batch_norm_1d(
                "denoiser.conv2.1",
                "sc_denoiser.bn2",
            ))
            .add_rules(PyTorchRemapRule::conv1d_or_norm_pair(
                "denoiser.conv_out",
                "sc_denoiser.conv_out",
            ))
            // DARTsort VAE posterior heads (fc_mu / fc_logvar)
            .add_rules(PyTorchRemapRule::linear_pair(
                "encoder.fc_mu",
                "dartsort_vae.fc_mu",
            ))
            .add_rules(PyTorchRemapRule::linear_pair(
                "encoder.fc_logvar",
                "dartsort_vae.fc_logvar",
            ))
    }

    /// Wraps a DARTsort `.onnx` waveform denoiser into a Stage 2 [`OnnxWaveformDenoiser`].
    pub fn wrap_denoiser(&self, runner: OnnxGraphRunner) -> OnnxWaveformDenoiser {
        OnnxWaveformDenoiser::new(runner)
            .with_layout(self.profile.snippet_layout)
            .with_normalization(self.profile.normalization)
    }

    /// Loads a DARTsort `.onnx` waveform denoiser from a file path.
    pub fn load_onnx_denoiser(
        &self,
        path: impl AsRef<Path>,
        device: SynapseMlDevice,
    ) -> Result<OnnxWaveformDenoiser> {
        let runner = OnnxGraphRunner::from_file(path, device)?;
        Ok(self.wrap_denoiser(runner))
    }

    /// Wraps a DARTsort `.onnx` encoder into a Stage 3 [`OnnxFeatureEmbedder`].
    pub fn wrap_embedder(&self, runner: OnnxGraphRunner) -> OnnxFeatureEmbedder {
        OnnxFeatureEmbedder::new(runner, self.profile.embedding_dim)
            .with_layout(self.profile.snippet_layout)
            .with_normalization(self.profile.normalization)
            .with_l2_normalize(false)
    }

    /// Wraps a DARTsort `.onnx` localization network into a Stage 4 [`OnnxPeakLocalizer`].
    pub fn wrap_localizer(&self, runner: OnnxGraphRunner) -> OnnxPeakLocalizer {
        let mut loc = OnnxPeakLocalizer::new(runner);
        loc.layout = self.profile.snippet_layout;
        loc.normalization = self.profile.normalization;
        loc
    }
}
