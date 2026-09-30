//! **Bombcell / UnitMatch** external quality control & automated unit curation profile
//! (Fabre et al. 2023 /van Beest et al. 2024).
//!
//! Bridges Bombcell and UnitMatch automated single-unit quality classifiers (`SUA`, `MUA`, `Noise` /
//! non-somatic) into `dsp-synapse-ml` via [`OnnxUnitCurator`] and [`PyTorchWeightAdapter`].

use std::path::Path;
use anyhow::Result;
use crate::backend::SynapseMlDevice;
use crate::hub::{PyTorchRemapRule, PyTorchWeightAdapter};
use crate::onnx::adapters::{OnnxNormalization, OnnxSnippetLayout, OnnxUnitCurator};
use crate::onnx::ir_runner::OnnxGraphRunner;
use super::{ExternalSorterFamily, ExternalSorterProfile};

/// Pre-configured I/O and weight-remapping profile for **Bombcell / UnitMatch** unit quality classifiers.
#[derive(Debug, Clone)]
pub struct BombcellProfile {
    pub profile: ExternalSorterProfile,
}

impl Default for BombcellProfile {
    fn default() -> Self {
        Self::new()
    }
}

impl BombcellProfile {
    pub fn new() -> Self {
        Self {
            profile: ExternalSorterProfile {
                family: ExternalSorterFamily::Bombcell,
                name: "bombcell-unitmatch-quality-curator".to_string(),
                num_channels: 1,
                num_samples: 8,
                embedding_dim: 3, // [P(SUA), P(MUA), P(Noise)]
                snippet_layout: OnnxSnippetLayout::ChannelsFirstNkt,
                normalization: OnnxNormalization::RawMicrovolts,
                l2_normalize_embeddings: false,
            },
        }
    }

    /// Returns a [`PyTorchWeightAdapter`] mapping Bombcell / UnitMatch MLP `state_dict` keys
    /// into `UnitQualityClassifier` (`unit_classifier.mlp.*`) in `dsp-synapse-ml`.
    pub fn weight_adapter() -> PyTorchWeightAdapter {
        PyTorchWeightAdapter::new()
            .with_passthrough(true)
            .add_rules(PyTorchRemapRule::linear_pair(
                "fc1",
                "unit_classifier.mlp.hidden.0.linear",
            ))
            .add_rules(PyTorchRemapRule::conv1d_or_norm_pair(
                "ln1",
                "unit_classifier.mlp.hidden.0.norm",
            ))
            .add_rules(PyTorchRemapRule::linear_pair(
                "fc2",
                "unit_classifier.mlp.hidden.1.linear",
            ))
            .add_rules(PyTorchRemapRule::conv1d_or_norm_pair(
                "ln2",
                "unit_classifier.mlp.hidden.1.norm",
            ))
            .add_rules(PyTorchRemapRule::linear_pair(
                "fc_out",
                "unit_classifier.mlp.out_layer",
            ))
    }

    /// Wraps an [`OnnxGraphRunner`] into an [`OnnxUnitCurator`].
    pub fn wrap_curator(&self, runner: OnnxGraphRunner) -> OnnxUnitCurator {
        OnnxUnitCurator::new(runner)
    }

    /// Loads a Bombcell / UnitMatch `.onnx` classifier from disk.
    pub fn load_onnx_curator(
        &self,
        path: impl AsRef<Path>,
        device: SynapseMlDevice,
    ) -> Result<OnnxUnitCurator> {
        let runner = OnnxGraphRunner::from_file(path, device)?;
        Ok(self.wrap_curator(runner))
    }
}
