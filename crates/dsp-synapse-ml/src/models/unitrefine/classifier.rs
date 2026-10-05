//! SpikeInterface UnitRefine & Bombcell automated cluster quality curation (`unitrefine/curation-v1`).

use std::path::Path;

use dsp_core::{ComputeTarget, DspError, DspResult};
use dsp_synapse::UnitQualityLabel;
use serde::{Deserialize, Serialize};

use crate::hub::TensorIoSpec;
use crate::runtime::{
    OnnxRuntimeSession, RuntimeTensor, default_compute_target, pull_model, validate_tensor_port,
};

pub const UNITREFINE_CURATION_MODEL_ID: &str = "unitrefine/curation-v1";
pub const UNITREFINE_BOMBCELL_MODEL_ID: &str = "unitrefine/bombcell-metrics-v1";

/// Prediction emitted by [`UnitRefineClassifier`] for a single unit cluster.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnitCurationResult {
    pub label: UnitQualityLabel,
    /// Calibrated softmax probabilities `[p_single_unit, p_multi_unit, p_noise]`.
    pub probabilities: [f32; 3],
}

/// Pretrained ONNX unit quality curator (`unitrefine/curation-v1`).
pub struct UnitRefineClassifier {
    session: OnnxRuntimeSession,
    io_spec: Option<TensorIoSpec>,
}

impl UnitRefineClassifier {
    /// Pulls and initializes `unitrefine/curation-v1` from [`crate::hub::ModelHub`].
    pub fn from_hub(target: Option<ComputeTarget>) -> DspResult<Self> {
        let (manifest, path) = pull_model(UNITREFINE_CURATION_MODEL_ID)?;
        let compute_target = match target {
            Some(t) => t.checked().map_err(|e| DspError::Model(e.to_string()))?,
            None => default_compute_target()?,
        };
        let session = OnnxRuntimeSession::from_file(&path, compute_target)?;
        Ok(Self {
            session,
            io_spec: Some(manifest.io_spec),
        })
    }

    /// Loads a UnitRefine `.onnx` model from disk.
    pub fn from_onnx_file(
        path: impl AsRef<Path>,
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
        })
    }

    /// Classifies a `[num_units, num_features]` quality metrics matrix into
    /// `[SingleUnit, MultiUnit, Noise]` labels and softmax probabilities.
    pub fn classify_metrics(
        &self,
        metrics: &[f32],
        num_units: usize,
        num_features: usize,
    ) -> DspResult<Vec<UnitCurationResult>> {
        let in_shape = vec![num_units, num_features];
        if let Some(io) = &self.io_spec {
            if let Some(in_port) = io.inputs.first() {
                validate_tensor_port(in_port, &in_shape)?;
            }
        }

        let input = RuntimeTensor::new(metrics.to_vec(), in_shape)?;
        let output = self.session.run(input)?;

        if let Some(io) = &self.io_spec {
            if let Some(out_port) = io.outputs.first() {
                validate_tensor_port(out_port, &output.shape)?;
            }
        }

        let num_classes = *output.shape.last().unwrap_or(&0);
        if num_classes != 3 {
            return Err(DspError::Model(format!(
                "UnitRefineClassifier expects 3 output classes [SUA, MUA, Noise], got {num_classes}"
            )));
        }

        let mut results = Vec::with_capacity(num_units);
        for row in output.data.chunks_exact(3) {
            let max_v = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e0 = (row[0] - max_v).exp();
            let e1 = (row[1] - max_v).exp();
            let e2 = (row[2] - max_v).exp();
            let sum = (e0 + e1 + e2).max(1e-12);
            let probs = [e0 / sum, e1 / sum, e2 / sum];

            let label = if probs[0] >= probs[1] && probs[0] >= probs[2] {
                UnitQualityLabel::SingleUnit
            } else if probs[1] >= probs[2] {
                UnitQualityLabel::MultiUnit
            } else {
                UnitQualityLabel::Noise
            };

            results.push(UnitCurationResult {
                label,
                probabilities: probs,
            });
        }
        Ok(results)
    }
}
