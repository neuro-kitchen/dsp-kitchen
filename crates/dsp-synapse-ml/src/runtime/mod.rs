//! Tensor execution for the pretrained models: Burn operations and an ONNX (`onnx-ir`) graph
//! evaluator, on a [`dsp_core::ComputeTarget`] the caller chooses. To be replaced by `burn-onnx`
//! with device-resident weights once a model artifact is validated (review RT1–RT3, ONNX1–4).

pub mod burn_engine;
pub mod onnx_engine;

#[cfg(feature = "hub")]
use std::path::PathBuf;
use dsp_core::{DspError, DspResult};
#[cfg(feature = "hub")]
use crate::hub::ModelManifest;
use crate::hub::TensorPortSpec;

pub use burn_engine::{burn_conv1d, burn_linear_2d};
pub use onnx_engine::{OnnxRuntimeSession, RuntimeTensor};

/// Pulls and verifies `model_id` of the catalog (feature `hub`), returning its manifest and local
/// path.
#[cfg(feature = "hub")]
pub fn pull_model(model_id: &str) -> DspResult<(ModelManifest, PathBuf)> {
    let hub = crate::hub::ModelHub::new().map_err(|e| DspError::Io(e.to_string()))?;
    hub.pull(model_id, false).map_err(|e| DspError::Io(e.to_string()))
}

/// Validates a runtime tensor's dimensions against a [`TensorPortSpec`] from [`crate::hub::ModelManifest`].
///
/// Any positive dimension `d > 0` in `spec.shape` must match `actual_shape[i]` exactly;
/// `-1` denotes a dynamic dimension (e.g., batch size, variable channel count, or basis rank).
pub fn validate_tensor_port(spec: &TensorPortSpec, actual_shape: &[usize]) -> DspResult<()> {
    if spec.shape.len() != actual_shape.len() {
        return Err(DspError::InvalidConfig(format!(
            "tensor port '{}' expects rank {} ({:?}), got rank {} ({:?})",
            spec.name,
            spec.shape.len(),
            spec.shape,
            actual_shape.len(),
            actual_shape
        )));
    }

    for (axis, (&expected, &actual)) in spec.shape.iter().zip(actual_shape.iter()).enumerate() {
        if expected > 0 && (expected as usize) != actual {
            return Err(DspError::InvalidConfig(format!(
                "tensor port '{}' axis {axis} expects dimension {expected} (spec {:?}), got {actual} (actual {:?})",
                spec.name, spec.shape, actual_shape
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_tensor_port_dynamic_and_fixed_dims() {
        let spec = TensorPortSpec {
            name: "waveforms".to_string(),
            dtype: "float32".to_string(),
            shape: vec![-1, -1, 61],
            description: Some("batch x channels x window".to_string()),
        };

        assert!(validate_tensor_port(&spec, &[32, 4, 61]).is_ok());
        assert!(validate_tensor_port(&spec, &[1, 384, 61]).is_ok());
        assert!(validate_tensor_port(&spec, &[32, 4, 40]).is_err());
        assert!(validate_tensor_port(&spec, &[32, 61]).is_err());
    }
}
