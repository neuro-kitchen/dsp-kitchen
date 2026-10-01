//! Unified inference runtime for `dsp-synapse-ml` bridging [`dsp_core::ComputeTarget`],
//! CubeCL dynamic [`dsp_core::compute::LaunchGeometry`] kernels, Burn tensor execution,
//! and ONNX (`onnx-ir`) graph evaluation.

pub mod burn_engine;
pub mod kernels;
pub mod onnx_engine;

use std::path::PathBuf;
use dsp_core::{DspError, DspResult};
use crate::hub::{ModelHub, ModelManifest, TensorPortSpec};

pub use dsp_core::compute::{ComputeError, ComputeTarget, ComputeTask, LaunchGeometry};
pub use burn_engine::{burn_conv1d, burn_linear_2d};
pub use kernels::{
    ProjectBasisTask, ReconstructBasisTask, TemplateFilterTask,
    execute_temporal_basis_project, execute_temporal_basis_reconstruct,
    execute_universal_template_filter, run_on_target,
};
pub use onnx_engine::{OnnxRuntimeSession, RuntimeTensor};

/// Resolves the default [`ComputeTarget`] from `DSP_KITCHEN_RUNTIME` or the highest-priority
/// compiled runtime (`Wgpu`, `Cuda`, `Hip`, `Cpu`).
pub fn default_compute_target() -> DspResult<ComputeTarget> {
    ComputeTarget::from_env().map_err(|e| DspError::Model(e.to_string()))
}

/// Pulls and SHA-256 verifies `model_id` via [`ModelHub`], returning its [`ModelManifest`] and
/// local weights path as a [`DspResult`].
pub fn pull_model(model_id: &str) -> DspResult<(ModelManifest, PathBuf)> {
    let hub = ModelHub::new().map_err(|e| DspError::Model(e.to_string()))?;
    let entry = hub
        .pull(model_id, false)
        .map_err(|e| DspError::Model(e.to_string()))?;
    Ok((entry.manifest, entry.local_weights_path))
}

/// Validates a runtime tensor's dimensions against a [`TensorPortSpec`] from [`crate::hub::ModelManifest`].
///
/// Any positive dimension `d > 0` in `spec.shape` must match `actual_shape[i]` exactly;
/// `-1` denotes a dynamic dimension (e.g., batch size, variable channel count, or basis rank).
pub fn validate_tensor_port(spec: &TensorPortSpec, actual_shape: &[usize]) -> DspResult<()> {
    if spec.shape.len() != actual_shape.len() {
        return Err(DspError::Model(format!(
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
            return Err(DspError::Model(format!(
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

    #[test]
    fn test_cubecl_basis_project_and_reconstruct_roundtrip() {
        let target = default_compute_target().unwrap();
        // 2 orthonormal basis vectors of length 4
        let basis: Vec<f32> = vec![
            0.5, 0.5, 0.5, 0.5, // component 0
            0.5, -0.5, 0.5, -0.5, // component 1
        ];
        // 1 spike, 2 channels, window_len = 4
        // ch 0 = 2 * b0 + 1 * b1 = [1.5, 0.5, 1.5, 0.5]
        // ch 1 = -1 * b0 + 3 * b1 = [1.0, -2.0, 1.0, -2.0]
        let snippets: Vec<f32> = vec![1.5, 0.5, 1.5, 0.5, 1.0, -2.0, 1.0, -2.0];

        let coeffs = run_on_target(
            target,
            ProjectBasisTask {
                snippets: &snippets,
                num_spikes: 1,
                num_channels: 2,
                window_len: 4,
                basis: &basis,
                num_components: 2,
            },
        )
        .unwrap();

        assert_eq!(coeffs.len(), 4);
        assert!((coeffs[0] - 2.0).abs() < 1e-4);
        assert!((coeffs[1] - 1.0).abs() < 1e-4);
        assert!((coeffs[2] - (-1.0)).abs() < 1e-4);
        assert!((coeffs[3] - 3.0).abs() < 1e-4);

        let recon = run_on_target(
            target,
            ReconstructBasisTask {
                coeffs: &coeffs,
                num_spikes: 1,
                num_channels: 2,
                num_components: 2,
                basis: &basis,
                window_len: 4,
            },
        )
        .unwrap();

        for (a, b) in snippets.iter().zip(recon.iter()) {
            assert!((a - b).abs() < 1e-4, "mismatch: {a} vs {b}");
        }
    }
}
