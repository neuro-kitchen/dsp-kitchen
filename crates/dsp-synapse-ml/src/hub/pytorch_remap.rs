//! PyTorch `state_dict` / `.safetensors` key remapping and tensor layout transposition
//! utilities for importing weights from external Python spike sorters (Kilosort4, DARTsort,
//! CEBRA, Bombcell/UnitMatch).

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use super::safetensors::SafetensorsMap;

/// Transformation applied to a tensor during PyTorch weight import.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightTransform {
    /// Keep shape and element order unchanged.
    Identity,
    /// Transpose a 2D weight matrix `[out_features, in_features] -> [in_features, out_features]`.
    Transpose2D,
    /// Flip the temporal kernel axis (dim 2) of a 3D `Conv1d` weight `[out_ch, in_ch, K]`.
    FlipConv1dKernel,
    /// Permute 3D `Conv1d` weight from `[kernel_size, in_channels, out_channels]`
    /// to `[out_channels, in_channels, kernel_size]`.
    PermuteConv1dKioToOik,
}

/// Rule mapping a source key (or prefix) in an external PyTorch `.safetensors` file
/// to a target key, with an optional tensor layout transform.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PyTorchRemapRule {
    pub source_key: String,
    pub target_key: String,
    pub transform: WeightTransform,
}

impl PyTorchRemapRule {
    pub fn identity(source_key: impl Into<String>, target_key: impl Into<String>) -> Self {
        Self {
            source_key: source_key.into(),
            target_key: target_key.into(),
            transform: WeightTransform::Identity,
        }
    }

    pub fn linear_weight(source_key: impl Into<String>, target_key: impl Into<String>) -> Self {
        Self {
            source_key: source_key.into(),
            target_key: target_key.into(),
            transform: WeightTransform::Transpose2D,
        }
    }

    /// Generates both `.weight` (`Transpose2D`) and `.bias` (`Identity`) remap rules for a `nn.Linear` layer.
    pub fn linear_pair(source_prefix: &str, target_prefix: &str) -> [Self; 2] {
        [
            Self::linear_weight(
                format!("{}.weight", source_prefix),
                format!("{}.weight", target_prefix),
            ),
            Self::identity(
                format!("{}.bias", source_prefix),
                format!("{}.bias", target_prefix),
            ),
        ]
    }

    /// Generates `.weight` and `.bias` (`Identity`) remap rules for a `nn.Conv1d` or `nn.LayerNorm` layer.
    pub fn conv1d_or_norm_pair(source_prefix: &str, target_prefix: &str) -> [Self; 2] {
        [
            Self::identity(
                format!("{}.weight", source_prefix),
                format!("{}.weight", target_prefix),
            ),
            Self::identity(
                format!("{}.bias", source_prefix),
                format!("{}.bias", target_prefix),
            ),
        ]
    }

    /// Generates the 4 remap rules (`.weight`, `.bias`, `.running_mean`, `.running_var`) for `nn.BatchNorm1d`.
    pub fn batch_norm_1d(source_prefix: &str, target_prefix: &str) -> [Self; 4] {
        [
            Self::identity(
                format!("{}.weight", source_prefix),
                format!("{}.weight", target_prefix),
            ),
            Self::identity(
                format!("{}.bias", source_prefix),
                format!("{}.bias", target_prefix),
            ),
            Self::identity(
                format!("{}.running_mean", source_prefix),
                format!("{}.running_mean", target_prefix),
            ),
            Self::identity(
                format!("{}.running_var", source_prefix),
                format!("{}.running_var", target_prefix),
            ),
        ]
    }
}

/// Transposes a contiguous 2D row-major matrix `[rows, cols]` into `[cols, rows]`.
pub fn transpose_2d_slice(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(
        data.len(),
        rows * cols,
        "transpose_2d_slice: expected {} elements for [{} x {}], got {}",
        rows * cols,
        rows,
        cols,
        data.len()
    );
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row_off = r * cols;
        for c in 0..cols {
            out[c * rows + r] = data[row_off + c];
        }
    }
    out
}

/// Flips the last axis `K` of a 3D tensor `[C_out, C_in, K]`.
pub fn flip_conv1d_kernel_slice(data: &[f32], c_out: usize, c_in: usize, k: usize) -> Vec<f32> {
    assert_eq!(data.len(), c_out * c_in * k);
    let mut out = vec![0.0f32; data.len()];
    for co in 0..c_out {
        for ci in 0..c_in {
            let off = (co * c_in + ci) * k;
            for i in 0..k {
                out[off + i] = data[off + (k - 1 - i)];
            }
        }
    }
    out
}

/// Permutes a 3D conv kernel from `[K, C_in, C_out]` to `[C_out, C_in, K]`.
pub fn permute_kio_to_oik(data: &[f32], k: usize, c_in: usize, c_out: usize) -> Vec<f32> {
    assert_eq!(data.len(), k * c_in * c_out);
    let mut out = vec![0.0f32; data.len()];
    for ki in 0..k {
        for ci in 0..c_in {
            for co in 0..c_out {
                let src = (ki * c_in + ci) * c_out + co;
                let dst = (co * c_in + ci) * k + ki;
                out[dst] = data[src];
            }
        }
    }
    out
}

/// Configurable adapter that converts a raw PyTorch `.safetensors` checkpoint into a
/// remapped `SafetensorsMap`.
#[derive(Debug, Clone, Default)]
pub struct PyTorchWeightAdapter {
    pub rules: Vec<PyTorchRemapRule>,
    /// Optional prefix stripped from incoming PyTorch keys before rule matching
    /// (e.g. `"module."` from `DistributedDataParallel` or `"_orig_mod."` from `torch.compile`).
    pub strip_prefixes: Vec<String>,
    /// If true, keys not matched by any explicit rule are copied through unchanged.
    pub passthrough_unmatched: bool,
}

impl PyTorchWeightAdapter {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            strip_prefixes: vec![
                "module.".to_string(),
                "_orig_mod.".to_string(),
                "model.".to_string(),
            ],
            passthrough_unmatched: false,
        }
    }

    pub fn with_passthrough(mut self, passthrough: bool) -> Self {
        self.passthrough_unmatched = passthrough;
        self
    }

    pub fn add_rule(mut self, rule: PyTorchRemapRule) -> Self {
        self.rules.push(rule);
        self
    }

    pub fn add_rules(mut self, rules: impl IntoIterator<Item = PyTorchRemapRule>) -> Self {
        self.rules.extend(rules);
        self
    }

    fn normalize_source_key<'a>(&self, raw_key: &'a str) -> &'a str {
        let mut key = raw_key;
        for prefix in &self.strip_prefixes {
            if let Some(stripped) = key.strip_prefix(prefix.as_str()) {
                key = stripped;
            }
        }
        key
    }

    /// Applies all remap and transposition rules to `source`, returning a new `SafetensorsMap`.
    pub fn adapt(&self, source: &SafetensorsMap) -> Result<SafetensorsMap> {
        let mut out = SafetensorsMap::new();

        for (raw_key, (shape, data)) in source.iter() {
            let norm_key = self.normalize_source_key(raw_key);
            let matched_rule = self
                .rules
                .iter()
                .find(|r| r.source_key.as_str() == raw_key.as_str() || r.source_key == norm_key);

            if let Some(rule) = matched_rule {
                let (new_shape, new_data) =
                    apply_transform(rule.transform, shape, data, &rule.source_key)?;
                out.insert_raw(rule.target_key.clone(), new_shape, new_data);
            } else if self.passthrough_unmatched {
                out.insert_raw(norm_key.to_string(), shape.clone(), data.clone());
            }
        }

        Ok(out)
    }

    /// Loads and adapts a `.safetensors` byte slice in one step.
    pub fn adapt_from_bytes(&self, bytes: &[u8]) -> Result<SafetensorsMap> {
        let raw_map = SafetensorsMap::from_bytes(bytes)
            .context("Failed to parse source PyTorch .safetensors bytes")?;
        self.adapt(&raw_map)
    }
}

fn apply_transform(
    transform: WeightTransform,
    shape: &[usize],
    data: &[f32],
    key: &str,
) -> Result<(Vec<usize>, Vec<f32>)> {
    match transform {
        WeightTransform::Identity => Ok((shape.to_vec(), data.to_vec())),
        WeightTransform::Transpose2D => {
            if shape.len() != 2 {
                bail!(
                    "WeightTransform::Transpose2D requires rank-2 tensor for '{}', got shape {:?}",
                    key,
                    shape
                );
            }
            let rows = shape[0];
            let cols = shape[1];
            Ok((vec![cols, rows], transpose_2d_slice(data, rows, cols)))
        }
        WeightTransform::FlipConv1dKernel => {
            if shape.len() != 3 {
                bail!(
                    "WeightTransform::FlipConv1dKernel requires rank-3 tensor for '{}', got shape {:?}",
                    key,
                    shape
                );
            }
            Ok((
                shape.to_vec(),
                flip_conv1d_kernel_slice(data, shape[0], shape[1], shape[2]),
            ))
        }
        WeightTransform::PermuteConv1dKioToOik => {
            if shape.len() != 3 {
                bail!(
                    "WeightTransform::PermuteConv1dKioToOik requires rank-3 tensor for '{}', got shape {:?}",
                    key,
                    shape
                );
            }
            let (k, c_in, c_out) = (shape[0], shape[1], shape[2]);
            Ok((
                vec![c_out, c_in, k],
                permute_kio_to_oik(data, k, c_in, c_out),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pytorch_linear_transpose_and_key_remap() {
        let mut pt_map = SafetensorsMap::new();
        pt_map.insert_raw(
            "module.encoder.fc1.weight",
            vec![2, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        pt_map.insert_raw("module.encoder.fc1.bias", vec![2], vec![0.5, -0.5]);

        let adapter = PyTorchWeightAdapter::new()
            .add_rules(PyTorchRemapRule::linear_pair("encoder.fc1", "fc1"));

        let adapted = adapter.adapt(&pt_map).unwrap();
        let (w_shape, w_data) = adapted.get_raw("fc1.weight").unwrap();
        assert_eq!(w_shape, &[3, 2]);
        assert_eq!(w_data, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        let (b_shape, b_data) = adapted.get_raw("fc1.bias").unwrap();
        assert_eq!(b_shape, &[2]);
        assert_eq!(b_data, &[0.5, -0.5]);
    }
}
