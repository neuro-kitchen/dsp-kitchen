//! Parser and serializer for the HuggingFace `.safetensors` binary format powered by `safetensors` (`0.8.0`).
//!
//! Binary layout specification:
//! - Bytes `0..8`: `u64` little-endian header byte length $N$.
//! - Bytes `8..8+N`: UTF-8 JSON header mapping tensor keys to `{ "dtype": "F32", "shape": [...], "data_offsets": [start, end] }`.
//! - Bytes `8+N..`: Contiguous little-endian raw tensor buffers.

use std::collections::BTreeMap;
use std::path::Path;
use anyhow::{Context, Result, bail};
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use serde::{Deserialize, Serialize};
use crate::backend::{SynapseMlDevice, Tensor};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetensorEntryHeader {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

/// In-memory collection of named tensors compatible with PyTorch / HuggingFace `.safetensors`.
#[derive(Debug, Clone, Default)]
pub struct SafetensorsMap {
    tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
}

impl SafetensorsMap {
    pub fn new() -> Self {
        Self {
            tensors: BTreeMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Inserts a rank-`D` `Tensor` into the map.
    pub fn insert_tensor<const D: usize>(&mut self, name: impl Into<String>, tensor: &Tensor<D>) {
        self.tensors.insert(
            name.into(),
            (tensor.shape.to_vec(), tensor.as_slice().to_vec()),
        );
    }

    /// Retrieves a rank-`D` `Tensor` by name, validating its rank and shape.
    pub fn get_tensor<const D: usize>(
        &self,
        name: &str,
        device: SynapseMlDevice,
    ) -> Result<Tensor<D>> {
        let (shape_vec, data) = self
            .tensors
            .get(name)
            .with_context(|| format!("Tensor '{}' not found in safetensors map", name))?;

        if shape_vec.len() != D {
            bail!(
                "Tensor '{}' has rank {}, expected rank {}",
                name,
                shape_vec.len(),
                D
            );
        }

        let mut shape = [0usize; D];
        shape.copy_from_slice(shape_vec);
        Ok(Tensor::from_floats(data.clone(), shape, device))
    }

    /// Serializes all stored tensors into the standard `.safetensors` binary format via `safetensors::serialize`.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut views = Vec::with_capacity(self.tensors.len());
        for (name, (shape, data)) in &self.tensors {
            let raw_bytes: &[u8] = bytemuck::cast_slice(data.as_slice());
            let view = TensorView::new(Dtype::F32, shape.clone(), raw_bytes)
                .with_context(|| format!("Failed to construct TensorView for '{}'", name))?;
            views.push((name.as_str(), view));
        }
        let encoded = safetensors::serialize(views, None)
            .context("Failed to serialize SafetensorsMap")?;
        Ok(encoded)
    }

    /// Parses a `.safetensors` byte buffer into a `SafetensorsMap` via `safetensors::SafeTensors::deserialize`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let st = SafeTensors::deserialize(bytes)
            .context("Failed to deserialize .safetensors buffer")?;

        let mut tensors = BTreeMap::new();
        for (key, view) in st.tensors() {
            if view.dtype() != Dtype::F32 {
                bail!(
                    "Unsupported dtype {:?} for tensor '{}', only F32 is supported",
                    view.dtype(),
                    key
                );
            }
            let raw_slice = view.data();
            let mut vec = Vec::with_capacity(raw_slice.len() / 4);
            for chunk in raw_slice.chunks_exact(4) {
                vec.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            tensors.insert(key, (view.shape().to_vec(), vec));
        }

        Ok(Self { tensors })
    }

    pub fn save_to_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let bytes = self.to_bytes()?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safetensors_binary_roundtrip() {
        let mut map = SafetensorsMap::new();
        let w = Tensor::<2>::from_floats(vec![1.0, -2.0, 3.5, 4.25], [2, 2], SynapseMlDevice::Cpu);
        let b = Tensor::<1>::from_floats(vec![0.5, -0.5], [2], SynapseMlDevice::Cpu);

        map.insert_tensor("linear.weight", &w);
        map.insert_tensor("linear.bias", &b);

        let encoded = map.to_bytes().unwrap();
        let decoded = SafetensorsMap::from_bytes(&encoded).unwrap();

        let w_loaded: Tensor<2> = decoded.get_tensor("linear.weight", SynapseMlDevice::Cpu).unwrap();
        let b_loaded: Tensor<1> = decoded.get_tensor("linear.bias", SynapseMlDevice::Cpu).unwrap();

        assert_eq!(w_loaded, w);
        assert_eq!(b_loaded, b);
    }
}
