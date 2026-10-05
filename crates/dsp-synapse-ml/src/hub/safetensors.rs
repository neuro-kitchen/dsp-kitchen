//! Parser and serializer for the HuggingFace `.safetensors` binary format powered by `safetensors` (`0.8.0`).
//!
//! Binary layout specification:
//! - Bytes `0..8`: `u64` little-endian header byte length $N$.
//! - Bytes `8..8+N`: UTF-8 JSON header mapping tensor keys to `{ "dtype": "F32", "shape": [...], "data_offsets": [start, end] }`.
//! - Bytes `8+N..`: Contiguous little-endian raw tensor buffers.

use std::collections::BTreeMap;
use std::path::Path;
use dsp_core::{DspError, DspResult as Result};
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetensorEntryHeader {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

/// In-memory collection of named `(shape, Vec<f32>)` tensors compatible with PyTorch / HuggingFace `.safetensors`.
#[derive(Debug, Clone, Default, PartialEq)]
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

    /// Iterates over all `(key, (shape, data))` entries in deterministic key order.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &(Vec<usize>, Vec<f32>))> {
        self.tensors.iter()
    }

    /// Inserts a dynamic shape + `Vec<f32>` buffer into the map.
    pub fn insert_raw(&mut self, name: impl Into<String>, shape: Vec<usize>, data: Vec<f32>) {
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "SafetensorsMap::insert_raw shape {:?} expects {} elements, got {}",
            shape,
            expected,
            data.len()
        );
        self.tensors.insert(name.into(), (shape, data));
    }

    /// Retrieves a `(shape, data)` slice entry by name.
    pub fn get_raw(&self, name: &str) -> Result<(&[usize], &[f32])> {
        let (shape, data) = self
            .tensors
            .get(name)
            .ok_or_else(|| DspError::InvalidConfig(format!("tensor '{name}' not found in safetensors map")))?;
        Ok((shape.as_slice(), data.as_slice()))
    }

    /// Serializes all stored tensors into the standard `.safetensors` binary format via `safetensors::serialize`.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut views = Vec::with_capacity(self.tensors.len());
        for (name, (shape, data)) in &self.tensors {
            let raw_bytes: &[u8] = bytemuck::cast_slice(data.as_slice());
            let view = TensorView::new(Dtype::F32, shape.clone(), raw_bytes)
                .map_err(|e| DspError::InvalidConfig(format!("failed to construct TensorView for '{name}': {e}")))?;
            views.push((name.as_str(), view));
        }
        let encoded = safetensors::serialize(views, None)
            .map_err(|e| DspError::InvalidConfig(format!("Failed to serialize SafetensorsMap: {e}")))?;
        Ok(encoded)
    }

    /// Parses a `.safetensors` byte buffer into a `SafetensorsMap` via `safetensors::SafeTensors::deserialize`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let st = SafeTensors::deserialize(bytes)
            .map_err(|e| DspError::InvalidConfig(format!("Failed to deserialize .safetensors buffer: {e}")))?;

        let mut tensors = BTreeMap::new();
        for (key, view) in st.tensors() {
            if view.dtype() != Dtype::F32 {
                return Err(DspError::UnsupportedFormat(format!(
                    "unsupported dtype {:?} for tensor '{}', only F32 is supported",
                    view.dtype(),
                    key
                )));
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
        std::fs::write(path.as_ref(), bytes).map_err(|e| DspError::Io(format!("{}: {e}", path.as_ref().display())))
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref()).map_err(|e| DspError::Io(format!("{}: {e}", path.as_ref().display())))?;
        Self::from_bytes(&bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safetensors_binary_roundtrip() {
        let mut map = SafetensorsMap::new();
        map.insert_raw("linear.weight", vec![2, 2], vec![1.0, -2.0, 3.5, 4.25]);
        map.insert_raw("linear.bias", vec![2], vec![0.5, -0.5]);

        let encoded = map.to_bytes().unwrap();
        let decoded = SafetensorsMap::from_bytes(&encoded).unwrap();

        let (w_shape, w_data) = decoded.get_raw("linear.weight").unwrap();
        assert_eq!(w_shape, &[2, 2]);
        assert_eq!(w_data, &[1.0, -2.0, 3.5, 4.25]);

        let (b_shape, b_data) = decoded.get_raw("linear.bias").unwrap();
        assert_eq!(b_shape, &[2]);
        assert_eq!(b_data, &[0.5, -0.5]);
    }
}
