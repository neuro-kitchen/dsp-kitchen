//! Native zero-dependency parser and serializer for the HuggingFace `.safetensors` binary format.
//!
//! Binary layout specification:
//! - Bytes `0..8`: `u64` little-endian header byte length $N$.
//! - Bytes `8..8+N`: UTF-8 JSON header mapping tensor keys to `{ "dtype": "F32", "shape": [...], "data_offsets": [start, end] }`.
//! - Bytes `8+N..`: Contiguous little-endian raw tensor buffers.

use std::collections::BTreeMap;
use std::path::Path;
use anyhow::{Context, Result, bail};
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

    /// Serializes all stored tensors into the standard `.safetensors` binary format.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut header_map: BTreeMap<String, SafetensorEntryHeader> = BTreeMap::new();
        let mut current_offset = 0usize;

        for (name, (shape, data)) in &self.tensors {
            let byte_len = data.len() * std::mem::size_of::<f32>();
            header_map.insert(
                name.clone(),
                SafetensorEntryHeader {
                    dtype: "F32".to_string(),
                    shape: shape.clone(),
                    data_offsets: [current_offset, current_offset + byte_len],
                },
            );
            current_offset += byte_len;
        }

        let mut json_bytes = serde_json::to_vec(&header_map)?;
        // Pad JSON header to 8-byte alignment with spaces per safetensors spec
        while json_bytes.len() % 8 != 0 {
            json_bytes.push(b' ');
        }

        let header_len = json_bytes.len() as u64;
        let mut out = Vec::with_capacity(8 + json_bytes.len() + current_offset);
        out.extend_from_slice(&header_len.to_le_bytes());
        out.extend_from_slice(&json_bytes);

        for (_name, (_shape, data)) in &self.tensors {
            for &val in data {
                out.extend_from_slice(&val.to_le_bytes());
            }
        }

        Ok(out)
    }

    /// Parses a `.safetensors` byte buffer into a `SafetensorsMap`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 8 {
            bail!("Invalid safetensors buffer: shorter than 8-byte header length");
        }

        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&bytes[0..8]);
        let header_len = u64::from_le_bytes(len_bytes) as usize;

        if bytes.len() < 8 + header_len {
            bail!(
                "Invalid safetensors buffer: header_len {} exceeds buffer size {}",
                header_len,
                bytes.len()
            );
        }

        let header_slice = &bytes[8..8 + header_len];
        let raw_body = &bytes[8 + header_len..];

        let raw_json: serde_json::Value = serde_json::from_slice(header_slice)
            .context("Failed to parse safetensors JSON header")?;
        let obj = raw_json
            .as_object()
            .context("Safetensors header must be a JSON object")?;

        let mut tensors = BTreeMap::new();
        for (key, val) in obj {
            if key == "__metadata__" {
                continue;
            }
            let entry: SafetensorEntryHeader = serde_json::from_value(val.clone())
                .with_context(|| format!("Invalid tensor entry header for '{}'", key))?;

            if entry.dtype != "F32" {
                bail!(
                    "Unsupported dtype '{}' for tensor '{}', only F32 is supported",
                    entry.dtype,
                    key
                );
            }

            let [start, end] = entry.data_offsets;
            if end > raw_body.len() || start > end || (end - start) % 4 != 0 {
                bail!("Invalid data_offsets {:?} for tensor '{}'", entry.data_offsets, key);
            }

            let num_floats = (end - start) / 4;
            let expected_numel: usize = entry.shape.iter().product();
            if num_floats != expected_numel {
                bail!(
                    "Shape {:?} mismatch with byte length {} for tensor '{}'",
                    entry.shape,
                    end - start,
                    key
                );
            }

            let mut vec = Vec::with_capacity(num_floats);
            let slice = &raw_body[start..end];
            for chunk in slice.chunks_exact(4) {
                vec.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }

            tensors.insert(key.clone(), (entry.shape, vec));
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
