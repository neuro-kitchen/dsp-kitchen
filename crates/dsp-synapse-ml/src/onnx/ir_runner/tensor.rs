//! [`DynTensor`]: the dynamic-rank values flowing through the interpreter.

use std::sync::Arc;

use anyhow::{Result, bail};

use crate::backend::{SynapseMlDevice, Tensor};

/// Dynamic-rank `f32` tensor value flowing through [`super::OnnxGraphRunner`].
///
/// The data is shared: looking a value up, reshaping or flattening it never copies the elements.
#[derive(Debug, Clone, PartialEq)]
pub struct DynTensor {
    pub shape: Vec<usize>,
    pub data: Arc<Vec<f32>>,
    pub device: SynapseMlDevice,
}

impl DynTensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>, device: SynapseMlDevice) -> Result<Self> {
        Self::shared(shape, Arc::new(data), device)
    }

    /// A tensor over already shared data (no copy).
    pub fn shared(shape: Vec<usize>, data: Arc<Vec<f32>>, device: SynapseMlDevice) -> Result<Self> {
        let expected = if shape.is_empty() { 1 } else { shape.iter().product::<usize>() };
        if data.len() != expected {
            bail!("DynTensor shape {shape:?} expects {expected} elements, got {}", data.len());
        }
        Ok(Self { shape, data, device })
    }

    /// The same elements under `shape` (no copy).
    pub fn reshaped(&self, shape: Vec<usize>) -> Result<Self> {
        Self::shared(shape, self.data.clone(), self.device)
    }

    /// `f` applied to every element.
    pub fn map(&self, f: impl Fn(f32) -> f32) -> Self {
        Self { shape: self.shape.clone(), data: Arc::new(self.data.iter().map(|&v| f(v)).collect()), device: self.device }
    }

    pub fn from_tensor<const D: usize>(t: &Tensor<D>) -> Self {
        Self { shape: t.shape.to_vec(), data: Arc::new(t.data.clone()), device: t.device }
    }

    /// Converts to a fixed-rank [`Tensor`], moving the data when this is its only owner.
    pub fn into_tensor<const D: usize>(self) -> Result<Tensor<D>> {
        if self.shape.len() != D {
            bail!("Expected rank-{} tensor, got rank-{} with shape {:?}", D, self.shape.len(), self.shape);
        }
        let mut arr = [0usize; D];
        arr.copy_from_slice(&self.shape);
        let data = Arc::try_unwrap(self.data).unwrap_or_else(|shared| (*shared).clone());
        Ok(Tensor::from_floats(data, arr, self.device))
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }
}
