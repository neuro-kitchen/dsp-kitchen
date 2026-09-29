//! Fully-connected `LinearLayer`, `LayerNorm1D`, and configurable Multi-Layer Perceptron (`MlpBackbone`).

use anyhow::Result;
use crate::backend::{SynapseMlDevice, Tensor1D, Tensor2D};
use crate::hub::SafetensorsMap;

/// Dense linear projection `y = x W + b` operating on 2D tensors `[batch, in_features] -> [batch, out_features]`.
#[derive(Debug, Clone)]
pub struct LinearLayer {
    /// Weight matrix of shape `[in_features, out_features]`.
    pub weight: Tensor2D,
    /// Bias vector of shape `[out_features]`.
    pub bias: Tensor1D,
}

impl LinearLayer {
    pub fn new_initialized(
        in_features: usize,
        out_features: usize,
        seed: u64,
        device: SynapseMlDevice,
    ) -> Self {
        let weight = Tensor2D::xavier_uniform(
            [in_features, out_features],
            in_features,
            out_features,
            seed,
            device,
        );
        let bias = Tensor1D::zeros([out_features], device);
        Self { weight, bias }
    }

    pub fn in_features(&self) -> usize {
        self.weight.shape[0]
    }

    pub fn out_features(&self) -> usize {
        self.weight.shape[1]
    }

    pub fn forward(&self, input: &Tensor2D) -> Tensor2D {
        input.matmul(&self.weight).add_bias_1d(&self.bias)
    }

    pub fn save_weights(&self, prefix: &str, map: &mut SafetensorsMap) {
        map.insert_tensor(format!("{}.weight", prefix), &self.weight);
        map.insert_tensor(format!("{}.bias", prefix), &self.bias);
    }

    pub fn load_weights(
        &mut self,
        prefix: &str,
        map: &SafetensorsMap,
        device: SynapseMlDevice,
    ) -> Result<()> {
        self.weight = map.get_tensor(&format!("{}.weight", prefix), device)?;
        self.bias = map.get_tensor(&format!("{}.bias", prefix), device)?;
        Ok(())
    }
}

/// Row-wise Layer Normalization over the last dimension `[batch, features]`.
#[derive(Debug, Clone)]
pub struct LayerNorm1D {
    pub gamma: Tensor1D,
    pub beta: Tensor1D,
    pub eps: f32,
}

impl LayerNorm1D {
    pub fn new(features: usize, device: SynapseMlDevice) -> Self {
        Self {
            gamma: Tensor1D::ones([features], device),
            beta: Tensor1D::zeros([features], device),
            eps: 1e-5,
        }
    }

    pub fn forward(&self, input: &Tensor2D) -> Tensor2D {
        let [m, n] = input.shape;
        assert_eq!(n, self.gamma.shape[0], "LayerNorm1D feature mismatch");
        let mut out = vec![0.0f32; m * n];
        if n == 0 {
            return Tensor2D::from_floats(out, [m, n], input.device);
        }

        let inv_n = 1.0 / (n as f32);
        for r in 0..m {
            let row = &input.data[r * n..(r + 1) * n];
            let mean: f32 = row.iter().sum::<f32>() * inv_n;
            let var: f32 = row
                .iter()
                .map(|&v| {
                    let d = v - mean;
                    d * d
                })
                .sum::<f32>()
                * inv_n;
            let inv_std = 1.0 / (var + self.eps).sqrt();

            for c in 0..n {
                let norm = (row[c] - mean) * inv_std;
                out[r * n + c] = norm * self.gamma.data[c] + self.beta.data[c];
            }
        }
        Tensor2D::from_floats(out, [m, n], input.device)
    }

    pub fn save_weights(&self, prefix: &str, map: &mut SafetensorsMap) {
        map.insert_tensor(format!("{}.weight", prefix), &self.gamma);
        map.insert_tensor(format!("{}.bias", prefix), &self.beta);
    }

    pub fn load_weights(
        &mut self,
        prefix: &str,
        map: &SafetensorsMap,
        device: SynapseMlDevice,
    ) -> Result<()> {
        self.gamma = map.get_tensor(&format!("{}.weight", prefix), device)?;
        self.beta = map.get_tensor(&format!("{}.bias", prefix), device)?;
        Ok(())
    }
}

/// Multi-Layer Perceptron with LayerNorm + GELU on hidden layers and a linear output layer.
#[derive(Debug, Clone)]
pub struct MlpBackbone {
    pub hidden_layers: Vec<(LinearLayer, LayerNorm1D)>,
    pub out_layer: LinearLayer,
}

impl MlpBackbone {
    pub fn new(
        in_features: usize,
        hidden_dims: &[usize],
        out_features: usize,
        seed: u64,
        device: SynapseMlDevice,
    ) -> Self {
        let mut hidden_layers = Vec::with_capacity(hidden_dims.len());
        let mut prev_dim = in_features;

        for (i, &h_dim) in hidden_dims.iter().enumerate() {
            let lin = LinearLayer::new_initialized(
                prev_dim,
                h_dim,
                seed.wrapping_add((i as u64 + 1) * 313),
                device,
            );
            let ln = LayerNorm1D::new(h_dim, device);
            hidden_layers.push((lin, ln));
            prev_dim = h_dim;
        }

        let out_layer = LinearLayer::new_initialized(
            prev_dim,
            out_features,
            seed.wrapping_add(9_999),
            device,
        );

        Self {
            hidden_layers,
            out_layer,
        }
    }

    pub fn forward(&self, input: &Tensor2D) -> Tensor2D {
        let mut x = input.clone();
        for (lin, ln) in &self.hidden_layers {
            x = ln.forward(&lin.forward(&x)).gelu();
        }
        self.out_layer.forward(&x)
    }

    pub fn save_weights(&self, prefix: &str, map: &mut SafetensorsMap) {
        for (i, (lin, ln)) in self.hidden_layers.iter().enumerate() {
            lin.save_weights(&format!("{}.hidden.{}.linear", prefix, i), map);
            ln.save_weights(&format!("{}.hidden.{}.norm", prefix, i), map);
        }
        self.out_layer
            .save_weights(&format!("{}.out_layer", prefix), map);
    }

    pub fn load_weights(
        &mut self,
        prefix: &str,
        map: &SafetensorsMap,
        device: SynapseMlDevice,
    ) -> Result<()> {
        for (i, (lin, ln)) in self.hidden_layers.iter_mut().enumerate() {
            lin.load_weights(&format!("{}.hidden.{}.linear", prefix, i), map, device)?;
            ln.load_weights(&format!("{}.hidden.{}.norm", prefix, i), map, device)?;
        }
        self.out_layer
            .load_weights(&format!("{}.out_layer", prefix), map, device)?;
        Ok(())
    }
}
