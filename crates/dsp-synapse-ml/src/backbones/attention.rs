//! Cross-electrode spatial-temporal self-attention block (`CrossElectrodeAttention`).
//!
//! Treats each of the $K$ neighboring electrodes as a token with feature vector of length $T$,
//! computing scaled dot-product attention $\text{softmax}(Q K^T / \sqrt{d_k}) V$ + residual LayerNorm.

use anyhow::Result;
use crate::backend::{SynapseMlDevice, Tensor2D, Tensor3D};
use crate::hub::SafetensorsMap;
use super::mlp::{LayerNorm1D, LinearLayer};

/// Spatial cross-electrode self-attention over `[batch, num_electrodes, model_dim]`.
#[derive(Debug, Clone)]
pub struct CrossElectrodeAttention {
    pub q_proj: LinearLayer,
    pub k_proj: LinearLayer,
    pub v_proj: LinearLayer,
    pub out_proj: LinearLayer,
    pub norm: LayerNorm1D,
    pub model_dim: usize,
}

impl CrossElectrodeAttention {
    pub fn new(model_dim: usize, seed: u64, device: SynapseMlDevice) -> Self {
        Self {
            q_proj: LinearLayer::new_initialized(model_dim, model_dim, seed, device),
            k_proj: LinearLayer::new_initialized(
                model_dim,
                model_dim,
                seed.wrapping_add(11),
                device,
            ),
            v_proj: LinearLayer::new_initialized(
                model_dim,
                model_dim,
                seed.wrapping_add(22),
                device,
            ),
            out_proj: LinearLayer::new_initialized(
                model_dim,
                model_dim,
                seed.wrapping_add(33),
                device,
            ),
            norm: LayerNorm1D::new(model_dim, device),
            model_dim,
        }
    }

    /// Forward pass `[batch, num_electrodes, model_dim] -> [batch, num_electrodes, model_dim]`.
    pub fn forward(&self, input: &Tensor3D) -> Tensor3D {
        let [batch, k_elec, d] = input.shape;
        assert_eq!(d, self.model_dim, "CrossElectrodeAttention model_dim mismatch");

        let scale = 1.0 / (d.max(1) as f32).sqrt();
        let mut out_flat = Vec::with_capacity(batch * k_elec * d);

        for b in 0..batch {
            let slice = &input.data[b * k_elec * d..(b + 1) * k_elec * d];
            let x_b = Tensor2D::from_floats(slice.to_vec(), [k_elec, d], input.device);

            let q = self.q_proj.forward(&x_b); // [k_elec, d]
            let k = self.k_proj.forward(&x_b); // [k_elec, d]
            let v = self.v_proj.forward(&x_b); // [k_elec, d]

            // Transpose K: [k_elec, d] -> [d, k_elec]
            let mut k_t_data = vec![0.0f32; d * k_elec];
            for r in 0..k_elec {
                for c in 0..d {
                    k_t_data[c * k_elec + r] = k.data[r * d + c];
                }
            }
            let k_t = Tensor2D::from_floats(k_t_data, [d, k_elec], input.device);

            // Scores: [k_elec, k_elec]
            let scores = q.matmul(&k_t).mul_scalar(scale).softmax();
            // Context: [k_elec, d]
            let context = scores.matmul(&v);
            let projected = self.out_proj.forward(&context);
            let residual = self.norm.forward(&projected.add(&x_b));
            out_flat.extend_from_slice(&residual.data);
        }

        Tensor3D::from_floats(out_flat, [batch, k_elec, d], input.device)
    }

    pub fn save_weights(&self, prefix: &str, map: &mut SafetensorsMap) {
        self.q_proj.save_weights(&format!("{}.q_proj", prefix), map);
        self.k_proj.save_weights(&format!("{}.k_proj", prefix), map);
        self.v_proj.save_weights(&format!("{}.v_proj", prefix), map);
        self.out_proj
            .save_weights(&format!("{}.out_proj", prefix), map);
        self.norm.save_weights(&format!("{}.norm", prefix), map);
    }

    pub fn load_weights(
        &mut self,
        prefix: &str,
        map: &SafetensorsMap,
        device: SynapseMlDevice,
    ) -> Result<()> {
        self.q_proj
            .load_weights(&format!("{}.q_proj", prefix), map, device)?;
        self.k_proj
            .load_weights(&format!("{}.k_proj", prefix), map, device)?;
        self.v_proj
            .load_weights(&format!("{}.v_proj", prefix), map, device)?;
        self.out_proj
            .load_weights(&format!("{}.out_proj", prefix), map, device)?;
        self.norm
            .load_weights(&format!("{}.norm", prefix), map, device)?;
        Ok(())
    }
}
