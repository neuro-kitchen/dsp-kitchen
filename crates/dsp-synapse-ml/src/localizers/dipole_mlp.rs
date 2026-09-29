//! `DipoleMlpLocalizer`: Neural regressor predicting both 3D physical source position
//! `[x_um, y_um, z_um]` and 3D current dipole moment vector `[px, py, pz]`.
//!
//! Implements `dsp_synapse::traits::PeakLocalizer`.

use anyhow::Result;
use dsp_core::SensorLayout;
use dsp_synapse::{PeakLocalizer, SnippetBatch};
use crate::backbones::MlpBackbone;
use crate::backend::{SynapseMlDevice, Tensor2D};
use crate::hub::SafetensorsMap;

/// Full 6-DOF dipole source estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DipoleSourceEstimate {
    pub position_um: [f32; 3],
    pub dipole_moment: [f32; 3],
}

#[derive(Debug, Clone)]
pub struct DipoleMlpLocalizer {
    pub mlp: MlpBackbone,
    pub k_neighbors: usize,
    pub device: SynapseMlDevice,
}

impl DipoleMlpLocalizer {
    pub fn new(k_neighbors: usize, seed: u64, device: SynapseMlDevice) -> Self {
        let k = k_neighbors.max(1);
        // Features per channel: [trough_v, peak_v, dx, dy] -> 4 * K
        let in_features = k * 4;
        // Outputs: [dx, dy, dz, px, py, pz] -> 6
        let mlp = MlpBackbone::new(in_features, &[48, 32], 6, seed, device);
        Self {
            mlp,
            k_neighbors: k,
            device,
        }
    }

    /// Predicts full 6-DOF dipole position and orientation for each spike in `batch`.
    pub fn localize_dipoles(
        &self,
        batch: &SnippetBatch,
        layout: &SensorLayout,
    ) -> Vec<DipoleSourceEstimate> {
        let n = batch.num_spikes;
        if n == 0 {
            return Vec::new();
        }

        let k_model = self.k_neighbors;
        let in_dim = k_model * 4;
        let mut features = vec![0.0f32; n * in_dim];
        let mut anchors = Vec::with_capacity(n);

        for i in 0..n {
            let prim_ch = batch.primary_channels[i];
            let (px, py) = layout
                .get_site(prim_ch)
                .map(|s| (s.position.x_um, s.position.y_um))
                .unwrap_or((0.0, 0.0));
            anchors.push((px, py));

            let ch_ids = batch.spike_channel_ids(i);
            let k_actual = ch_ids.len().min(k_model);

            for ch_idx in 0..k_actual {
                let wave = batch.channel_slice(i, ch_idx);
                let min_v = wave.iter().copied().fold(f32::INFINITY, f32::min);
                let max_v = wave.iter().copied().fold(f32::NEG_INFINITY, f32::max);

                let (cx, cy) = layout
                    .get_site(ch_ids[ch_idx])
                    .map(|s| (s.position.x_um, s.position.y_um))
                    .unwrap_or((px, py));
                let off = i * in_dim + ch_idx * 4;
                features[off] = min_v / 100.0;
                features[off + 1] = max_v / 100.0;
                features[off + 2] = (cx - px) / 100.0;
                features[off + 3] = (cy - py) / 100.0;
            }
        }

        let x_tensor = Tensor2D::from_floats(features, [n, in_dim], self.device);
        let raw_out = self.mlp.forward(&x_tensor);

        let mut results = Vec::with_capacity(n);
        for i in 0..n {
            let (ax, ay) = anchors[i];
            let row = &raw_out.data[i * 6..(i + 1) * 6];
            let pos = [
                ax + row[0].tanh() * 30.0,
                ay + row[1].tanh() * 30.0,
                (20.0 + row[2].tanh() * 15.0).max(1.0),
            ];
            let p_norm = (row[3] * row[3] + row[4] * row[4] + row[5] * row[5])
                .sqrt()
                .max(1e-6);
            let moment = [row[3] / p_norm, row[4] / p_norm, row[5] / p_norm];
            results.push(DipoleSourceEstimate {
                position_um: pos,
                dipole_moment: moment,
            });
        }
        results
    }

    pub fn save_weights(&self, map: &mut SafetensorsMap) {
        self.mlp.save_weights("dipole_mlp", map);
    }

    pub fn load_weights(&mut self, map: &SafetensorsMap) -> Result<()> {
        self.mlp.load_weights("dipole_mlp", map, self.device)
    }
}

impl PeakLocalizer for DipoleMlpLocalizer {
    fn localize(&self, batch: &SnippetBatch, layout: &SensorLayout) -> Vec<[f32; 3]> {
        self.localize_dipoles(batch, layout)
            .into_iter()
            .map(|d| d.position_um)
            .collect()
    }
}
