//! `MonopolarMlpLocalizer`: Neural network regressor predicting 3D physical source
//! coordinates `(x_um, y_um, z_um)` from $K$-nearest neighbor peak-to-peak amplitudes.
//!
//! Implements `dsp_synapse::traits::PeakLocalizer`.

use anyhow::Result;
use dsp_core::SensorLayout;
use dsp_synapse::{PeakLocalizer, SnippetBatch};
use crate::backbones::MlpBackbone;
use crate::backend::{SynapseMlDevice, Tensor2D};
use crate::hub::SafetensorsMap;

#[derive(Debug, Clone)]
pub struct MonopolarMlpLocalizer {
    pub mlp: MlpBackbone,
    pub k_neighbors: usize,
    pub device: SynapseMlDevice,
}

impl MonopolarMlpLocalizer {
    pub fn new(k_neighbors: usize, seed: u64, device: SynapseMlDevice) -> Self {
        let k = k_neighbors.max(1);
        // Input features per neighbor: [normalized_ptp_amp, dx_um / 100.0, dy_um / 100.0] -> 3 * K
        let in_features = k * 3;
        // Outputs: [dx_offset_scale, dy_offset_scale, z_distance_scale] -> 3
        let mlp = MlpBackbone::new(in_features, &[32, 32], 3, seed, device);
        Self {
            mlp,
            k_neighbors: k,
            device,
        }
    }

    pub fn save_weights(&self, map: &mut SafetensorsMap) {
        self.mlp.save_weights("monopolar_mlp", map);
    }

    pub fn load_weights(&mut self, map: &SafetensorsMap) -> Result<()> {
        self.mlp.load_weights("monopolar_mlp", map, self.device)
    }
}

impl PeakLocalizer for MonopolarMlpLocalizer {
    fn localize(&self, batch: &SnippetBatch, layout: &SensorLayout) -> dsp_core::DspResult<Vec<[f32; 3]>> {
        let n = batch.num_spikes;
        if n == 0 {
            return Ok(Vec::new());
        }

        let k_model = self.k_neighbors;
        let in_dim = k_model * 3;
        let mut features = vec![0.0f32; n * in_dim];
        let mut anchors = Vec::with_capacity(n);
        let mut com_offsets = Vec::with_capacity(n);

        for i in 0..n {
            let prim_ch = batch.primary_channels[i];
            let (px, py) = layout
                .get_site(prim_ch)
                .map(|s| (s.position.x_um, s.position.y_um))
                .unwrap_or((0.0, 0.0));
            anchors.push((px, py));

            let ch_ids = batch.spike_channel_ids(i);
            let k_actual = ch_ids.len().min(k_model);

            // Compute peak-to-peak amplitude per channel and Center-of-Mass baseline
            let mut ptp = vec![0.0f32; k_actual];
            let mut max_ptp = 1e-6f32;
            for ch_idx in 0..k_actual {
                let wave = batch.channel_slice(i, ch_idx);
                let mut min_v = f32::INFINITY;
                let mut max_v = f32::NEG_INFINITY;
                for &v in wave {
                    if v < min_v {
                        min_v = v;
                    }
                    if v > max_v {
                        max_v = v;
                    }
                }
                let amp = (max_v - min_v).max(0.0);
                ptp[ch_idx] = amp;
                if amp > max_ptp {
                    max_ptp = amp;
                }
            }

            let mut sum_w = 0.0f32;
            let mut com_dx = 0.0f32;
            let mut com_dy = 0.0f32;

            for ch_idx in 0..k_actual {
                let ch_id = ch_ids[ch_idx];
                let (cx, cy) = layout
                    .get_site(ch_id)
                    .map(|s| (s.position.x_um, s.position.y_um))
                    .unwrap_or((px, py));
                let dx = cx - px;
                let dy = cy - py;
                let norm_amp = ptp[ch_idx] / max_ptp;

                sum_w += norm_amp;
                com_dx += norm_amp * dx;
                com_dy += norm_amp * dy;

                let feat_off = i * in_dim + ch_idx * 3;
                features[feat_off] = norm_amp;
                features[feat_off + 1] = dx / 100.0;
                features[feat_off + 2] = dy / 100.0;
            }

            if sum_w > 1e-6 {
                com_dx /= sum_w;
                com_dy /= sum_w;
            }
            com_offsets.push((com_dx, com_dy));
        }

        let input_tensor = Tensor2D::from_floats(features, [n, in_dim], self.device);
        // Residual MLP correction around physical Center-of-Mass
        let delta = self.mlp.forward(&input_tensor).tanh();

        let mut coords = Vec::with_capacity(n);
        for i in 0..n {
            let (px, py) = anchors[i];
            let (cdx, cdy) = com_offsets[i];
            let x = px + cdx + delta.data[i * 3] * 15.0;
            let y = py + cdy + delta.data[i * 3 + 1] * 15.0;
            // Perpendicular distance z >= 1.0 um from electrode plane
            let z = (15.0 + delta.data[i * 3 + 2] * 10.0).max(1.0);
            coords.push([x, y, z]);
        }
        Ok(coords)
    }
}
