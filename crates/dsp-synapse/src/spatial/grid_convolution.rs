//! Grid Convolution Spike Source Localization (`grid_convolution.rs`).
//!
//! Evaluates a 3D grid of synthetic monopole footprints around the primary channel
//! and computes an exponentially weighted soft-argmax over cosine similarity scores
//! (matching `spikeinterface.sortingcomponents.peak_localization.GridConvolution`).

use dsp_core::SensorLayout;
use crate::extraction::SnippetBatch;
use crate::traits::PeakLocalizer;
use super::center_of_mass::waveform_peak_to_peak;

/// Localizes a single spike via Grid Convolution soft-argmax over a local 3D grid.
pub fn localize_spike_grid_convolution(
    channel_ids: &[usize],
    ptp_amplitudes: &[f32],
    layout: &SensorLayout,
    xy_radius_um: f32,
    xy_step_um: f32,
    z_values_um: &[f32],
    softmax_temperature: f32,
) -> [f32; 3] {
    if channel_ids.is_empty() || ptp_amplitudes.is_empty() {
        return [0.0, 0.0, 15.0];
    }

    let prim_site = match layout.get_site(channel_ids[0]) {
        Ok(s) => s,
        Err(_) => return [0.0, 0.0, 15.0],
    };
    let anchor_x = prim_site.position.x_um;
    let anchor_y = prim_site.position.y_um;

    let mut sites = Vec::with_capacity(channel_ids.len());
    let mut obs = Vec::with_capacity(channel_ids.len());
    for (&ch, &amp) in channel_ids.iter().zip(ptp_amplitudes.iter()) {
        if let Ok(s) = layout.get_site(ch) {
            sites.push([s.position.x_um, s.position.y_um, s.position.z_um]);
            obs.push(amp.max(0.0));
        }
    }

    let obs_norm = obs.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
    let step = xy_step_um.max(2.0);
    let n_steps = (xy_radius_um / step).round() as isize;
    let temp = softmax_temperature.max(0.01);

    let mut grid_pts = Vec::new();
    let mut scores = Vec::new();
    let mut max_score = f32::NEG_INFINITY;

    for ix in -n_steps..=n_steps {
        let gx = anchor_x + (ix as f32) * step;
        for iy in -n_steps..=n_steps {
            let gy = anchor_y + (iy as f32) * step;
            for &gz in z_values_um {
                let mut dot = 0.0f32;
                let mut syn_norm_sq = 0.0f32;
                for (s, &v_obs) in sites.iter().zip(obs.iter()) {
                    let dx = gx - s[0];
                    let dy = gy - s[1];
                    let dz = gz - s[2];
                    let w = 1.0 / (dx * dx + dy * dy + dz * dz + 1.0).sqrt();
                    dot += w * v_obs;
                    syn_norm_sq += w * w;
                }
                let cos_sim = dot / (syn_norm_sq.sqrt().max(1e-8) * obs_norm);
                let logit = cos_sim / temp;
                if logit > max_score {
                    max_score = logit;
                }
                grid_pts.push([gx, gy, gz]);
                scores.push(logit);
            }
        }
    }

    let mut sum_w = 0.0f32;
    let mut est = [0.0f32; 3];
    for (pt, &sc) in grid_pts.iter().zip(scores.iter()) {
        let w = (sc - max_score).exp();
        sum_w += w;
        est[0] += w * pt[0];
        est[1] += w * pt[1];
        est[2] += w * pt[2];
    }

    if sum_w > 1e-12 {
        [est[0] / sum_w, est[1] / sum_w, est[2] / sum_w]
    } else {
        [anchor_x, anchor_y, 15.0]
    }
}

/// Grid Convolution localizer implementing [`PeakLocalizer`].
#[derive(Debug, Clone)]
pub struct GridConvolutionLocalizer {
    pub xy_radius_um: f32,
    pub xy_step_um: f32,
    pub z_values_um: Vec<f32>,
    pub softmax_temperature: f32,
}

impl Default for GridConvolutionLocalizer {
    fn default() -> Self {
        Self {
            xy_radius_um: 30.0,
            xy_step_um: 5.0,
            z_values_um: vec![5.0, 10.0, 20.0, 35.0, 50.0],
            softmax_temperature: 0.05,
        }
    }
}

impl PeakLocalizer for GridConvolutionLocalizer {
    fn localize(&self, batch: &SnippetBatch, layout: &SensorLayout) -> dsp_core::DspResult<Vec<[f32; 3]>> {
        let mut out = Vec::with_capacity(batch.num_spikes);
        let mut ptp_buf = vec![0.0f32; batch.num_channels];

        for i in 0..batch.num_spikes {
            for k in 0..batch.num_channels {
                ptp_buf[k] = waveform_peak_to_peak(batch.channel_slice(i, k));
            }
            let ch_ids = batch.spike_channel_ids(i);
            out.push(localize_spike_grid_convolution(
                ch_ids,
                &ptp_buf,
                layout,
                self.xy_radius_um,
                self.xy_step_um,
                &self.z_values_um,
                self.softmax_temperature,
            ));
        }
        Ok(out)
    }
}
