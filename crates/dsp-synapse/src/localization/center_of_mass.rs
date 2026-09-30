//! Center-of-Mass (CoM) 2D/3D spike source localization (`center_of_mass.rs`).
//!
//! Computes the amplitude-weighted centroid of $K$-nearest neighbor electrode positions:
//! $$\mathbf{r}_{\text{CoM}} = \frac{\sum_{k=1}^K a_k^p \mathbf{r}_k}{\sum_{k=1}^K a_k^p}$$

use dsp_core::SensorLayout;
use crate::extraction::SnippetBatch;
use crate::traits::PeakLocalizer;

/// Computes the peak-to-peak amplitude of a 1D waveform slice.
pub fn waveform_peak_to_peak(wave: &[f32]) -> f32 {
    if wave.is_empty() {
        return 0.0;
    }
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
    (max_v - min_v).max(0.0)
}

/// Localizes a single spike's `[x_um, y_um, z_um]` coordinate via Center-of-Mass.
pub fn localize_spike_center_of_mass(
    channel_ids: &[usize],
    ptp_amplitudes: &[f32],
    layout: &SensorLayout,
    power: f32,
) -> [f32; 3] {
    let mut sum_w = 0.0f32;
    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;
    let mut sum_z = 0.0f32;

    for (&ch_id, &amp) in channel_ids.iter().zip(ptp_amplitudes.iter()) {
        if let Ok(site) = layout.get_site(ch_id) {
            let w = amp.max(0.0).powf(power);
            sum_w += w;
            sum_x += w * site.position.x_um;
            sum_y += w * site.position.y_um;
            sum_z += w * site.position.z_um;
        }
    }

    if sum_w > 1e-8 {
        [sum_x / sum_w, sum_y / sum_w, sum_z / sum_w]
    } else if let Some(&first_ch) = channel_ids.first() {
        layout
            .get_site(first_ch)
            .map(|s| [s.position.x_um, s.position.y_um, s.position.z_um])
            .unwrap_or([0.0, 0.0, 0.0])
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// Center-of-Mass localizer implementing [`PeakLocalizer`].
#[derive(Debug, Clone, Copy)]
pub struct CenterOfMassLocalizer {
    /// Exponent $p$ applied to peak-to-peak amplitudes ($p = 1.0$ standard, $p = 2.0$ power-weighted).
    pub amplitude_power: f32,
}

impl Default for CenterOfMassLocalizer {
    fn default() -> Self {
        Self {
            amplitude_power: 1.0,
        }
    }
}

impl PeakLocalizer for CenterOfMassLocalizer {
    fn localize(&self, batch: &SnippetBatch, layout: &SensorLayout) -> dsp_core::DspResult<Vec<[f32; 3]>> {
        let mut out = Vec::with_capacity(batch.num_spikes);
        let mut ptp_buf = vec![0.0f32; batch.num_channels];

        for i in 0..batch.num_spikes {
            for k in 0..batch.num_channels {
                ptp_buf[k] = waveform_peak_to_peak(batch.channel_slice(i, k));
            }
            let ch_ids = batch.spike_channel_ids(i);
            out.push(localize_spike_center_of_mass(
                ch_ids,
                &ptp_buf,
                layout,
                self.amplitude_power,
            ));
        }
        Ok(out)
    }
}
