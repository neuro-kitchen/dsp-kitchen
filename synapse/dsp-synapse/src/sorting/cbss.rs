//! Convolutive Blind Source Separation (cBSS) for High-Density Surface EMG (`cbss.rs`).
//!
//! Decomposes multi-channel HD-EMG grids (`[channels, samples]`) into constituent Motor Unit
//! Pulse Trains (IPTs) using the Holobar & Zazula (CKC) / Negro et al. (2016) pipeline:
//! 1. **Temporal Extension**: Extends $M$ channels by $L$ delayed replicas into an $M \cdot L \times S$
//!    block-Hankel observation matrix $\tilde{\mathbf{X}}$.
//! 2. **Spatial-Temporal ZCA Whitening & FastICA**: Orthogonalizes extended observations and extracts
//!    sparse, super-Gaussian source Innovation Pulse Trains $s_k(t)$ via [`dsp_base::linalg::FastIcaModel`].
//! 3. **Peak Picking & Discharge Quality**: Extracts motor unit discharge timestamps from squared IPTs
//!    $s_k^2(t)$, deduplicates duplicate delayed replicas, and computes the Coefficient of Variation
//!    of Inter-Spike Intervals ($\text{CoV}_{\text{ISI}} = \sigma_{\text{ISI}} / \mu_{\text{ISI}}$) and
//!    Pulse-to-Noise Ratio ($\text{PNR}$ in dB).

use cubecl::prelude::Client;
use dsp_base::linalg::{FastIcaModel, IcaContrast};
use serde::{Deserialize, Serialize};

/// FastICA convergence tolerance on the change of each unmixing vector.
pub const ICA_TOLERANCE: f32 = 1e-4;

/// Decomposed single Motor Unit (MU) pulse train and discharge quality metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MotorUnitPulseTrain {
    pub unit_id: usize,
    /// Discharge sample indices (`0..samples`).
    pub spike_samples: Vec<u64>,
    /// Innovation Pulse Train (IPT) $s_k(t)$ of length `samples`.
    pub ipt: Vec<f32>,
    /// Coefficient of Variation of Inter-Spike Intervals ($\sigma_{\text{ISI}} / \mu_{\text{ISI}}$).
    /// Physiological motor units typically have $\text{CoV}_{\text{ISI}} < 0.35$.
    pub cov_isi: f32,
    /// Pulse-to-Noise Ratio (PNR) in dB ($10 \log_{10}(\mathbb{E}[s^2(t_k)] / \mathbb{E}[s^2(t \notin t_k)])$).
    pub pnr_db: f32,
}

/// Convolutive Blind Source Separation (cBSS) decomposer for HD-EMG motor unit identification.
#[derive(Debug, Clone)]
pub struct ConvolutiveBssDecomposer {
    /// Extension factor $L$ (number of delayed replicas per channel, typically $4..16$).
    pub extension_factor: usize,
    /// Number of candidate ICA sources to extract.
    pub num_sources: usize,
    /// Minimum refractory distance between discharges of the same motor unit (in ms, typically $20\,\text{ms}$ for MUs).
    pub refractory_ms: f64,
    /// Threshold factor (in standard deviations of $|s(t)|$) for discharge peak detection.
    pub peak_threshold_sigma: f32,
    /// Maximum FastICA iterations.
    pub max_iterations: usize,
}

impl Default for ConvolutiveBssDecomposer {
    fn default() -> Self {
        Self {
            extension_factor: 6,
            num_sources: 8,
            refractory_ms: 20.0,
            peak_threshold_sigma: 3.5,
            max_iterations: 80,
        }
    }
}

impl ConvolutiveBssDecomposer {
    pub fn new(extension_factor: usize, num_sources: usize, refractory_ms: f64) -> Self {
        Self {
            extension_factor: extension_factor.max(1),
            num_sources: num_sources.max(1),
            refractory_ms: refractory_ms.max(2.0),
            ..Self::default()
        }
    }

    /// Decomposes a multi-channel HD-EMG recording (`[channels, samples]`) into deduplicated Motor
    /// Unit Pulse Trains, fitting ICA on `client`.
    pub fn decompose(
        &self,
        client: &Client,
        data: &[f32],
        channels: usize,
        samples: usize,
        sample_rate_hz: f64,
    ) -> Vec<MotorUnitPulseTrain> {
        assert_eq!(data.len(), channels * samples);
        if channels == 0 || samples < 16 {
            return Vec::new();
        }

        let l_ext = self.extension_factor.clamp(1, samples / 4);
        let ext_channels = channels * l_ext;
        let k_sources = self.num_sources.clamp(1, ext_channels);

        // 1. Construct temporally extended observation matrix X_tilde of shape [channels * L, samples]
        let mut extended = vec![0.0f32; ext_channels * samples];
        for c in 0..channels {
            let src_row = &data[c * samples..(c + 1) * samples];
            for lag in 0..l_ext {
                let dst_row = &mut extended[(c * l_ext + lag) * samples..(c * l_ext + lag + 1) * samples];
                dst_row[lag..].copy_from_slice(&src_row[..samples - lag]);
            }
        }

        // 2. Fit FastICA with Cube (kurtosis) contrast over the extended observations
        let ica = FastIcaModel::fit::<f32>(
            client,
            &extended,
            ext_channels,
            samples,
            k_sources,
            IcaContrast::Cube,
            self.max_iterations,
            ICA_TOLERANCE,
        );
        let sources = ica.transform_cpu(&extended, ext_channels, samples);

        let ref_samples = ((sample_rate_hz * self.refractory_ms * 1e-3).round() as usize).max(2);
        let mut candidates: Vec<MotorUnitPulseTrain> = Vec::new();

        for s_idx in 0..ica.num_components {
            let mut ipt = sources[s_idx * samples..(s_idx + 1) * samples].to_vec();
            // Orient sign so dominant skewness/extrema are positive
            let max_pos = ipt.iter().copied().fold(0.0f32, f32::max);
            let max_neg = ipt.iter().copied().fold(0.0f32, f32::min).abs();
            if max_neg > max_pos {
                for v in &mut ipt {
                    *v = -*v;
                }
            }

            let rms = (ipt.iter().map(|v| v * v).sum::<f32>() / (samples as f32)).sqrt().max(1e-8);
            let thresh = self.peak_threshold_sigma * rms;

            let mut spikes = Vec::new();
            let mut last_t: Option<usize> = None;
            for t in 1..(samples - 1) {
                let v = ipt[t];
                if v > thresh && v >= ipt[t - 1] && v > ipt[t + 1] {
                    if last_t.map_or(true, |prev| t > prev + ref_samples) {
                        spikes.push(t as u64);
                        last_t = Some(t);
                    } else if let Some(prev) = last_t {
                        if v > ipt[prev] {
                            *spikes.last_mut().unwrap() = t as u64;
                            last_t = Some(t);
                        }
                    }
                }
            }

            if spikes.len() < 3 {
                continue;
            }

            let cov_isi = compute_cov_isi(&spikes);
            let pnr_db = compute_pnr_db(&ipt, &spikes);

            candidates.push(MotorUnitPulseTrain {
                unit_id: candidates.len(),
                spike_samples: spikes,
                ipt,
                cov_isi,
                pnr_db,
            });
        }

        // 3. Sort by PNR (highest quality first) and deduplicate delayed replica trains (>50% spike coincidence within +/- L samples)
        candidates.sort_by(|a, b| b.pnr_db.partial_cmp(&a.pnr_db).unwrap_or(std::cmp::Ordering::Equal));
        let mut unique_units: Vec<MotorUnitPulseTrain> = Vec::new();
        let tol_samples = (l_ext as u64) + 2;

        for mut cand in candidates {
            let is_dup = unique_units.iter().any(|u| {
                spike_train_coincidence(&u.spike_samples, &cand.spike_samples, tol_samples) > 0.5
            });
            if !is_dup {
                cand.unit_id = unique_units.len();
                unique_units.push(cand);
            }
        }

        unique_units
    }
}

fn compute_cov_isi(spikes: &[u64]) -> f32 {
    if spikes.len() < 3 {
        return f32::NAN;
    }
    let isis: Vec<f32> = spikes.windows(2).map(|w| (w[1] - w[0]) as f32).collect();
    let n = isis.len() as f32;
    let mean = isis.iter().sum::<f32>() / n;
    if mean <= 1e-6 {
        return f32::NAN;
    }
    let var = isis.iter().map(|&d| (d - mean) * (d - mean)).sum::<f32>() / n;
    var.sqrt() / mean
}

fn compute_pnr_db(ipt: &[f32], spikes: &[u64]) -> f32 {
    if ipt.is_empty() || spikes.is_empty() {
        return 0.0;
    }
    let mut is_peak = vec![false; ipt.len()];
    let mut peak_sum = 0.0f32;
    for &t in spikes {
        let idx = t as usize;
        if idx < ipt.len() {
            peak_sum += ipt[idx] * ipt[idx];
            for w in idx.saturating_sub(2)..=(idx + 2).min(ipt.len() - 1) {
                is_peak[w] = true;
            }
        }
    }
    let mean_peak_sq = peak_sum / (spikes.len() as f32);

    let mut noise_sum = 0.0f32;
    let mut noise_count = 0usize;
    for (i, &v) in ipt.iter().enumerate() {
        if !is_peak[i] {
            noise_sum += v * v;
            noise_count += 1;
        }
    }
    let mean_noise_sq = (noise_sum / (noise_count.max(1) as f32)).max(1e-12);
    10.0 * (mean_peak_sq / mean_noise_sq).max(1e-6).log10()
}

fn spike_train_coincidence(a: &[u64], b: &[u64], tol: u64) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    // Check best lag within [-tol, tol]
    let max_shift = tol as i64;
    let mut best_matches = 0usize;
    for shift in -max_shift..=max_shift {
        let mut matches = 0usize;
        let mut j = 0usize;
        for &ta in a {
            let shifted_a = (ta as i64 + shift).max(0) as u64;
            while j < b.len() && b[j] + 2 < shifted_a {
                j += 1;
            }
            if j < b.len() && b[j].abs_diff(shifted_a) <= 2 {
                matches += 1;
            }
        }
        if matches > best_matches {
            best_matches = matches;
        }
    }
    (best_matches as f32) / (a.len().min(b.len()) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cbss_decomposer_recovers_motor_unit_pulse_trains() {
        let channels = 4;
        let samples = 2000;
        let fs = 2000.0;

        // Two motor units firing regularly with different periods (80 samples = 25 Hz, 115 samples = 17.4 Hz)
        let mu1_times: Vec<usize> = (60..samples - 40).step_by(80).collect();
        let mu2_times: Vec<usize> = (90..samples - 40).step_by(115).collect();

        let mut data = vec![0.0f32; channels * samples];
        let w1 = [1.0f32, 0.7, 0.2, 0.05];
        let w2 = [0.05f32, 0.25, 0.8, 1.0];

        for &t0 in &mu1_times {
            for c in 0..channels {
                // Convolutive delayed biphasic MUAP
                let center = t0 + c;
                data[c * samples + center] += 25.0 * w1[c];
                data[c * samples + center + 1] -= 12.0 * w1[c];
            }
        }
        for &t0 in &mu2_times {
            for c in 0..channels {
                let center = t0 + (channels - 1 - c);
                data[c * samples + center] += 25.0 * w2[c];
                data[c * samples + center + 1] -= 12.0 * w2[c];
            }
        }

        struct Decompose<'a>(&'a [f32], usize, usize, f64);
        impl dsp_core::compute::ComputeTask for Decompose<'_> {
            type Output = Vec<MotorUnitPulseTrain>;
            fn run(self, client: Client) -> Self::Output {
                ConvolutiveBssDecomposer::new(3, 4, 20.0).decompose(&client, self.0, self.1, self.2, self.3)
            }
        }
        for target in dsp_core::compute::ComputeTarget::available() {
            let units = target.run(Decompose(&data, channels, samples, fs)).unwrap();
            assert!(units.len() >= 2, "{}: expected at least 2 motor units, got {}", target.name(), units.len());
            assert!(units[0].cov_isi < 0.15, "cov_isi={}", units[0].cov_isi);
            assert!(units[0].pnr_db > 15.0, "pnr_db={}", units[0].pnr_db);
        }
    }
}
