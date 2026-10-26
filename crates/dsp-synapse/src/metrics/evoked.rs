//! Stimulus-Triggered Averaging (`STA`), Peri-Stimulus Time Histograms (`PSTH`),
//! and Motor Evoked Potential (`MEP`) Quantification (`evoked.rs`).
//!
//! Aligns spike trains and multi-channel continuous recordings (such as HD-EMG grids with
//! cortical/spinal stimulation triggers `MET` / `eS1r`) to stimulus onset timestamps, reporting
//! trial-averaged responses $\mu(t) \pm \text{SE}(t)$ ($\text{SE} = \text{SD} / \sqrt{N_{\text{trials}}}$)
//! and per-channel MEP onset latency, peak-to-peak amplitude, RMS, and rectified AUC.

use dsp_base::math::{standard_error, RunningMoments};
use serde::{Deserialize, Serialize};

/// Peri-Stimulus Time Histogram (PSTH) across $N$ stimulus trials.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PsthResult {
    /// Bin centers relative to stimulus onset in milliseconds (`[-pre_ms, +post_ms]`).
    pub time_bins_ms: Vec<f64>,
    /// Trial-averaged firing rate $\mu(t)$ in Hz.
    pub mean_rate_hz: Vec<f32>,
    /// Across-trial Standard Error of the Mean $\text{SE}(t) = \text{SD}(t) / \sqrt{N_{\text{trials}}}$ in Hz.
    pub se_rate_hz: Vec<f32>,
    pub num_trials: usize,
    pub bin_width_ms: f64,
}

/// Multi-channel Stimulus-Triggered Average (STA) waveform with Standard Error ($\text{SE}$).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StimulusTriggeredAverage {
    pub num_channels: usize,
    pub num_samples: usize,
    pub num_trials: usize,
    /// Relative time axis in milliseconds (`[-pre_ms, +post_ms]`) of length `num_samples`.
    pub time_ms: Vec<f64>,
    /// Trial-averaged evoked potential `[num_channels, num_samples]` in $\mu\text{V}$.
    pub mean_uv: Vec<f32>,
    /// Across-trial sample standard deviation `[num_channels, num_samples]` in $\mu\text{V}$.
    pub std_uv: Vec<f32>,
    /// Across-trial Standard Error of the Mean $\text{SE} = \text{SD} / \sqrt{N_{\text{trials}}}$ `[num_channels, num_samples]`.
    pub se_uv: Vec<f32>,
}

/// Quantified Motor Evoked Potential (MEP) metrics for a single channel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MepMetrics {
    pub channel_id: usize,
    /// Latency from stimulus ($t = 0\,\text{ms}$) to first deflection exceeding `threshold_sigma` pre-stimulus SD (`NaN` if no response).
    pub onset_latency_ms: f32,
    /// Peak-to-peak amplitude ($\max V - \min V$) within the active MEP response window in $\mu\text{V}$.
    pub peak_to_peak_uv: f32,
    /// Root-mean-square amplitude within the active MEP response window in $\mu\text{V}$.
    pub rms_uv: f32,
    /// Rectified Area Under the Curve ($\int |V(t)|\,dt$) within the response window in $\mu\text{V}\cdot\text{ms}$.
    pub rectified_auc_uv_ms: f32,
}

/// Computes a Peri-Stimulus Time Histogram (PSTH) with trial-by-trial $\text{SE}$ around `stim_times_sec`.
pub fn compute_psth(
    spike_times_sec: &[f64],
    stim_times_sec: &[f64],
    pre_ms: f64,
    post_ms: f64,
    bin_width_ms: f64,
) -> PsthResult {
    let dt_ms = bin_width_ms.max(MIN_PSTH_BIN_MS);
    let span_ms = (pre_ms + post_ms).max(dt_ms);
    let num_bins = ((span_ms / dt_ms).round() as usize).max(1);
    let dt_sec = dt_ms * 1e-3;
    let n_trials = stim_times_sec.len();

    let time_bins_ms: Vec<f64> = (0..num_bins)
        .map(|b| -pre_ms + (b as f64 + 0.5) * dt_ms)
        .collect();

    if n_trials == 0 {
        return PsthResult {
            time_bins_ms,
            mean_rate_hz: vec![0.0; num_bins],
            se_rate_hz: vec![0.0; num_bins],
            num_trials: 0,
            bin_width_ms: dt_ms,
        };
    }

    let mut sorted_spikes = spike_times_sec.to_vec();
    sorted_spikes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Per-trial rate matrix [n_trials, num_bins]
    let mut trial_rates = vec![0.0f32; n_trials * num_bins];
    let inv_dt = (1.0 / dt_sec) as f32;

    for (tr_idx, &t0) in stim_times_sec.iter().enumerate() {
        let win_start = t0 - pre_ms * 1e-3;
        let win_end = t0 + post_ms * 1e-3;
        let lo = sorted_spikes.partition_point(|&ts| ts < win_start);
        for &ts in &sorted_spikes[lo..] {
            if ts >= win_end {
                break;
            }
            let rel_ms = (ts - t0) * 1e3 + pre_ms;
            let b = ((rel_ms / dt_ms).floor() as usize).min(num_bins - 1);
            trial_rates[tr_idx * num_bins + b] += inv_dt;
        }
    }

    let mut mean_rate_hz = vec![0.0f32; num_bins];
    let mut se_rate_hz = vec![0.0f32; num_bins];
    let n_f = n_trials as f32;

    for b in 0..num_bins {
        let mut sum = 0.0f32;
        for tr in 0..n_trials {
            sum += trial_rates[tr * num_bins + b];
        }
        let mu = sum / n_f;
        mean_rate_hz[b] = mu;

        if n_trials > 1 {
            let mut var_sum = 0.0f32;
            for tr in 0..n_trials {
                let d = trial_rates[tr * num_bins + b] - mu;
                var_sum += d * d;
            }
            let sd = (var_sum / ((n_trials - 1) as f32)).sqrt();
            se_rate_hz[b] = standard_error(sd, n_trials);
        }
    }

    PsthResult {
        time_bins_ms,
        mean_rate_hz,
        se_rate_hz,
        num_trials: n_trials,
        bin_width_ms: dt_ms,
    }
}

/// Narrowest PSTH bin (ms).
const MIN_PSTH_BIN_MS: f64 = 0.1;
/// Baseline σ floor (µV) of MEP onset detection, and smallest onset threshold (in baseline σ).
const MIN_BASELINE_SD_UV: f32 = 1e-4;
const MIN_THRESHOLD_SIGMA: f32 = 1.0;

/// Degrees-of-freedom correction of the across-trial SD of [`compute_stimulus_triggered_average`]:
/// 1 (sample SD; zero with a single trial).
pub const STA_STD_DDOF: u64 = 1;

/// Computes the multi-channel Stimulus-Triggered Average (`STA`) and across-trial Standard Error (`SE`)
/// around stimulus sample indices `stim_samples`.
pub fn compute_stimulus_triggered_average(
    data: &[f32],
    channels: usize,
    samples: usize,
    stim_samples: &[u64],
    pre_samples: usize,
    post_samples: usize,
    sample_rate_hz: f64,
) -> StimulusTriggeredAverage {
    assert_eq!(data.len(), channels * samples);
    let win_len = pre_samples + post_samples;
    let dt_ms = 1e3 / sample_rate_hz.max(1.0);
    let time_ms: Vec<f64> = (0..win_len)
        .map(|i| (i as isize - pre_samples as isize) as f64 * dt_ms)
        .collect();

    let valid_stims: Vec<usize> = stim_samples
        .iter()
        .copied()
        .map(|s| s as usize)
        .filter(|&s| s >= pre_samples && s + post_samples <= samples)
        .collect();
    let n_trials = valid_stims.len();

    let total_out = channels * win_len;
    let mut mean_uv = vec![0.0f32; total_out];
    let mut std_uv = vec![0.0f32; total_out];
    let mut se_uv = vec![0.0f32; total_out];

    if n_trials == 0 || win_len == 0 {
        return StimulusTriggeredAverage {
            num_channels: channels,
            num_samples: win_len,
            num_trials: 0,
            time_ms,
            mean_uv,
            std_uv,
            se_uv,
        };
    }

    // Across-trial moments; SD with ddof = 1 (sample SD), as the SE it feeds
    let mut moments = RunningMoments::new(total_out);
    let mut trial = vec![0.0f32; total_out];
    for &s0 in &valid_stims {
        let start = s0 - pre_samples;
        for ch in 0..channels {
            trial[ch * win_len..(ch + 1) * win_len].copy_from_slice(&data[ch * samples + start..ch * samples + start + win_len]);
        }
        moments.push(&trial);
    }
    let std = moments.std(STA_STD_DDOF);
    for idx in 0..total_out {
        mean_uv[idx] = moments.mean()[idx] as f32;
        std_uv[idx] = std[idx] as f32;
        se_uv[idx] = standard_error(std_uv[idx], n_trials);
    }

    StimulusTriggeredAverage {
        num_channels: channels,
        num_samples: win_len,
        num_trials: n_trials,
        time_ms,
        mean_uv,
        std_uv,
        se_uv,
    }
}

/// Quantifies per-channel Motor Evoked Potential (MEP) onset latency, peak-to-peak amplitude,
/// RMS, and rectified AUC from a [`StimulusTriggeredAverage`] within `[response_start_ms, response_end_ms]`.
pub fn quantify_mep(
    sta: &StimulusTriggeredAverage,
    response_start_ms: f64,
    response_end_ms: f64,
    threshold_sigma: f32,
) -> Vec<MepMetrics> {
    let win_len = sta.num_samples;
    if win_len == 0 || sta.num_channels == 0 {
        return Vec::new();
    }

    let dt_ms = if win_len >= 2 {
        (sta.time_ms[1] - sta.time_ms[0]).abs() as f32
    } else {
        1.0
    };

    let mut out = Vec::with_capacity(sta.num_channels);
    for ch in 0..sta.num_channels {
        let wave = &sta.mean_uv[ch * win_len..(ch + 1) * win_len];

        // Baseline statistics from pre-stimulus samples (time_ms < 0.0)
        let mut base_vals = Vec::new();
        for (t, &tms) in sta.time_ms.iter().enumerate() {
            if tms < 0.0 {
                base_vals.push(wave[t]);
            }
        }
        let (base_mean, base_sd) = if !base_vals.is_empty() {
            let n = base_vals.len() as f32;
            let m = base_vals.iter().sum::<f32>() / n;
            let v = base_vals.iter().map(|&x| (x - m) * (x - m)).sum::<f32>() / n;
            (m, v.sqrt().max(MIN_BASELINE_SD_UV))
        } else {
            // No pre-stimulus samples: no baseline, so no onset can be found
            (0.0, f32::NAN)
        };

        let thresh = threshold_sigma.max(MIN_THRESHOLD_SIGMA) * base_sd;
        let mut onset_latency_ms = f32::NAN;
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        let mut sum_sq = 0.0f32;
        let mut rect_sum = 0.0f32;
        let mut resp_count = 0usize;

        for (t, &tms) in sta.time_ms.iter().enumerate() {
            if tms >= response_start_ms && tms <= response_end_ms {
                let v = wave[t] - base_mean;
                if onset_latency_ms.is_nan() && v.abs() > thresh {
                    onset_latency_ms = tms as f32;
                }
                if v < min_v {
                    min_v = v;
                }
                if v > max_v {
                    max_v = v;
                }
                sum_sq += v * v;
                rect_sum += v.abs();
                resp_count += 1;
            }
        }

        let (ptp, rms, auc) = if resp_count > 0 {
            (
                (max_v - min_v).max(0.0),
                (sum_sq / (resp_count as f32)).sqrt(),
                rect_sum * dt_ms,
            )
        } else {
            // Empty response window: undefined
            (f32::NAN, f32::NAN, f32::NAN)
        };

        out.push(MepMetrics {
            channel_id: ch,
            onset_latency_ms,
            peak_to_peak_uv: ptp,
            rms_uv: rms,
            rectified_auc_uv_ms: auc,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sta_se_and_mep_quantification() {
        let fs = 2000.0; // 0.5 ms per sample
        let samples = 2000;
        let mut data = vec![0.0f32; samples];
        let stim_samples = vec![200u64, 600, 1000, 1400];

        // Inject an evoked biphasic MEP starting +12.0 ms (+24 samples) after each trigger
        // with peak-to-peak = 150 - (-200) = 350 uV
        for &s0 in &stim_samples {
            let base = s0 as usize;
            data[base + 24] = -200.0;
            data[base + 28] = 150.0;
        }

        let sta = compute_stimulus_triggered_average(&data, 1, samples, &stim_samples, 40, 80, fs);
        assert_eq!(sta.num_trials, 4);
        // Since all 4 trials are identical, SE should be 0.0 everywhere
        assert!(sta.se_uv.iter().all(|&v| v.abs() < 1e-5));

        let meps = quantify_mep(&sta, 5.0, 35.0, 3.0);
        assert_eq!(meps.len(), 1);
        assert!((meps[0].onset_latency_ms - 12.0).abs() < 0.6, "latency={}", meps[0].onset_latency_ms);
        assert!((meps[0].peak_to_peak_uv - 350.0).abs() < 1.0, "ptp={}", meps[0].peak_to_peak_uv);
    }
}
