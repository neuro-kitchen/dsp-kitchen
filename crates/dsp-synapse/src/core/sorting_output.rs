//! Unified Spike Sorting Output Container (`sorting_output.rs`).
//!
//! Provides [`SortedUnit`] and [`SortingOutput`] as the canonical domain containers for
//! sorted spike trains, templates (`mean`, `std`, `se`), 3D positions, drift estimates,
//! probe layouts, and IBL/SpikeInterface quality metrics across all `dsp-synapse` sorters
//! (`StreamingSpikeRunner`, `GmmClusterer`, `cluster_isosplit`, `cluster_density_peaks`,
//! `OmpSpikeMatcher`, `ConvolutiveBssDecomposer`, and external Phy/Kilosort/NWB runs).

use std::collections::BTreeMap;

use dsp_core::SensorLayout;
use serde::{Deserialize, Serialize};

use crate::metrics::{
    compute_amplitude_cutoff, compute_isi_violations, compute_presence_ratio, compute_snr,
};
use crate::sorting::MotorUnitPulseTrain;
use crate::spatial::DriftEstimate;
use super::events::DeduplicatedSpike;
use super::snippets::WaveformSnippet;
use super::template::{compute_mean_template, UnitQualityLabel, WaveformTemplate};

/// A single sorted unit (neuron or motor unit) with its spike train, template, and quality metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SortedUnit {
    /// Non-negative integer unit/cluster ID.
    pub unit_id: usize,
    /// Primary recording channel with the largest template amplitude.
    pub primary_channel: usize,
    /// Sorted spike timestamps in sample indices (`0..total_samples`).
    pub spike_samples: Vec<u64>,
    /// Per-spike peak amplitudes in $\mu\text{V}$ (same length as `spike_samples`, or empty).
    pub amplitudes_uv: Vec<f32>,
    /// Per-spike 3D source coordinates `[x_um, y_um, z_um]` (same length as `spike_samples`, or empty).
    pub locations_um: Vec<[f32; 3]>,
    /// Multi-channel waveform template (`mean`, `std`, and `se = std / sqrt(n)`).
    pub template: Option<WaveformTemplate>,
    /// Automated or curated quality classification (`SingleUnit`, `MultiUnit`, `Noise`).
    pub quality_label: UnitQualityLabel,
    /// Peak-to-noise ratio (`|peak_uv| / sigma_noise_uv`).
    pub snr: f32,
    /// Mean firing rate in Hz across the recording duration.
    pub firing_rate_hz: f64,
    /// Hill et al. (2011) refractory ISI violation ratio (`isi_violations_ratio`).
    pub isi_violation_ratio: f64,
    /// Fraction of time bins in which the unit is present (`[0.0, 1.0]`).
    pub presence_ratio: f64,
    /// Estimated fraction of missing spikes below detection threshold (`[0.0, 0.5]`).
    pub amplitude_cutoff: f64,
}

impl SortedUnit {
    /// Creates a `SortedUnit` and computes standard firing & IBL quality metrics automatically.
    pub fn from_spikes(
        unit_id: usize,
        primary_channel: usize,
        mut spike_samples: Vec<u64>,
        mut amplitudes_uv: Vec<f32>,
        mut locations_um: Vec<[f32; 3]>,
        template: Option<WaveformTemplate>,
        sample_rate_hz: f64,
        total_samples: u64,
        channel_noise_std_uv: f32,
    ) -> Self {
        // Ensure spikes (and parallel per-spike vectors) are sorted chronologically
        if !spike_samples.windows(2).all(|w| w[0] <= w[1]) {
            let mut order: Vec<usize> = (0..spike_samples.len()).collect();
            order.sort_by_key(|&i| spike_samples[i]);
            spike_samples = order.iter().map(|&i| spike_samples[i]).collect();
            if amplitudes_uv.len() == order.len() {
                amplitudes_uv = order.iter().map(|&i| amplitudes_uv[i]).collect();
            }
            if locations_um.len() == order.len() {
                locations_um = order.iter().map(|&i| locations_um[i]).collect();
            }
        }

        let duration_sec = if total_samples > 0 && sample_rate_hz > 0.0 {
            (total_samples as f64) / sample_rate_hz
        } else {
            spike_samples
                .last()
                .map_or(1.0, |&last| ((last + 1) as f64) / sample_rate_hz.max(1.0))
        };

        let isi = compute_isi_violations(&spike_samples, sample_rate_hz, duration_sec, 1.5, 0.0);
        let presence_bin_s = (duration_sec / 10.0).clamp(0.5, 60.0);
        let presence = compute_presence_ratio(
            &spike_samples,
            total_samples.max(spike_samples.last().copied().unwrap_or(0) + 1),
            sample_rate_hz,
            presence_bin_s,
            0.0,
        );
        let amp_cutoff = if amplitudes_uv.is_empty() {
            f64::NAN
        } else {
            compute_amplitude_cutoff(&amplitudes_uv)
        };

        let peak_uv = template
            .as_ref()
            .and_then(|t| {
                t.mean
                    .iter()
                    .copied()
                    .map(f32::abs)
                    .max_by(|a, b| a.total_cmp(b))
            })
            .or_else(|| {
                amplitudes_uv
                    .iter()
                    .copied()
                    .map(f32::abs)
                    .max_by(|a, b| a.total_cmp(b))
            })
            .unwrap_or(0.0);
        let snr = compute_snr(peak_uv, channel_noise_std_uv.max(1e-3));

        // Automated IBL / Allen quality label heuristic
        let quality_label = if snr < 1.5 || spike_samples.len() < 3 {
            UnitQualityLabel::Noise
        } else if (isi.isi_violations_ratio.is_nan() || isi.isi_violations_ratio < 0.5)
            && snr >= 3.0
        {
            UnitQualityLabel::SingleUnit
        } else {
            UnitQualityLabel::MultiUnit
        };

        Self {
            unit_id,
            primary_channel,
            spike_samples,
            amplitudes_uv,
            locations_um,
            template,
            quality_label,
            snr,
            firing_rate_hz: isi.firing_rate_hz,
            isi_violation_ratio: isi.isi_violations_ratio,
            presence_ratio: presence,
            amplitude_cutoff: amp_cutoff,
        }
    }

    /// Number of spikes assigned to this unit.
    pub fn num_spikes(&self) -> usize {
        self.spike_samples.len()
    }
}

/// Complete output of a spike sorting or motor-unit decomposition run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SortingOutput {
    /// Identifier of the sorter or pipeline (e.g., `"kilosort4"`, `"ppca_gmm"`, `"isosplit"`, `"cbss"`).
    pub sorter_name: String,
    /// Recording sample rate in Hz.
    pub sample_rate_hz: f64,
    /// Total number of samples in the analyzed recording span.
    pub total_samples: u64,
    /// Optional probe / electrode grid geometry.
    pub probe: Option<SensorLayout>,
    /// Sorted units ordered by `unit_id`.
    pub units: Vec<SortedUnit>,
    /// Optional rigid vertical probe drift estimate.
    pub drift: Option<DriftEstimate>,
}

impl SortingOutput {
    /// Creates an empty or custom `SortingOutput`.
    pub fn new(
        sorter_name: impl Into<String>,
        sample_rate_hz: f64,
        total_samples: u64,
        probe: Option<SensorLayout>,
        mut units: Vec<SortedUnit>,
        drift: Option<DriftEstimate>,
    ) -> Self {
        units.sort_by_key(|u| u.unit_id);
        Self {
            sorter_name: sorter_name.into(),
            sample_rate_hz,
            total_samples,
            probe,
            units,
            drift,
        }
    }

    /// Builds a `SortingOutput` from deduplicated spikes, cluster `labels` (`-1` = noise/unassigned),
    /// optional extracted `snippets`, and optional 3D `locations_um`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_clustered_spikes(
        sorter_name: impl Into<String>,
        sample_rate_hz: f64,
        total_samples: u64,
        probe: Option<SensorLayout>,
        spikes: &[DeduplicatedSpike],
        labels: &[i32],
        snippets: Option<&[WaveformSnippet]>,
        locations_um: Option<&[[f32; 3]]>,
        channel_sigmas_uv: &[f32],
        drift: Option<DriftEstimate>,
    ) -> Self {
        let n = spikes.len().min(labels.len());
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for i in 0..n {
            if labels[i] >= 0 {
                groups.entry(labels[i] as usize).or_default().push(i);
            }
        }

        let mut units = Vec::with_capacity(groups.len());
        for (unit_id, indices) in groups {
            let mut spike_samples = Vec::with_capacity(indices.len());
            let mut amplitudes_uv = Vec::with_capacity(indices.len());
            let mut unit_locs = Vec::new();
            let mut unit_snips = Vec::new();
            let mut ch_counts: BTreeMap<usize, usize> = BTreeMap::new();

            for &idx in &indices {
                let spk = &spikes[idx];
                spike_samples.push(spk.sample_index);
                amplitudes_uv.push(spk.peak_amplitude_uv);
                *ch_counts.entry(spk.primary_channel).or_insert(0) += 1;
                if let Some(locs) = locations_um {
                    if let Some(&loc) = locs.get(idx) {
                        unit_locs.push(loc);
                    }
                }
                if let Some(snips) = snippets {
                    if let Some(s) = snips.get(idx) {
                        unit_snips.push(s.clone());
                    }
                }
            }

            let primary_channel = ch_counts
                .into_iter()
                .max_by_key(|&(_, count)| count)
                .map_or(0, |(ch, _)| ch);
            let template = if unit_snips.is_empty() {
                None
            } else {
                compute_mean_template(&unit_snips)
            };
            let noise_sd = channel_sigmas_uv
                .get(primary_channel)
                .copied()
                .unwrap_or(10.0);

            units.push(SortedUnit::from_spikes(
                unit_id,
                primary_channel,
                spike_samples,
                amplitudes_uv,
                unit_locs,
                template,
                sample_rate_hz,
                total_samples,
                noise_sd,
            ));
        }

        Self::new(sorter_name, sample_rate_hz, total_samples, probe, units, drift)
    }

    /// Builds a `SortingOutput` from HD-EMG Convolutive Blind Source Separation (`cBSS`) motor unit pulse trains.
    pub fn from_motor_units(
        sorter_name: impl Into<String>,
        sample_rate_hz: f64,
        total_samples: u64,
        probe: Option<SensorLayout>,
        motor_units: &[MotorUnitPulseTrain],
    ) -> Self {
        let mut units = Vec::with_capacity(motor_units.len());
        for mu in motor_units {
            let amps: Vec<f32> = mu
                .spike_samples
                .iter()
                .map(|&s| mu.ipt.get(s as usize).copied().unwrap_or(1.0))
                .collect();
            let mut unit = SortedUnit::from_spikes(
                mu.unit_id,
                0,
                mu.spike_samples.clone(),
                amps,
                Vec::new(),
                None,
                sample_rate_hz,
                total_samples,
                1.0,
            );
            unit.snr = mu.pnr_db;
            unit.quality_label = if mu.pnr_db >= 20.0 && (mu.cov_isi.is_nan() || mu.cov_isi <= 0.35)
            {
                UnitQualityLabel::SingleUnit
            } else {
                UnitQualityLabel::MultiUnit
            };
            units.push(unit);
        }
        Self::new(sorter_name, sample_rate_hz, total_samples, probe, units, None)
    }

    /// Number of sorted units.
    pub fn num_units(&self) -> usize {
        self.units.len()
    }

    /// Total number of spikes across all units.
    pub fn total_spikes(&self) -> usize {
        self.units.iter().map(|u| u.spike_samples.len()).sum()
    }

    /// Looks up a unit by its `unit_id`.
    pub fn unit(&self, unit_id: usize) -> Option<&SortedUnit> {
        self.units.iter().find(|u| u.unit_id == unit_id)
    }

    /// Returns globally time-sorted `(spike_sample, unit_id, amplitude_uv, location_um)` tuples
    /// across all units (used when exporting flat Phy/Kilosort arrays).
    pub fn flattened_spikes(&self) -> (Vec<u64>, Vec<i32>, Vec<f32>, Vec<[f32; 3]>) {
        let total = self.total_spikes();
        let mut rows: Vec<(u64, i32, f32, [f32; 3])> = Vec::with_capacity(total);
        for u in &self.units {
            for (i, &s) in u.spike_samples.iter().enumerate() {
                let amp = u.amplitudes_uv.get(i).copied().unwrap_or(1.0);
                let loc = u.locations_um.get(i).copied().unwrap_or([0.0, 0.0, 0.0]);
                rows.push((s, u.unit_id as i32, amp, loc));
            }
        }
        rows.sort_by_key(|r| (r.0, r.1));
        let mut samples = Vec::with_capacity(rows.len());
        let mut clusters = Vec::with_capacity(rows.len());
        let mut amps = Vec::with_capacity(rows.len());
        let mut locs = Vec::with_capacity(rows.len());
        for (s, c, a, l) in rows {
            samples.push(s);
            clusters.push(c);
            amps.push(a);
            locs.push(l);
        }
        (samples, clusters, amps, locs)
    }
}
