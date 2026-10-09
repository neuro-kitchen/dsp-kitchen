//! Unified Spike Sorting Output Container (`sorting_output.rs`).
//!
//! Provides [`SortedUnit`] and [`SortingOutput`] as the canonical domain containers for
//! sorted spike trains, templates (`mean`, `std`, `se`), 3D positions, drift estimates,
//! probe layouts, and IBL/SpikeInterface quality metrics across all `dsp-synapse` sorters
//! (`StreamingDetector`, `GmmClusterer`, `cluster_kde_merge`, `cluster_density_peaks`,
//! `MatchingPursuitMatcher`, `ConvolutiveBssDecomposer`, and external Phy/Kilosort/NWB runs).

use std::collections::BTreeMap;

use dsp_io::neuro::probe::SensorLayout;
use serde::{Deserialize, Serialize};

use crate::metrics::{
    QualityCriteria, compute_amplitude_cutoff, compute_isi_violations, compute_presence_ratio,
    compute_snr,
};
use crate::sorting::MotorUnitPulseTrain;
use crate::spatial::DriftEstimate;
use super::events::DeduplicatedSpike;
use super::snippets::WaveformSnippet;
use super::template::{compute_mean_template, UnitQualityLabel, WaveformTemplate};

/// Optional source recording provenance associated with a [`SortingOutput`] (e.g., from `params.py`
/// or the recording source that was sorted). Avoids hardcoding dummy `recording.dat` / `int16`
/// values when exporting to Phy.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct RecordingMeta {
    pub dat_path: Option<String>,
    pub dtype: Option<String>,
    pub offset: u64,
    pub hp_filtered: bool,
    pub n_channels_dat: Option<usize>,
}

/// Amplitude and location written for a spike that has none in flat (Phy-style) arrays, which
/// need one value per spike: NaN, so missing values are never mistaken for measurements.
pub const MISSING_AMPLITUDE: f32 = f32::NAN;
pub const MISSING_LOCATION: [f32; 3] = [f32::NAN; 3];

pub mod serde_nan {
    use serde::{Deserialize, Deserializer};

    /// Default of a metric missing from a file: not computed.
    pub fn nan() -> f64 {
        f64::NAN
    }

    pub fn deserialize_f64<'de, D>(deserializer: D) -> Result<f64, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt = Option::<f64>::deserialize(deserializer)?;
        Ok(opt.unwrap_or(f64::NAN))
    }

    pub fn deserialize_f32<'de, D>(deserializer: D) -> Result<f32, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt = Option::<f32>::deserialize(deserializer)?;
        Ok(opt.unwrap_or(f32::NAN))
    }
}

/// A single sorted unit (neuron or motor unit) with its spike train, template, and quality metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Automated or curated quality classification (`SingleUnit`, `MultiUnit`, `Noise`, `Unsorted`).
    pub quality_label: UnitQualityLabel,
    /// Peak-to-noise ratio (`|peak_uv| / sigma_noise_uv`).
    #[serde(default, deserialize_with = "serde_nan::deserialize_f32")]
    pub snr: f32,
    /// Mean firing rate in Hz across the recording duration.
    #[serde(default, deserialize_with = "serde_nan::deserialize_f64")]
    pub firing_rate_hz: f64,
    /// Hill et al. (2011) refractory ISI violation ratio (`isi_violations_ratio`).
    #[serde(default, deserialize_with = "serde_nan::deserialize_f64")]
    pub isi_violation_ratio: f64,
    /// Fraction of time bins in which the unit is present (`[0.0, 1.0]`).
    #[serde(default, deserialize_with = "serde_nan::deserialize_f64")]
    pub presence_ratio: f64,
    /// Estimated fraction of missing spikes below detection threshold (`[0.0, 0.5]`).
    #[serde(default, deserialize_with = "serde_nan::deserialize_f64")]
    pub amplitude_cutoff: f64,
    /// EMUsort's composite quality score in `[0, 1]` ([`crate::metrics::composite_score`]); NaN when
    /// the sorter does not compute it.
    #[serde(default = "serde_nan::nan", deserialize_with = "serde_nan::deserialize_f64")]
    pub composite_score: f64,
}

impl PartialEq for SortedUnit {
    fn eq(&self, other: &Self) -> bool {
        let f32_eq = |a: f32, b: f32| a == b || (a.is_nan() && b.is_nan());
        let f64_eq = |a: f64, b: f64| a == b || (a.is_nan() && b.is_nan());

        self.unit_id == other.unit_id
            && self.primary_channel == other.primary_channel
            && self.spike_samples == other.spike_samples
            && self.amplitudes_uv == other.amplitudes_uv
            && self.locations_um == other.locations_um
            && self.template == other.template
            && self.quality_label == other.quality_label
            && f32_eq(self.snr, other.snr)
            && f64_eq(self.firing_rate_hz, other.firing_rate_hz)
            && f64_eq(self.isi_violation_ratio, other.isi_violation_ratio)
            && f64_eq(self.presence_ratio, other.presence_ratio)
            && f64_eq(self.amplitude_cutoff, other.amplitude_cutoff)
            && f64_eq(self.composite_score, other.composite_score)
    }
}

impl SortedUnit {
    /// Creates a `SortedUnit` and computes standard firing & IBL quality metrics using default
    /// [`QualityCriteria`] and a known `channel_noise_std_uv`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_spikes(
        unit_id: usize,
        primary_channel: usize,
        spike_samples: Vec<u64>,
        amplitudes_uv: Vec<f32>,
        locations_um: Vec<[f32; 3]>,
        template: Option<WaveformTemplate>,
        sample_rate_hz: f64,
        total_samples: u64,
        channel_noise_std_uv: f32,
    ) -> Self {
        Self::from_spikes_with(
            unit_id,
            Some(primary_channel),
            spike_samples,
            amplitudes_uv,
            locations_um,
            template,
            sample_rate_hz,
            total_samples,
            Some(channel_noise_std_uv),
            QualityCriteria::default(),
        )
    }

    /// Creates a `SortedUnit` with configurable [`QualityCriteria`], optional `primary_channel`
    /// (inferred from `template.best_channel()` when `None`), and optional `channel_noise_std_uv`
    /// (leaving `snr` as `NaN` rather than fabricating a dummy noise floor when `None`).
    #[allow(clippy::too_many_arguments)]
    pub fn from_spikes_with(
        unit_id: usize,
        primary_channel: Option<usize>,
        mut spike_samples: Vec<u64>,
        mut amplitudes_uv: Vec<f32>,
        mut locations_um: Vec<[f32; 3]>,
        template: Option<WaveformTemplate>,
        sample_rate_hz: f64,
        total_samples: u64,
        channel_noise_std_uv: Option<f32>,
        criteria: QualityCriteria,
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

        let isi = compute_isi_violations(
            &spike_samples,
            sample_rate_hz,
            duration_sec,
            criteria.refractory_ms,
            criteria.censored_ms,
        );
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
        let snr = match channel_noise_std_uv {
            Some(sd) => compute_snr(peak_uv, sd),
            None => f32::NAN,
        };
        let resolved_primary = primary_channel
            .or_else(|| template.as_ref().and_then(|t| t.best_channel()))
            .unwrap_or(0);
        let quality_label = criteria.classify(spike_samples.len(), snr, isi.isi_violations_ratio);

        Self {
            unit_id,
            primary_channel: resolved_primary,
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
            composite_score: f64::NAN,
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
    /// Identifier of the sorter or pipeline (e.g., `"kilosort4"`, `"ppca_gmm"`, `"kde_merge"`, `"cbss"`).
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
    /// Optional recording provenance (e.g. `dat_path`, `dtype`, `offset`, `hp_filtered`).
    #[serde(default)]
    pub recording_meta: RecordingMeta,
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
            recording_meta: RecordingMeta::default(),
        }
    }

    /// Attaches recording provenance metadata (`dat_path`, `dtype`, `offset`, `hp_filtered`).
    pub fn with_recording_meta(mut self, recording_meta: RecordingMeta) -> Self {
        self.recording_meta = recording_meta;
        self
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
                .unwrap_or(f32::NAN);

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
                .map(|&s| mu.ipt.get(s as usize).copied().unwrap_or(f32::NAN))
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
                let amp = u.amplitudes_uv.get(i).copied().unwrap_or(MISSING_AMPLITUDE);
                let loc = u.locations_um.get(i).copied().unwrap_or(MISSING_LOCATION);
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

    /// Groups flat per-spike arrays `(times, clusters, amplitudes, locations)` by non-negative
    /// cluster ID, returning `(spike_samples, amplitudes_uv, locations_um)` per cluster.
    pub fn group_spikes_by_cluster(
        times: &[u64],
        clusters: &[i32],
        amplitudes: &[f32],
        locations: &[[f32; 3]],
    ) -> BTreeMap<usize, (Vec<u64>, Vec<f32>, Vec<[f32; 3]>)> {
        let n = times.len().min(clusters.len());
        let mut groups: BTreeMap<usize, (Vec<u64>, Vec<f32>, Vec<[f32; 3]>)> = BTreeMap::new();
        for i in 0..n {
            let c = clusters[i];
            if c >= 0 {
                let entry = groups.entry(c as usize).or_default();
                entry.0.push(times[i]);
                if let Some(&amp) = amplitudes.get(i) {
                    entry.1.push(amp);
                }
                if let Some(&loc) = locations.get(i) {
                    entry.2.push(loc);
                }
            }
        }
        groups
    }

    /// Packs unit spike trains into NWB's ragged `(unit_ids, spike_times_sec, spike_times_index)`
    /// representation with full `f64` timestamp precision.
    pub fn to_ragged_spikes(&self) -> (Vec<i64>, Vec<f64>, Vec<u64>) {
        let fs = self.sample_rate_hz.max(f64::MIN_POSITIVE);
        let mut unit_ids = Vec::with_capacity(self.units.len());
        let mut spike_times_sec = Vec::with_capacity(self.total_spikes());
        let mut spike_times_index = Vec::with_capacity(self.units.len());
        let mut cum = 0u64;
        for u in &self.units {
            unit_ids.push(u.unit_id as i64);
            for &s in &u.spike_samples {
                spike_times_sec.push((s as f64) / fs);
            }
            cum += u.spike_samples.len() as u64;
            spike_times_index.push(cum);
        }
        (unit_ids, spike_times_sec, spike_times_index)
    }

    /// Unpacks NWB's ragged `(unit_ids, spike_times_sec, spike_times_index)` arrays into
    /// `(unit_id, spike_samples)` pairs at `sample_rate_hz`.
    pub fn unpack_ragged_spikes(
        unit_ids: &[i64],
        spike_times_sec: &[f64],
        spike_times_index: &[u64],
        sample_rate_hz: f64,
    ) -> Vec<(usize, Vec<u64>)> {
        let fs = sample_rate_hz.max(f64::MIN_POSITIVE);
        let mut prev = 0usize;
        let mut out = Vec::with_capacity(unit_ids.len());
        for (pos, &uid) in unit_ids.iter().enumerate() {
            let end = spike_times_index.get(pos).copied().unwrap_or(0) as usize;
            let slice = &spike_times_sec[prev.min(spike_times_sec.len())..end.min(spike_times_sec.len())];
            let samples: Vec<u64> = slice.iter().map(|&t| (t * fs).round().max(0.0) as u64).collect();
            prev = end;
            out.push((uid.max(0) as usize, samples));
        }
        out
    }
}

