//! What every sorter produces, and its one conversion to a [`SortingOutput`]: spikes with their
//! units, a template per unit, and how the amplitudes are scaled. Labels, the composite score and
//! the export (Phy, `.sorting.zarr`, NWB) then work the same for every sorter.
//!
//! [`SorterResult::to_sorting_output`]: one unit per non-empty unit id. Its primary channel is the
//! template channel with the most energy; it is labelled *good* when its auto-correlogram is
//! refractory ([`dsp_synapse::metrics::acg_refractory`]), else *mua* (Phy's `cluster_KSLabel`); it
//! carries EMUsort's composite score, whose SNR follows [`AmplitudeScale`].

use dsp_io::neuro::probe::SensorLayout;
use dsp_synapse::core::{SortedUnit, SortingOutput, WaveformTemplate};
use dsp_synapse::metrics::{acg_refractory, composite_score, CompositeScoreOptions, RefractoryOptions};
use dsp_synapse::{QualityCriteria, UnitQualityLabel};

/// What the spike amplitudes (and templates) are measured in, which sets each unit's SNR.
#[derive(Debug, Clone, PartialEq)]
pub enum AmplitudeScale {
    /// Whitened σ (Kilosort4, EMUsort): the noise is 1 by construction, so the SNR is the median
    /// `|amplitude|`.
    Whitened,
    /// The recording's unit (µV), with each channel's noise level: the SNR is the template's
    /// largest `|value|` on the primary channel over that channel's noise.
    Recording { noise_levels: Vec<f32> },
}

/// A sorter's spikes and units (module docs). Spike `i`: `spike_samples[i]` (recording samples),
/// unit `spike_units[i]` (`< n_units`; others left out), `spike_amplitudes[i]`,
/// `spike_locations[i]` (µm).
#[derive(Debug, Clone, PartialEq)]
pub struct SorterResult {
    pub sorter: String,
    pub sample_rate_hz: f64,
    pub total_samples: u64,
    pub n_units: usize,
    pub spike_samples: Vec<u64>,
    pub spike_units: Vec<u32>,
    pub spike_amplitudes: Vec<f32>,
    pub spike_locations: Vec<[f32; 3]>,
    /// One per unit (`None`: no template).
    pub templates: Vec<Option<WaveformTemplate>>,
    pub amplitude_scale: AmplitudeScale,
    /// The auto-correlogram test of the labels.
    pub label_refractory: RefractoryOptions,
    pub composite: CompositeScoreOptions,
}

impl SorterResult {
    /// The units in [`SortingOutput`] form (module docs).
    ///
    /// # Panics
    ///
    /// If the per-spike arrays differ in length, or there are not `n_units` templates.
    pub fn to_sorting_output(&self, probe: Option<SensorLayout>) -> SortingOutput {
        let n = self.spike_samples.len();
        assert!(
            self.spike_units.len() == n && self.spike_amplitudes.len() == n && self.spike_locations.len() == n,
            "one unit, amplitude and location per spike"
        );
        assert_eq!(self.templates.len(), self.n_units, "one template slot per unit");
        let mut samples: Vec<Vec<u64>> = vec![Vec::new(); self.n_units];
        let mut amps: Vec<Vec<f32>> = vec![Vec::new(); self.n_units];
        let mut locs: Vec<Vec<[f32; 3]>> = vec![Vec::new(); self.n_units];
        for i in 0..n {
            let u = self.spike_units[i] as usize;
            if u < self.n_units {
                samples[u].push(self.spike_samples[i]);
                amps[u].push(self.spike_amplitudes[i]);
                locs[u].push(self.spike_locations[i]);
            }
        }
        let units = samples
            .into_iter()
            .zip(amps)
            .zip(locs)
            .enumerate()
            .filter(|(_, ((s, _), _))| !s.is_empty())
            .map(|(unit_id, ((s, a), l))| {
                let template = self.templates[unit_id].clone();
                let primary = template.as_ref().and_then(primary_channel);
                let snr = match &self.amplitude_scale {
                    AmplitudeScale::Whitened => median_abs(&a),
                    AmplitudeScale::Recording { noise_levels } => match (&template, primary) {
                        (Some(t), Some(ch)) => {
                            let peak = t.channel_row(ch).map_or(0.0, |r| r.iter().fold(0.0f32, |m, v| m.max(v.abs())));
                            noise_levels.get(ch).map_or(f64::NAN, |&z| (peak / z) as f64)
                        }
                        _ => f64::NAN,
                    },
                };
                let mut sorted = s.clone();
                sorted.sort_unstable();
                let good = acg_refractory(&sorted, self.sample_rate_hz, &self.label_refractory).1;
                let mut unit = SortedUnit::from_spikes_with(
                    unit_id,
                    primary,
                    s,
                    a,
                    l,
                    template,
                    self.sample_rate_hz,
                    self.total_samples,
                    None,
                    QualityCriteria::default(),
                );
                unit.quality_label = if good { UnitQualityLabel::SingleUnit } else { UnitQualityLabel::MultiUnit };
                unit.composite_score =
                    composite_score(&unit.spike_samples, &unit.amplitudes_uv, snr, self.total_samples, self.sample_rate_hz, &self.composite).score;
                unit
            })
            .collect();
        SortingOutput::new(self.sorter.clone(), self.sample_rate_hz, self.total_samples, probe, units, None)
    }
}

/// The template channel with the most energy (`Σ mean²`).
fn primary_channel(t: &WaveformTemplate) -> Option<usize> {
    let w = t.num_samples;
    (0..t.channel_ids.len())
        .max_by(|&i, &j| {
            let energy = |r: usize| t.mean[r * w..(r + 1) * w].iter().map(|v| v * v).sum::<f32>();
            energy(i).total_cmp(&energy(j))
        })
        .map(|r| t.channel_ids[r])
}

/// Median of `|v|` (NaN when empty).
fn median_abs(v: &[f32]) -> f64 {
    let mut a: Vec<f64> = v.iter().map(|x| x.abs() as f64).collect();
    if a.is_empty() {
        return f64::NAN;
    }
    a.sort_by(f64::total_cmp);
    let m = a.len() / 2;
    if a.len().is_multiple_of(2) { 0.5 * (a[m - 1] + a[m]) } else { a[m] }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f64 = 30_000.0;
    const TOTAL: u64 = 300 * 30_000;

    /// Unit 0: a 10 Hz train with a 3 ms dead time (refractory); unit 1: spikes in pairs 1 ms
    /// apart (not refractory); unit 2: no spikes.
    fn result(scale: AmplitudeScale) -> SorterResult {
        let mut spike_samples = Vec::new();
        let mut spike_units = Vec::new();
        for k in 0..3000u64 {
            spike_samples.push(k * 3000 + (k * 7919 % 1500));
            spike_units.push(0);
        }
        // Pairs 1 ms apart (bin 1): many coincidences inside the refractory period. (Pairs only in
        // bin 2 would leave bin 1 empty, and the paper's minimum over k would call that refractory.)
        for k in 0..3000u64 {
            spike_samples.extend([k * 3000 + 700, k * 3000 + 730]);
            spike_units.extend([1, 1]);
        }
        let n = spike_samples.len();
        // Unit 0's template peaks on channel 2, unit 1's on channel 0
        let tmpl = |big: usize| {
            let mean: Vec<f32> = (0..3 * 10).map(|i| if i / 10 == big { -40.0 } else { -5.0 }).collect();
            WaveformTemplate::with_count(vec![0, 1, 2], 10, 100, mean, vec![0.0; 30])
        };
        SorterResult {
            sorter: "test".into(),
            sample_rate_hz: FS,
            total_samples: TOTAL,
            n_units: 3,
            spike_samples,
            spike_units,
            spike_amplitudes: vec![-8.0; n],
            spike_locations: vec![[0.0, 10.0, 0.0]; n],
            templates: vec![Some(tmpl(2)), Some(tmpl(0)), None],
            amplitude_scale: scale,
            label_refractory: RefractoryOptions::default(),
            composite: CompositeScoreOptions::default(),
        }
    }

    #[test]
    fn units_get_their_channel_label_and_score() {
        let out = result(AmplitudeScale::Whitened).to_sorting_output(None);
        assert_eq!(out.units.len(), 2, "the empty unit is left out");
        let (u0, u1) = (&out.units[0], &out.units[1]);
        assert_eq!((u0.unit_id, u0.primary_channel, u0.spike_samples.len()), (0, 2, 3000));
        assert_eq!((u1.unit_id, u1.primary_channel), (1, 0));
        assert_eq!(u0.quality_label, UnitQualityLabel::SingleUnit);
        assert_eq!(u1.quality_label, UnitQualityLabel::MultiUnit);
        assert!(u0.composite_score > 0.0 && u0.composite_score <= 1.0, "{}", u0.composite_score);
    }

    /// Whitened: SNR = median |amplitude| (8 → high SNR); recording scale with large noise: low.
    #[test]
    fn the_amplitude_scale_sets_the_snr() {
        let whitened = result(AmplitudeScale::Whitened).to_sorting_output(None).units[0].composite_score;
        let noisy = result(AmplitudeScale::Recording { noise_levels: vec![1.0, 1.0, 40.0] }).to_sorting_output(None).units[0].composite_score;
        assert!(noisy < whitened * 0.1, "SNR 1 (40 / 40) must cut the score: {noisy} vs {whitened}");
    }
}
