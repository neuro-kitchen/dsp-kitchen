//! Per-unit numbers the Curation views plot besides the spike trains (Phy's firing-rate view,
//! cluster scatter and amplitude histogram): a unit's rate over the recording, the histogram of
//! its spike amplitudes (is it cut off by the detection threshold?), and every unit as a point
//! of depth, amplitude, rate and spike count. Host arithmetic on the sorting already in memory.

use super::sorting::SortingData;

/// One unit as a point of the unit scatter.
#[derive(Debug, Clone, PartialEq)]
pub struct UnitPoint {
    /// Unit index (into [`SortingData::units`]).
    pub unit: usize,
    /// Position on the probe (µm), when the sorting has the probe.
    pub x: f32,
    pub depth: f32,
    /// Typical amplitude ([`SortingData::unit_amplitude`]).
    pub amplitude: f32,
    /// Spikes per second over the recording.
    pub rate_hz: f32,
    pub spikes: usize,
}

impl SortingData {
    /// Spikes per second of `unit` in bins of `bin_s` seconds over `[0, duration_s)` (the last bin
    /// as long as what is left of the recording).
    pub fn firing_rate(&self, unit: usize, duration_s: f64, bin_s: f64) -> Vec<f32> {
        if duration_s <= 0.0 || bin_s <= 0.0 {
            return Vec::new();
        }
        let bins = (duration_s / bin_s).ceil().max(1.0) as usize;
        let mut counts = vec![0u32; bins];
        for t in self.unit_times(unit) {
            let s = t as f64 / self.sample_rate;
            if (0.0..duration_s).contains(&s) {
                counts[((s / bin_s) as usize).min(bins - 1)] += 1;
            }
        }
        counts
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let len = (duration_s - i as f64 * bin_s).min(bin_s);
                (c as f64 / len) as f32
            })
            .collect()
    }

    /// The stored amplitudes of `unit` in `bins` bins: (lower edge, bin width, counts). `None`
    /// when the sorting stores no amplitudes or the unit has no spikes.
    pub fn amplitude_histogram(&self, unit: usize, bins: usize) -> Option<(f32, f32, Vec<u32>)> {
        let values = self.unit_amplitude_values(unit);
        if values.is_empty() || bins == 0 {
            return None;
        }
        let (lo, hi) = values.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
        let width = ((hi - lo) / bins as f32).max(f32::EPSILON);
        let mut counts = vec![0u32; bins];
        for v in values {
            counts[(((v - lo) / width) as usize).min(bins - 1)] += 1;
        }
        Some((lo, width, counts))
    }

    /// The typical amplitude of `unit`: the median of its stored spike amplitudes, else its
    /// template's peak-to-peak on its first channel. `None` without either.
    pub fn unit_amplitude(&self, unit: usize) -> Option<f32> {
        let mut values = self.unit_amplitude_values(unit);
        if !values.is_empty() {
            values.sort_by(f32::total_cmp);
            return Some(values[values.len() / 2]);
        }
        let row = self.units.get(unit)?.template.first()?;
        let (lo, hi) = row.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
        (hi > lo).then_some(hi - lo)
    }

    /// Every unit not merged away that has a position on the probe, as a point of the unit
    /// scatter over a recording `duration_s` long.
    pub fn unit_points(&self, duration_s: f64) -> Vec<UnitPoint> {
        self.alive()
            .filter_map(|u| {
                let (x, depth) = self.unit_position(u)?;
                let spikes = self.by_unit.get(u).map_or(0, Vec::len);
                let rate_hz = if duration_s > 0.0 { (spikes as f64 / duration_s) as f32 } else { 0.0 };
                Some(UnitPoint { unit: u, x, depth, amplitude: self.unit_amplitude(u).unwrap_or(0.0), rate_hz, spikes })
            })
            .collect()
    }

    /// The stored amplitudes of `unit`'s spikes (empty without amplitudes).
    fn unit_amplitude_values(&self, unit: usize) -> Vec<f32> {
        if self.amplitudes.is_empty() {
            return Vec::new();
        }
        self.by_unit.get(unit).map_or_else(Vec::new, |ks| ks.iter().filter_map(|&k| self.amplitudes.get(k as usize).copied()).collect())
    }

    /// `unit`'s mean template on recording channel `channel`, when that channel is in its extent.
    pub fn template_on(&self, unit: usize, channel: usize) -> Option<&[f32]> {
        let u = self.units.get(unit)?;
        let row = u.extent.channels.iter().position(|&c| c == channel)?;
        u.template.get(row).map(Vec::as_slice)
    }
}

#[cfg(test)]
mod tests {
    use super::super::sorting::testing::sorting_of_units;

    /// Units 5 and 6 on channel 0, spikes 10 samples apart at 1 kHz (so 100 spikes per second).
    #[test]
    fn a_units_rate_counts_its_spikes_per_second_of_each_bin() {
        let s = sorting_of_units(&[(5, 0, 4), (6, 0, 2)]);
        // Unit 5: spikes at 0, 10, 20, 30 ms; bins of 20 ms over 60 ms
        let rate = s.firing_rate(0, 0.06, 0.02);
        assert_eq!(rate.len(), 3);
        assert_eq!(rate, vec![100.0, 100.0, 0.0]);
        // A last bin shorter than the others counts per its own length
        let rate = s.firing_rate(0, 0.035, 0.02);
        assert_eq!(rate.len(), 2);
        assert!((rate[1] - 2.0 / 0.015).abs() < 1e-3, "{rate:?}");
    }

    #[test]
    fn amplitudes_give_a_histogram_and_a_median_and_templates_a_fallback() {
        let mut s = sorting_of_units(&[(5, 0, 4), (6, 0, 2)]);
        // No stored amplitudes: no histogram; the template's peak-to-peak (0.5 − (−1.5))
        assert!(s.amplitude_histogram(0, 4).is_none());
        assert!((s.unit_amplitude(0).unwrap() - 2.0).abs() < 1e-6);
        s.amplitudes = vec![1.0, 2.0, 3.0, 4.0, 10.0, 10.0];
        let (lo, width, counts) = s.amplitude_histogram(0, 3).unwrap();
        assert_eq!((lo, counts.iter().sum::<u32>()), (1.0, 4));
        assert!((width - 1.0).abs() < 1e-6);
        assert_eq!(counts, vec![1, 1, 2]);
        assert_eq!(s.unit_amplitude(0), Some(3.0));
    }

    #[test]
    fn units_with_a_position_are_points() {
        let mut s = sorting_of_units(&[(5, 0, 4), (6, 1, 2)]);
        assert!(s.unit_points(1.0).is_empty(), "no probe, no positions");
        s.positions = vec![Some((0.0, 20.0)), Some((16.0, 40.0))];
        let points = s.unit_points(2.0);
        assert_eq!(points.len(), 2);
        assert_eq!((points[1].spikes, points[1].rate_hz), (2, 1.0));
        assert_eq!(s.template_on(1, 1).map(<[f32]>::len), Some(4));
        assert!(s.template_on(1, 0).is_none());
    }
}
