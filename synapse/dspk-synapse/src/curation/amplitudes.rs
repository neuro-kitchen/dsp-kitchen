//! Spike amplitudes of a unit over time, as stored with the sorting.

use super::sorting::SortingData;

impl SortingData {
    /// (time s, amplitude) of up to `max` spikes of `unit`, evenly spread.
    pub fn unit_amplitudes(&self, unit: usize, max: usize) -> Vec<(f32, f32)> {
        if self.amplitudes.is_empty() {
            return Vec::new();
        }
        let idx: Vec<usize> = self.by_unit.get(unit).map_or_else(Vec::new, |ks| ks.iter().map(|&k| k as usize).collect());
        let step = (idx.len() / max.max(1)).max(1);
        idx.iter().step_by(step).map(|&k| ((self.times[k] as f64 / self.sample_rate) as f32, self.amplitudes[k])).collect()
    }
}
