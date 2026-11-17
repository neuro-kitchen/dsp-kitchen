//! Choosing spikes to split off a unit: those a lasso holds in a scatter view.

use super::SortingData;

/// Whether `p` lies inside `polygon` (ray casting; a polygon of fewer than 3 points holds nothing).
pub fn inside(p: (f32, f32), polygon: &[(f32, f32)]) -> bool {
    if polygon.len() < 3 {
        return false;
    }
    let mut on = false;
    let mut j = polygon.len() - 1;
    for i in 0..polygon.len() {
        let ((xi, yi), (xj, yj)) = (polygon[i], polygon[j]);
        if (yi > p.1) != (yj > p.1) && p.0 < (xj - xi) * (p.1 - yi) / (yj - yi) + xi {
            on = !on;
        }
        j = i;
    }
    on
}

/// The spike times of unit `unit` (an index into `sorting.units`) whose (time s, amplitude)
/// lies inside `polygon`: the amplitude view's lasso.
pub fn spikes_in_polygon(sorting: &SortingData, unit: usize, polygon: &[(f32, f32)]) -> Vec<u64> {
    if sorting.amplitudes.is_empty() {
        return Vec::new();
    }
    sorting.by_unit[unit]
        .iter()
        .map(|&k| k as usize)
        .filter(|&k| inside(((sorting.times[k] as f64 / sorting.sample_rate) as f32, sorting.amplitudes[k]), polygon))
        .map(|k| sorting.times[k])
        .collect()
}

/// The spikes of `unit` (sample times) whose snippet on a channel, projected on `basis` (PC1,
/// PC2), falls inside `polygon`: a lasso drawn on the features view. Every spike of the unit is
/// read, not only those drawn: through `read(samples, out)` (that channel, as the view reads it),
/// one second of signal at a time and only the seconds holding spikes of the unit, so a remote
/// signal is asked a few dozen times, not once per spike. `before` / `after`: the snippet's
/// samples around the spike, as the basis was made with.
pub fn spikes_in_feature_polygon(
    sorting: &SortingData,
    unit: usize,
    before: u64,
    after: u64,
    basis: &super::waveforms::PcaBasis,
    polygon: &[(f32, f32)],
    total_samples: u64,
    read: impl Fn(std::ops::Range<u64>, &mut [f32]) -> dspk_core::DspResult<()>,
) -> dspk_core::DspResult<Vec<u64>> {
    let chunk = (sorting.sample_rate.round() as u64).max(before + after + 1);
    let times: Vec<u64> = sorting.unit_times(unit).into_iter().filter(|&t| t >= before && t + after <= total_samples).collect();
    let mut kept = Vec::new();
    let mut i = 0;
    while i < times.len() {
        // The spikes whose snippet lies in [start, start + chunk), read once
        let start = times[i] - before;
        let end = (start + chunk).min(total_samples);
        let mut buf = vec![0f32; (end - start) as usize];
        read(start..end, &mut buf)?;
        while i < times.len() && times[i] + after <= end {
            let a = (times[i] - before - start) as usize;
            let p = basis.project(&buf[a..a + (before + after) as usize]);
            if inside((p[0], p[1]), polygon) {
                kept.push(times[i]);
            }
            i += 1;
        }
    }
    Ok(kept)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lasso_holds_the_points_inside_it() {
        let square = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)];
        assert!(inside((1.0, 1.0), &square));
        assert!(!inside((3.0, 1.0), &square));
        assert!(!inside((1.0, -0.5), &square));
        assert!(!inside((1.0, 1.0), &square[..2]), "two points hold nothing");
        // A concave lasso: the notch is outside
        let notched = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (2.0, 1.0), (0.0, 4.0)];
        assert!(inside((1.0, 0.5), &notched));
        assert!(!inside((2.0, 3.0), &notched));
    }

    /// A lasso on the features keeps the spikes whose snippet projects inside it, all of them
    /// read (here two shapes of spike: up and down), in chunks.
    #[test]
    fn a_feature_lasso_keeps_every_spike_projected_inside_it() {
        use super::super::sorting::testing::sorting_of_units;
        use super::super::waveforms::PcaBasis;
        // Unit 0: 6 spikes 10 samples apart (at 0, 10, …); the signal a bump up at even spikes,
        // down at odd ones, over 3 samples (1 before, 2 after)
        let s = sorting_of_units(&[(5, 0, 6)]);
        let times = s.unit_times(0);
        let total = 200;
        let mut signal = vec![0f32; total];
        for (k, &t) in times.iter().enumerate() {
            let v = if k % 2 == 0 { 5.0 } else { -5.0 };
            if t >= 1 {
                signal[t as usize] = v;
            }
        }
        let basis = PcaBasis { mean: vec![0.0; 3], pc1: vec![0.0, 1.0, 0.0], pc2: vec![1.0, 0.0, 0.0] };
        // PC1 > 0: the spikes going up (the first one is too close to the start to cut)
        let upper = [(0.5, -10.0), (10.0, -10.0), (10.0, 10.0), (0.5, 10.0)];
        let reads = std::cell::Cell::new(0);
        let kept = spikes_in_feature_polygon(&s, 0, 1, 2, &basis, &upper, total as u64, |range, out| {
            reads.set(reads.get() + 1);
            out.copy_from_slice(&signal[range.start as usize..range.end as usize]);
            Ok(())
        })
        .unwrap();
        assert_eq!(kept, vec![times[2], times[4]]);
        assert_eq!(reads.get(), 1, "every spike in one chunk: one read");
    }
}
