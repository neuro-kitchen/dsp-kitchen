//! SpikeInterface's detection prototype (`sortingcomponents/tools.py`
//! `get_prototype_and_waveforms_*`, MIT): the per-sample median (NaNs ignored) of peak waveforms on
//! their own channel, each divided by `|w[nbefore]|`. The waveforms come from `locally_exclusive`
//! detection ([`super::LocallyExclusiveDetector`]), at most `n_peaks` of them drawn at random.

/// The prototype of `waveforms` (row-major `[count, len]`, peak at `nbefore`). A waveform whose
/// peak sample is 0 gives NaN (`0 / 0`, ignored as by `np.nanmedian`) or ±∞ (kept, as NumPy keeps
/// it); a sample with no value but NaN is NaN.
///
/// # Panics
///
/// If `waveforms.len()` is not a multiple of `len`, or `nbefore ≥ len`.
pub fn prototype(waveforms: &[f32], len: usize, nbefore: usize) -> Vec<f32> {
    assert!(len > 0 && waveforms.len() % len == 0 && nbefore < len, "waveforms must be [count, len]");
    let count = waveforms.len() / len;
    let mut column: Vec<f64> = Vec::with_capacity(count);
    (0..len)
        .map(|s| {
            column.clear();
            for w in waveforms.chunks_exact(len) {
                let v = w[s] as f64 / (w[nbefore] as f64).abs();
                if !v.is_nan() {
                    column.push(v);
                }
            }
            let n = column.len();
            if n == 0 {
                return f32::NAN;
            }
            let (lower, mid, _) = column.select_nth_unstable_by(n / 2, f64::total_cmp);
            let mid = *mid;
            (if n % 2 == 1 { mid } else { 0.5 * (lower.iter().copied().fold(f64::NEG_INFINITY, f64::max) + mid) }) as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_of_scaled_waveforms() {
        // Two waveforms of one shape at different sizes, one odd one out, one all zero (NaNs)
        let shape = [0.5f32, -1.0, 0.25];
        let mut w = Vec::new();
        for a in [2.0f32, 5.0] {
            w.extend(shape.iter().map(|v| v * a));
        }
        w.extend([3.0f32, -1.0, 1.0]);
        w.extend([0.0f32, 0.0, 0.0]);
        assert_eq!(prototype(&w, 3, 1), vec![0.5, -1.0, 0.25]);
    }
}
