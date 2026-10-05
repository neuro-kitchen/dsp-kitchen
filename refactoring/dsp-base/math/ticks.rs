//! "Nice" axis ticks: steps from the 1-2-5 series, so labels read as round numbers.

/// Smallest value of the 1-2-5 series (× 10^k) that is ≥ `raw` (1 for non-positive or non-finite
/// input).
pub fn nice_step(raw: f64) -> f64 {
    if raw.is_nan() || raw <= 0.0 || !raw.is_finite() {
        return 1.0;
    }
    let mag = 10f64.powf(raw.log10().floor());
    let norm = raw / mag;
    let nice = if norm <= 1.0 { 1.0 } else if norm <= 2.0 { 2.0 } else if norm <= 5.0 { 5.0 } else { 10.0 };
    nice * mag
}

/// Tick positions over `[min, max]`: multiples of a nice step giving about `target` ticks, and the
/// decimals their labels need.
pub fn ticks(min: f64, max: f64, target: usize) -> (Vec<f64>, usize) {
    let span = (max - min).max(1e-12);
    let step = nice_step(span / target.max(2) as f64);
    let decimals = (-step.log10().floor()).max(0.0) as usize;
    let first = (min / step).ceil() as i64;
    // At most a few dozen: `target` is a count of labels on screen
    let values = (first..).map(|i| i as f64 * step).take_while(|&v| v <= max + 1e-12).take(64).collect();
    (values, decimals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nice_step() {
        assert_eq!(nice_step(0.013), 0.02);
        assert_eq!(nice_step(3.0), 5.0);
        assert_eq!(nice_step(7.0), 10.0);
        assert_eq!(nice_step(0.0), 1.0);
        assert_eq!(nice_step(f64::NAN), 1.0);
    }

    #[test]
    fn test_ticks() {
        // 0.94 / 4 rounds up to a 0.5 step
        assert_eq!(ticks(0.03, 0.97, 4), (vec![0.5], 1));
        let (v, d) = ticks(0.0, 1.0, 5);
        assert_eq!((v.len(), d), (6, 1));
        assert!((v[1] - 0.2).abs() < 1e-12 && (v[5] - 1.0).abs() < 1e-12);
        let (v, d) = ticks(-10.0, 10.0, 4);
        assert_eq!((v, d), (vec![-10.0, -5.0, 0.0, 5.0, 10.0], 0));
    }
}
