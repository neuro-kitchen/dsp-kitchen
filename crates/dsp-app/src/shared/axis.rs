//! Axis helpers shared by the plot modules.

/// Smallest value from the 1-2-5 series (× 10^k) that is ≥ `raw`.
pub fn nice_step(raw: f64) -> f64 {
    if !(raw > 0.0) || !raw.is_finite() {
        return 1.0;
    }
    let mag = 10f64.powf(raw.log10().floor());
    let norm = raw / mag;
    let nice = if norm <= 1.0 { 1.0 } else if norm <= 2.0 { 2.0 } else if norm <= 5.0 { 5.0 } else { 10.0 };
    nice * mag
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
    }
}
