/// Computes the sub-sample peak offset using 3-point parabolic interpolation:
/// `delta_t = (y[t-1] - y[t+1]) / (2 * (y[t-1] - 2*y[t] + y[t+1]))`
pub fn parabolic_subsample_offset(y_prev: f32, y_peak: f32, y_next: f32) -> f32 {
    let denom = 2.0 * (y_prev - 2.0 * y_peak + y_next);
    if denom.abs() < 1e-6 {
        return 0.0;
    }
    ((y_prev - y_next) / denom).clamp(-0.5, 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parabolic_alignment() {
        let y0 = -80.0;
        let y1 = -100.0;
        let y2 = -90.0;
        let offset = parabolic_subsample_offset(y0, y1, y2);
        assert!(offset > 0.0 && offset < 0.5);
    }
}
