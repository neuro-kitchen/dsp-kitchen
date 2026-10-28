//! Lagged cross-correlation of two short signals and its peak (host; for template, delay and
//! registration searches over a few lags).

use num_traits::Float;

/// Denominator below which a parabola through three samples counts as flat.
const PARABOLA_FLAT: f64 = 1e-12;

/// `Σₜ x[t] · y[t + lag]` over the samples where both exist.
pub fn lagged_dot<T: Float>(x: &[T], y: &[T], lag: isize) -> T {
    let start = (-lag).max(0) as usize;
    let end = (y.len() as isize - lag).clamp(0, x.len() as isize) as usize;
    (start..end.max(start)).fold(T::zero(), |acc, t| acc + x[t] * y[(t as isize + lag) as usize])
}

/// [`lagged_dot`] for every lag in `−max_lag..=max_lag` (entry `k` is lag `k − max_lag`): the
/// `mode="full"` cross-correlation `scipy.signal.correlate(y, x)` restricted to those lags.
pub fn cross_correlation<T: Float>(x: &[T], y: &[T], max_lag: usize) -> Vec<T> {
    let m = max_lag as isize;
    (-m..=m).map(|lag| lagged_dot(x, y, lag)).collect()
}

/// Offset in `[−0.5, 0.5]` of the vertex of the parabola through `(−1, y_prev)`, `(0, y_mid)`,
/// `(1, y_next)` (a peak or a trough alike); 0 when the three are collinear.
pub fn parabolic_vertex_offset<T: Float>(y_prev: T, y_mid: T, y_next: T) -> T {
    let two = T::one() + T::one();
    let denom = two * (y_prev - two * y_mid + y_next);
    let half = T::from(0.5).expect("0.5");
    if denom.abs() <= T::from(PARABOLA_FLAT).expect("tolerance") {
        return T::zero();
    }
    ((y_prev - y_next) / denom).max(-half).min(half)
}

/// Largest entry of a [`cross_correlation`] over `−max_lag..=max_lag`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LagPeak<T> {
    /// Integer lag of the largest entry (the earliest on ties).
    pub lag: isize,
    pub value: T,
    /// Parabolic refinement: the peak lies at `lag + offset` (0 at either end).
    pub offset: T,
}

impl<T: Float> LagPeak<T> {
    /// `lag + offset`.
    pub fn fractional_lag(&self) -> T {
        T::from(self.lag).expect("lag") + self.offset
    }
}

/// The largest entry of `corr` (lags `−max_lag..=max_lag`), refined parabolically; `None` when
/// `corr` is empty.
pub fn peak_lag<T: Float>(corr: &[T], max_lag: usize) -> Option<LagPeak<T>> {
    let (best, &value) = corr
        .iter()
        .enumerate()
        .fold(None, |best: Option<(usize, &T)>, (i, v)| match best {
            Some((_, b)) if *v <= *b => best,
            _ => Some((i, v)),
        })?;
    let offset = if best > 0 && best + 1 < corr.len() {
        parabolic_vertex_offset(corr[best - 1], corr[best], corr[best + 1])
    } else {
        T::zero()
    };
    Some(LagPeak { lag: best as isize - max_lag as isize, value, offset })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_scipy_correlate() {
        // scipy.signal.correlate(y, x, "full")[1:6] (lags −2..=2) = [0, 0, 2, 5, 2]
        let x = [1.0, 2.0, 0.0, 0.0];
        let y = [0.0, 1.0, 2.0, 0.0];
        assert_eq!(cross_correlation(&x, &y, 2), vec![0.0, 0.0, 2.0, 5.0, 2.0]);
        let peak = peak_lag(&cross_correlation(&x, &y, 2), 2).unwrap();
        assert_eq!((peak.lag, peak.value), (1, 5.0), "y is x delayed by one sample");
    }

    #[test]
    fn parabola_vertex() {
        // y = −(t − 0.25)²
        let y = |t: f64| -(t - 0.25) * (t - 0.25);
        assert!((parabolic_vertex_offset(y(-1.0), y(0.0), y(1.0)) - 0.25).abs() < 1e-12);
        assert_eq!(parabolic_vertex_offset(1.0, 1.0, 1.0), 0.0);
    }
}
