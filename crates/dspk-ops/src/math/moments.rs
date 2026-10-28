//! Element-wise running mean and variance of equal-length vectors (window / epoch averaging:
//! spike templates, stimulus-triggered averages), in `f64` with Welford updates and Chan merges,
//! so batches reduced elsewhere (e.g. on the device) combine exactly.

/// Running count, mean and sum of squared deviations (`M₂`) per element. See the module docs.
#[derive(Debug, Clone, PartialEq)]
pub struct RunningMoments {
    count: u64,
    mean: Vec<f64>,
    m2: Vec<f64>,
}

impl RunningMoments {
    /// Empty moments of vectors of `len` elements.
    pub fn new(len: usize) -> Self {
        Self { count: 0, mean: vec![0.0; len], m2: vec![0.0; len] }
    }

    /// Elements per vector.
    pub fn len(&self) -> usize {
        self.mean.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mean.is_empty()
    }

    /// Vectors accumulated.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Adds one vector (Welford). Panics on a length mismatch.
    pub fn push(&mut self, x: &[f32]) {
        assert_eq!(x.len(), self.len(), "vector length mismatch");
        self.count += 1;
        let n = self.count as f64;
        for ((m, m2), &v) in self.mean.iter_mut().zip(&mut self.m2).zip(x) {
            let v = v as f64;
            let delta = v - *m;
            *m += delta / n;
            *m2 += delta * (v - *m);
        }
    }

    /// Merges a batch of `count` vectors given its per-element mean and `M₂` (Chan et al.).
    /// Panics on a length mismatch.
    pub fn merge(&mut self, count: u64, mean: &[f64], m2: &[f64]) {
        assert!(mean.len() == self.len() && m2.len() == self.len(), "vector length mismatch");
        if count == 0 {
            return;
        }
        let (n_a, n_b) = (self.count as f64, count as f64);
        let n = n_a + n_b;
        for i in 0..self.len() {
            let delta = mean[i] - self.mean[i];
            self.mean[i] += delta * (n_b / n);
            self.m2[i] += m2[i].max(0.0) + delta * delta * (n_a * n_b / n);
        }
        self.count += count;
    }

    /// Element-wise mean.
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Element-wise `M₂ / (count − ddof)` (`ddof = 0`: population variance, numpy's default;
    /// `1`: sample variance). Zeros while `count ≤ ddof`.
    pub fn variance(&self, ddof: u64) -> Vec<f64> {
        let dof = self.count.saturating_sub(ddof);
        if dof == 0 {
            return vec![0.0; self.len()];
        }
        self.m2.iter().map(|&m2| (m2 / dof as f64).max(0.0)).collect()
    }

    /// Element-wise standard deviation with `ddof` (see [`Self::variance`]).
    pub fn std(&self, ddof: u64) -> Vec<f64> {
        self.variance(ddof).into_iter().map(f64::sqrt).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_merge_match_two_pass() {
        let rows: [[f32; 2]; 5] = [[1.0, -2.0], [3.0, 0.0], [2.0, 4.0], [10.0, 1.0], [-1.0, 1.0]];
        let mut all = RunningMoments::new(2);
        rows.iter().for_each(|r| all.push(r));
        // numpy: mean [3, 0.8], var ddof=0 [14, 3.76], ddof=1 [17.5, 4.7]
        assert!((all.mean()[0] - 3.0).abs() < 1e-12 && (all.mean()[1] - 0.8).abs() < 1e-12);
        assert!((all.variance(0)[0] - 14.0).abs() < 1e-12 && (all.variance(1)[1] - 4.7).abs() < 1e-12);

        let (mut a, mut b) = (RunningMoments::new(2), RunningMoments::new(2));
        rows[..2].iter().for_each(|r| a.push(r));
        rows[2..].iter().for_each(|r| b.push(r));
        a.merge(b.count(), b.mean(), &b.variance(0).iter().map(|v| v * b.count() as f64).collect::<Vec<_>>());
        for (x, y) in a.variance(0).iter().zip(all.variance(0)) {
            assert!((x - y).abs() < 1e-12);
        }
    }
}
