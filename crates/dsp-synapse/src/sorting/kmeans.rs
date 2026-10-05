//! k-means clustering with the semantics of `sklearn.cluster.KMeans` (Lloyd): greedy k-means++
//! seeding, `n_init` restarts keeping the lowest inertia, convergence when the centres move by
//! less than `tol · mean feature variance`, empty clusters re-seeded at the point farthest from
//! its centre. Seeded and deterministic.

/// sklearn defaults.
pub const DEFAULT_N_INIT: usize = 10;
pub const DEFAULT_MAX_ITER: usize = 300;
pub const DEFAULT_TOL: f64 = 1e-4;

/// Options of [`kmeans`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KMeansOptions {
    pub n_init: usize,
    pub max_iter: usize,
    /// Relative to the mean per-feature variance of the data.
    pub tol: f64,
    pub seed: u64,
}

impl Default for KMeansOptions {
    fn default() -> Self {
        Self { n_init: DEFAULT_N_INIT, max_iter: DEFAULT_MAX_ITER, tol: DEFAULT_TOL, seed: 0 }
    }
}

/// Result of [`kmeans`].
#[derive(Debug, Clone, PartialEq)]
pub struct KMeansResult {
    /// `[k, d]` centres.
    pub centers: Vec<f32>,
    /// Nearest centre of each point.
    pub labels: Vec<usize>,
    /// Sum of squared distances to the nearest centre.
    pub inertia: f64,
}

/// SplitMix64: small, seeded, reproducible.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn sq_dist(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum()
}

/// Index drawn with probability proportional to `weights` (the last positive one on round-off).
fn draw(weights: &[f64], rng: &mut Rng) -> usize {
    let total: f64 = weights.iter().sum();
    if !(total > 0.0) {
        return (rng.next_u64() as usize) % weights.len();
    }
    let mut r = rng.uniform() * total;
    for (i, &w) in weights.iter().enumerate() {
        r -= w;
        if r < 0.0 {
            return i;
        }
    }
    weights.iter().rposition(|&w| w > 0.0).unwrap_or(0)
}

/// Greedy k-means++ (sklearn): each new centre is the best of `2 + ln k` candidates drawn by
/// squared distance.
fn kmeans_plus_plus(x: &[f32], n: usize, d: usize, k: usize, rng: &mut Rng) -> Vec<f32> {
    let trials = 2 + (k as f64).ln() as usize;
    let mut centers = Vec::with_capacity(k * d);
    let first = (rng.next_u64() as usize) % n;
    centers.extend_from_slice(&x[first * d..(first + 1) * d]);
    let mut closest: Vec<f64> = (0..n).map(|i| sq_dist(&x[i * d..(i + 1) * d], &centers[..d])).collect();
    for _ in 1..k {
        let mut best: Option<(usize, f64, Vec<f64>)> = None;
        for _ in 0..trials {
            let c = draw(&closest, rng);
            let cand = &x[c * d..(c + 1) * d];
            let updated: Vec<f64> = (0..n).map(|i| closest[i].min(sq_dist(&x[i * d..(i + 1) * d], cand))).collect();
            let pot: f64 = updated.iter().sum();
            if best.as_ref().is_none_or(|b| pot < b.1) {
                best = Some((c, pot, updated));
            }
        }
        let (c, _, updated) = best.expect("at least one trial");
        centers.extend_from_slice(&x[c * d..(c + 1) * d]);
        closest = updated;
    }
    centers
}

/// One Lloyd run from `centers`; returns the final centres, labels, inertia.
fn lloyd(x: &[f32], n: usize, d: usize, k: usize, mut centers: Vec<f32>, max_iter: usize, tol_abs: f64) -> KMeansResult {
    let mut labels = vec![0usize; n];
    let mut dist = vec![0.0f64; n];
    for _ in 0..max_iter.max(1) {
        for i in 0..n {
            let xi = &x[i * d..(i + 1) * d];
            let (best, bd) = (0..k).map(|c| (c, sq_dist(xi, &centers[c * d..(c + 1) * d]))).fold((0, f64::INFINITY), |b, c| if c.1 < b.1 { c } else { b });
            labels[i] = best;
            dist[i] = bd;
        }
        let mut sums = vec![0.0f64; k * d];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            counts[labels[i]] += 1;
            for f in 0..d {
                sums[labels[i] * d + f] += x[i * d + f] as f64;
            }
        }
        let mut new_centers = centers.clone();
        // Empty clusters move to the points farthest from their centre
        let mut far: Vec<usize> = (0..n).collect();
        far.sort_by(|&a, &b| dist[b].total_cmp(&dist[a]));
        let mut far_iter = far.into_iter();
        for c in 0..k {
            if counts[c] > 0 {
                for f in 0..d {
                    new_centers[c * d + f] = (sums[c * d + f] / counts[c] as f64) as f32;
                }
            } else if let Some(p) = far_iter.next() {
                new_centers[c * d..(c + 1) * d].copy_from_slice(&x[p * d..(p + 1) * d]);
            }
        }
        let shift: f64 = (0..k).map(|c| sq_dist(&centers[c * d..(c + 1) * d], &new_centers[c * d..(c + 1) * d])).sum();
        centers = new_centers;
        if shift <= tol_abs {
            break;
        }
    }
    // Final assignment to the final centres
    let mut inertia = 0.0;
    for i in 0..n {
        let xi = &x[i * d..(i + 1) * d];
        let (best, bd) = (0..k).map(|c| (c, sq_dist(xi, &centers[c * d..(c + 1) * d]))).fold((0, f64::INFINITY), |b, c| if c.1 < b.1 { c } else { b });
        labels[i] = best;
        inertia += bd;
    }
    KMeansResult { centers, labels, inertia }
}

/// k-means of `x` (`[n, d]`) into `k` clusters (`k ≤ n`). See the module docs.
pub fn kmeans(x: &[f32], n: usize, d: usize, k: usize, options: &KMeansOptions) -> KMeansResult {
    assert_eq!(x.len(), n * d, "kmeans: data size mismatch");
    assert!(k >= 1 && k <= n, "kmeans: need 1 ≤ k ≤ n");
    // sklearn: tol scaled by the mean variance of the features
    let mean_var = {
        let mut v = 0.0f64;
        for f in 0..d {
            let mean = (0..n).map(|i| x[i * d + f] as f64).sum::<f64>() / n as f64;
            v += (0..n).map(|i| (x[i * d + f] as f64 - mean).powi(2)).sum::<f64>() / n as f64;
        }
        v / d.max(1) as f64
    };
    let tol_abs = options.tol * mean_var;
    let mut rng = Rng(options.seed);
    (0..options.n_init.max(1))
        .map(|_| {
            let init = kmeans_plus_plus(x, n, d, k, &mut rng);
            lloyd(x, n, d, k, init, options.max_iter, tol_abs)
        })
        .min_by(|a, b| a.inertia.total_cmp(&b.inertia))
        .expect("n_init ≥ 1")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separates_three_blobs() {
        let centres = [[0.0f32, 0.0], [10.0, 0.0], [0.0, 10.0]];
        let mut x = Vec::new();
        for c in &centres {
            for i in 0..30 {
                let o = (i as f32 * 0.37).sin() * 0.5;
                x.extend_from_slice(&[c[0] + o, c[1] - o]);
            }
        }
        let r = kmeans(&x, 90, 2, 3, &KMeansOptions { seed: 7, ..Default::default() });
        for block in 0..3 {
            let l = r.labels[block * 30];
            assert!(r.labels[block * 30..(block + 1) * 30].iter().all(|&v| v == l), "block {block}");
        }
        assert!(r.inertia < 90.0);
    }
}
