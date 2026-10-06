//! k-means clustering with the semantics of `sklearn.cluster.KMeans` (Lloyd): greedy k-means++
//! seeding, `n_init` restarts keeping the lowest inertia, convergence when the centres move by
//! less than `tol · mean feature variance`, empty clusters re-seeded at the point farthest from
//! its centre. Seeded and deterministic.
//!
//! On the device ([`DevicePoints`], uploaded once): distances, assignments, per-cluster sums,
//! k-means++ trial potentials, inertia. The host keeps the random draws (same stream as the
//! reference) and the `[k, d]` centres; per Lloyd iteration it reads back the `k · d` sums, per
//! k-means++ draw one block of weights. Distances are `f32`; sums are split and added in `f64`.

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

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_base::core::{buffer, reduce};
use dsp_core::compute::LaunchGeometry;

use super::kernels::points::{closest_update_kernel, cluster_sums_kernel, nearest_centre_kernel};
use super::points::{block_sums, device_sum, DevicePoints, SUM_BLOCK};

/// Points each unit of the cluster sums adds before the host adds the splits in `f64`.
const SUM_SPLIT: usize = 4096;

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

/// `‖a − b‖²` of two host vectors.
fn sq_dist(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum()
}

/// Index drawn with probability proportional to the `n` device `weights` (as the reference
/// `draw`: one uniform, the first index where the running sum passes it, the last positive weight
/// on round-off), reading back only the block sums and one block.
fn draw<R: Runtime>(client: &ComputeClient<R>, weights: &Handle, sums: &[f32], n: usize, rng: &mut Rng) -> usize {
    let total: f64 = sums.iter().map(|&v| v as f64).sum();
    if !(total > 0.0) {
        return (rng.next_u64() as usize) % n;
    }
    let mut r = rng.uniform() * total;
    let block_of = |b: usize| (b * SUM_BLOCK, (n - b * SUM_BLOCK).min(SUM_BLOCK));
    for (b, &s) in sums.iter().enumerate() {
        if r - s as f64 >= 0.0 {
            r -= s as f64;
            continue;
        }
        let (start, len) = block_of(b);
        for (i, &w) in buffer::download_range::<R, f32>(client, weights.clone(), start, len).iter().enumerate() {
            r -= w as f64;
            if r < 0.0 {
                return start + i;
            }
        }
    }
    // Round-off: the last positive weight
    for b in (0..sums.len()).rev().filter(|&b| sums[b] > 0.0) {
        let (start, len) = block_of(b);
        let block = buffer::download_range::<R, f32>(client, weights.clone(), start, len);
        if let Some(i) = block.iter().rposition(|&w| w > 0.0) {
            return start + i;
        }
    }
    0
}

/// `updated = min(closest, ‖x − x_candidate‖²)` on the device.
fn closest_after<R: Runtime>(client: &ComputeClient<R>, points: &DevicePoints, closest: &Handle, candidate: usize) -> Handle {
    let (n, d) = (points.n, points.d);
    let updated = buffer::empty::<R, f32>(client, n);
    let geom = LaunchGeometry::elementwise(client, n);
    // SAFETY: `points` holds `d · n`, `closest` and `updated` `n` values
    unsafe {
        closest_update_kernel::launch::<f32, R>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ArrayArg::from_raw_parts(points.handle.clone(), d * n),
            ArrayArg::from_raw_parts(closest.clone(), n),
            ArrayArg::from_raw_parts(updated.clone(), n),
            candidate as u32,
            n as u32,
            d as u32,
        );
    }
    updated
}

/// Greedy k-means++ (sklearn): each new centre is the best of `2 + ln k` candidates drawn by
/// squared distance. `unreached` holds `n` values of `f32::MAX`.
fn kmeans_plus_plus<R: Runtime>(client: &ComputeClient<R>, points: &DevicePoints, k: usize, unreached: &Handle, rng: &mut Rng) -> Vec<f32> {
    let n = points.n;
    let trials = 2 + (k as f64).ln() as usize;
    let first = (rng.next_u64() as usize) % n;
    let mut centers = points.point(client, first);
    let mut closest = closest_after(client, points, unreached, first);
    for _ in 1..k {
        let sums = block_sums(client, &closest, n);
        let mut best: Option<(usize, f64, Handle)> = None;
        for _ in 0..trials {
            let c = draw(client, &closest, &sums, n, rng);
            let updated = closest_after(client, points, &closest, c);
            let pot = device_sum(client, &updated, n);
            if best.as_ref().is_none_or(|b| pot < b.1) {
                best = Some((c, pot, updated));
            }
        }
        let (c, _, updated) = best.expect("at least one trial");
        centers.extend(points.point(client, c));
        closest = updated;
    }
    centers
}

/// Device scratch of a Lloyd run.
struct Assignment {
    label: Handle,
    dist: Handle,
}

/// Nearest centre and its squared distance for every point.
fn assign<R: Runtime>(client: &ComputeClient<R>, points: &DevicePoints, centers: &[f32], k: usize, out: &Assignment) {
    let (n, d) = (points.n, points.d);
    let geom = LaunchGeometry::elementwise(client, n);
    // SAFETY: `points` holds `d · n`, the centres `k · d`, `label` and `dist` `n` values
    unsafe {
        nearest_centre_kernel::launch::<f32, R>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ArrayArg::from_raw_parts(points.handle.clone(), d * n),
            ArrayArg::from_raw_parts(buffer::upload(client, centers), k * d),
            ArrayArg::from_raw_parts(out.label.clone(), n),
            ArrayArg::from_raw_parts(out.dist.clone(), n),
            n as u32,
            d as u32,
            k as u32,
        );
    }
}

/// `(sums [k, d], counts [k])` of the current assignment, split on the device, added in `f64`.
fn cluster_sums<R: Runtime>(client: &ComputeClient<R>, points: &DevicePoints, label: &Handle, k: usize) -> (Vec<f64>, Vec<usize>) {
    let (n, d) = (points.n, points.d);
    let splits = n.div_ceil(SUM_SPLIT).max(1);
    let (sums, counts) = (buffer::empty::<R, f32>(client, splits * k * d), buffer::empty::<R, u32>(client, splits * k));
    let geom = LaunchGeometry::elementwise(client, splits * k * d);
    // SAFETY: `points` holds `d · n`, `label` `n`, `sums` `splits · k · d`, `counts` `splits · k`
    unsafe {
        cluster_sums_kernel::launch::<f32, R>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ArrayArg::from_raw_parts(points.handle.clone(), d * n),
            ArrayArg::from_raw_parts(label.clone(), n),
            ArrayArg::from_raw_parts(sums.clone(), splits * k * d),
            ArrayArg::from_raw_parts(counts.clone(), splits * k),
            n as u32,
            d as u32,
            k as u32,
            SUM_SPLIT as u32,
            splits as u32,
        );
    }
    let (sums, counts) = (buffer::download_prefix::<R, f32>(client, sums, splits * k * d), buffer::download_prefix::<R, u32>(client, counts, splits * k));
    let mut total = vec![0.0f64; k * d];
    let mut count = vec![0usize; k];
    for s in 0..splits {
        for (t, &v) in total.iter_mut().zip(&sums[s * k * d..(s + 1) * k * d]) {
            *t += v as f64;
        }
        for (c, &v) in count.iter_mut().zip(&counts[s * k..(s + 1) * k]) {
            *c += v as usize;
        }
    }
    (total, count)
}

/// One Lloyd run from `centers`: the final centres, their inertia, and the assignment (on the
/// device).
fn lloyd<R: Runtime>(client: &ComputeClient<R>, points: &DevicePoints, k: usize, mut centers: Vec<f32>, max_iter: usize, tol_abs: f64) -> (Vec<f32>, f64, Assignment) {
    let (n, d) = (points.n, points.d);
    let assignment = Assignment { label: buffer::empty::<R, u32>(client, n), dist: buffer::empty::<R, f32>(client, n) };
    for _ in 0..max_iter.max(1) {
        assign(client, points, &centers, k, &assignment);
        let (sums, counts) = cluster_sums(client, points, &assignment.label, k);
        let mut new_centers = centers.clone();
        // Empty clusters move to the points farthest from their centre (distances read only then)
        let mut far: Option<std::vec::IntoIter<usize>> = None;
        for c in 0..k {
            if counts[c] > 0 {
                for f in 0..d {
                    new_centers[c * d + f] = (sums[c * d + f] / counts[c] as f64) as f32;
                }
            } else {
                let far = far.get_or_insert_with(|| {
                    let dist = buffer::download_prefix::<R, f32>(client, assignment.dist.clone(), n);
                    let mut order: Vec<usize> = (0..n).collect();
                    order.sort_by(|&a, &b| dist[b].total_cmp(&dist[a]));
                    order.into_iter()
                });
                if let Some(p) = far.next() {
                    new_centers[c * d..(c + 1) * d].copy_from_slice(&points.point(client, p));
                }
            }
        }
        let shift: f64 = (0..k).map(|c| sq_dist(&centers[c * d..(c + 1) * d], &new_centers[c * d..(c + 1) * d])).sum();
        centers = new_centers;
        if shift <= tol_abs {
            break;
        }
    }
    // Final assignment to the final centres
    assign(client, points, &centers, k, &assignment);
    let inertia = device_sum(client, &assignment.dist, n);
    (centers, inertia, assignment)
}

/// k-means of host points `x` (`[n, d]` row-major) into `k` clusters (`k ≤ n`), uploaded once.
pub fn kmeans<R: Runtime>(client: &ComputeClient<R>, x: &[f32], n: usize, d: usize, k: usize, options: &KMeansOptions) -> KMeansResult {
    kmeans_points(client, &DevicePoints::upload(client, x, n, d), k, options)
}

/// k-means of device points into `k` clusters (`k ≤ n`). See the module docs.
pub fn kmeans_points<R: Runtime>(client: &ComputeClient<R>, points: &DevicePoints, k: usize, options: &KMeansOptions) -> KMeansResult {
    kmeans_points_with_progress(client, points, k, options, &mut |_, _| {})
}

/// [`kmeans_points`], calling `progress(done, total)` after each of the `n_init` restarts.
pub fn kmeans_points_with_progress<R: Runtime>(
    client: &ComputeClient<R>,
    points: &DevicePoints,
    k: usize,
    options: &KMeansOptions,
    progress: &mut dyn FnMut(u64, u64),
) -> KMeansResult {
    let (n, d) = (points.n, points.d);
    assert!(k >= 1 && k <= n, "kmeans: need 1 ≤ k ≤ n");
    // sklearn: tol scaled by the mean variance of the features (rows of the feature-major points)
    let (mean, std) = (buffer::empty::<R, f32>(client, d), buffer::empty::<R, f32>(client, d));
    reduce::row_mean_std::<R, f32>(client, &points.handle, &mean, &std, d, n);
    let mean_var = buffer::download_prefix::<R, f32>(client, std, d).iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / d.max(1) as f64;
    let tol_abs = options.tol * mean_var;

    let unreached = buffer::upload(client, &vec![f32::MAX; n]);
    let mut rng = Rng(options.seed);
    let mut best: Option<(Vec<f32>, f64, Assignment)> = None;
    let restarts = options.n_init.max(1);
    for restart in 0..restarts {
        let init = kmeans_plus_plus(client, points, k, &unreached, &mut rng);
        let run = lloyd(client, points, k, init, options.max_iter, tol_abs);
        if best.as_ref().is_none_or(|b| run.1 < b.1) {
            best = Some(run);
        }
        progress(restart as u64 + 1, restarts as u64);
    }
    let (centers, inertia, assignment) = best.expect("n_init ≥ 1");
    let labels = buffer::download_prefix::<R, u32>(client, assignment.label, n).into_iter().map(|l| l as usize).collect();
    KMeansResult { centers, labels, inertia }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::compute::{ComputeTarget, ComputeTask};

    // The reference (host, `f64`) the device path is checked against
    fn sq_dist(a: &[f32], b: &[f32]) -> f64 {
        a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum()
    }

    /// Index drawn with probability proportional to `weights` (the last positive one on round-off).
    fn draw_host(weights: &[f64], rng: &mut Rng) -> usize {
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
    fn kmeans_plus_plus_host(x: &[f32], n: usize, d: usize, k: usize, rng: &mut Rng) -> Vec<f32> {
        let trials = 2 + (k as f64).ln() as usize;
        let mut centers = Vec::with_capacity(k * d);
        let first = (rng.next_u64() as usize) % n;
        centers.extend_from_slice(&x[first * d..(first + 1) * d]);
        let mut closest: Vec<f64> = (0..n).map(|i| sq_dist(&x[i * d..(i + 1) * d], &centers[..d])).collect();
        for _ in 1..k {
            let mut best: Option<(usize, f64, Vec<f64>)> = None;
            for _ in 0..trials {
                let c = draw_host(&closest, rng);
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
    fn lloyd_host(x: &[f32], n: usize, d: usize, k: usize, mut centers: Vec<f32>, max_iter: usize, tol_abs: f64) -> KMeansResult {
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
    fn kmeans_host(x: &[f32], n: usize, d: usize, k: usize, options: &KMeansOptions) -> KMeansResult {
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
                let init = kmeans_plus_plus_host(x, n, d, k, &mut rng);
                lloyd_host(x, n, d, k, init, options.max_iter, tol_abs)
            })
            .min_by(|a, b| a.inertia.total_cmp(&b.inertia))
            .expect("n_init ≥ 1")
    }

    fn three_blobs() -> Vec<f32> {
        let centres = [[0.0f32, 0.0], [10.0, 0.0], [0.0, 10.0]];
        let mut x = Vec::new();
        for c in &centres {
            for i in 0..30 {
                let o = (i as f32 * 0.37).sin() * 0.5;
                x.extend_from_slice(&[c[0] + o, c[1] - o]);
            }
        }
        x
    }

    struct Fit(Vec<f32>, usize, usize, usize, KMeansOptions);
    impl ComputeTask for Fit {
        type Output = KMeansResult;
        fn run<R: Runtime>(self, client: ComputeClient<R>) -> Self::Output {
            kmeans(&client, &self.0, self.1, self.2, self.3, &self.4)
        }
    }

    #[test]
    fn separates_three_blobs() {
        let Ok(target) = ComputeTarget::from_env() else { return };
        let r = target.run(Fit(three_blobs(), 90, 2, 3, KMeansOptions { seed: 7, ..Default::default() })).expect("target run");
        for block in 0..3 {
            let l = r.labels[block * 30];
            assert!(r.labels[block * 30..(block + 1) * 30].iter().all(|&v| v == l), "block {block}");
        }
        assert!(r.inertia < 90.0);
    }

    #[test]
    fn device_matches_host_reference() {
        let Ok(target) = ComputeTarget::from_env() else { return };
        // More points than one block of weights, so draws cross blocks
        let (n, d, k) = (3000usize, 6usize, 5usize);
        let mut state = 0x2545_f491u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let x: Vec<f32> = (0..n).flat_map(|i| (0..d).map(|f| ((i % k) * 7 + f) as f32 + next()).collect::<Vec<_>>()).collect();
        let options = KMeansOptions { seed: 3, ..Default::default() };
        let want = kmeans_host(&x, n, d, k, &options);
        let got = target.run(Fit(x, n, d, k, options)).expect("target run");
        assert!((got.inertia - want.inertia).abs() <= 1e-3 * want.inertia, "inertia {} vs {}", got.inertia, want.inertia);
        // Same partition (centre order may differ when an f32 weight tips a draw)
        let mut map = vec![usize::MAX; k];
        for (&g, &w) in got.labels.iter().zip(&want.labels) {
            assert!(map[g] == usize::MAX || map[g] == w, "partitions differ");
            map[g] = w;
        }
    }
}
