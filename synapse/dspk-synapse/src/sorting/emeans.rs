//! e-means: mini-batch k-means whose centres follow an exponential moving average of the batches'
//! per-cluster counts and sums, with the step annealed on a cosine and set, with the batch size, by
//! fitted scaling laws (Bessa et al., "e-means: outperforming k-means (in one-tenth the epochs)",
//! arXiv:2611.09501; the constants and the details below are those of the authors' reference,
//! faiss `contrib/e_means`). Same inputs and result as [`super::kmeans`], to compare them.
//!
//! Per run of `T` epochs (passes over the points in a fresh shuffle each), in batches of `B`:
//! - **Init**: `k` distinct points drawn at random (or k-means++), each with mass `N_j = 2⁻¹⁶`.
//! - **Batch**: assign each point to the nearest current centre; per cluster, its count `n_j` and
//!   the mean `m_j` of its points. With `α` the step and `γ = 1 − α`: `N_j ← γ N_j + α n_j`, and a
//!   cluster with points moves `c_j ← c_j + (α n_j / N_j)(m_j − c_j)` (the same as the EMA of sums
//!   over counts); one without keeps its position while its mass decays.
//! - **Step**: `α_t = α₀ (0.1 + 0.9 · ½(1 + cos(π t / M)))` over the `M` batches of the run.
//! - **Revival**, every 5 batches (not for `d ≤ 2`): clusters with `N_j < 0.1 · mean(N)` take the
//!   place of a donor drawn without replacement with probability ∝ `N` (Gumbel keys); donor and
//!   child split the donor's mass and sit at `c ⊙ (1 ∓ 10⁻⁴ s)`, `s = (+1, −1, +1, …)`.
//! - **Close**: each centre becomes the mean of the points assigned to it during the last epoch
//!   (centres without any keep their position).
//! - **Laws** (`n` points, `k` clusters): `B = C_B k^0.1347 n^0.8765 T^0.8133` rounded to a power of
//!   two (`C_B = 0.035134`, "throughput", for `d > 16`; an eighth of it, "quality", for `d ≤ 16`),
//!   capped at `n/2`; `α₀ = 0.8244 k^-0.4617 B^0.9003 n^-0.4498 T^-0.1057`, within `[10⁻⁴, 1]`.
//!
//! Unlike the reference, the result's labels and inertia are a final assignment to the closed
//! centres (as [`super::kmeans`] reports them), so the two can be compared on the same terms.
//!
//! Data stays on the device ([`DevicePoints`], uploaded once): the epoch's shuffle is a keyed
//! permutation computed per point (only four keys cross per epoch); the batch assignments and sums,
//! the EMA update of the centres and masses, the last epoch's sums and the closing means are
//! kernels on device buffers. A batch launches four kernels and transfers nothing but scalar
//! arguments. Revival reads the `k` masses every fifth batch, and the centres (written back with
//! the masses) only when a cluster is revived. The result reads the labels, the centres and one
//! sum (the inertia).

use cubecl::prelude::*;
use cubecl::server::Handle;
use dspk_core::compute::LaunchGeometry;
use dspk_ops::core::buffer;

use super::kernels::emeans::{batch_nearest_kernel, batch_sums_kernel, close_centres_kernel, ema_update_kernel, gather_rows_kernel, gather_shuffled_kernel};
use super::kernels::points::nearest_centre_kernel;
use super::kmeans::{kmeans_plus_plus, KMeansResult, Rng};
use super::points::{device_sum, DevicePoints};

/// The reference's default epoch budget `T`.
pub const DEFAULT_EPOCHS: f64 = 5.0;
/// Initial mass of every centre (the paper: 10⁻¹⁰; the reference: 2⁻¹⁶).
const INIT_MASS: f64 = 1.0 / 65536.0;
/// The step's floor, as a fraction of `α₀`, at the end of the cosine.
const ALPHA_FLOOR: f64 = 0.1;
/// Revival: every this many batches, clusters below `REVIVAL_THRESHOLD · mean mass`.
const REVIVAL_EVERY: usize = 5;
const REVIVAL_THRESHOLD: f64 = 0.1;
/// No revival at or below this dimension (it cost held-out error on 2-D data in the reference).
const NO_REVIVAL_MAX_D: usize = 2;
/// Relative coordinate offset of a split's donor and child.
const SPLIT_EPSILON: f64 = 1e-4;
/// The quality batch law at or below this dimension, the throughput law above it.
const QUALITY_MAX_D: usize = 16;
/// Batch law `(C_B, b_k, b_n, b_T)`; quality is an eighth of throughput's coefficient.
const THROUGHPUT_LAW: (f64, f64, f64, f64) = (0.035134, 0.134749, 0.876473, 0.813344);
const QUALITY_LAW: (f64, f64, f64, f64) = (0.00439175, 0.134749, 0.876473, 0.813344);
/// Step law `(C_α, a_k, a_B, a_n, a_T)`.
const STEP_LAW: (f64, f64, f64, f64, f64) = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057);
/// Batch points each unit of the batch sums adds before the host adds the splits in `f64`.
const SUM_SPLIT: usize = 4096;

/// Which batch law sets the batch size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BatchLaw {
    /// Quality for `d ≤ 16`, throughput above (the reference's default).
    #[default]
    Auto,
    /// Larger batches: fewer updates per epoch.
    Throughput,
    /// Batches eight times smaller: more updates per epoch.
    Quality,
}

/// How the centres start.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EMeansInit {
    /// `k` distinct points drawn at random (the paper's choice).
    #[default]
    Random,
    /// Greedy k-means++, as [`super::kmeans`] (slower; slightly better in the paper).
    PlusPlus,
}

/// Options of [`cluster_emeans`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EMeansOptions {
    /// Epoch budget `T` (fractions run a prefix of one more shuffle).
    pub epochs: f64,
    /// Batch size `B` (`None`: the batch law).
    pub batch_size: Option<usize>,
    /// Initial step `α₀` (`None`: the step law at `B`).
    pub alpha0: Option<f64>,
    pub batch_law: BatchLaw,
    /// Revival threshold `τ` (`None`: 0.1, or none for `d ≤ 2`; `Some(0.0)`: never).
    pub revival: Option<f64>,
    pub init: EMeansInit,
    pub seed: u64,
}

impl Default for EMeansOptions {
    fn default() -> Self {
        Self { epochs: DEFAULT_EPOCHS, batch_size: None, alpha0: None, batch_law: BatchLaw::Auto, revival: None, init: EMeansInit::Random, seed: 0 }
    }
}

/// The batch size and initial step a run uses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EMeansRecipe {
    pub batch_size: usize,
    pub alpha0: f64,
    /// The batches of the whole run (`M`).
    pub total_batches: usize,
}

/// Result of [`cluster_emeans`]: as [`KMeansResult`], with the recipe used and the splits made.
#[derive(Debug, Clone, PartialEq)]
pub struct EMeansResult {
    pub kmeans: KMeansResult,
    pub recipe: EMeansRecipe,
    /// Dead clusters revived (donor splits).
    pub splits: usize,
}

/// `value` rounded to the nearest power of two (at least 1).
fn round_power_of_two(value: f64) -> usize {
    if value <= 1.0 {
        return 1;
    }
    let low = 2f64.powi(value.log2().floor() as i32);
    let high = 2.0 * low;
    (if value / low < high / value { low } else { high }) as usize
}

/// Rows of each epoch: whole passes, a fractional prefix (rounded half to even), then a whole
/// last pass whose assignments close the run (only the prefix when below one pass).
fn epoch_rows(n: usize, epochs: f64) -> Vec<usize> {
    let whole = epochs.floor() as usize;
    let remainder = ((epochs - whole as f64) * n as f64).round_ties_even() as usize;
    if whole == 0 {
        return vec![remainder.max(1).min(n)];
    }
    let mut rows = vec![n; whole - 1];
    if remainder > 0 {
        rows.push(remainder);
    }
    rows.push(n);
    rows
}

/// The recipe of a run on `n` points of `d` features into `k` clusters.
pub fn emeans_recipe(n: usize, d: usize, k: usize, options: &EMeansOptions) -> EMeansRecipe {
    let t = options.epochs.max(f64::MIN_POSITIVE);
    let law = match options.batch_law {
        BatchLaw::Auto if d <= QUALITY_MAX_D => BatchLaw::Quality,
        BatchLaw::Auto => BatchLaw::Throughput,
        other => other,
    };
    let batch_size = match options.batch_size {
        Some(b) => b.clamp(1, n),
        None => {
            let (c, bk, bn, bt) = if law == BatchLaw::Throughput { THROUGHPUT_LAW } else { QUALITY_LAW };
            let raw = c * (k as f64).powf(bk) * (n as f64).powf(bn) * t.powf(bt);
            let b = round_power_of_two(raw);
            // The two fitted recipes cap differently
            let b = if law == BatchLaw::Throughput { b.min(round_power_of_two(n as f64 / 2.0)) } else { b.min(n / 2) };
            b.clamp(1, n)
        }
    };
    let alpha0 = options.alpha0.unwrap_or_else(|| {
        let (c, ak, ab, an, at) = STEP_LAW;
        (c * (k as f64).powf(ak) * (batch_size as f64).powf(ab) * (n as f64).powf(an) * t.powf(at)).clamp(1e-4, 1.0)
    });
    let total_batches = epoch_rows(n, options.epochs).iter().map(|r| r.div_ceil(batch_size)).sum();
    EMeansRecipe { batch_size, alpha0, total_batches }
}

/// The step of batch `t` of `total`: the cosine from `α₀` down to `0.1 α₀`.
fn scheduled_alpha(alpha0: f64, t: usize, total: usize) -> f64 {
    let fraction = t.min(total) as f64 / total.max(1) as f64;
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * fraction).cos());
    alpha0 * (ALPHA_FLOOR + (1.0 - ALPHA_FLOOR) * cosine)
}

/// Dead clusters to revive and their donors, from the masses: the clusters below `τ · mean mass`
/// take the place of donors drawn without replacement ∝ mass (Gumbel keys on one uniform per
/// cluster, all drawn as the reference does). Pairs `(donor, revived)`.
fn plan_splits(mass: &[f64], tau: f64, rng: &mut Rng) -> Vec<(usize, usize)> {
    let k = mass.len();
    if tau <= 0.0 {
        return Vec::new();
    }
    let threshold = tau * mass.iter().sum::<f64>() / k as f64;
    let unused: Vec<usize> = (0..k).filter(|&c| mass[c] < threshold).collect();
    let uniform: Vec<f64> = (0..k).map(|_| rng.uniform().max(f64::MIN_POSITIVE)).collect();
    let mut donors: Vec<(f64, usize)> = (0..k).filter(|&c| mass[c] >= threshold && mass[c] > 0.0).map(|c| (mass[c].ln() - (-uniform[c].ln()).ln(), c)).collect();
    donors.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.cmp(&a.1)));
    donors.into_iter().map(|(_, c)| c).zip(unused).collect()
}

/// Splits each donor of `pairs` with its revived cluster: the two share the donor's mass and sit
/// at `c ⊙ (1 ∓ 10⁻⁴ s)`, `s = (+1, −1, …)`; a zero donor moves the child's first coordinate by one
/// `f32` step so the two differ. `centres` is `[k, d]`.
fn apply_splits(centres: &mut [f64], mass: &mut [f64], d: usize, pairs: &[(usize, usize)]) {
    for &(source, target) in pairs {
        let half = mass[source] / 2.0;
        for f in 0..d {
            let s = if f % 2 == 0 { 1.0 } else { -1.0 };
            let c = centres[source * d + f];
            centres[source * d + f] = c * (1.0 - SPLIT_EPSILON * s);
            centres[target * d + f] = c * (1.0 + SPLIT_EPSILON * s);
        }
        if d > 0 && (0..d).all(|f| centres[source * d + f] == centres[target * d + f]) {
            let v = centres[target * d] as f32;
            centres[target * d] = f32::from_bits(if v >= 0.0 { v.to_bits() + 1 } else { v.to_bits() - 1 }) as f64;
        }
        mass[source] = half;
        mass[target] = half;
    }
}

/// The epoch's shuffle: a Feistel network on `2 · half` bits with four round keys.
#[derive(Clone, Copy)]
struct Shuffle {
    half: u32,
    mask: u32,
    keys: [u32; 4],
}

impl Shuffle {
    /// The shuffle of `0..n` for one epoch (four keys from `rng`).
    fn new(n: usize, rng: &mut Rng) -> Self {
        let bits = usize::BITS - (n.max(4) - 1).leading_zeros();
        let half = bits.div_ceil(2);
        let keys = [rng.next_u64() as u32, rng.next_u64() as u32, rng.next_u64() as u32, rng.next_u64() as u32];
        Self { half, mask: (1u32 << half) - 1, keys }
    }

    /// Position `j` of the shuffle of `0..n` (the kernels' `shuffle_index`, bit for bit).
    #[cfg_attr(not(test), allow(dead_code))]
    fn index(&self, j: u32, n: u32) -> u32 {
        let mix = |x: u32, key: u32| {
            let mut h = x ^ key;
            h = h.wrapping_mul(0x9E37_79B1);
            h ^= h >> 15;
            h = h.wrapping_mul(0x85EB_CA6B);
            h ^ (h >> 13)
        };
        let mut x = j;
        loop {
            let (mut l, mut r) = (x >> self.half, x & self.mask);
            for key in self.keys {
                let t = l ^ (mix(r, key) & self.mask);
                l = r;
                r = t;
            }
            x = (l << self.half) | r;
            if x < n {
                return x;
            }
        }
    }
}

/// `k` distinct indices of `0..n`, drawn from `rng` (a partial Fisher-Yates).
fn sample_distinct(n: usize, k: usize, rng: &mut Rng) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + (rng.next_u64() % (n - i) as u64) as usize;
        idx.swap(i, j);
    }
    idx.truncate(k);
    idx
}

/// The run's device state: centres `[k, d]`, masses, the last epoch's sums and counts, the
/// labels, and the batch's split sums (allocated once).
struct DeviceRun {
    centres: Handle,
    mass: Handle,
    close_sums: Handle,
    close_counts: Handle,
    label: Handle,
    sums: Handle,
    counts: Handle,
    k: usize,
    d: usize,
}

impl DeviceRun {
    /// The masses on the host (`k` values).
    fn masses(&self, client: &Client) -> Vec<f64> {
        buffer::download_prefix::<f32>(client, self.mass.clone(), self.k).into_iter().map(f64::from).collect()
    }

    /// Revives dead clusters: reads the masses; only when a cluster is revived, the centres too,
    /// and writes both back. The number revived.
    fn revive(&mut self, client: &Client, tau: f64, rng: &mut Rng) -> usize {
        if tau <= 0.0 {
            return 0;
        }
        let mut mass = self.masses(client);
        let pairs = plan_splits(&mass, tau, rng);
        if pairs.is_empty() {
            return 0;
        }
        let (k, d) = (self.k, self.d);
        let mut centres: Vec<f64> = buffer::download_prefix::<f32>(client, self.centres.clone(), k * d).into_iter().map(f64::from).collect();
        apply_splits(&mut centres, &mut mass, d, &pairs);
        self.centres = buffer::upload(client, &centres.iter().map(|&v| v as f32).collect::<Vec<_>>());
        self.mass = buffer::upload(client, &mass.iter().map(|&v| v as f32).collect::<Vec<_>>());
        pairs.len()
    }

    /// One batch, all on the device: assign `[start, start + len)` of `shuffled`, sum per cluster,
    /// update the centres and masses with step `alpha` (and the closing sums when `close`).
    fn batch(&self, client: &Client, shuffled: &DevicePoints, start: usize, len: usize, alpha: f64, close: bool) {
        let (n, d, k) = (shuffled.n, self.d, self.k);
        let geom = LaunchGeometry::elementwise(client, len);
        // SAFETY: `shuffled` holds `d · n`, `centres` `k · d`, `label` `n` values; `start + len ≤ n`
        unsafe {
            batch_nearest_kernel::launch::<f32>(
                client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(shuffled.handle.clone(), d * n),
                BufferArg::from_raw_parts(self.centres.clone(), k * d),
                BufferArg::from_raw_parts(self.label.clone(), n),
                n as u32,
                start as u32,
                len as u32,
                d as u32,
                k as u32,
            );
        }
        let splits = len.div_ceil(SUM_SPLIT).max(1);
        let geom = LaunchGeometry::elementwise(client, splits * k * d);
        // SAFETY: `shuffled` holds `d · n`, `label` `n`; `sums` and `counts` were allocated for at
        // least `splits` splits (the most a batch of the run has)
        unsafe {
            batch_sums_kernel::launch::<f32>(
                client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(shuffled.handle.clone(), d * n),
                BufferArg::from_raw_parts(self.label.clone(), n),
                BufferArg::from_raw_parts(self.sums.clone(), splits * k * d),
                BufferArg::from_raw_parts(self.counts.clone(), splits * k),
                n as u32,
                start as u32,
                len as u32,
                d as u32,
                k as u32,
                SUM_SPLIT as u32,
                splits as u32,
            );
        }
        let geom = LaunchGeometry::elementwise(client, k);
        // SAFETY: `sums` holds `splits · k · d`, `counts` `splits · k`; `centres`, `close_sums`
        // `k · d`; `mass`, `close_counts` `k` values
        unsafe {
            ema_update_kernel::launch::<f32>(
                client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(self.sums.clone(), splits * k * d),
                BufferArg::from_raw_parts(self.counts.clone(), splits * k),
                BufferArg::from_raw_parts(self.centres.clone(), k * d),
                BufferArg::from_raw_parts(self.mass.clone(), k),
                BufferArg::from_raw_parts(self.close_sums.clone(), k * d),
                BufferArg::from_raw_parts(self.close_counts.clone(), k),
                k as u32,
                d as u32,
                splits as u32,
                alpha as f32,
                u32::from(close),
            );
        }
    }
}

/// The first `rows` points of the epoch's shuffle of `points`, gathered on the device.
fn gather_shuffled(client: &Client, points: &DevicePoints, rows: usize, shuffle: &Shuffle) -> DevicePoints {
    let (n, d) = (points.n, points.d);
    let out = buffer::empty::<f32>(client, d * rows);
    let geom = LaunchGeometry::elementwise(client, rows);
    let [k0, k1, k2, k3] = shuffle.keys;
    // SAFETY: `points` holds `d · n`, `out` `d · rows` values; the shuffle maps `0..rows` into `0..n`
    unsafe {
        gather_shuffled_kernel::launch::<f32>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(points.handle.clone(), d * n),
            BufferArg::from_raw_parts(out.clone(), d * rows),
            n as u32,
            rows as u32,
            d as u32,
            shuffle.half,
            shuffle.mask,
            k0,
            k1,
            k2,
            k3,
        );
    }
    DevicePoints { handle: out, n: rows, d }
}

/// e-means of host points `x` (`[n, d]` row-major) into `k` clusters (`k ≤ n`), uploaded once.
pub fn cluster_emeans(client: &Client, x: &[f32], n: usize, d: usize, k: usize, options: &EMeansOptions) -> EMeansResult {
    execute_emeans(client, &DevicePoints::upload(client, x, n, d), k, options)
}

/// e-means of device points into `k` clusters (`k ≤ n`). See the module docs.
pub fn execute_emeans(client: &Client, points: &DevicePoints, k: usize, options: &EMeansOptions) -> EMeansResult {
    execute_emeans_with_progress(client, points, k, options, &mut |_, _| {})
}

/// [`execute_emeans`], calling `progress(done, total)` after each epoch.
pub fn execute_emeans_with_progress(client: &Client, points: &DevicePoints, k: usize, options: &EMeansOptions, progress: &mut dyn FnMut(u64, u64)) -> EMeansResult {
    let (n, d) = (points.n, points.d);
    assert!(k >= 1 && k <= n, "emeans: need 1 ≤ k ≤ n");
    let recipe = emeans_recipe(n, d, k, options);
    let mut rng = Rng(options.seed);
    // Starting centres, on the device
    let centres = match options.init {
        EMeansInit::Random => {
            let index: Vec<u32> = sample_distinct(n, k, &mut rng).into_iter().map(|i| i as u32).collect();
            let centres = buffer::empty::<f32>(client, k * d);
            let geom = LaunchGeometry::elementwise(client, k * d);
            // SAFETY: `points` holds `d · n`, `index` `k` values (all < n), `centres` `k · d`
            unsafe {
                gather_rows_kernel::launch::<f32>(
                    client,
                    geom.cube_count,
                    geom.cube_dim,
                    BufferArg::from_raw_parts(points.handle.clone(), d * n),
                    BufferArg::from_raw_parts(buffer::upload(client, &index), k),
                    BufferArg::from_raw_parts(centres.clone(), k * d),
                    n as u32,
                    k as u32,
                    d as u32,
                );
            }
            centres
        }
        EMeansInit::PlusPlus => {
            let unreached = buffer::upload(client, &vec![f32::MAX; n]);
            buffer::upload(client, &kmeans_plus_plus(client, points, k, &unreached, &mut rng))
        }
    };
    let epochs = epoch_rows(n, options.epochs);
    let max_splits = recipe.batch_size.min(n).div_ceil(SUM_SPLIT).max(1);
    let mut run = DeviceRun {
        centres,
        mass: buffer::upload(client, &vec![INIT_MASS as f32; k]),
        close_sums: buffer::upload(client, &vec![0.0f32; k * d]),
        close_counts: buffer::upload(client, &vec![0u32; k]),
        label: buffer::empty::<u32>(client, n),
        sums: buffer::empty::<f32>(client, max_splits * k * d),
        counts: buffer::empty::<u32>(client, max_splits * k),
        k,
        d,
    };
    let tau = options.revival.unwrap_or(if d <= NO_REVIVAL_MAX_D { 0.0 } else { REVIVAL_THRESHOLD });
    // Revival draws its own stream (independent of the shuffles)
    let mut split_rng = Rng(options.seed ^ 0x5DEE_CE66_D1CE_4E5B);
    let mut splits = 0;
    let mut t = 0usize;
    for (e, &rows) in epochs.iter().enumerate() {
        let shuffle = Shuffle::new(n, &mut rng);
        let shuffled = gather_shuffled(client, points, rows, &shuffle);
        let last = e + 1 == epochs.len();
        let mut start = 0;
        while start < rows {
            let len = recipe.batch_size.min(rows - start);
            run.batch(client, &shuffled, start, len, scheduled_alpha(recipe.alpha0, t, recipe.total_batches), last);
            t += 1;
            if t % REVIVAL_EVERY == 0 {
                splits += run.revive(client, tau, &mut split_rng);
            }
            start += len;
        }
        progress(e as u64 + 1, epochs.len() as u64);
    }
    // Close, then the final assignment to the closed centres (labels and inertia as k-means)
    let geom = LaunchGeometry::elementwise(client, k * d);
    // SAFETY: `centres`, `close_sums` hold `k · d`, `close_counts` `k` values
    unsafe {
        close_centres_kernel::launch::<f32>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(run.centres.clone(), k * d),
            BufferArg::from_raw_parts(run.close_sums.clone(), k * d),
            BufferArg::from_raw_parts(run.close_counts.clone(), k),
            k as u32,
            d as u32,
        );
    }
    let dist = buffer::empty::<f32>(client, n);
    let geom = LaunchGeometry::elementwise(client, n);
    // SAFETY: `points` holds `d · n`, `centres` `k · d`, `label` and `dist` `n` values
    unsafe {
        nearest_centre_kernel::launch::<f32>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(points.handle.clone(), d * n),
            BufferArg::from_raw_parts(run.centres.clone(), k * d),
            BufferArg::from_raw_parts(run.label.clone(), n),
            BufferArg::from_raw_parts(dist.clone(), n),
            n as u32,
            d as u32,
            k as u32,
        );
    }
    let inertia = device_sum(client, &dist, n);
    let labels = buffer::download_prefix::<u32>(client, run.label, n).into_iter().map(|l| l as usize).collect();
    let centers = buffer::download_prefix::<f32>(client, run.centres, k * d);
    EMeansResult { kmeans: KMeansResult { centers, labels, inertia }, recipe, splits }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sorting::kmeans::{cluster_kmeans, KMeansOptions};
    use dspk_core::compute::{ComputeTarget, ComputeTask};

    fn sq_dist(a: &[f32], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (*x as f64 - y).powi(2)).sum()
    }

    /// e-means on the host in `f64`: the same shuffles, batches, updates and revivals as the
    /// device (which keeps its state in `f32`).
    fn emeans_host(x: &[f32], n: usize, d: usize, k: usize, options: &EMeansOptions) -> (Vec<f32>, Vec<usize>, f64) {
        assert!(matches!(options.init, EMeansInit::Random), "the host reference starts at random points");
        let recipe = emeans_recipe(n, d, k, options);
        let mut rng = Rng(options.seed);
        let mut centres: Vec<f64> = sample_distinct(n, k, &mut rng).into_iter().flat_map(|i| x[i * d..(i + 1) * d].iter().map(|&v| v as f64).collect::<Vec<_>>()).collect();
        let mut mass = vec![INIT_MASS; k];
        let tau = options.revival.unwrap_or(if d <= NO_REVIVAL_MAX_D { 0.0 } else { REVIVAL_THRESHOLD });
        let mut split_rng = Rng(options.seed ^ 0x5DEE_CE66_D1CE_4E5B);
        let epochs = epoch_rows(n, options.epochs);
        let mut t = 0;
        let (mut close_sums, mut close_counts) = (vec![0.0f64; k * d], vec![0usize; k]);
        for (e, &rows) in epochs.iter().enumerate() {
            let shuffle = Shuffle::new(n, &mut rng);
            let last = e + 1 == epochs.len();
            let mut start = 0;
            while start < rows {
                let len = recipe.batch_size.min(rows - start);
                // The device compares f32 distances to f32 centres
                let c32: Vec<f32> = centres.iter().map(|&v| v as f32).collect();
                let (mut sums, mut counts) = (vec![0.0f64; k * d], vec![0usize; k]);
                for j in start..start + len {
                    let i = shuffle.index(j as u32, n as u32) as usize;
                    let xi = &x[i * d..(i + 1) * d];
                    let c = (0..k)
                        .map(|c| (c, xi.iter().zip(&c32[c * d..(c + 1) * d]).map(|(a, b)| (a - b) * (a - b)).sum::<f32>()))
                        .fold((0, f32::MAX), |b, c| if c.1 < b.1 { c } else { b })
                        .0;
                    counts[c] += 1;
                    for f in 0..d {
                        sums[c * d + f] += xi[f] as f64;
                    }
                }
                let alpha = scheduled_alpha(recipe.alpha0, t, recipe.total_batches);
                for c in 0..k {
                    let n_c = counts[c] as f64;
                    let m = (1.0 - alpha) * mass[c] + alpha * n_c;
                    if counts[c] > 0 {
                        for f in 0..d {
                            centres[c * d + f] += alpha * n_c / m * (sums[c * d + f] / n_c - centres[c * d + f]);
                        }
                    }
                    mass[c] = m;
                }
                if last {
                    close_sums.iter_mut().zip(&sums).for_each(|(a, b)| *a += b);
                    close_counts.iter_mut().zip(&counts).for_each(|(a, b)| *a += b);
                }
                t += 1;
                if t % REVIVAL_EVERY == 0 {
                    // The device reads its masses as f32
                    let m32: Vec<f64> = mass.iter().map(|&v| v as f32 as f64).collect();
                    let pairs = plan_splits(&m32, tau, &mut split_rng);
                    apply_splits(&mut centres, &mut mass, d, &pairs);
                }
                start += len;
            }
        }
        for c in 0..k {
            if close_counts[c] > 0 {
                for f in 0..d {
                    centres[c * d + f] = close_sums[c * d + f] / close_counts[c] as f64;
                }
            }
        }
        let mut labels = vec![0; n];
        let mut inertia = 0.0;
        for i in 0..n {
            let xi = &x[i * d..(i + 1) * d];
            let (c, dist) = (0..k).map(|c| (c, sq_dist(xi, &centres[c * d..(c + 1) * d]))).fold((0, f64::INFINITY), |b, c| if c.1 < b.1 { c } else { b });
            labels[i] = c;
            inertia += dist;
        }
        (centres.iter().map(|&v| v as f32).collect(), labels, inertia)
    }

    /// The epoch's shuffle is a permutation of `0..n`, for sizes on and off a power of two.
    #[test]
    fn the_shuffle_is_a_permutation() {
        for n in [1usize, 2, 3, 7, 64, 1000, 4097] {
            let s = Shuffle::new(n, &mut Rng(n as u64));
            let mut seen = vec![false; n];
            for j in 0..n {
                let i = s.index(j as u32, n as u32) as usize;
                assert!(i < n && !seen[i], "n {n}: {j} → {i}");
                seen[i] = true;
            }
        }
    }

    struct Gather(Vec<f32>, usize, usize, usize);
    impl ComputeTask for Gather {
        type Output = (Vec<f32>, Vec<u32>);
        fn run(self, client: Client) -> Self::Output {
            let points = DevicePoints::upload(&client, &self.0, self.1, self.2);
            let shuffle = Shuffle::new(self.1, &mut Rng(9));
            let out = gather_shuffled(&client, &points, self.3, &shuffle);
            let want = (0..self.3 as u32).map(|j| shuffle.index(j, self.1 as u32)).collect();
            (buffer::download_prefix::<f32>(&client, out.handle, self.2 * self.3), want)
        }
    }

    /// The device's shuffle is the host's, bit for bit (a prefix, as a fractional epoch takes).
    #[test]
    fn the_device_shuffles_as_the_host() {
        let Ok(target) = ComputeTarget::from_env() else { return };
        let (n, d, m) = (1000usize, 3usize, 700usize);
        let x: Vec<f32> = (0..n * d).map(|v| v as f32).collect();
        let (got, index) = target.run(Gather(x.clone(), n, d, m)).expect("target run");
        for (j, &i) in index.iter().enumerate() {
            for f in 0..d {
                assert_eq!(got[f * m + j], x[i as usize * d + f], "position {j}, feature {f}");
            }
        }
    }

    /// `n` points around `k` well-apart centres in `d` dimensions (seeded).
    fn blobs(n: usize, d: usize, k: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        (0..n).flat_map(|i| (0..d).map(|f| ((i % k) * 7 + f) as f32 + next()).collect::<Vec<_>>()).collect()
    }

    struct Fit(Vec<f32>, usize, usize, usize, EMeansOptions);
    impl ComputeTask for Fit {
        type Output = EMeansResult;
        fn run(self, client: Client) -> Self::Output {
            cluster_emeans(&client, &self.0, self.1, self.2, self.3, &self.4)
        }
    }

    struct Lloyd(Vec<f32>, usize, usize, usize);
    impl ComputeTask for Lloyd {
        type Output = KMeansResult;
        fn run(self, client: Client) -> Self::Output {
            cluster_kmeans(&client, &self.0, self.1, self.2, self.3, &KMeansOptions { seed: 1, ..Default::default() })
        }
    }

    /// The laws as the reference computes them (values from its Python, `n = 100000`).
    #[test]
    fn the_recipe_follows_the_reference_laws() {
        let throughput = emeans_recipe(100_000, 128, 1024, &EMeansOptions::default());
        assert_eq!(throughput.batch_size, 8192);
        assert!((throughput.alpha0 - 0.532876).abs() < 1e-5, "alpha0 {}", throughput.alpha0);
        let quality = emeans_recipe(100_000, 8, 1024, &EMeansOptions::default());
        assert_eq!(quality.batch_size, 1024);
        assert!((quality.alpha0 - 0.081955).abs() < 1e-5, "alpha0 {}", quality.alpha0);
        // Whole epochs, then a fraction runs a prefix of one more
        assert_eq!(epoch_rows(1000, 2.0), vec![1000, 1000]);
        assert_eq!(epoch_rows(1000, 1.5), vec![500, 1000]);
        assert_eq!(epoch_rows(1000, 0.25), vec![250]);
        // The step falls from α₀ to a tenth of it
        assert!((scheduled_alpha(0.5, 0, 100) - 0.5).abs() < 1e-12);
        assert!((scheduled_alpha(0.5, 100, 100) - 0.05).abs() < 1e-12);
    }

    #[test]
    fn device_matches_host_reference() {
        let Ok(target) = ComputeTarget::from_env() else { return };
        // Several batches per epoch and revivals (d > 2)
        let (n, d, k) = (5000usize, 6usize, 8usize);
        let x = blobs(n, d, k, 0x2545_f491);
        let options = EMeansOptions { epochs: 3.0, batch_size: Some(512), seed: 5, ..Default::default() };
        let (_, want_labels, want_inertia) = emeans_host(&x, n, d, k, &options);
        let got = target.run(Fit(x, n, d, k, options)).expect("target run");
        assert!((got.kmeans.inertia - want_inertia).abs() <= 1e-3 * want_inertia, "inertia {} vs {}", got.kmeans.inertia, want_inertia);
        let mut map = vec![usize::MAX; k];
        for (&g, &w) in got.kmeans.labels.iter().zip(&want_labels) {
            assert!(map[g] == usize::MAX || map[g] == w, "partitions differ");
            map[g] = w;
        }
    }

    /// In the regime of the paper (many clusters in unstructured data), 3 epochs of e-means come
    /// within 2 % of Lloyd's inertia (k-means++, 10 restarts). The reference implementation on the
    /// same points: e-means 17,168–17,185 at 3 epochs, Lloyd at 25 iterations 17,265.
    #[test]
    fn many_clusters_as_well_as_lloyd() {
        let Ok(target) = ComputeTarget::from_env() else { return };
        let (n, d, k) = (20_000usize, 16usize, 64usize);
        let mut state = 0x2545_f491u32;
        let x: Vec<f32> = (0..n * d)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect();
        let e = target.run(Fit(x.clone(), n, d, k, EMeansOptions { epochs: 3.0, seed: 1, ..Default::default() })).expect("target run");
        let l = target.run(Lloyd(x, n, d, k)).expect("target run");
        assert!(e.kmeans.inertia <= 1.02 * l.inertia, "e-means {} vs Lloyd {}", e.kmeans.inertia, l.inertia);
    }

    /// Few, well-separated clusters: from random starts e-means can merge two blobs (the reference
    /// does the same on these points: inertia 162,158–318,882 against Lloyd's 5,329), so a small
    /// `k` starts from k-means++, which finds every blob.
    #[test]
    fn few_blobs_from_kmeans_plus_plus() {
        let Ok(target) = ComputeTarget::from_env() else { return };
        let (n, d, k) = (8000usize, 8usize, 10usize);
        let x = blobs(n, d, k, 0x9e37_79b9);
        let e = target.run(Fit(x.clone(), n, d, k, EMeansOptions { epochs: 3.0, init: EMeansInit::PlusPlus, seed: 2, ..Default::default() })).expect("target run");
        let l = target.run(Lloyd(x, n, d, k)).expect("target run");
        assert!(e.kmeans.inertia <= 1.02 * l.inertia, "e-means {} vs Lloyd {}", e.kmeans.inertia, l.inertia);
        for block in 0..k {
            let first = e.kmeans.labels[block];
            assert!((0..n).filter(|i| i % k == block).all(|i| e.kmeans.labels[i] == first), "blob {block} split");
        }
    }

    struct Time(Vec<f32>, usize, usize, usize, bool);
    impl ComputeTask for Time {
        type Output = (f64, f64);
        fn run(self, client: Client) -> Self::Output {
            let points = DevicePoints::upload(&client, &self.0, self.1, self.2);
            let t0 = std::time::Instant::now();
            let inertia = if self.4 {
                execute_emeans(&client, &points, self.3, &EMeansOptions { epochs: 3.0, init: EMeansInit::PlusPlus, seed: 1, ..Default::default() }).kmeans.inertia
            } else {
                crate::sorting::kmeans::execute_kmeans(&client, &points, self.3, &KMeansOptions { seed: 1, ..Default::default() }).inertia
            };
            (t0.elapsed().as_secs_f64(), inertia)
        }
    }

    /// Times k-means (k-means++, 10 restarts) against e-means (k-means++, 3 epochs) on the same
    /// device points, at a kilosort-like small `k` and a larger one. Run by hand:
    /// `cargo test -p dspk-synapse --release --lib emeans_against_kmeans_timings -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn emeans_against_kmeans_timings() {
        let Ok(target) = ComputeTarget::from_env() else { return };
        for (n, d, k) in [(50_000usize, 6usize, 6usize), (200_000, 16, 64), (500_000, 32, 256)] {
            let mut state = 0x2545_f491u32;
            let x: Vec<f32> = (0..n * d)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
                })
                .collect();
            // Warm the kernels (compilation and tuning out of the timings)
            let _ = target.run(Time(x[..2000 * d].to_vec(), 2000, d, k.min(20), true));
            let _ = target.run(Time(x[..2000 * d].to_vec(), 2000, d, k.min(20), false));
            let (te, ie) = target.run(Time(x.clone(), n, d, k, true)).expect("target run");
            let (tk, ik) = target.run(Time(x, n, d, k, false)).expect("target run");
            println!("n {n} d {d} k {k}: e-means {te:.3} s (inertia {ie:.1}) · k-means {tk:.3} s (inertia {ik:.1}) · {:.1}× faster, inertia {:+.2} %", tk / te, 100.0 * (ie - ik) / ik);
        }
    }
}
