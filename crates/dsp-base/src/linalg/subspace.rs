//! The leading `k` principal components of `L` rows of `D` features, streamed to the device in
//! batches ([`RowSource`]): PCA of more rows than fit in device memory at once (MountainSort 5's
//! snippets, its subdivision features).
//!
//! The rows are centred on their mean (one pass). Then, after scikit-learn's solver choice
//! (MountainSort 5 `pca_solver.py`):
//!
//! - **`D ≤ exact_cap`** (8000): the covariance `C = Σ (x − μ)(x − μ)ᵀ` is formed on the device
//!   (one product per batch of at most [`GRAM_ROWS`] rows, summed). Its leading eigenvectors are
//!   exact: all of them by Jacobi ([`fn@super::symmetric_eigen`]) when `k + oversamples ≥ D / 2`,
//!   otherwise by subspace iteration on `C` until the leading values change by less than
//!   `tolerance` (relative), then Rayleigh–Ritz.
//! - **`D > exact_cap`**: `C` is not formed (`D²` values); subspace iteration applies `C` as
//!   `Σ_b X_bᵀ (X_b Q)` over the batches, for scikit-learn's randomized count of iterations (7
//!   when `k < 0.1 · min(L, D)`, else 4) from a seeded Gaussian start of `k + oversamples` columns,
//!   then Rayleigh–Ritz. Approximate, as scikit-learn's randomized solver; the random stream
//!   differs from NumPy's, so the components are not bitwise scikit-learn's.
//!
//! Orthonormalisation is by the eigendecomposition of the Gram matrix `YᵀY` (`k + oversamples`
//! square; on the host up to [`HOST_EIGEN_MAX`], else on the device): directions whose singular
//! value is below `√ε` of the largest are set to zero instead of being amplified (rank-deficient
//! data, e.g. masked snippets). `k` is capped at
//! `min(k, L, D)` as scikit-learn caps it. Signs: each component's largest `|value|` is positive
//! (scikit-learn ≥ 1.5 `svd_flip`, `u_based_decision=False`).

use std::ops::Range;

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_core::compute::LaunchGeometry;

use super::eigen::{symmetric_eigen, EigenOptions, SymmetricEigen};
use super::tridiagonal::symmetric_eigen_cpu;
use super::kernels::{add_assign_kernel, centre_columns_kernel, gather_rows_kernel};
use super::matmul::{matmul, MatrixView};
use crate::core::buffer;

/// Matrices up to this size are eigendecomposed on the host ([`symmetric_eigen_cpu`], `f64`): the
/// device Jacobi solver's launches cost more than the arithmetic there.
pub const HOST_EIGEN_MAX: usize = 512;

/// Rows per partial product of the covariance (each sum adds at most this many terms in `f32`).
pub const GRAM_ROWS: usize = 4096;

/// Rows of `dim` features, handed to the device a batch at a time.
pub trait RowSource {
    fn rows(&self) -> usize;
    fn dim(&self) -> usize;
    /// Rows `range` as a new row-major `[range.len(), dim]` `f32` device buffer.
    fn batch(&self, client: &Client, range: Range<usize>) -> Handle;
}

/// Rows already on the host, row-major `[rows, dim]`.
#[derive(Debug, Clone, Copy)]
pub struct HostRows<'a> {
    pub data: &'a [f32],
    pub dim: usize,
}

impl RowSource for HostRows<'_> {
    fn rows(&self) -> usize {
        if self.dim == 0 { 0 } else { self.data.len() / self.dim }
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn batch(&self, client: &Client, range: Range<usize>) -> Handle {
        buffer::upload(client, &self.data[range.start * self.dim..range.end * self.dim])
    }
}

/// Some rows (`rows`, by index) of a host matrix `[_, dim]`, gathered a batch at a time: no copy of
/// the subset is made up front.
#[derive(Debug, Clone, Copy)]
pub struct IndexedRows<'a> {
    pub data: &'a [f32],
    pub dim: usize,
    pub rows: &'a [usize],
}

impl RowSource for IndexedRows<'_> {
    fn rows(&self) -> usize {
        self.rows.len()
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn batch(&self, client: &Client, range: Range<usize>) -> Handle {
        let d = self.dim;
        let mut out = Vec::with_capacity(range.len() * d);
        for &i in &self.rows[range] {
            out.extend_from_slice(&self.data[i * d..(i + 1) * d]);
        }
        buffer::upload(client, &out)
    }
}

/// Some rows (`rows`, by index) of a `[_, dim]` `f32` matrix already in device memory (`data`, `len`
/// values), gathered on the device a batch at a time: nothing crosses to the device but the indices.
#[derive(Debug, Clone, Copy)]
pub struct DeviceRows<'a> {
    pub data: &'a Handle,
    pub len: usize,
    pub dim: usize,
    pub rows: &'a [u32],
}

impl RowSource for DeviceRows<'_> {
    fn rows(&self) -> usize {
        self.rows.len()
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn batch(&self, client: &Client, range: Range<usize>) -> Handle {
        let b = range.len();
        let total = b * self.dim;
        let out = buffer::empty::<f32>(client, total.max(1));
        if total > 0 {
            let geom = LaunchGeometry::elementwise(client, total);
            // SAFETY: `data` holds `len` values; the indices are below `len / dim`
            unsafe {
                gather_rows_kernel::launch::<f32>(
                    client,
                    geom.cube_count,
                    geom.cube_dim,
                    BufferArg::from_raw_parts(self.data.clone(), self.len),
                    BufferArg::from_raw_parts(buffer::upload(client, &self.rows[range]), b),
                    BufferArg::from_raw_parts(out.clone(), total),
                    self.dim as u32,
                    total as u32,
                );
            }
        }
        out
    }
}

/// Settings of [`TopComponents::fit`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopComponentsOptions {
    /// Above this many features the covariance is not formed (scikit-learn's randomized solver).
    pub exact_cap: usize,
    /// Extra columns of the iterated subspace (scikit-learn's `n_oversamples`).
    pub oversamples: usize,
    /// Iterations above `exact_cap`; `None`: scikit-learn's (module docs).
    pub randomized_iterations: Option<usize>,
    /// Relative change of the leading values that ends the exact iteration.
    pub tolerance: f64,
    pub max_iterations: usize,
    pub seed: u64,
    /// Device elements per batch of rows.
    pub batch_elements: usize,
}

impl Default for TopComponentsOptions {
    fn default() -> Self {
        Self {
            exact_cap: 8000,
            oversamples: 10,
            randomized_iterations: None,
            tolerance: 1e-6,
            max_iterations: 300,
            seed: 0,
            batch_elements: 1 << 25,
        }
    }
}

/// Fitted components (module docs).
#[derive(Debug, Clone)]
pub struct TopComponents {
    pub dim: usize,
    pub k: usize,
    /// Column means.
    pub mean: Vec<f32>,
    /// `[k, dim]`, row `c` the `c`-th component.
    pub components: Vec<f32>,
    /// Variance along each component (`/ (L − 1)`).
    pub variances: Vec<f64>,
    /// Subspace iterations run (0: direct eigendecomposition).
    pub iterations: usize,
    mean_h: Handle,
    /// `[dim, k]` (components as columns) on the device.
    v_h: Handle,
}

/// Seeded standard normal draws (splitmix64, Box–Muller).
struct Normal(u64);

impl Normal {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (((z ^ (z >> 31)) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn next(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// How `C` is applied.
enum Operator<'a, S: RowSource> {
    /// `[d, d]` on the device.
    Formed(Handle),
    Implicit(&'a S),
}

impl TopComponents {
    /// The leading `k` components of `src` (module docs).
    ///
    /// # Panics
    ///
    /// If `src` has no rows or no features.
    pub fn fit<S: RowSource>(client: &Client, src: &S, k: usize, opts: &TopComponentsOptions) -> Self {
        let (l, d) = (src.rows(), src.dim());
        assert!(l > 0 && d > 0, "PCA of {l} rows of {d} features");
        let k = k.min(l).min(d);
        let batch_rows = (opts.batch_elements / d).clamp(1, GRAM_ROWS);

        // Column means
        let mut sums = vec![0.0f64; d];
        let ones = buffer::filled::<f32>(client, batch_rows, 1.0);
        let row = buffer::empty::<f32>(client, d);
        for r in batches(l, batch_rows) {
            let b = r.len();
            let x = src.batch(client, r);
            matmul::<f32>(client, &MatrixView::row_major(&ones, batch_rows, 1, b), &MatrixView::row_major(&x, b * d, b, d), &row, d);
            for (s, v) in sums.iter_mut().zip(buffer::download::<f32>(client, row.clone())) {
                *s += v as f64;
            }
        }
        let mean: Vec<f32> = sums.iter().map(|s| (s / l as f64) as f32).collect();
        let mean_h = buffer::upload(client, &mean);
        let denom = (l.max(2) - 1) as f64;

        let r = (k + opts.oversamples).min(d);
        let exact = d <= opts.exact_cap;
        let operator = if exact {
            let c = buffer::zeros::<f32>(client, d * d);
            let part = buffer::empty::<f32>(client, d * d);
            for rr in batches(l, batch_rows) {
                let b = rr.len();
                let x = centred(client, src, rr, &mean_h);
                let xv = MatrixView::row_major(&x, b * d, b, d);
                matmul::<f32>(client, &xv.transposed(), &xv, &part, d * d);
                add_assign(client, &c, &part, d * d);
            }
            if 2 * r >= d {
                let eig = eigen_of(client, &c, d);
                let vectors: Vec<f64> = (0..d * k).map(|e| eig.vectors[(e / k) * d + e % k]).collect();
                let values = eig.values[..k].iter().map(|v| v.max(0.0) / denom).collect();
                return Self::finish(client, d, k, mean, mean_h, vectors, values, 0);
            }
            Operator::Formed(c)
        } else {
            Operator::Implicit(src)
        };
        let iterations = if exact {
            opts.max_iterations
        } else {
            opts.randomized_iterations.unwrap_or(if (k as f64) < 0.1 * l.min(d) as f64 { 7 } else { 4 })
        };

        let mut rng = Normal(opts.seed);
        let start: Vec<f32> = (0..d * r).map(|_| rng.next() as f32).collect();
        let (mut q, _) = orthonormalise(client, &buffer::upload(client, &start), d, r);
        let mut previous: Option<Vec<f64>> = None;
        let mut ran = 0;
        for _ in 0..iterations {
            ran += 1;
            let y = apply(client, &operator, &q, d, r, l, batch_rows, &mean_h);
            let (next, sigmas) = orthonormalise(client, &y, d, r);
            q = next;
            if exact {
                let lead = &sigmas[..k];
                if let Some(p) = &previous {
                    let change = lead.iter().zip(p).map(|(a, b)| (a - b).abs() / a.abs().max(f64::MIN_POSITIVE)).fold(0.0, f64::max);
                    if change < opts.tolerance {
                        break;
                    }
                }
                previous = Some(lead.to_vec());
            }
        }
        // Rayleigh–Ritz: B = Qᵀ C Q, V = Q U
        let y = apply(client, &operator, &q, d, r, l, batch_rows, &mean_h);
        let bmat = buffer::empty::<f32>(client, r * r);
        let qv = MatrixView::row_major(&q, d * r, d, r);
        matmul::<f32>(client, &qv.transposed(), &MatrixView::row_major(&y, d * r, d, r), &bmat, r * r);
        // Symmetrise: rounding leaves B slightly asymmetric
        let mut bh = buffer::download::<f32>(client, bmat);
        for i in 0..r {
            for j in 0..i {
                let m = 0.5 * (bh[i * r + j] + bh[j * r + i]);
                bh[i * r + j] = m;
                bh[j * r + i] = m;
            }
        }
        let eig = if r <= HOST_EIGEN_MAX {
            symmetric_eigen_cpu(&bh.iter().map(|&v| v as f64).collect::<Vec<_>>(), r)
        } else {
            symmetric_eigen::<f32>(client, &buffer::upload(client, &bh), r, EigenOptions::default())
        };
        let u: Vec<f32> = (0..r * k).map(|e| eig.vectors[(e / k) * r + e % k] as f32).collect();
        let v = buffer::empty::<f32>(client, d * k);
        matmul::<f32>(client, &qv, &MatrixView::row_major(&buffer::upload(client, &u), r * k, r, k), &v, d * k);
        let vectors: Vec<f64> = buffer::download::<f32>(client, v).into_iter().map(f64::from).collect();
        let values = eig.values[..k].iter().map(|v| v.max(0.0) / denom).collect();
        Self::finish(client, d, k, mean, mean_h, vectors, values, ran)
    }

    /// Sign convention and device copies; `vectors` is `[d, k]` (components as columns).
    #[allow(clippy::too_many_arguments)]
    fn finish(client: &Client, d: usize, k: usize, mean: Vec<f32>, mean_h: Handle, vectors: Vec<f64>, variances: Vec<f64>, iterations: usize) -> Self {
        let mut components = vec![0.0f32; k * d];
        for c in 0..k {
            let big = (0..d).map(|t| vectors[t * k + c]).fold(0.0f64, |m, v| if v.abs() > m.abs() { v } else { m });
            let sign = if big < 0.0 { -1.0 } else { 1.0 };
            for t in 0..d {
                components[c * d + t] = (sign * vectors[t * k + c]) as f32;
            }
        }
        let columns: Vec<f32> = (0..d * k).map(|e| components[(e % k) * d + e / k]).collect();
        Self { dim: d, k, mean, components, variances, iterations, mean_h, v_h: buffer::upload(client, &columns) }
    }

    /// `(x − μ) · Vᵀ` of every row of `src`, `[rows, k]` on the host.
    ///
    /// # Panics
    ///
    /// If `src` has another number of features.
    pub fn transform<S: RowSource>(&self, client: &Client, src: &S, batch_elements: usize) -> Vec<f32> {
        let (l, d, k) = (src.rows(), self.dim, self.k);
        assert_eq!(src.dim(), d, "rows of {} features, fitted on {d}", src.dim());
        let batch_rows = (batch_elements / d.max(k)).max(1);
        let mut out = Vec::with_capacity(l * k);
        let f = buffer::empty::<f32>(client, batch_rows * k);
        for r in batches(l, batch_rows) {
            let b = r.len();
            let x = centred(client, src, r, &self.mean_h);
            matmul::<f32>(client, &MatrixView::row_major(&x, b * d, b, d), &MatrixView::row_major(&self.v_h, d * k, d, k), &f, batch_rows * k);
            out.extend(buffer::download_prefix::<f32>(client, f.clone(), b * k));
        }
        out
    }
}

/// Eigendecomposition of the symmetric `[n, n]` device matrix `m`: on the host up to
/// [`HOST_EIGEN_MAX`], else on the device.
fn eigen_of(client: &Client, m: &Handle, n: usize) -> SymmetricEigen {
    if n <= HOST_EIGEN_MAX {
        let host: Vec<f64> = buffer::download::<f32>(client, m.clone()).into_iter().take(n * n).map(f64::from).collect();
        // Rounding leaves the device product slightly asymmetric
        let sym: Vec<f64> = (0..n * n).map(|e| 0.5 * (host[e] + host[(e % n) * n + e / n])).collect();
        symmetric_eigen_cpu(&sym, n)
    } else {
        symmetric_eigen::<f32>(client, m, n, EigenOptions::default())
    }
}

fn batches(l: usize, rows: usize) -> impl Iterator<Item = Range<usize>> {
    (0..l.div_ceil(rows)).map(move |i| i * rows..((i + 1) * rows).min(l))
}

/// Rows `r` of `src`, centred, on the device.
fn centred<S: RowSource>(client: &Client, src: &S, r: Range<usize>, mean: &Handle) -> Handle {
    let (b, d) = (r.len(), src.dim());
    let x = src.batch(client, r);
    let geom = LaunchGeometry::elementwise(client, b * d);
    // SAFETY: `x` holds `b · d` values, `mean` `d`
    unsafe {
        centre_columns_kernel::launch::<f32>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(x.clone(), b * d),
            BufferArg::from_raw_parts(mean.clone(), d),
            d as u32,
            (b * d) as u32,
        );
    }
    x
}

fn add_assign(client: &Client, acc: &Handle, x: &Handle, n: usize) {
    let geom = LaunchGeometry::elementwise(client, n);
    // SAFETY: both hold `n` values
    unsafe {
        add_assign_kernel::launch::<f32>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(acc.clone(), n),
            BufferArg::from_raw_parts(x.clone(), n),
            n as u32,
        );
    }
}

/// `C · Q` (`[d, r]`).
#[allow(clippy::too_many_arguments)]
fn apply<S: RowSource>(client: &Client, op: &Operator<'_, S>, q: &Handle, d: usize, r: usize, l: usize, batch_rows: usize, mean: &Handle) -> Handle {
    let qv = MatrixView::row_major(q, d * r, d, r);
    let y = buffer::empty::<f32>(client, d * r);
    match op {
        Operator::Formed(c) => matmul::<f32>(client, &MatrixView::row_major(c, d * d, d, d), &qv, &y, d * r),
        Operator::Implicit(src) => {
            let acc = buffer::zeros::<f32>(client, d * r);
            let z = buffer::empty::<f32>(client, batch_rows * r);
            for rr in batches(l, batch_rows) {
                let b = rr.len();
                let x = centred(client, *src, rr, mean);
                let xv = MatrixView::row_major(&x, b * d, b, d);
                matmul::<f32>(client, &xv, &qv, &z, batch_rows * r);
                matmul::<f32>(client, &xv.transposed(), &MatrixView::row_major(&z, b * r, b, r), &y, d * r);
                add_assign(client, &acc, &y, d * r);
            }
            return acc;
        }
    }
    y
}

/// Orthonormal columns spanning `y` (`[d, r]`), and the singular values of `y` (descending), by
/// the eigendecomposition of `yᵀy` (module docs).
fn orthonormalise(client: &Client, y: &Handle, d: usize, r: usize) -> (Handle, Vec<f64>) {
    let yv = MatrixView::row_major(y, d * r, d, r);
    let g = buffer::empty::<f32>(client, r * r);
    matmul::<f32>(client, &yv.transposed(), &yv, &g, r * r);
    let eig = eigen_of(client, &g, r);
    let top = eig.values.first().copied().unwrap_or(0.0).max(0.0);
    let floor = top * f32::EPSILON as f64;
    let mut t = vec![0.0f32; r * r];
    for c in 0..r {
        let lambda = eig.values[c];
        if lambda > floor && lambda > 0.0 {
            let s = 1.0 / lambda.sqrt();
            for i in 0..r {
                t[i * r + c] = (eig.vectors[i * r + c] * s) as f32;
            }
        }
    }
    let q = buffer::empty::<f32>(client, d * r);
    matmul::<f32>(client, &yv, &MatrixView::row_major(&buffer::upload(client, &t), r * r, r, r), &q, d * r);
    (q, eig.values.iter().map(|v| v.max(0.0).sqrt()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::symmetric_eigen_host;

    /// Rows with a known spectrum: `L` rows of `D` features, variance decaying along random axes.
    fn data(l: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = Normal(seed);
        let scales: Vec<f64> = (0..d).map(|i| 10.0 * 0.8f64.powi(i as i32) + 0.01).collect();
        // A random rotation would be costly; mixing two coordinates keeps the axes non-trivial
        (0..l)
            .flat_map(|_| {
                let z: Vec<f64> = scales.iter().map(|s| s * rng.next()).collect();
                (0..d).map(move |j| (z[j] + 0.5 * z[(j + 1) % d] + 3.0) as f32).collect::<Vec<_>>()
            })
            .collect()
    }

    /// Host covariance, its leading eigenvalues (device Jacobi on the host's `f64` matrix).
    fn reference(client: &Client, x: &[f32], l: usize, d: usize, k: usize) -> Vec<f64> {
        let mean: Vec<f64> = (0..d).map(|j| (0..l).map(|i| x[i * d + j] as f64).sum::<f64>() / l as f64).collect();
        let mut c = vec![0.0f64; d * d];
        for i in 0..l {
            for a in 0..d {
                for b in 0..d {
                    c[a * d + b] += (x[i * d + a] as f64 - mean[a]) * (x[i * d + b] as f64 - mean[b]);
                }
            }
        }
        c.iter_mut().for_each(|v| *v /= (l - 1) as f64);
        symmetric_eigen_host::<f32>(client, &c, d, EigenOptions::default()).values[..k].to_vec()
    }

    fn check(client: &Client, opts: TopComponentsOptions, iterate: bool) {
        let (l, d, k) = (3000usize, 64usize, 5usize);
        let x = data(l, d, 1);
        let src = HostRows { data: &x, dim: d };
        let fit = TopComponents::fit(client, &src, k, &opts);
        assert_eq!(fit.iterations > 0, iterate, "{opts:?}");
        let want = reference(client, &x, l, d, k);
        for (g, w) in fit.variances.iter().zip(&want) {
            assert!((g - w).abs() < 2e-3 * w, "variances {:?} vs {want:?}", fit.variances);
        }
        // Orthonormal components, features with those variances
        for a in 0..k {
            for b in 0..k {
                let dot: f32 = (0..d).map(|t| fit.components[a * d + t] * fit.components[b * d + t]).sum();
                assert!((dot - if a == b { 1.0 } else { 0.0 }).abs() < 1e-3, "<{a},{b}> = {dot}");
            }
        }
        let f = fit.transform(client, &src, 1 << 12);
        assert_eq!(f.len(), l * k);
        let var0 = (0..l).map(|i| (f[i * k] as f64).powi(2)).sum::<f64>() / (l - 1) as f64;
        assert!((var0 - want[0]).abs() < 2e-3 * want[0], "{var0} vs {}", want[0]);
    }

    fn all_paths(client: &Client) {
        // Small batches: the sums run over many batches
        let base = TopComponentsOptions { batch_elements: 64 * 100, ..Default::default() };
        // Direct (k + oversamples ≥ D / 2)
        check(client, TopComponentsOptions { oversamples: 30, ..base }, false);
        // Exact subspace iteration (oversamples small against D)
        check(client, TopComponentsOptions { oversamples: 3, ..base }, true);
        // Implicit ("randomized"), more iterations than scikit-learn's to reach the test tolerance
        check(client, TopComponentsOptions { exact_cap: 10, oversamples: 10, randomized_iterations: Some(12), ..base }, true);
    }
    runtime_test!(test_top_components_match_the_covariance, all_paths);
}
