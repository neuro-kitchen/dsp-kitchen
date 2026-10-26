//! Symmetric eigendecomposition on the device: parallel cyclic Jacobi, batched.
//!
//! A sweep visits every off-diagonal pair once. The pairs are scheduled as a round-robin
//! tournament (circle method): each of the `m − 1` rounds (`m` = `n` rounded up to even) holds `m / 2`
//! disjoint pairs, so all rotations of a round are applied at once as `A ← Jᵀ A J`, `V ← V J`.
//!
//! **Convergence** (Demmel & Veselić 1992, "Jacobi's method is more accurate than QR"): a pair has
//! converged when `|a_pq| ≤ tol · √|a_pp · a_qq|`; converged pairs are not rotated, and sweeps stop
//! when every pair has converged. This relative, per-pair test lets Jacobi compute the small
//! eigenvalues of a positive definite matrix to high relative accuracy, which whitening needs
//! (`1 / √λ` magnifies their errors). The tolerance defaults to `√n · ε` of the float type (the
//! scaling of LAPACK's one-sided Jacobi): a fixed value below that rounding floor is never reached.
//!
//! Two paths, chosen by size against the runtime's shared memory:
//! - **shared**: one cube per matrix holds `A`, `V` and a scratch copy in shared memory and runs every
//!   sweep in one launch (small matrices, e.g. local whitening neighbourhoods);
//! - **global**: three launches per round over all `(matrix, row, column)` elements, with a
//!   convergence check per sweep (large matrices, e.g. all channels of a probe).

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_core::compute::LaunchGeometry;

use super::kernels::eigen::{
    jacobi_cols_kernel, jacobi_offdiag_ratio_kernel, jacobi_rotations_kernel, jacobi_rows_kernel, jacobi_shared_kernel,
};

use crate::core::{buffer, cast, to_f64, DspFloat};
use crate::math::execute_scaling;

/// Default cap on sweeps (cyclic Jacobi converges quadratically; well-conditioned matrices need
/// fewer than ten).
pub const EIGEN_MAX_SWEEPS: usize = 30;

/// Matrices that need this many `n × n` shared buffers (`A`, `V`, scratch) to fit the runtime's
/// shared memory take the single-launch shared path.
const SHARED_MATRICES: usize = 3;

/// Convergence settings of [`symmetric_eigen_batched`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EigenOptions {
    /// Pair tolerance `tol` of `|a_pq| ≤ tol · √|a_pp · a_qq|`. `None`: [`rounding_tolerance`] of
    /// the float type and size; a smaller value is raised to it (it could never be reached).
    pub tolerance: Option<f64>,
    pub max_sweeps: usize,
}

impl Default for EigenOptions {
    fn default() -> Self {
        Self { tolerance: None, max_sweeps: EIGEN_MAX_SWEEPS }
    }
}

/// `√n · ε` of `F`: the size of the off-diagonal entries rounding leaves after a rotation of an
/// `n × n` matrix, relative to its diagonal. Pair tests below it cannot be met.
pub fn rounding_tolerance<F: DspFloat>(n: usize) -> f64 {
    (n.max(1) as f64).sqrt() * to_f64(F::EPSILON)
}

/// Eigendecomposition of one symmetric `n × n` matrix, eigenvalues in descending order.
#[derive(Debug, Clone, PartialEq)]
pub struct SymmetricEigen {
    pub n: usize,
    /// Eigenvalues, largest first.
    pub values: Vec<f64>,
    /// Row-major `[n, n]`: column `k` (`vectors[row · n + k]`) is the unit eigenvector of `values[k]`.
    pub vectors: Vec<f64>,
}

// ---------------------------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------------------------

/// Whether `n × n` matrices of `F` fit the shared path on this runtime.
fn fits_shared<F: DspFloat>(client: &Client, n: usize) -> bool {
    let hw = &client.properties().hardware;
    // A, V, scratch, plus per-index rotation data and the reduction buffers
    let bytes = SHARED_MATRICES * n * n * size_of::<F>() + n * (2 * size_of::<F>() + size_of::<u32>()) + 2 * hw.max_units_per_cube as usize * size_of::<F>();
    bytes <= hw.max_shared_memory_size
}

/// Eigendecompositions of `batch` symmetric `n × n` matrices stored row-major and contiguous in
/// `matrices` (device buffer of `F`, left unchanged). Matrices converge independently; sweeps stop
/// once all have converged or after `options.max_sweeps`.
pub fn symmetric_eigen_batched<F: DspFloat>(
    client: &Client,
    matrices: &Handle,
    batch: usize,
    n: usize,
    options: EigenOptions,
) -> Vec<SymmetricEigen> {
    if batch == 0 || n == 0 {
        return vec![SymmetricEigen { n, values: Vec::new(), vectors: Vec::new() }; batch];
    }
    let len = batch * n * n;
    let floor = rounding_tolerance::<F>(n);
    let tolerance = options.tolerance.map_or(floor, |t| t.max(floor));

    // Working copy on the device: the solver rotates `a` in place
    let a = buffer::empty::<F>(client, len);
    let v = buffer::empty::<F>(client, len);
    execute_scaling::<F>(client, matrices, &a, len, cast(1.0), cast(0.0));
    let (a, v) = if fits_shared::<F>(client, n) {
        run_shared::<F>(client, a, v, batch, n, tolerance, options.max_sweeps)
    } else {
        run_global::<F>(client, a, v, batch, n, tolerance, options.max_sweeps)
    };

    let a = buffer::download::<F>(client, a);
    let v = buffer::download::<F>(client, v);
    (0..batch)
        .map(|b| {
            let (ab, vb) = (&a[b * n * n..(b + 1) * n * n], &v[b * n * n..(b + 1) * n * n]);
            let mut order: Vec<usize> = (0..n).collect();
            let diag: Vec<f64> = (0..n).map(|i| to_f64(ab[i * n + i])).collect();
            order.sort_by(|&x, &y| diag[y].total_cmp(&diag[x]));
            let values = order.iter().map(|&k| diag[k]).collect();
            let mut vectors = vec![0.0f64; n * n];
            for (new_k, &k) in order.iter().enumerate() {
                for row in 0..n {
                    vectors[row * n + new_k] = to_f64(vb[row * n + k]);
                }
            }
            SymmetricEigen { n, values, vectors }
        })
        .collect()
}

/// Eigendecomposition of one symmetric `n × n` device matrix of `F`.
pub fn symmetric_eigen<F: DspFloat>(client: &Client, matrix: &Handle, n: usize, options: EigenOptions) -> SymmetricEigen {
    symmetric_eigen_batched::<F>(client, matrix, 1, n, options).remove(0)
}

/// Eigendecomposition of a host matrix (row-major, `n × n`, symmetric) on the device in `F`.
pub fn symmetric_eigen_host<F: DspFloat>(client: &Client, matrix: &[f64], n: usize, options: EigenOptions) -> SymmetricEigen {
    assert_eq!(matrix.len(), n * n, "matrix size mismatch");
    let handle = buffer::upload(client, &matrix.iter().map(|&x| cast::<F>(x)).collect::<Vec<F>>());
    symmetric_eigen::<F>(client, &handle, n, options)
}

fn run_shared<F: DspFloat>(client: &Client, a: Handle, v: Handle, batch: usize, n: usize, tolerance: f64, max_sweeps: usize) -> (Handle, Handle) {
    let geom = LaunchGeometry::per_row(client, batch, n * n);
    unsafe {
        jacobi_shared_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim.clone(),
            BufferArg::from_raw_parts(a.clone(), batch * n * n),
            BufferArg::from_raw_parts(v.clone(), batch * n * n),
            batch as u32,
            cast::<F>(tolerance),
            max_sweeps as u32,
            n as u32,
            geom.cube_dim.x,
        );
    }
    (a, v)
}

fn run_global<F: DspFloat>(client: &Client, a: Handle, v: Handle, batch: usize, n: usize, tolerance: f64, max_sweeps: usize) -> (Handle, Handle) {
    let len = batch * n * n;
    let m = n + n % 2;
    let identity: Vec<F> = (0..len).map(|e| cast::<F>(if (e % (n * n)) / n == e % n { 1.0 } else { 0.0 })).collect();
    let (a, a_tmp) = (a, buffer::empty::<F>(client, len));
    let (mut v, mut v_tmp) = (buffer::upload(client, &identity), v);
    let partner = buffer::empty::<u32>(client, batch * n);
    let coef_self = buffer::empty::<F>(client, batch * n);
    let coef_other = buffer::empty::<F>(client, batch * n);
    let ratio = buffer::empty::<F>(client, batch);
    let elems = LaunchGeometry::elementwise(client, len);
    let pairs = LaunchGeometry::elementwise(client, batch * m / 2);
    let per_matrix = LaunchGeometry::per_row(client, batch, n * n);

    for sweep in 0..=max_sweeps {
        unsafe {
            jacobi_offdiag_ratio_kernel::launch::<F>(
                client,
                per_matrix.cube_count.clone(),
                per_matrix.cube_dim.clone(),
                BufferArg::from_raw_parts(a.clone(), len),
                BufferArg::from_raw_parts(ratio.clone(), batch),
                batch as u32,
                n as u32,
                per_matrix.cube_dim.x,
            );
        }
        let worst = buffer::download::<F>(client, ratio.clone());
        let converged = worst.iter().all(|&r| to_f64(r) <= tolerance);
        if converged || sweep == max_sweeps {
            break;
        }
        for round in 0..m - 1 {
            unsafe {
                jacobi_rotations_kernel::launch::<F>(
                    client,
                    pairs.cube_count.clone(),
                    pairs.cube_dim.clone(),
                    BufferArg::from_raw_parts(a.clone(), len),
                    BufferArg::from_raw_parts(partner.clone(), batch * n),
                    BufferArg::from_raw_parts(coef_self.clone(), batch * n),
                    BufferArg::from_raw_parts(coef_other.clone(), batch * n),
                    batch as u32,
                    n as u32,
                    m as u32,
                    round as u32,
                    cast::<F>(tolerance),
                );
                jacobi_rows_kernel::launch::<F>(
                    client,
                    elems.cube_count.clone(),
                    elems.cube_dim.clone(),
                    BufferArg::from_raw_parts(a.clone(), len),
                    BufferArg::from_raw_parts(partner.clone(), batch * n),
                    BufferArg::from_raw_parts(coef_self.clone(), batch * n),
                    BufferArg::from_raw_parts(coef_other.clone(), batch * n),
                    BufferArg::from_raw_parts(a_tmp.clone(), len),
                    batch as u32,
                    n as u32,
                );
                jacobi_cols_kernel::launch::<F>(
                    client,
                    elems.cube_count.clone(),
                    elems.cube_dim.clone(),
                    BufferArg::from_raw_parts(a_tmp.clone(), len),
                    BufferArg::from_raw_parts(v.clone(), len),
                    BufferArg::from_raw_parts(partner.clone(), batch * n),
                    BufferArg::from_raw_parts(coef_self.clone(), batch * n),
                    BufferArg::from_raw_parts(coef_other.clone(), batch * n),
                    BufferArg::from_raw_parts(a.clone(), len),
                    BufferArg::from_raw_parts(v_tmp.clone(), len),
                    batch as u32,
                    n as u32,
                );
            }
            std::mem::swap(&mut v, &mut v_tmp);
        }
    }
    (a, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Host check: `A v_k = λ_k v_k`, orthonormal vectors, descending values.
    fn check(name: &str, a: &[f64], e: &SymmetricEigen, tol: f64) {
        let n = e.n;
        let scale = a.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1.0);
        for k in 0..n {
            for i in 0..n {
                let av: f64 = (0..n).map(|j| a[i * n + j] * e.vectors[j * n + k]).sum();
                assert!((av - e.values[k] * e.vectors[i * n + k]).abs() < tol * scale, "{name}: A v_{k} ≠ λ v_{k} at row {i}");
            }
            for l in 0..n {
                let dot: f64 = (0..n).map(|i| e.vectors[i * n + k] * e.vectors[i * n + l]).sum();
                let want = if k == l { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < tol, "{name}: vectors {k}·{l} = {dot}");
            }
        }
        assert!(e.values.windows(2).all(|w| w[0] >= w[1]), "{name}: values not descending");
    }

    /// Symmetric test matrix `Mᵀ M + shift` with a spread of eigenvalues.
    fn spd(n: usize, seed: usize) -> Vec<f64> {
        let m: Vec<f64> = (0..n * n).map(|i| (((i + seed) * 7919) % 211) as f64 / 211.0 - 0.5).collect();
        (0..n * n)
            .map(|e| {
                let (i, j) = (e / n, e % n);
                (0..n).map(|k| m[k * n + i] * m[k * n + j]).sum::<f64>() + if i == j { 0.1 * i as f64 } else { 0.0 }
            })
            .collect()
    }

    fn solves(client: &Client) {
        // Sizes on both paths (odd and even): shared for small, global past the shared limit
        for n in [1usize, 2, 5, 16, 33, 160] {
            let a = spd(n, n);
            let e = symmetric_eigen_host::<f32>(client, &a, n, EigenOptions::default());
            check(&format!("{} n={n} shared={}", client.name(), fits_shared::<f32>(client, n)), &a, &e, 2e-3);
        }
        // Batched: every matrix of the batch converges on its own
        let (batch, n) = (7usize, 12usize);
        let mats: Vec<f64> = (0..batch).flat_map(|b| spd(n, 31 * b)).collect();
        let handle = buffer::upload(client, &mats.iter().map(|&x| x as f32).collect::<Vec<_>>());
        let all = symmetric_eigen_batched::<f32>(client, &handle, batch, n, EigenOptions::default());
        for (b, e) in all.iter().enumerate() {
            check(&format!("{} batch {b}", client.name()), &mats[b * n * n..(b + 1) * n * n], e, 2e-3);
        }
    }
    runtime_test!(test_symmetric_eigen, solves);
}
