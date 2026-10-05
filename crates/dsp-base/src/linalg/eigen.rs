//! Symmetric eigendecomposition on the device: parallel cyclic Jacobi, batched.
//!
//! A sweep visits every off-diagonal pair once. The pairs are scheduled as a round-robin
//! tournament (circle method): each of the `m − 1` rounds (`m` = `n` rounded up to even) holds `m / 2`
//! disjoint pairs, so all rotations of a round are applied at once as `A ← Jᵀ A J`, `V ← V J`. Sweeps
//! repeat until the off-diagonal Frobenius norm is at most `tolerance ·` the matrix's Frobenius norm
//! (both invariant under the rotations).
//!
//! Two paths, chosen by size against the runtime's shared memory:
//! - **shared**: one cube per matrix holds `A`, `V` and a scratch copy in shared memory and runs every
//!   sweep in one launch (small matrices, e.g. local whitening neighbourhoods);
//! - **global**: three launches per round over all `(matrix, row, column)` elements, with a
//!   convergence check per sweep (large matrices, e.g. all channels of a probe).

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_core::compute::{row_position, LaunchGeometry};

use crate::core::{buffer, cast, to_f64, DspFloat};
use crate::math::execute_scaling;

/// Default convergence threshold: off-diagonal norm relative to the matrix norm.
pub const EIGEN_RELATIVE_TOLERANCE: f64 = 1e-6;

/// Default cap on sweeps (cyclic Jacobi converges quadratically; well-conditioned matrices need
/// fewer than ten).
pub const EIGEN_MAX_SWEEPS: usize = 30;

/// Matrices that need this many `n × n` shared buffers (`A`, `V`, scratch) to fit the runtime's
/// shared memory take the single-launch shared path.
const SHARED_MATRICES: usize = 3;

/// Convergence settings of [`symmetric_eigen_batched`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EigenOptions {
    /// Stop when `‖offdiag(A)‖_F ≤ tolerance · ‖A‖_F`.
    pub tolerance: f64,
    pub max_sweeps: usize,
}

impl Default for EigenOptions {
    fn default() -> Self {
        Self { tolerance: EIGEN_RELATIVE_TOLERANCE, max_sweeps: EIGEN_MAX_SWEEPS }
    }
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

/// Pair `k` of round `round` in the circle-method schedule of `m` (even) players.
#[cube]
fn round_pair(round: u32, k: u32, m: u32) -> (u32, u32) {
    let last = m - 1u32;
    let mut a = last;
    let mut b = round;
    if k != 0u32 {
        a = (round + k) % last;
        b = (round + last - k) % last;
    }
    (u32::min(a, b), u32::max(a, b))
}

/// Jacobi rotation zeroing `a_pq`: returns `(c, s)` with `A' = Jᵀ A J`, `J_pp = J_qq = c`,
/// `J_pq = s`, `J_qp = −s`.
#[cube]
fn rotation<F: Float>(app: F, aqq: F, apq: F) -> (F, F) {
    let mut c = F::new(1.0f32);
    let mut s = F::new(0.0f32);
    if apq != F::new(0.0f32) {
        let theta = (aqq - app) / (F::new(2.0f32) * apq);
        let mut t = F::new(1.0f32) / (F::abs(theta) + F::sqrt(theta * theta + F::new(1.0f32)));
        if theta < F::new(0.0f32) {
            t = -t;
        }
        c = F::new(1.0f32) / F::sqrt(t * t + F::new(1.0f32));
        s = t * c;
    }
    (c, s)
}

// ---------------------------------------------------------------------------------------------
// Global path
// ---------------------------------------------------------------------------------------------

/// Rotations of round `round` for every matrix: for each index `i`, its partner and the coefficients
/// of row / column `i` in the update (`new_i = self_i · old_i + other_i · old_partner`). An index
/// paired with the padding player (odd `n`) keeps itself. One unit per `(matrix, pair)`.
#[cube(launch)]
pub fn jacobi_rotations_kernel<F: Float>(
    a: &Array<F>,
    partner: &mut Array<u32>,
    coef_self: &mut Array<F>,
    coef_other: &mut Array<F>,
    batch: u32,
    n: u32,
    m: u32,
    round: u32,
) {
    let unit = ABSOLUTE_POS as u32;
    let half = m / 2u32;
    if unit < batch * half {
        let b = unit / half;
        let (p, q) = round_pair(round, unit - b * half, m);
        let (mat, idx) = ((b * n * n) as usize, (b * n) as usize);
        if q < n {
            let (c, s) = rotation::<F>(a[mat + (p * n + p) as usize], a[mat + (q * n + q) as usize], a[mat + (p * n + q) as usize]);
            partner[idx + p as usize] = q;
            partner[idx + q as usize] = p;
            coef_self[idx + p as usize] = c;
            coef_self[idx + q as usize] = c;
            coef_other[idx + p as usize] = -s;
            coef_other[idx + q as usize] = s;
        } else {
            partner[idx + p as usize] = p;
            coef_self[idx + p as usize] = F::new(1.0f32);
            coef_other[idx + p as usize] = F::new(0.0f32);
        }
    }
}

/// Row stage `B = Jᵀ A`: `B[i, j] = self_i · A[i, j] + other_i · A[partner_i, j]`. One unit per element.
#[cube(launch)]
pub fn jacobi_rows_kernel<F: Float>(
    a: &Array<F>,
    partner: &Array<u32>,
    coef_self: &Array<F>,
    coef_other: &Array<F>,
    out: &mut Array<F>,
    batch: u32,
    n: u32,
) {
    let e = ABSOLUTE_POS as u32;
    if e < batch * n * n {
        let b = e / (n * n);
        let r = e - b * n * n;
        let (i, j) = (r / n, r % n);
        let idx = b * n + i;
        let pi = partner[idx as usize];
        out[e as usize] = coef_self[idx as usize] * a[e as usize] + coef_other[idx as usize] * a[(b * n * n + pi * n + j) as usize];
    }
}

/// Column stage `A = B J` (the rotated pairs set exactly to 0) and `V ← V J` into `v_out`. One unit
/// per element.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn jacobi_cols_kernel<F: Float>(
    b_in: &Array<F>,
    v_in: &Array<F>,
    partner: &Array<u32>,
    coef_self: &Array<F>,
    coef_other: &Array<F>,
    a_out: &mut Array<F>,
    v_out: &mut Array<F>,
    batch: u32,
    n: u32,
) {
    let e = ABSOLUTE_POS as u32;
    if e < batch * n * n {
        let b = e / (n * n);
        let r = e - b * n * n;
        let (i, j) = (r / n, r % n);
        let jdx = (b * n + j) as usize;
        let pj = partner[jdx];
        let row = b * n * n + i * n;
        let (cs, co) = (coef_self[jdx], coef_other[jdx]);
        let mut value = cs * b_in[e as usize] + co * b_in[(row + pj) as usize];
        if pj == i && i != j {
            value = F::new(0.0f32);
        }
        a_out[e as usize] = value;
        v_out[e as usize] = cs * v_in[e as usize] + co * v_in[(row + pj) as usize];
    }
}

/// `norms[2b] = ‖offdiag(A_b)‖²`, `norms[2b + 1] = ‖A_b‖²`. One cube per matrix
/// ([`LaunchGeometry::per_row`]).
#[cube(launch)]
pub fn jacobi_norms_kernel<F: Float>(a: &Array<F>, norms: &mut Array<F>, batch: u32, n: u32, #[comptime] units: u32) {
    let b = row_position();
    if b < batch {
        let unit = UNIT_POS_X;
        let base = (b * n * n) as usize;
        let mut off = F::new(0.0f32);
        let mut all = F::new(0.0f32);
        let mut e = unit;
        while e < n * n {
            let x = a[base + e as usize];
            all += x * x;
            if e / n != e % n {
                off += x * x;
            }
            e += units;
        }
        let mut off_s = SharedMemory::<F>::new(comptime!(units as usize));
        let mut all_s = SharedMemory::<F>::new(comptime!(units as usize));
        off_s[unit as usize] = off;
        all_s[unit as usize] = all;
        sync_cube();
        let mut stride = comptime!(units / 2);
        while stride > 0u32 {
            if unit < stride {
                let (o, t) = (off_s[(unit + stride) as usize], all_s[(unit + stride) as usize]);
                off_s[unit as usize] += o;
                all_s[unit as usize] += t;
            }
            sync_cube();
            stride /= 2u32;
        }
        if unit == 0u32 {
            norms[(2u32 * b) as usize] = off_s[0];
            norms[(2u32 * b + 1u32) as usize] = all_s[0];
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Shared path
// ---------------------------------------------------------------------------------------------

/// Every sweep of one matrix per cube, in shared memory: rotations, row stage and column stage of a
/// round are separated by cube barriers, and the convergence test is a cube reduction. `a` is
/// overwritten with the converged (diagonal) matrix, `v` with the eigenvectors. `n` is comptime so the
/// shared buffers are sized at compile time.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn jacobi_shared_kernel<F: Float + CubeElement>(
    a: &mut Array<F>,
    v: &mut Array<F>,
    batch: u32,
    tolerance_sq: F,
    max_sweeps: u32,
    #[comptime] n: u32,
    #[comptime] units: u32,
) {
    let b = row_position();
    if b < batch {
        let unit = UNIT_POS_X;
        let nn = comptime!(n * n);
        let m = comptime!(n + n % 2);
        let base = (b * nn) as usize;

        let mut sa = SharedMemory::<F>::new(comptime!(nn as usize));
        let mut sv = SharedMemory::<F>::new(comptime!(nn as usize));
        let mut tmp = SharedMemory::<F>::new(comptime!(nn as usize));
        let mut partner = SharedMemory::<u32>::new(comptime!(n as usize));
        let mut coef_self = SharedMemory::<F>::new(comptime!(n as usize));
        let mut coef_other = SharedMemory::<F>::new(comptime!(n as usize));
        let mut off_s = SharedMemory::<F>::new(comptime!(units as usize));
        let mut all_s = SharedMemory::<F>::new(comptime!(units as usize));

        let mut e = unit;
        while e < nn {
            sa[e as usize] = a[base + e as usize];
            sv[e as usize] = F::new(0.0f32);
            if e / n == e % n {
                sv[e as usize] = F::new(1.0f32);
            }
            e += units;
        }
        sync_cube();

        let mut sweep: u32 = 0u32;
        let mut done: u32 = 0u32;
        while done == 0u32 {
            // Convergence: off-diagonal and total Frobenius norms
            let mut off = F::new(0.0f32);
            let mut all = F::new(0.0f32);
            let mut e = unit;
            while e < nn {
                let x = sa[e as usize];
                all += x * x;
                if e / n != e % n {
                    off += x * x;
                }
                e += units;
            }
            off_s[unit as usize] = off;
            all_s[unit as usize] = all;
            sync_cube();
            let mut stride = comptime!(units / 2);
            while stride > 0u32 {
                if unit < stride {
                    let (o, t) = (off_s[(unit + stride) as usize], all_s[(unit + stride) as usize]);
                    off_s[unit as usize] += o;
                    all_s[unit as usize] += t;
                }
                sync_cube();
                stride /= 2u32;
            }
            // Every unit reads the same reduced values: the loop exit is uniform
            if off_s[0] <= tolerance_sq * all_s[0] || sweep >= max_sweeps {
                done = 1u32;
            }
            sync_cube();

            if done == 0u32 {
                let mut round = 0u32;
                while round < m - 1u32 {
                    let mut k = unit;
                    while k < m / 2u32 {
                        let (p, q) = round_pair(round, k, m);
                        if q < n {
                            let (c, s) = rotation::<F>(sa[(p * n + p) as usize], sa[(q * n + q) as usize], sa[(p * n + q) as usize]);
                            partner[p as usize] = q;
                            partner[q as usize] = p;
                            coef_self[p as usize] = c;
                            coef_self[q as usize] = c;
                            coef_other[p as usize] = -s;
                            coef_other[q as usize] = s;
                        } else {
                            partner[p as usize] = p;
                            coef_self[p as usize] = F::new(1.0f32);
                            coef_other[p as usize] = F::new(0.0f32);
                        }
                        k += units;
                    }
                    sync_cube();
                    // Row stage: tmp = Jᵀ A
                    let mut e = unit;
                    while e < nn {
                        let (i, j) = (e / n, e % n);
                        tmp[e as usize] = coef_self[i as usize] * sa[e as usize] + coef_other[i as usize] * sa[(partner[i as usize] * n + j) as usize];
                        e += units;
                    }
                    sync_cube();
                    // Column stage: A = tmp J
                    let mut e = unit;
                    while e < nn {
                        let (i, j) = (e / n, e % n);
                        let pj = partner[j as usize];
                        let mut value = coef_self[j as usize] * tmp[e as usize] + coef_other[j as usize] * tmp[(i * n + pj) as usize];
                        if pj == i && i != j {
                            value = F::new(0.0f32);
                        }
                        sa[e as usize] = value;
                        e += units;
                    }
                    sync_cube();
                    // V ← V J through tmp
                    let mut e = unit;
                    while e < nn {
                        let (i, j) = (e / n, e % n);
                        tmp[e as usize] = coef_self[j as usize] * sv[e as usize] + coef_other[j as usize] * sv[(i * n + partner[j as usize]) as usize];
                        e += units;
                    }
                    sync_cube();
                    let mut e = unit;
                    while e < nn {
                        sv[e as usize] = tmp[e as usize];
                        e += units;
                    }
                    sync_cube();
                    round += 1u32;
                }
                sweep += 1u32;
            }
        }

        let mut e = unit;
        while e < nn {
            a[base + e as usize] = sa[e as usize];
            v[base + e as usize] = sv[e as usize];
            e += units;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------------------------

/// Whether `n × n` matrices of `F` fit the shared path on this runtime.
fn fits_shared<R: Runtime, F: DspFloat>(client: &ComputeClient<R>, n: usize) -> bool {
    let hw = &client.properties().hardware;
    // A, V, scratch, plus per-index rotation data and the reduction buffers
    let bytes = SHARED_MATRICES * n * n * size_of::<F>() + n * (2 * size_of::<F>() + size_of::<u32>()) + 2 * hw.max_units_per_cube as usize * size_of::<F>();
    bytes <= hw.max_shared_memory_size
}

/// Eigendecompositions of `batch` symmetric `n × n` matrices stored row-major and contiguous in
/// `matrices` (device buffer of `F`, left unchanged). Matrices converge independently; sweeps stop
/// once all have converged or after `options.max_sweeps`.
pub fn symmetric_eigen_batched<R: Runtime, F: DspFloat>(
    client: &ComputeClient<R>,
    matrices: &Handle,
    batch: usize,
    n: usize,
    options: EigenOptions,
) -> Vec<SymmetricEigen> {
    if batch == 0 || n == 0 {
        return vec![SymmetricEigen { n, values: Vec::new(), vectors: Vec::new() }; batch];
    }
    let len = batch * n * n;
    let tolerance_sq = options.tolerance * options.tolerance;

    // Working copy on the device: the solver rotates `a` in place
    let a = buffer::empty::<R, F>(client, len);
    let v = buffer::empty::<R, F>(client, len);
    execute_scaling::<R, F>(client, matrices, &a, len, cast(1.0), cast(0.0));
    let (a, v) = if fits_shared::<R, F>(client, n) {
        run_shared::<R, F>(client, a, v, batch, n, tolerance_sq, options.max_sweeps)
    } else {
        run_global::<R, F>(client, a, v, batch, n, tolerance_sq, options.max_sweeps)
    };

    let a = buffer::download::<R, F>(client, a);
    let v = buffer::download::<R, F>(client, v);
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
pub fn symmetric_eigen<R: Runtime, F: DspFloat>(client: &ComputeClient<R>, matrix: &Handle, n: usize, options: EigenOptions) -> SymmetricEigen {
    symmetric_eigen_batched::<R, F>(client, matrix, 1, n, options).remove(0)
}

/// Eigendecomposition of a host matrix (row-major, `n × n`, symmetric) on the device in `F`.
pub fn symmetric_eigen_host<R: Runtime, F: DspFloat>(client: &ComputeClient<R>, matrix: &[f64], n: usize, options: EigenOptions) -> SymmetricEigen {
    assert_eq!(matrix.len(), n * n, "matrix size mismatch");
    let handle = buffer::upload(client, &matrix.iter().map(|&x| cast::<F>(x)).collect::<Vec<F>>());
    symmetric_eigen::<R, F>(client, &handle, n, options)
}

fn run_shared<R: Runtime, F: DspFloat>(client: &ComputeClient<R>, a: Handle, v: Handle, batch: usize, n: usize, tolerance_sq: f64, max_sweeps: usize) -> (Handle, Handle) {
    let geom = LaunchGeometry::per_row(client, batch, n * n);
    unsafe {
        jacobi_shared_kernel::launch::<F, R>(
            client,
            geom.cube_count,
            geom.cube_dim.clone(),
            ArrayArg::from_raw_parts(a.clone(), batch * n * n),
            ArrayArg::from_raw_parts(v.clone(), batch * n * n),
            batch as u32,
            cast::<F>(tolerance_sq),
            max_sweeps as u32,
            n as u32,
            geom.cube_dim.x,
        );
    }
    (a, v)
}

fn run_global<R: Runtime, F: DspFloat>(client: &ComputeClient<R>, a: Handle, v: Handle, batch: usize, n: usize, tolerance_sq: f64, max_sweeps: usize) -> (Handle, Handle) {
    let len = batch * n * n;
    let m = n + n % 2;
    let identity: Vec<F> = (0..len).map(|e| cast::<F>(if (e % (n * n)) / n == e % n { 1.0 } else { 0.0 })).collect();
    let (a, a_tmp) = (a, buffer::empty::<R, F>(client, len));
    let (mut v, mut v_tmp) = (buffer::upload(client, &identity), v);
    let partner = buffer::empty::<R, u32>(client, batch * n);
    let coef_self = buffer::empty::<R, F>(client, batch * n);
    let coef_other = buffer::empty::<R, F>(client, batch * n);
    let norms = buffer::empty::<R, F>(client, 2 * batch);
    let elems = LaunchGeometry::elementwise(client, len);
    let pairs = LaunchGeometry::elementwise(client, batch * m / 2);
    let per_matrix = LaunchGeometry::per_row(client, batch, n * n);

    for sweep in 0..=max_sweeps {
        unsafe {
            jacobi_norms_kernel::launch::<F, R>(
                client,
                per_matrix.cube_count.clone(),
                per_matrix.cube_dim.clone(),
                ArrayArg::from_raw_parts(a.clone(), len),
                ArrayArg::from_raw_parts(norms.clone(), 2 * batch),
                batch as u32,
                n as u32,
                per_matrix.cube_dim.x,
            );
        }
        let nrm = buffer::download::<R, F>(client, norms.clone());
        let converged = nrm.chunks(2).all(|c| to_f64(c[0]) <= tolerance_sq * to_f64(c[1]));
        if converged || sweep == max_sweeps {
            break;
        }
        for round in 0..m - 1 {
            unsafe {
                jacobi_rotations_kernel::launch::<F, R>(
                    client,
                    pairs.cube_count.clone(),
                    pairs.cube_dim.clone(),
                    ArrayArg::from_raw_parts(a.clone(), len),
                    ArrayArg::from_raw_parts(partner.clone(), batch * n),
                    ArrayArg::from_raw_parts(coef_self.clone(), batch * n),
                    ArrayArg::from_raw_parts(coef_other.clone(), batch * n),
                    batch as u32,
                    n as u32,
                    m as u32,
                    round as u32,
                );
                jacobi_rows_kernel::launch::<F, R>(
                    client,
                    elems.cube_count.clone(),
                    elems.cube_dim.clone(),
                    ArrayArg::from_raw_parts(a.clone(), len),
                    ArrayArg::from_raw_parts(partner.clone(), batch * n),
                    ArrayArg::from_raw_parts(coef_self.clone(), batch * n),
                    ArrayArg::from_raw_parts(coef_other.clone(), batch * n),
                    ArrayArg::from_raw_parts(a_tmp.clone(), len),
                    batch as u32,
                    n as u32,
                );
                jacobi_cols_kernel::launch::<F, R>(
                    client,
                    elems.cube_count.clone(),
                    elems.cube_dim.clone(),
                    ArrayArg::from_raw_parts(a_tmp.clone(), len),
                    ArrayArg::from_raw_parts(v.clone(), len),
                    ArrayArg::from_raw_parts(partner.clone(), batch * n),
                    ArrayArg::from_raw_parts(coef_self.clone(), batch * n),
                    ArrayArg::from_raw_parts(coef_other.clone(), batch * n),
                    ArrayArg::from_raw_parts(a.clone(), len),
                    ArrayArg::from_raw_parts(v_tmp.clone(), len),
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

    fn solves<R: Runtime>(client: &ComputeClient<R>) {
        // Sizes on both paths (odd and even): shared for small, global past the shared limit
        for n in [1usize, 2, 5, 16, 33, 160] {
            let a = spd(n, n);
            let e = symmetric_eigen_host::<R, f32>(client, &a, n, EigenOptions::default());
            check(&format!("{} n={n} shared={}", R::name(client), fits_shared::<R, f32>(client, n)), &a, &e, 2e-3);
        }
        // Batched: every matrix of the batch converges on its own
        let (batch, n) = (7usize, 12usize);
        let mats: Vec<f64> = (0..batch).flat_map(|b| spd(n, 31 * b)).collect();
        let handle = buffer::upload(client, &mats.iter().map(|&x| x as f32).collect::<Vec<_>>());
        let all = symmetric_eigen_batched::<R, f32>(client, &handle, batch, n, EigenOptions::default());
        for (b, e) in all.iter().enumerate() {
            check(&format!("{} batch {b}", R::name(client)), &mats[b * n * n..(b + 1) * n * n], e, 2e-3);
        }
    }
    runtime_test!(test_symmetric_eigen, solves);
}
