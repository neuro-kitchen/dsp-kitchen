//! Kernels of the parallel cyclic Jacobi eigensolver (see `linalg/eigen.rs`).
//!
//! Convergence and rotations follow Demmel & Veselić (1992): a pair `(p, q)` counts as converged
//! when `|a_pq| ≤ tol · √|a_pp · a_qq|`, such pairs are not rotated, and the matrix has converged
//! when every pair has. This relative test (rather than a norm of the whole off-diagonal part) is
//! what lets Jacobi compute small eigenvalues to high relative accuracy.

use cubecl::prelude::*;
use dsp_core::compute::row_position;

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

/// `|a_pq| / √|a_pp · a_qq|`: 0 for a zero `a_pq`, the largest finite value when the diagonal
/// product is zero but `a_pq` is not.
#[cube]
pub fn offdiag_ratio<F: Float>(apq: F, app: F, aqq: F) -> F {
    let scale = F::sqrt(F::abs(app * aqq));
    let mut ratio = F::new(0.0f32);
    if apq != F::new(0.0f32) {
        if scale > F::new(0.0f32) {
            ratio = F::abs(apq) / scale;
        } else {
            ratio = F::max_value();
        }
    }
    ratio
}

/// Jacobi rotation zeroing `a_pq`: returns `(c, s)` with `A' = Jᵀ A J`, `J_pp = J_qq = c`,
/// `J_pq = s`, `J_qp = −s`; the identity (`s = 0`) for a pair that has converged.
#[cube]
fn rotation<F: Float>(app: F, aqq: F, apq: F, tolerance: F) -> (F, F) {
    let mut c = F::new(1.0f32);
    let mut s = F::new(0.0f32);
    if offdiag_ratio::<F>(apq, app, aqq) > tolerance {
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
#[allow(clippy::too_many_arguments)]
pub fn jacobi_rotations_kernel<F: Float + CubeElement + LaunchArg>(
    a: &[F],
    partner: &mut [u32],
    coef_self: &mut [F],
    coef_other: &mut [F],
    batch: u32,
    n: u32,
    m: u32,
    round: u32,
    tolerance: F,
) {
    let unit = ABSOLUTE_POS as u32;
    let half = m / 2u32;
    if unit < batch * half {
        let b = unit / half;
        let (p, q) = round_pair(round, unit - b * half, m);
        let (mat, idx) = ((b * n * n) as usize, (b * n) as usize);
        if q < n {
            let (c, s) =
                rotation::<F>(a[mat + (p * n + p) as usize], a[mat + (q * n + q) as usize], a[mat + (p * n + q) as usize], tolerance);
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
    a: &[F],
    partner: &[u32],
    coef_self: &[F],
    coef_other: &[F],
    out: &mut [F],
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

/// Column stage `A = B J` (the pairs rotated this round set exactly to 0) and `V ← V J` into `v_out`.
/// One unit per element.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn jacobi_cols_kernel<F: Float>(
    b_in: &[F],
    v_in: &[F],
    partner: &[u32],
    coef_self: &[F],
    coef_other: &[F],
    a_out: &mut [F],
    v_out: &mut [F],
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
        // Only a rotated pair is zeroed: a converged one keeps its (small) value
        if pj == i && i != j && co != F::new(0.0f32) {
            value = F::new(0.0f32);
        }
        a_out[e as usize] = value;
        v_out[e as usize] = cs * v_in[e as usize] + co * v_in[(row + pj) as usize];
    }
}

/// `ratio[b]` = the largest [`fn@offdiag_ratio`] over the off-diagonal entries of matrix `b` (the
/// matrix has converged when it is at most the tolerance). One cube per matrix
/// ([`dsp_core::compute::LaunchGeometry::per_row`]).
#[cube(launch)]
pub fn jacobi_offdiag_ratio_kernel<F: Float>(a: &[F], ratio: &mut [F], batch: u32, n: u32, #[comptime] units: u32) {
    let b = row_position();
    if b < batch {
        let unit = UNIT_POS_X;
        let base = (b * n * n) as usize;
        let mut worst = F::new(0.0f32);
        let mut e = unit;
        while e < n * n {
            let (i, j) = (e / n, e % n);
            if i != j {
                let r = offdiag_ratio::<F>(a[base + e as usize], a[base + (i * n + i) as usize], a[base + (j * n + j) as usize]);
                worst = F::max(worst, r);
            }
            e += units;
        }
        let mut worst_s = Shared::<[F]>::new_slice(comptime!(units as usize));
        worst_s[unit as usize] = worst;
        sync_cube();
        let stride = RuntimeCell::<u32>::new(units / 2u32);
        while stride.read() > 0u32 {
            let s = stride.read();
            if unit < s {
                let other = worst_s[(unit + s) as usize];
                worst_s[unit as usize] = F::max(worst_s[unit as usize], other);
            }
            sync_cube();
            stride.store(s / 2u32);
        }
        if unit == 0u32 {
            ratio[b as usize] = worst_s[0];
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Shared path
// ---------------------------------------------------------------------------------------------

/// Every sweep of one matrix per cube, in shared memory: rotations, row stage and column stage of a
/// round are separated by cube barriers, and the convergence test (the largest [`fn@offdiag_ratio`]
/// at most `tolerance`) is a cube reduction. `a` is overwritten with the converged (diagonal)
/// matrix, `v` with the eigenvectors. `n` is comptime so the shared buffers are sized at compile
/// time.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn jacobi_shared_kernel<F: Float + CubeElement + LaunchArg>(
    a: &mut [F],
    v: &mut [F],
    batch: u32,
    tolerance: F,
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

        let mut sa = Shared::<[F]>::new_slice(comptime!(nn as usize));
        let mut sv = Shared::<[F]>::new_slice(comptime!(nn as usize));
        let mut tmp = Shared::<[F]>::new_slice(comptime!(nn as usize));
        let mut partner = Shared::<[u32]>::new_slice(comptime!(n as usize));
        let mut coef_self = Shared::<[F]>::new_slice(comptime!(n as usize));
        let mut coef_other = Shared::<[F]>::new_slice(comptime!(n as usize));
        let mut worst_s = Shared::<[F]>::new_slice(comptime!(units as usize));

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

        let mut sweep = 0u32;
        let mut done = 0u32;
        // Bounded: at most `max_sweeps + 1` convergence tests
        while sweep <= max_sweeps {
            if done == 0u32 {
                // Convergence: the largest relative off-diagonal entry
                let mut worst = F::new(0.0f32);
                let mut e = unit;
                while e < nn {
                    let (i, j) = (e / n, e % n);
                    if i != j {
                        worst = F::max(worst, offdiag_ratio::<F>(sa[e as usize], sa[(i * n + i) as usize], sa[(j * n + j) as usize]));
                    }
                    e += units;
                }
                worst_s[unit as usize] = worst;
                sync_cube();
                let stride = RuntimeCell::<u32>::new(units / 2u32);
                while stride.read() > 0u32 {
                    let s = stride.read();
                    if unit < s {
                        let other = worst_s[(unit + s) as usize];
                        worst_s[unit as usize] = F::max(worst_s[unit as usize], other);
                    }
                    sync_cube();
                    stride.store(s / 2u32);
                }
                // Every unit reads the same reduced value: the decision is uniform
                if worst_s[0] <= tolerance || sweep == max_sweeps {
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
                                let (c, s) = rotation::<F>(sa[(p * n + p) as usize], sa[(q * n + q) as usize], sa[(p * n + q) as usize], tolerance);
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
                        // Column stage: A = tmp J (only rotated pairs are zeroed)
                        let mut e = unit;
                        while e < nn {
                            let (i, j) = (e / n, e % n);
                            let pj = partner[j as usize];
                            let co = coef_other[j as usize];
                            let mut value = coef_self[j as usize] * tmp[e as usize] + co * tmp[(i * n + pj) as usize];
                            if pj == i && i != j && co != F::new(0.0f32) {
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
                }
            }
            sweep += 1u32;
        }

        let mut e = unit;
        while e < nn {
            a[base + e as usize] = sa[e as usize];
            v[base + e as usize] = sv[e as usize];
            e += units;
        }
    }
}
