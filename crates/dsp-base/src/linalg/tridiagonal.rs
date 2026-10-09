//! Symmetric eigendecomposition on the host in `f64`: Householder reduction to tridiagonal form,
//! then the implicit QL algorithm (EISPACK `tred2` / `tql2`, as in JAMA's public-domain
//! `EigenvalueDecomposition`). `O(n³)` with a small constant: the right tool for the mid-sized
//! matrices of subspace methods (tens to a few hundred rows), where the device Jacobi solver
//! ([`fn@super::symmetric_eigen`]) spends its time on launches.

use super::eigen::SymmetricEigen;

/// Eigenvalues (largest first) and unit eigenvectors of the symmetric row-major `[n, n]` matrix
/// `a`, on the host.
///
/// # Panics
///
/// If `a.len() != n²`.
pub fn symmetric_eigen_cpu(a: &[f64], n: usize) -> SymmetricEigen {
    assert_eq!(a.len(), n * n, "matrix must be [n, n]");
    if n == 0 {
        return SymmetricEigen { n, values: Vec::new(), vectors: Vec::new() };
    }
    let mut v = a.to_vec();
    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n];
    tred2(&mut v, &mut d, &mut e, n);
    tql2(&mut v, &mut d, &mut e, n);
    // Descending order, vectors as columns
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| d[j].total_cmp(&d[i]));
    let values = order.iter().map(|&i| d[i]).collect();
    let mut vectors = vec![0.0; n * n];
    for (c, &src) in order.iter().enumerate() {
        for r in 0..n {
            vectors[r * n + c] = v[r * n + src];
        }
    }
    SymmetricEigen { n, values, vectors }
}

/// Householder reduction of `v` (in: the matrix; out: the accumulated transform) to the
/// tridiagonal `d` (diagonal) and `e` (sub-diagonal, `e[0] = 0`).
fn tred2(v: &mut [f64], d: &mut [f64], e: &mut [f64], n: usize) {
    let at = |r: usize, c: usize| r * n + c;
    for j in 0..n {
        d[j] = v[at(n - 1, j)];
    }
    for i in (1..n).rev() {
        let mut scale = 0.0;
        let mut h = 0.0;
        for k in 0..i {
            scale += d[k].abs();
        }
        if scale == 0.0 {
            e[i] = d[i - 1];
            for j in 0..i {
                d[j] = v[at(i - 1, j)];
                v[at(i, j)] = 0.0;
                v[at(j, i)] = 0.0;
            }
        } else {
            for k in 0..i {
                d[k] /= scale;
                h += d[k] * d[k];
            }
            let mut f = d[i - 1];
            let mut g = h.sqrt();
            if f > 0.0 {
                g = -g;
            }
            e[i] = scale * g;
            h -= f * g;
            d[i - 1] = f - g;
            for ej in e.iter_mut().take(i) {
                *ej = 0.0;
            }
            for j in 0..i {
                f = d[j];
                v[at(j, i)] = f;
                g = e[j] + v[at(j, j)] * f;
                for k in j + 1..i {
                    g += v[at(k, j)] * d[k];
                    e[k] += v[at(k, j)] * f;
                }
                e[j] = g;
            }
            f = 0.0;
            for j in 0..i {
                e[j] /= h;
                f += e[j] * d[j];
            }
            let hh = f / (h + h);
            for j in 0..i {
                e[j] -= hh * d[j];
            }
            for j in 0..i {
                f = d[j];
                g = e[j];
                for k in j..i {
                    v[at(k, j)] -= f * e[k] + g * d[k];
                }
                d[j] = v[at(i - 1, j)];
                v[at(i, j)] = 0.0;
            }
        }
        d[i] = h;
    }
    // Accumulate the transformations
    for i in 0..n - 1 {
        v[at(n - 1, i)] = v[at(i, i)];
        v[at(i, i)] = 1.0;
        let h = d[i + 1];
        if h != 0.0 {
            for k in 0..=i {
                d[k] = v[at(k, i + 1)] / h;
            }
            for j in 0..=i {
                let mut g = 0.0;
                for k in 0..=i {
                    g += v[at(k, i + 1)] * v[at(k, j)];
                }
                for k in 0..=i {
                    v[at(k, j)] -= g * d[k];
                }
            }
        }
        for k in 0..=i {
            v[at(k, i + 1)] = 0.0;
        }
    }
    for j in 0..n {
        d[j] = v[at(n - 1, j)];
        v[at(n - 1, j)] = 0.0;
    }
    v[at(n - 1, n - 1)] = 1.0;
    e[0] = 0.0;
}

/// Implicit QL iterations on the tridiagonal (`d`, `e`), rotating the eigenvectors in `v`.
fn tql2(v: &mut [f64], d: &mut [f64], e: &mut [f64], n: usize) {
    let at = |r: usize, c: usize| r * n + c;
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;
    let mut f = 0.0;
    let mut tst1: f64 = 0.0;
    let eps = f64::EPSILON;
    for l in 0..n {
        tst1 = tst1.max(d[l].abs() + e[l].abs());
        let mut m = l;
        while m < n {
            if e[m].abs() <= eps * tst1 {
                break;
            }
            m += 1;
        }
        let m = m.min(n - 1);
        if m > l {
            // Bounded: a NaN input must not spin forever
            for _ in 0..(60 * n).max(60) {
                let g = d[l];
                let mut p = (d[l + 1] - g) / (2.0 * e[l]);
                let mut r = p.hypot(1.0);
                if p < 0.0 {
                    r = -r;
                }
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                let dl1 = d[l + 1];
                let mut h = g - d[l];
                for di in d.iter_mut().take(n).skip(l + 2) {
                    *di -= h;
                }
                f += h;
                p = d[m];
                let mut c = 1.0;
                let mut c2 = c;
                let mut c3 = c;
                let el1 = e[l + 1];
                let mut s = 0.0;
                let mut s2 = 0.0;
                for i in (l..m).rev() {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    let g = c * e[i];
                    h = c * p;
                    r = p.hypot(e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    for k in 0..n {
                        let hk = v[at(k, i + 1)];
                        v[at(k, i + 1)] = s * v[at(k, i)] + c * hk;
                        v[at(k, i)] = c * v[at(k, i)] - s * hk;
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;
                if e[l].abs() <= eps * tst1 {
                    break;
                }
            }
        }
        d[l] += f;
        e[l] = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(a: &[f64], n: usize) {
        let eig = symmetric_eigen_cpu(a, n);
        assert!(eig.values.windows(2).all(|w| w[0] >= w[1]), "descending: {:?}", eig.values);
        for c in 0..n {
            // A v = λ v, |v| = 1
            let v: Vec<f64> = (0..n).map(|r| eig.vectors[r * n + c]).collect();
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-10, "column {c}: |v| = {norm}");
            for r in 0..n {
                let av: f64 = (0..n).map(|k| a[r * n + k] * v[k]).sum();
                assert!((av - eig.values[c] * v[r]).abs() < 1e-9 * (1.0 + eig.values[0].abs()), "column {c}, row {r}");
            }
        }
    }

    #[test]
    fn decomposes_symmetric_matrices() {
        check(&[2.0, 1.0, 1.0, 2.0], 2);
        let eig = symmetric_eigen_cpu(&[2.0, 1.0, 1.0, 2.0], 2);
        assert!((eig.values[0] - 3.0).abs() < 1e-12 && (eig.values[1] - 1.0).abs() < 1e-12);
        check(&[5.0], 1);
        // Diagonal, repeated eigenvalues, a rank-deficient Gram matrix, a random one
        check(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 3.0], 3);
        let n = 40;
        let mut state = 7u64;
        let mut rnd = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let x: Vec<f64> = (0..n * 5).map(|_| rnd()).collect();
        let gram: Vec<f64> = (0..n * n).map(|e| (0..5).map(|k| x[(e / n) * 5 + k] * x[(e % n) * 5 + k]).sum()).collect();
        check(&gram, n);
        let m: Vec<f64> = (0..n * n).map(|_| rnd()).collect();
        let sym: Vec<f64> = (0..n * n).map(|e| m[e] + m[(e % n) * n + e / n]).collect();
        check(&sym, n);
    }
}
