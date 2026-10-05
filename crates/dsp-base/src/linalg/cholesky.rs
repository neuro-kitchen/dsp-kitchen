//! Host Cholesky factorization of small symmetric positive-definite matrices (`O(n³)`; for the
//! `n ≲ 100` systems of model fitting, e.g. Gaussian mixture covariances and kriging kernels).
//! Exact: no iteration.

/// Lower-triangular `L` with `A = L Lᵀ` (row-major `[n, n]`, upper triangle zero), or `None` when
/// `A` is not positive definite. Reads the lower triangle of `a` only.
pub fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    assert_eq!(a.len(), n * n, "matrix size mismatch");
    let mut l = vec![0.0f64; n * n];
    for j in 0..n {
        let mut diag = a[j * n + j];
        for k in 0..j {
            diag -= l[j * n + k] * l[j * n + k];
        }
        if !(diag > 0.0) || !diag.is_finite() {
            return None;
        }
        let ljj = diag.sqrt();
        l[j * n + j] = ljj;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = s / ljj;
        }
    }
    Some(l)
}

/// `A⁻¹` (row-major `[n, n]`, symmetric) and `ln det A` of a symmetric positive-definite `A`
/// through its Cholesky factor (`ln det A = 2 Σ ln Lᵢᵢ`), or `None` when `A` is not positive
/// definite.
pub fn spd_inverse_logdet(a: &[f64], n: usize) -> Option<(Vec<f64>, f64)> {
    let l = cholesky(a, n)?;
    let log_det = 2.0 * (0..n).map(|i| l[i * n + i].ln()).sum::<f64>();
    // W = L⁻¹ (lower triangular) by forward substitution, then A⁻¹ = Wᵀ W
    let mut w = vec![0.0f64; n * n];
    for c in 0..n {
        w[c * n + c] = 1.0 / l[c * n + c];
        for i in c + 1..n {
            let s: f64 = (c..i).map(|k| l[i * n + k] * w[k * n + c]).sum();
            w[i * n + c] = -s / l[i * n + i];
        }
    }
    let mut inv = vec![0.0f64; n * n];
    for i in 0..n {
        for j in i..n {
            let v: f64 = (j..n).map(|k| w[k * n + i] * w[k * n + j]).sum();
            inv[i * n + j] = v;
            inv[j * n + i] = v;
        }
    }
    Some((inv, log_det))
}

/// `X` with `A X = B` for symmetric positive-definite `A` (`[n, n]`) and `B` (`[n, m]`, row-major),
/// by Cholesky and two triangular solves, or `None` when `A` is not positive definite.
pub fn cholesky_solve(a: &[f64], b: &[f64], n: usize, m: usize) -> Option<Vec<f64>> {
    assert_eq!(b.len(), n * m, "right-hand side size mismatch");
    let l = cholesky(a, n)?;
    let mut x = b.to_vec();
    // L Y = B (forward), then Lᵀ X = Y (backward), column by column of B
    for c in 0..m {
        for i in 0..n {
            let s: f64 = (0..i).map(|k| l[i * n + k] * x[k * m + c]).sum();
            x[i * m + c] = (x[i * m + c] - s) / l[i * n + i];
        }
        for i in (0..n).rev() {
            let s: f64 = (i + 1..n).map(|k| l[k * n + i] * x[k * m + c]).sum();
            x[i * m + c] = (x[i * m + c] - s) / l[i * n + i];
        }
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_and_log_determinant() {
        // A = [[4, 2, 0.4], [2, 5, 1], [0.4, 1, 3]]
        let a = [4.0, 2.0, 0.4, 2.0, 5.0, 1.0, 0.4, 1.0, 3.0];
        let (inv, log_det) = spd_inverse_logdet(&a, 3).unwrap();
        let det = 4.0 * (5.0 * 3.0 - 1.0) - 2.0 * (2.0 * 3.0 - 0.4) + 0.4 * (2.0 * 1.0 - 5.0 * 0.4);
        assert!((log_det - f64::ln(det)).abs() < 1e-12);
        for i in 0..3 {
            for j in 0..3 {
                let p: f64 = (0..3).map(|k| a[i * 3 + k] * inv[k * 3 + j]).sum();
                assert!((p - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12, "A·A⁻¹ at ({i}, {j}) = {p}");
            }
        }
    }

    #[test]
    fn solves_several_right_hand_sides() {
        let a = [4.0, 2.0, 0.4, 2.0, 5.0, 1.0, 0.4, 1.0, 3.0];
        let x_true = [1.0, -2.0, 0.5, 3.0, -1.0, 2.0];
        let b: Vec<f64> = (0..3).flat_map(|i| (0..2).map(move |c| (0..3).map(|k| a[i * 3 + k] * x_true[k * 2 + c]).sum::<f64>())).collect();
        let x = cholesky_solve(&a, &b, 3, 2).unwrap();
        for (got, want) in x.iter().zip(x_true) {
            assert!((got - want).abs() < 1e-12);
        }
    }

    #[test]
    fn rejects_indefinite() {
        assert!(cholesky(&[1.0, 2.0, 2.0, 1.0], 2).is_none());
        assert!(cholesky(&[0.0], 1).is_none());
    }
}
