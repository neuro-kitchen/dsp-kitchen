/// Symmetric Jacobi Eigenvalue Decomposition.
/// Computes all eigenvalues and eigenvectors for a real symmetric matrix A of size [N, N].
/// Pure-Rust, zero-dependency, numerically stable algorithm.
pub struct SymmetricEig {
    pub eigenvalues: Vec<f32>,
    /// Column-major or row-major eigenvectors matrix of size [N, N].
    /// Vector k is stored at indices [row * n + k].
    pub eigenvectors: Vec<f32>,
    pub n: usize,
}

impl SymmetricEig {
    /// Computes eigendecomposition of a symmetric N x N matrix.
    /// `a` is a flat slice of length N * N in row-major order.
    pub fn decompose(a: &[f32], n: usize, max_iterations: usize) -> Self {
        assert_eq!(a.len(), n * n, "Matrix size mismatch");

        if n == 0 {
            return Self {
                eigenvalues: Vec::new(),
                eigenvectors: Vec::new(),
                n: 0,
            };
        }

        if n == 1 {
            return Self {
                eigenvalues: vec![a[0]],
                eigenvectors: vec![1.0],
                n: 1,
            };
        }

        let mut a_mat = a.to_vec();
        // Initialize V as identity matrix of size N x N
        let mut v = vec![0.0f32; n * n];
        for i in 0..n {
            v[i * n + i] = 1.0;
        }

        let eps = 1e-7f32;

        for _iter in 0..max_iterations {
            // Find max off-diagonal element
            let mut max_offdiag = 0.0f32;
            let mut p = 0;
            let mut q = 1;

            for i in 0..n {
                for j in (i + 1)..n {
                    let val = a_mat[i * n + j].abs();
                    if val > max_offdiag {
                        max_offdiag = val;
                        p = i;
                        q = j;
                    }
                }
            }

            if max_offdiag < eps {
                break;
            }

            // Compute Jacobi rotation angle
            let app = a_mat[p * n + p];
            let aqq = a_mat[q * n + q];
            let apq = a_mat[p * n + q];

            let theta = (aqq - app) / (2.0 * apq);
            let t = if theta >= 0.0 {
                1.0 / (theta + (1.0 + theta * theta).sqrt())
            } else {
                -1.0 / (-theta + (1.0 + theta * theta).sqrt())
            };

            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;
            let tau = s / (1.0 + c);

            // Update matrix A
            a_mat[p * n + p] = app - t * apq;
            a_mat[q * n + q] = aqq + t * apq;
            a_mat[p * n + q] = 0.0;
            a_mat[q * n + p] = 0.0;

            for r in 0..n {
                if r != p && r != q {
                    let arp = a_mat[r * n + p];
                    let arq = a_mat[r * n + q];
                    a_mat[r * n + p] = arp - s * (arq + tau * arp);
                    a_mat[p * n + r] = a_mat[r * n + p];
                    a_mat[r * n + q] = arq + s * (arp - tau * arq);
                    a_mat[q * n + r] = a_mat[r * n + q];
                }
            }

            // Update eigenvectors V
            for r in 0..n {
                let vrp = v[r * n + p];
                let vrq = v[r * n + q];
                v[r * n + p] = vrp - s * (vrq + tau * vrp);
                v[r * n + q] = vrq + s * (vrp - tau * vrq);
            }
        }

        // Extract eigenvalues from diagonal
        let mut eigenvalues = Vec::with_capacity(n);
        for i in 0..n {
            eigenvalues.push(a_mat[i * n + i]);
        }

        // Sort eigenvalues and corresponding eigenvectors in descending order
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap_or(std::cmp::Ordering::Equal));

        let mut sorted_evals = Vec::with_capacity(n);
        let mut sorted_evecs = vec![0.0f32; n * n];

        for (new_col, &old_col) in order.iter().enumerate() {
            sorted_evals.push(eigenvalues[old_col]);
            for row in 0..n {
                sorted_evecs[row * n + new_col] = v[row * n + old_col];
            }
        }

        Self {
            eigenvalues: sorted_evals,
            eigenvectors: sorted_evecs,
            n,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric_eig_2x2() {
        // Symmetric matrix:
        // [ 2, 1 ]
        // [ 1, 2 ]
        // Eigenvalues are 3 and 1
        let a = vec![2.0, 1.0, 1.0, 2.0];
        let eig = SymmetricEig::decompose(&a, 2, 50);

        assert_eq!(eig.n, 2);
        assert!((eig.eigenvalues[0] - 3.0).abs() < 1e-4);
        assert!((eig.eigenvalues[1] - 1.0).abs() < 1e-4);
    }
}
