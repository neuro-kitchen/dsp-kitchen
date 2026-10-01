use super::svd::SymmetricEig;

/// Contrast function $G(u)$ and its derivatives $g(u) = G'(u)$, $g'(u) = G''(u)$ for FastICA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IcaContrast {
    /// $g(u) = \tanh(u)$, $g'(u) = 1 - \tanh^2(u)$ — general-purpose robust non-Gaussianity.
    LogCosh,
    /// $g(u) = u^3$, $g'(u) = 3u^2$ — kurtosis-based super-Gaussian spiky sources.
    Cube,
    /// $g(u) = u^2$, $g'(u) = 2u$ — asymmetric skewed action potential / MUAP sources (HD-EMG cBSS).
    Skew,
}

impl IcaContrast {
    #[inline]
    pub fn eval(self, u: f32) -> (f32, f32) {
        match self {
            IcaContrast::LogCosh => {
                let th = u.tanh();
                (th, 1.0 - th * th)
            }
            IcaContrast::Cube => (u * u * u, 3.0 * u * u),
            IcaContrast::Skew => (u * u, 2.0 * u),
        }
    }
}

/// Fitted Fast Independent Component Analysis (FastICA) model.
#[derive(Debug, Clone)]
pub struct FastIcaModel {
    pub num_channels: usize,
    pub num_components: usize,
    pub mean: Vec<f32>,
    /// Unmixing matrix $\mathbf{U} = \mathbf{W}_{\text{ICA}} \mathbf{K}_{\text{whiten}}$ of shape
    /// `[num_components, num_channels]` (row-major) such that $\mathbf{S} = \mathbf{U}(\mathbf{X} - \boldsymbol{\mu})$.
    pub unmixing: Vec<f32>,
}

impl FastIcaModel {
    /// Fits deflationary fixed-point FastICA on `data` (`[channels, samples]`).
    pub fn fit(
        data: &[f32],
        channels: usize,
        samples: usize,
        mut num_components: usize,
        contrast: IcaContrast,
        max_iters: usize,
        tol: f32,
    ) -> Self {
        assert_eq!(data.len(), channels * samples, "Data size mismatch");
        assert!(channels > 0 && samples > 0);
        num_components = num_components.clamp(1, channels);

        let inv_s = 1.0 / (samples as f32);
        let mut mean = vec![0.0f32; channels];
        for c in 0..channels {
            mean[c] = data[c * samples..(c + 1) * samples].iter().sum::<f32>() * inv_s;
        }

        // 1. Center and compute PCA whitening matrix K of shape [num_components, channels]
        let mut cov = vec![0.0f32; channels * channels];
        for i in 0..channels {
            let ri = &data[i * samples..(i + 1) * samples];
            let mi = mean[i];
            for j in i..channels {
                let rj = &data[j * samples..(j + 1) * samples];
                let mj = mean[j];
                let mut acc = 0.0f64;
                for t in 0..samples {
                    acc += ((ri[t] - mi) as f64) * ((rj[t] - mj) as f64);
                }
                let val = (acc * (inv_s as f64)) as f32;
                cov[i * channels + j] = val;
                cov[j * channels + i] = val;
            }
        }

        let eig = SymmetricEig::decompose(&cov, channels, 120);
        let mut k_whiten = vec![0.0f32; num_components * channels];
        for comp in 0..num_components {
            let scale = 1.0 / (eig.eigenvalues[comp].max(1e-8)).sqrt();
            for c in 0..channels {
                k_whiten[comp * channels + c] = eig.eigenvectors[c * channels + comp] * scale;
            }
        }

        // Project centered data into whitened subspace Z of shape [num_components, samples]
        let mut z = vec![0.0f32; num_components * samples];
        for comp in 0..num_components {
            let z_row = &mut z[comp * samples..(comp + 1) * samples];
            for c in 0..channels {
                let kw = k_whiten[comp * channels + c];
                let mc = mean[c];
                let x_row = &data[c * samples..(c + 1) * samples];
                for t in 0..samples {
                    z_row[t] += kw * (x_row[t] - mc);
                }
            }
        }

        // 2. Deflationary fixed-point FastICA in whitened space (W of shape [num_components, num_components])
        let mut w_ica = vec![0.0f32; num_components * num_components];
        for p in 0..num_components {
            let mut wp = vec![0.0f32; num_components];
            wp[p] = 1.0;

            for _iter in 0..max_iters.max(1) {
                let mut wp_new = vec![0.0f32; num_components];
                let mut mean_gp = 0.0f32;

                for t in 0..samples {
                    let mut u = 0.0f32;
                    for d in 0..num_components {
                        u += wp[d] * z[d * samples + t];
                    }
                    let (g_u, gp_u) = contrast.eval(u);
                    mean_gp += gp_u;
                    for d in 0..num_components {
                        wp_new[d] += z[d * samples + t] * g_u;
                    }
                }

                mean_gp *= inv_s;
                for d in 0..num_components {
                    wp_new[d] = wp_new[d] * inv_s - mean_gp * wp[d];
                }

                // Gram-Schmidt deflation against previously converged components 0..p
                for j in 0..p {
                    let mut dot = 0.0f32;
                    for d in 0..num_components {
                        dot += wp_new[d] * w_ica[j * num_components + d];
                    }
                    for d in 0..num_components {
                        wp_new[d] -= dot * w_ica[j * num_components + d];
                    }
                }

                let norm = wp_new.iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-12);
                for v in &mut wp_new {
                    *v /= norm;
                }

                let cos_sim: f32 = wp.iter().zip(&wp_new).map(|(a, b)| a * b).sum::<f32>().abs();
                wp = wp_new;
                if (1.0 - cos_sim) < tol {
                    break;
                }
            }

            w_ica[p * num_components..(p + 1) * num_components].copy_from_slice(&wp);
        }

        // 3. Combine unmixing U = W_ica * K_whiten of shape [num_components, channels]
        let mut unmixing = vec![0.0f32; num_components * channels];
        for p in 0..num_components {
            for c in 0..channels {
                let mut sum = 0.0f32;
                for d in 0..num_components {
                    sum += w_ica[p * num_components + d] * k_whiten[d * channels + c];
                }
                unmixing[p * channels + c] = sum;
            }
        }

        Self {
            num_channels: channels,
            num_components,
            mean,
            unmixing,
        }
    }

    /// Applies the learned unmixing matrix to `data` (`[channels, samples]`) -> `[num_components, samples]`.
    pub fn transform_cpu(&self, data: &[f32], channels: usize, samples: usize) -> Vec<f32> {
        assert_eq!(channels, self.num_channels);
        assert_eq!(data.len(), channels * samples);
        let mut s = vec![0.0f32; self.num_components * samples];
        for p in 0..self.num_components {
            let s_row = &mut s[p * samples..(p + 1) * samples];
            for c in 0..channels {
                let u_pc = self.unmixing[p * channels + c];
                let mc = self.mean[c];
                let x_row = &data[c * samples..(c + 1) * samples];
                for t in 0..samples {
                    s_row[t] += u_pc * (x_row[t] - mc);
                }
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fastica_separates_two_supergaussian_sources() {
        let samples = 2000;
        let mut s1 = vec![0.0f32; samples];
        let mut s2 = vec![0.0f32; samples];
        for t in 0..samples {
            // Two independent sparse impulse-like signals
            if t % 37 == 5 {
                s1[t] = 6.0;
            }
            if t % 53 == 11 {
                s2[t] = 6.0;
            }
        }

        // Mix with 2x2 mixing matrix A = [[0.8, 0.6], [-0.5, 0.9]]
        let mut x = vec![0.0f32; 2 * samples];
        for t in 0..samples {
            x[t] = 0.8 * s1[t] + 0.6 * s2[t];
            x[samples + t] = -0.5 * s1[t] + 0.9 * s2[t];
        }

        let ica = FastIcaModel::fit(&x, 2, samples, 2, IcaContrast::Cube, 100, 1e-5);
        let y = ica.transform_cpu(&x, 2, samples);

        let corr = |a: &[f32], b: &[f32]| -> f32 {
            let ma = a.iter().sum::<f32>() / (a.len() as f32);
            let mb = b.iter().sum::<f32>() / (b.len() as f32);
            let mut num = 0.0f32;
            let mut da = 0.0f32;
            let mut db = 0.0f32;
            for i in 0..a.len() {
                let va = a[i] - ma;
                let vb = b[i] - mb;
                num += va * vb;
                da += va * va;
                db += vb * vb;
            }
            (num / (da.sqrt() * db.sqrt()).max(1e-12)).abs()
        };

        let y0 = &y[..samples];
        let y1 = &y[samples..];
        let c00 = corr(y0, &s1);
        let c01 = corr(y0, &s2);
        let c10 = corr(y1, &s1);
        let c11 = corr(y1, &s2);

        let best1 = c00.max(c10);
        let best2 = c01.max(c11);
        assert!(best1 > 0.95, "best1={best1}");
        assert!(best2 > 0.95, "best2={best2}");
    }
}
