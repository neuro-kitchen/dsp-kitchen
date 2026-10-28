//! Kernels of the learned-template preparation (see `kilosort4/learned.rs`). Templates are in PC
//! space: `a[(u · C + ch) · np + p]` (`U` units, `C` channels, `np` PCs); lags run `−(nt−1) ..= nt−1`
//! (`L = 2·nt − 1` of them, index `lag + nt − 1`).

use cubecl::prelude::*;

/// Alignment scores: `out[(u · K + k) · L + l] = max_ch |Σ_p a[u, ch, p] · proto[p, k, l]|`, where
/// `proto[p, k, l] = Σ_t wpca[p, t] · wtemp[k, t − lag]` (unit `u`'s match with prototype `k` at that
/// lag on its best channel). One unit per `(u, k, l)`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn prototype_match_kernel<F: Float>(a: &[F], proto: &[F], out: &mut [F], units: u32, channels: u32, np: u32, k: u32, lags: u32) {
    let unit = ABSOLUTE_POS as u32;
    if unit < units * k * lags {
        let u = unit / (k * lags);
        let rest = unit - u * k * lags;
        let kk = rest / lags;
        let l = rest - kk * lags;
        let mut best = F::new(0.0f32);
        let mut ch = 0u32;
        while ch < channels {
            let mut acc = F::new(0.0f32);
            let mut p = 0u32;
            while p < np {
                acc += a[((u * channels + ch) * np + p) as usize] * proto[((p * k + kk) * lags + l) as usize];
                p += 1u32;
            }
            best = F::max(best, F::abs(acc));
            ch += 1u32;
        }
        out[unit as usize] = best;
    }
}

/// Best lagged correlation of every pair of templates: `out[a · U + b] = max_lag Σ_{p,q}
/// utu[(a·np + p), (b·np + q)] · wtw[p, q, lag]`, with `utu = X·Xᵀ` (`X[(u·np + p), ch] = a[u, ch, p]`)
/// and `wtw[p, q, lag] = Σ_t wpca[p, t] · wpca[q, t + lag]`: the inner product of the two waveforms
/// with `b` shifted by `lag`. The diagonal holds each template's squared norm (lag 0 is its maximum).
/// One unit per `(a, b)`.
#[cube(launch)]
pub fn pair_similarity_kernel<F: Float>(utu: &[F], wtw: &[F], out: &mut [F], units: u32, np: u32, lags: u32) {
    let unit = ABSOLUTE_POS as u32;
    if unit < units * units {
        let a = unit / units;
        let b = unit - a * units;
        let stride = units * np;
        let mut best = F::min_value();
        let mut l = 0u32;
        while l < lags {
            let mut acc = F::new(0.0f32);
            let mut p = 0u32;
            while p < np {
                let mut q = 0u32;
                while q < np {
                    acc += utu[((a * np + p) * stride + b * np + q) as usize] * wtw[((p * np + q) * lags + l) as usize];
                    q += 1u32;
                }
                p += 1u32;
            }
            best = F::max(best, acc);
            l += 1u32;
        }
        out[unit as usize] = best;
    }
}
