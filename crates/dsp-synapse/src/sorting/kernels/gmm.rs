//! Gaussian-mixture EM kernels: the E-step per spike, the M-step sums per component.
//! Shapes: features `[n, d]`, means `[k, d]`, precisions `[k, d, d]`, responsibilities `[n, k]`.

use cubecl::prelude::*;

/// One unit per spike `i`: `lp_c = log_norms[c] − ½ (x − μ_c)ᵀ P_c (x − μ_c)` for every component,
/// responsibilities `exp(lp_c − logsumexp lp)` into `resp[i, ·]`, `log_lik[i] = logsumexp lp`, and
/// the squared Mahalanobis distance to the most likely component into `mahalanobis_sq[i]`.
/// `log_norms[c] = ln π_c − ½ (d ln 2π + ln det Σ_c)`.
#[cube(launch)]
pub fn gmm_e_step_kernel<F: Float>(
    features: &Array<F>,
    means: &Array<F>,
    precisions: &Array<F>,
    log_norms: &Array<F>,
    resp: &mut Array<F>,
    log_lik: &mut Array<F>,
    mahalanobis_sq: &mut Array<F>,
    n: u32,
    d: u32,
    k: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        let x = i * d;
        let mut best = F::new(f32::NEG_INFINITY);
        let mut best_q = F::new(0.0f32);
        let mut c: u32 = 0u32;
        while c < k {
            let m = c * d;
            let p = c * d * d;
            let mut q = F::new(0.0f32);
            let mut r: u32 = 0u32;
            while r < d {
                let dr = features[(x + r) as usize] - means[(m + r) as usize];
                let mut inner = F::new(0.0f32);
                let mut col: u32 = 0u32;
                while col < d {
                    let dc = features[(x + col) as usize] - means[(m + col) as usize];
                    inner += precisions[(p + r * d + col) as usize] * dc;
                    col += 1u32;
                }
                q += dr * inner;
                r += 1u32;
            }
            q = F::max(q, F::new(0.0f32));
            let lp = log_norms[c as usize] - F::new(0.5f32) * q;
            resp[(i * k + c) as usize] = lp;
            if lp > best {
                best = lp;
                best_q = q;
            }
            c += 1u32;
        }
        let mut sum = F::new(0.0f32);
        let mut c: u32 = 0u32;
        while c < k {
            let e = F::exp(resp[(i * k + c) as usize] - best);
            resp[(i * k + c) as usize] = e;
            sum += e;
            c += 1u32;
        }
        let mut c: u32 = 0u32;
        while c < k {
            let v = resp[(i * k + c) as usize];
            resp[(i * k + c) as usize] = v / sum;
            c += 1u32;
        }
        log_lik[i as usize] = best + F::ln(sum);
        mahalanobis_sq[i as usize] = best_q;
    }
}

/// One unit per `(component c, feature f)`: `Σᵢ rᵢc mᵢf xᵢf` into `weighted_sum[c, f]` and
/// `Σᵢ rᵢc mᵢf` into `mask_sum[c, f]` (`m = mask[i, f]` clamped to `[0, 1]` when `masked`, else 1),
/// and, from the `f = 0` unit, `Σᵢ rᵢc` into `resp_sum[c]`.
#[cube(launch)]
pub fn gmm_mean_sums_kernel<F: Float>(
    features: &Array<F>,
    resp: &Array<F>,
    mask: &Array<F>,
    weighted_sum: &mut Array<F>,
    mask_sum: &mut Array<F>,
    resp_sum: &mut Array<F>,
    n: u32,
    d: u32,
    k: u32,
    masked: u32,
) {
    let unit = ABSOLUTE_POS as u32;
    if unit < k * d {
        let c = unit / d;
        let f = unit - c * d;
        let mut s = F::new(0.0f32);
        let mut sm = F::new(0.0f32);
        let mut sr = F::new(0.0f32);
        let mut i: u32 = 0u32;
        while i < n {
            let r = resp[(i * k + c) as usize];
            let mut m = F::new(1.0f32);
            if masked != 0u32 {
                m = F::clamp(mask[(i * d + f) as usize], F::new(0.0f32), F::new(1.0f32));
            }
            s += r * m * features[(i * d + f) as usize];
            sm += r * m;
            sr += r;
            i += 1u32;
        }
        weighted_sum[unit as usize] = s;
        mask_sum[unit as usize] = sm;
        if f == 0u32 {
            resp_sum[c as usize] = sr;
        }
    }
}

/// One unit per `(component c, row r, column col)`: `Σᵢ rᵢc (xᵢr − μ_cr)(xᵢcol − μ_ccol)` into
/// `scatter[c, r, col]` (off-diagonal entries 0 when `diagonal`).
#[cube(launch)]
pub fn gmm_scatter_kernel<F: Float>(
    features: &Array<F>,
    resp: &Array<F>,
    means: &Array<F>,
    scatter: &mut Array<F>,
    n: u32,
    d: u32,
    k: u32,
    diagonal: u32,
) {
    let unit = ABSOLUTE_POS as u32;
    if unit < k * d * d {
        let c = unit / (d * d);
        let rem = unit - c * d * d;
        let r = rem / d;
        let col = rem - r * d;
        let mut s = F::new(0.0f32);
        if diagonal == 0u32 || r == col {
            let (mr, mc) = (means[(c * d + r) as usize], means[(c * d + col) as usize]);
            let mut i: u32 = 0u32;
            while i < n {
                let w = resp[(i * k + c) as usize];
                s += w * (features[(i * d + r) as usize] - mr) * (features[(i * d + col) as usize] - mc);
                i += 1u32;
            }
        }
        scatter[unit as usize] = s;
    }
}
