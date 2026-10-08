//! Learned-template matching kernels (see `kilosort4/matching.rs`). Scores are `s[j · T + t] =
//! ŵ_jᵀ D(t)`: unit-norm template `j` against the data window starting at `t − nt/2`. Lags of the
//! template products run `−(nt−1) ..= nt−1` (`L = 2·nt − 1`, index `lag + nt − 1`).

use cubecl::prelude::*;
use dsp_base::core::DspFloat;

/// Best template per sample by explained variance at its average norm, `V = 2·μ·c − μ²` (`c` the
/// score): `vmax[t]`, `best[t]` (template) and `amp[t] = c`. One unit per sample.
#[cube(launch)]
pub fn best_template_kernel<F: Float>(s: &[F], mu: &[F], vmax: &mut [F], best: &mut [u32], amp: &mut [F], n: u32, samples: u32) {
    let t = ABSOLUTE_POS as u32;
    if t < samples {
        let mut bv = F::min_value();
        let mut bj = 0u32;
        let mut bc = F::new(0.0f32);
        let mut j = 0u32;
        while j < n {
            let c = s[(j * samples + t) as usize];
            let m = mu[j as usize];
            let v = F::new(2.0f32) * m * c - m * m;
            if v > bv {
                bv = v;
                bj = j;
                bc = c;
            }
            j += 1u32;
        }
        vmax[t as usize] = bv;
        best[t as usize] = bj;
        amp[t as usize] = bc;
    }
}

/// Spike marks: `peak[t] = vmax[t]` where `t` (in `lo..hi`) has `amp[t] ≥ th`, `vmax[t] > 0` and is
/// the maximum of `vmax` over `t ± pool` (on ties the earliest), else 0. One unit per sample.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn mark_peaks_kernel<F: DspFloat>(vmax: &[F], amp: &[F], peak: &mut [F], samples: u32, pool: u32, lo: u32, hi: u32, th: F) {
    let t = ABSOLUTE_POS as u32;
    if t < samples {
        let v = vmax[t as usize];
        let mut keep = t >= lo && t < hi && amp[t as usize] >= th && v > F::new(0.0f32);
        if keep {
            let mut u = 0u32;
            if t >= pool {
                u = t - pool;
            }
            let end = u32::min(t + pool + 1u32, samples);
            while u < end {
                let w = vmax[u as usize];
                // Strictly larger anywhere, or equal earlier: not this sample's peak
                if w > v || (w == v && u < t) {
                    keep = false;
                }
                u += 1u32;
            }
        }
        let mut out = F::new(0.0f32);
        if keep {
            out = v;
        }
        peak[t as usize] = out;
    }
}

/// Removes the matched spikes of one class (`⌊t/(pool+1)⌋ mod 3 == class`, so their ranges do not
/// overlap) from the scores: `s[j, t + lag] −= amp_t · ctc[i, j, −lag]` (`i` the spike's template: a
/// spike at `t` changes `ŵ_jᵀ D(t + lag)` by `amp · Σ_τ ŵ_j[τ] ŵ_i[τ + lag]`).
/// One unit per `(spike, j, lag)`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn subtract_scores_kernel<F: Float>(
    s: &mut [F],
    ctc: &[F],
    times: &[u32],
    best: &[u32],
    amp: &[F],
    spikes: u32,
    n: u32,
    samples: u32,
    lags: u32,
    pool: u32,
    class: u32,
) {
    let unit = ABSOLUTE_POS as u32;
    if unit < spikes * n * lags {
        let k = unit / (n * lags);
        let rest = unit - k * n * lags;
        let j = rest / lags;
        let l = rest - j * lags;
        let t = times[k as usize];
        if (t / (pool + 1u32)) % 3u32 == class {
            let half = (lags - 1u32) / 2u32;
            // target sample t + lag, lag = l − half
            if t + l >= half && t + l - half < samples {
                let i = best[t as usize];
                let target = (j * samples + t + l - half) as usize;
                s[target] -= amp[t as usize] * ctc[((i * n + j) * lags + (lags - 1u32 - l)) as usize];
            }
        }
    }
}

/// Removes the matched spikes of one class from the data: `x[ch, t − nt/2 + τ] −= amp_t · ŵ_i[ch, τ]`.
/// One unit per `(spike, ch, τ)`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn subtract_data_kernel<F: Float>(
    x: &mut [F],
    w: &[F],
    times: &[u32],
    best: &[u32],
    amp: &[F],
    spikes: u32,
    channels: u32,
    samples: u32,
    nt: u32,
    pool: u32,
    class: u32,
) {
    let unit = ABSOLUTE_POS as u32;
    if unit < spikes * channels * nt {
        let k = unit / (channels * nt);
        let rest = unit - k * channels * nt;
        let ch = rest / nt;
        let tau = rest - ch * nt;
        let t = times[k as usize];
        if (t / (pool + 1u32)) % 3u32 == class {
            let half = nt / 2u32;
            if t + tau >= half && t + tau - half < samples {
                let i = best[t as usize];
                x[(ch * samples + t + tau - half) as usize] -= amp[t as usize] * w[((i * channels + ch) * nt + tau) as usize];
            }
        }
    }
}

/// Spikes of a round: `out_best[k] = best[times[k]]`, `out_amp[k] = amp[times[k]]`. One unit per spike.
#[cube(launch)]
pub fn gather_spikes_kernel<F: Float>(times: &[u32], best: &[u32], amp: &[F], out_best: &mut [u32], out_amp: &mut [F], spikes: u32) {
    let k = ABSOLUTE_POS as u32;
    if k < spikes {
        let t = times[k as usize];
        out_best[k as usize] = best[t as usize];
        out_amp[k as usize] = amp[t as usize];
    }
}

/// Template products at every lag: `ctc[i, j, l] = Σ_{p,q} utu[(i·np + p), (j·np + q)] · wtw[p, q, l]`
/// (`= Σ_τ ŵ_i[τ] · ŵ_j[τ + lag]` over all channels). One unit per `(i, j, l)`.
#[cube(launch)]
pub fn template_products_kernel<F: Float>(utu: &[F], wtw: &[F], ctc: &mut [F], n: u32, np: u32, lags: u32) {
    let unit = ABSOLUTE_POS as u32;
    if unit < n * n * lags {
        let i = unit / (n * lags);
        let rest = unit - i * n * lags;
        let j = rest / lags;
        let l = rest - j * lags;
        let stride = n * np;
        let mut acc = F::new(0.0f32);
        let mut p = 0u32;
        while p < np {
            let mut q = 0u32;
            while q < np {
                acc += utu[((i * np + p) * stride + j * np + q) as usize] * wtw[((p * np + q) * lags + l) as usize];
                q += 1u32;
            }
            p += 1u32;
        }
        ctc[unit as usize] = acc;
    }
}

/// `dst[i] = src[i]`. One unit per value.
#[cube(launch)]
pub fn copy_kernel<F: Float>(src: &[F], dst: &mut [F], n: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        dst[i as usize] = src[i as usize];
    }
}

/// Features of the matched spikes: residual PCs at the spike plus its own template's contribution,
/// `out[k, c, p] = b[ch, p, t_k] + amp_k · u[i_k, ch, p]` on the spike's `nc` channels
/// `chans[k · nc + c]` (`b`: `[channels, np, samples]` PC projections of the residual; `u`: unit-norm
/// template features). One unit per `(k, c, p)`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gather_features_kernel<F: Float>(
    b: &[F],
    u: &[F],
    times: &[u32],
    best: &[u32],
    amp: &[F],
    chans: &[u32],
    out: &mut [F],
    spikes: u32,
    nc: u32,
    np: u32,
    channels: u32,
    samples: u32,
) {
    let unit = ABSOLUTE_POS as u32;
    if unit < spikes * nc * np {
        let k = unit / (nc * np);
        let rest = unit - k * nc * np;
        let c = rest / np;
        let p = rest - c * np;
        let ch = chans[(k * nc + c) as usize];
        let t = times[k as usize];
        let i = best[k as usize];
        out[unit as usize] = b[((ch * np + p) * samples + t) as usize] + amp[k as usize] * u[((i * channels + ch) * np + p) as usize];
    }
}
