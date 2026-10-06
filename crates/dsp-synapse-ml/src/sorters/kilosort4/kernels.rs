//! Device kernels of Kilosort4 universal-template detection (see `detect.rs` for the steps).
//! Shapes: data `[channels, samples]`; templates `[n_templates, nt]`; per-centre channel table
//! `iC[c, centre]` (`[n_chans, n_centres]`); weights `[n_sizes, n_chans, n_centres]`; neighbour
//! table `iC2[j, centre]` (`[n_neighbours, n_centres]`).

use cubecl::prelude::*;
use dsp_core::compute::{channel_position, sample_position};

/// `B[ch, k, t] = Σ_j x[ch, t − nt/2 + j] · w[k, j]` (zero outside the row). One unit per
/// `(row = ch · n_templates + k, t)`.
#[cube(launch)]
pub fn correlate_templates_kernel<F: Float>(
    x: &Array<F>,
    templates: &Array<F>,
    b: &mut Array<F>,
    channels: u32,
    samples: u32,
    n_templates: u32,
    nt: u32,
) {
    let t = sample_position();
    let row = channel_position();
    if row < channels * n_templates && t < samples {
        let ch = row / n_templates;
        let k = row - ch * n_templates;
        let half = nt / 2u32;
        let mut acc = F::new(0.0f32);
        let mut j: u32 = 0u32;
        while j < nt {
            if t + j >= half && t + j - half < samples {
                acc += x[(ch * samples + t + j - half) as usize] * templates[(k * nt + j) as usize];
            }
            j += 1u32;
        }
        b[(row * samples + t) as usize] = acc;
    }
}

/// `As[centre, t] = max_{s,k} |Σ_c w[s, c, centre] · B[iC[c, centre], k, t]|`, and in `arg` the
/// signed `1 + s · n_templates + k` of the maximum (sign of the response). One unit per
/// `(centre, t)`.
#[cube(launch)]
pub fn centre_response_kernel<F: Float>(
    b: &Array<F>,
    weights: &Array<F>,
    ic: &Array<u32>,
    a_s: &mut Array<F>,
    arg: &mut Array<i32>,
    samples: u32,
    n_templates: u32,
    n_centres: u32,
    n_chans: u32,
    n_sizes: u32,
) {
    let t = sample_position();
    let centre = channel_position();
    if centre < n_centres && t < samples {
        let mut best = F::new(-1.0f32);
        let mut best_arg: i32 = 0i32;
        let mut s: u32 = 0u32;
        while s < n_sizes {
            let mut k: u32 = 0u32;
            while k < n_templates {
                let mut acc = F::new(0.0f32);
                let mut c: u32 = 0u32;
                while c < n_chans {
                    let ch = ic[(c * n_centres + centre) as usize];
                    acc += weights[((s * n_chans + c) * n_centres + centre) as usize]
                        * b[((ch * n_templates + k) * samples + t) as usize];
                    c += 1u32;
                }
                let mag = F::abs(acc);
                if mag > best {
                    best = mag;
                    let id = i32::cast_from(1u32 + s * n_templates + k);
                    best_arg = id;
                    if acc < F::new(0.0f32) {
                        best_arg = 0i32 - id;
                    }
                }
                k += 1u32;
            }
            s += 1u32;
        }
        a_s[(centre * samples + t) as usize] = best;
        arg[(centre * samples + t) as usize] = best_arg;
    }
}

/// `Amax[centre, t] = max_j As[iC2[j, centre], t]`, zero in the first / last `nt` samples. One
/// unit per `(centre, t)`.
#[cube(launch)]
pub fn neighbour_max_kernel<F: Float>(
    a_s: &Array<F>,
    ic2: &Array<u32>,
    a_max: &mut Array<F>,
    samples: u32,
    n_centres: u32,
    n_neighbours: u32,
    nt: u32,
) {
    let t = sample_position();
    let centre = channel_position();
    if centre < n_centres && t < samples {
        let mut m = F::new(0.0f32);
        if t >= nt && t + nt < samples {
            let mut j: u32 = 0u32;
            while j < n_neighbours {
                let other = ic2[(j * n_centres + centre) as usize];
                m = F::max(m, a_s[(other * samples + t) as usize]);
                j += 1u32;
            }
        }
        a_max[(centre * samples + t) as usize] = m;
    }
}

/// `score[centre, t] = As[centre, t]` where it equals the maximum of `Amax[centre, ·]` over
/// `t ± pool` (a local maximum over neighbouring centres and time), else 0. One unit per
/// `(centre, t)`.
#[cube(launch)]
pub fn local_peak_score_kernel<F: Float>(
    a_s: &Array<F>,
    a_max: &Array<F>,
    score: &mut Array<F>,
    samples: u32,
    n_centres: u32,
    pool: u32,
) {
    let t = sample_position();
    let centre = channel_position();
    if centre < n_centres && t < samples {
        let row = centre * samples;
        let mut pooled = F::new(0.0f32);
        let mut u: u32 = 0u32;
        if t >= pool {
            u = t - pool;
        }
        let mut end = t + pool;
        if end >= samples {
            end = samples - 1u32;
        }
        while u <= end {
            pooled = F::max(pooled, a_max[(row + u) as usize]);
            u += 1u32;
        }
        let v = a_s[(row + t) as usize];
        let mut out = F::new(0.0f32);
        if v == pooled && v > F::new(0.0f32) {
            out = v;
        }
        score[(row + t) as usize] = out;
    }
}

/// Per spike `i` (centre `centres[i]`, sample `times[i]`) and centre channel `c`: the signed
/// arg-max `picked[i] = arg[centre, t]` (written by `c = 0`), the template it encodes
/// (`(|a| − 1) mod n_templates`), features `feat[i, c, p] = Σ_j x[iC[c, centre], t − nt/2 + j] ·
/// wpca[p, j]` and the template response `amp[i, c] = B[iC[c, centre], template, t]`. One unit per
/// `(i, c)`; the spikes come straight from the device candidate lists.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn spike_features_kernel<F: Float>(
    x: &Array<F>,
    b: &Array<F>,
    wpca: &Array<F>,
    ic: &Array<u32>,
    arg: &Array<i32>,
    centres: &Array<u32>,
    times: &Array<u32>,
    picked: &mut Array<i32>,
    feat: &mut Array<F>,
    amp: &mut Array<F>,
    samples: u32,
    n_templates: u32,
    n_centres: u32,
    n_chans: u32,
    n_pcs: u32,
    nt: u32,
    n_spikes: u32,
) {
    let unit = ABSOLUTE_POS as u32;
    if unit < n_spikes * n_chans {
        let i = unit / n_chans;
        let c = unit - i * n_chans;
        let centre = centres[i as usize];
        let t = times[i as usize];
        let a = arg[(centre * samples + t) as usize];
        if c == 0u32 {
            picked[i as usize] = a;
        }
        let mut magnitude = u32::cast_from(a);
        if a < 0i32 {
            magnitude = u32::cast_from(-a);
        }
        let mut template = 0u32;
        if magnitude > 0u32 {
            template = (magnitude - 1u32) % n_templates;
        }
        let ch = ic[(c * n_centres + centre) as usize];
        let half = nt / 2u32;
        let mut p: u32 = 0u32;
        while p < n_pcs {
            let mut acc = F::new(0.0f32);
            let mut j: u32 = 0u32;
            while j < nt {
                if t + j >= half && t + j - half < samples {
                    acc += x[(ch * samples + t + j - half) as usize] * wpca[(p * nt + j) as usize];
                }
                j += 1u32;
            }
            feat[((i * n_chans + c) * n_pcs + p) as usize] = acc;
            p += 1u32;
        }
        amp[(i * n_chans + c) as usize] = b[((ch * n_templates + template) * samples + t) as usize];
    }
}
