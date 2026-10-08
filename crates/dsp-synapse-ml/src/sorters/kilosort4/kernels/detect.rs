//! Device kernels of Kilosort4 universal-template detection (see `detect.rs` for the steps).
//! Shapes: data `[channels, samples]`; templates `[n_templates, nt]`; per-centre channel table
//! `iC[c, centre]` (`[n_chans, n_centres]`); weights `[n_sizes, n_chans, n_centres]`; neighbour
//! table `iC2[j, centre]` (`[n_neighbours, n_centres]`).

use cubecl::prelude::*;
use dsp_core::compute::{channel_position, sample_position};

/// `B[ch, k, t] = Σ_j x[ch, t − nt/2 + j] · w[k, j]` (zero outside the row), summed over `j` in
/// order: every channel correlated with every template, a filter bank. One unit per `(ch, t)`
/// ([`fn@sample_position`], [`fn@channel_position`]) computing all `n_templates` outputs.
///
/// A cube shares its `tile_x` samples (plus `nt − 1` of context) of its `tile_y` channels and the
/// `[n_templates, nt]` templates through shared memory, loaded once; each sample read from the
/// tile then serves every template from registers. `tile_x` / `tile_y` are the cube's size.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn correlate_templates_kernel<F: Float>(
    x: &[F],
    templates: &[F],
    b: &mut [F],
    channels: u32,
    samples: u32,
    #[comptime] n_templates: u32,
    #[comptime] nt: u32,
    #[comptime] tile_x: u32,
    #[comptime] tile_y: u32,
) {
    let span = comptime!(tile_x + nt - 1);
    let half = comptime!(nt / 2);
    let mut xs = Shared::<[F]>::new_slice(comptime!((span * tile_y) as usize));
    let mut ws = Shared::<[F]>::new_slice(comptime!((n_templates * nt) as usize));
    let t = sample_position();
    let ch = channel_position();
    let t0 = t - UNIT_POS_X;

    // Templates, loaded by the cube's units together
    let unit = UNIT_POS_Y * tile_x + UNIT_POS_X;
    let mut i = unit;
    while i < comptime!(n_templates * nt) {
        ws[i as usize] = templates[i as usize];
        i += comptime!(tile_x * tile_y);
    }
    // This unit's channel row: samples t0 − half + k for k < span, zero outside the row
    let row = UNIT_POS_Y * span;
    let mut k = UNIT_POS_X;
    while k < span {
        let mut v = F::new(0.0f32);
        let pos = t0 + k;
        if ch < channels && pos >= half && pos - half < samples {
            v = x[(ch * samples + pos - half) as usize];
        }
        xs[(row + k) as usize] = v;
        k += tile_x;
    }
    sync_cube();

    if ch < channels && t < samples {
        let mut acc = Array::<F>::new(comptime!(n_templates as usize));
        #[unroll]
        for tk in 0..n_templates {
            acc[tk as usize] = F::new(0.0f32);
        }
        let mut j = 0u32;
        while j < nt {
            let xv = xs[(row + UNIT_POS_X + j) as usize];
            #[unroll]
            for tk in 0..n_templates {
                acc[tk as usize] += xv * ws[(tk * nt + j) as usize];
            }
            j += 1u32;
        }
        #[unroll]
        for tk in 0..n_templates {
            b[((ch * n_templates + tk) * samples + t) as usize] = acc[tk as usize];
        }
    }
}

/// `As[centre, t] = max_{s,k} |Σ_c w[s, c, centre] · B[iC[c, centre], k, t]|`, and in `arg` the
/// signed `1 + s · n_templates + k` of the maximum (sign of the response; the first maximum in
/// `(s, k)` order). One unit per `(centre, t)`.
///
/// Per unit this is a small product, `R[s, k] = Σ_c W[s, c] · B[c, k]` (`n_sizes × n_chans` by
/// `n_chans × n_templates`): each `B[c, k]` serves every size. The sizes are comptime, so the
/// loops unroll and the `n_chans · n_templates` values of `B` and the channel indices are loaded
/// once into registers (instead of once per size); sums keep the order `c = 0, 1, …`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn centre_response_kernel<F: Float>(
    b: &[F],
    weights: &[F],
    ic: &[u32],
    a_s: &mut [F],
    arg: &mut [i32],
    samples: u32,
    n_centres: u32,
    #[comptime] n_chans: u32,
    #[comptime] n_sizes: u32,
    #[comptime] n_templates: u32,
) {
    let t = sample_position();
    let centre = channel_position();
    if centre < n_centres && t < samples {
        let mut chan = Array::<u32>::new(comptime!(n_chans as usize));
        #[unroll]
        for c in 0..n_chans {
            chan[c as usize] = ic[(c * n_centres + centre) as usize];
        }
        let mut resp = Array::<F>::new(comptime!((n_sizes * n_templates) as usize));
        #[unroll]
        for k in 0..n_templates {
            let mut bv = Array::<F>::new(comptime!(n_chans as usize));
            #[unroll]
            for c in 0..n_chans {
                bv[c as usize] = b[((chan[c as usize] * n_templates + k) * samples + t) as usize];
            }
            #[unroll]
            for s in 0..n_sizes {
                let mut acc = F::new(0.0f32);
                #[unroll]
                for c in 0..n_chans {
                    acc += weights[((s * n_chans + c) * n_centres + centre) as usize] * bv[c as usize];
                }
                resp[(s * n_templates + k) as usize] = acc;
            }
        }
        let mut best = F::new(-1.0f32);
        let mut best_arg = 0i32;
        #[unroll]
        for s in 0..n_sizes {
            #[unroll]
            for k in 0..n_templates {
                let acc = resp[(s * n_templates + k) as usize];
                let mag = F::abs(acc);
                if mag > best {
                    best = mag;
                    let id = i32::cast_from(1u32 + s * n_templates + k);
                    best_arg = id;
                    if acc < F::new(0.0f32) {
                        best_arg = 0i32 - id;
                    }
                }
            }
        }
        a_s[(centre * samples + t) as usize] = best;
        arg[(centre * samples + t) as usize] = best_arg;
    }
}

/// `Amax[centre, t] = max_j As[iC2[j, centre], t]`, zero in the first / last `nt` samples.
///
/// Neighbouring centres share most of their neighbours, so the centres are taken in blocks of
/// `tile_y` (one cube per block × `tile_x` samples): the cube loads the **union** of its block's
/// neighbour rows (`union_rows[block, ..union_len[block]]`, at most `u_max`) for its samples into
/// shared memory once, and each unit takes its maximum there through `local_nb[j, centre]` (the
/// neighbour's position in its block's union). A maximum does not depend on order: the result is
/// exactly the gather over `iC2`, with a few global reads per output instead of `n_neighbours`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn neighbour_max_kernel<F: Float>(
    a_s: &[F],
    union_rows: &[u32],
    union_len: &[u32],
    local_nb: &[u32],
    a_max: &mut [F],
    samples: u32,
    n_centres: u32,
    n_neighbours: u32,
    nt: u32,
    #[comptime] tile_x: u32,
    #[comptime] tile_y: u32,
    #[comptime] u_max: u32,
) {
    let block = CUBE_POS_Y;
    let t = CUBE_POS_X * tile_x + UNIT_POS_X;
    let centre = block * tile_y + UNIT_POS_Y;
    let mut rows = Shared::<[F]>::new_slice(comptime!((u_max * tile_x) as usize));
    let count = union_len[block as usize];
    let mut r = UNIT_POS_Y;
    while r < count {
        let mut v = F::new(0.0f32);
        if t < samples {
            v = a_s[(union_rows[(block * u_max + r) as usize] * samples + t) as usize];
        }
        rows[(r * tile_x + UNIT_POS_X) as usize] = v;
        r += tile_y;
    }
    sync_cube();
    if centre < n_centres && t < samples {
        let mut m = F::new(0.0f32);
        if t >= nt && t + nt < samples {
            let mut j = 0u32;
            while j < n_neighbours {
                let local = local_nb[(j * n_centres + centre) as usize];
                m = F::max(m, rows[(local * tile_x + UNIT_POS_X) as usize]);
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
    a_s: &[F],
    a_max: &[F],
    score: &mut [F],
    samples: u32,
    n_centres: u32,
    pool: u32,
) {
    let t = sample_position();
    let centre = channel_position();
    if centre < n_centres && t < samples {
        let row = centre * samples;
        let mut pooled = F::new(0.0f32);
        let mut u = 0u32;
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
    x: &[F],
    b: &[F],
    wpca: &[F],
    ic: &[u32],
    arg: &[i32],
    centres: &[u32],
    times: &[u32],
    picked: &mut [i32],
    feat: &mut [F],
    amp: &mut [F],
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
        let mut p = 0u32;
        while p < n_pcs {
            let mut acc = F::new(0.0f32);
            let mut j = 0u32;
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

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_base::core::buffer;
    use dsp_core::compute::{ComputeTarget, LaunchGeometry};

    /// `correlate_templates_kernel` against the host in `f64` (edges zero-padded), odd sizes.
    #[test]
    fn correlate_templates_matches_host() {
        let Ok(target) = ComputeTarget::from_env() else { return };
        let client = target.client().expect("client");
        let (channels, samples, n_templates, nt) = (5usize, 301usize, 3usize, 21usize);
        let mut state = 0x2468_ace1u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let x: Vec<f32> = (0..channels * samples).map(|_| next()).collect();
        let w: Vec<f32> = (0..n_templates * nt).map(|_| next()).collect();
        let out = buffer::empty::<f32>(&client, channels * n_templates * samples);
        let geom = LaunchGeometry::channels_samples(&client, channels, samples);
        unsafe {
            correlate_templates_kernel::launch::<f32>(
                &client,
                geom.cube_count,
                geom.cube_dim.clone(),
                BufferArg::from_raw_parts(buffer::upload(&client, &x), x.len()),
                BufferArg::from_raw_parts(buffer::upload(&client, &w), w.len()),
                BufferArg::from_raw_parts(out.clone(), channels * n_templates * samples),
                channels as u32,
                samples as u32,
                n_templates as u32,
                nt as u32,
                geom.cube_dim.x,
                geom.cube_dim.y,
            );
        }
        let got = buffer::download::<f32>(&client, out);
        let half = nt / 2;
        for ch in 0..channels {
            for k in 0..n_templates {
                for t in 0..samples {
                    let want: f64 = (0..nt)
                        .filter(|&j| t + j >= half && t + j - half < samples)
                        .map(|j| x[ch * samples + t + j - half] as f64 * w[k * nt + j] as f64)
                        .sum();
                    let g = got[(ch * n_templates + k) * samples + t] as f64;
                    assert!((g - want).abs() < 1e-5, "ch {ch} k {k} t {t}: {g} vs {want}");
                }
            }
        }
    }

    /// `centre_response_kernel` against the host in `f64`: responses and the signed arg-max.
    #[test]
    fn centre_response_matches_host() {
        let Ok(target) = ComputeTarget::from_env() else { return };
        let client = target.client().expect("client");
        let (channels, samples, n_chans, n_sizes, n_templates, n_centres) = (7usize, 53usize, 3usize, 2usize, 3usize, 5usize);
        let mut state = 0x1357_9bdfu32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let b: Vec<f32> = (0..channels * n_templates * samples).map(|_| next()).collect();
        let weights: Vec<f32> = (0..n_sizes * n_chans * n_centres).map(|_| next()).collect();
        let ic: Vec<u32> = (0..n_chans * n_centres).map(|i| ((i * 5 + 1) % channels) as u32).collect();

        let (a_s, arg) = (buffer::empty::<f32>(&client, n_centres * samples), buffer::empty::<i32>(&client, n_centres * samples));
        let geom = LaunchGeometry::channels_samples(&client, n_centres, samples);
        unsafe {
            centre_response_kernel::launch::<f32>(
                &client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(buffer::upload(&client, &b), b.len()),
                BufferArg::from_raw_parts(buffer::upload(&client, &weights), weights.len()),
                BufferArg::from_raw_parts(buffer::upload(&client, &ic), ic.len()),
                BufferArg::from_raw_parts(a_s.clone(), n_centres * samples),
                BufferArg::from_raw_parts(arg.clone(), n_centres * samples),
                samples as u32,
                n_centres as u32,
                n_chans as u32,
                n_sizes as u32,
                n_templates as u32,
            );
        }
        let (got_a, got_arg) = (buffer::download::<f32>(&client, a_s), buffer::download::<i32>(&client, arg));
        for centre in 0..n_centres {
            for t in 0..samples {
                let (mut best, mut best_arg) = (-1.0f64, 0i32);
                for s in 0..n_sizes {
                    for k in 0..n_templates {
                        let acc: f64 = (0..n_chans)
                            .map(|c| {
                                let ch = ic[c * n_centres + centre] as usize;
                                weights[(s * n_chans + c) * n_centres + centre] as f64 * b[(ch * n_templates + k) * samples + t] as f64
                            })
                            .sum();
                        if acc.abs() > best {
                            best = acc.abs();
                            let id = (1 + s * n_templates + k) as i32;
                            best_arg = if acc < 0.0 { -id } else { id };
                        }
                    }
                }
                let e = centre * samples + t;
                assert!((got_a[e] as f64 - best).abs() < 1e-5, "centre {centre} t {t}: {} vs {best}", got_a[e]);
                assert_eq!(got_arg[e], best_arg, "centre {centre} t {t}");
            }
        }
    }
}
