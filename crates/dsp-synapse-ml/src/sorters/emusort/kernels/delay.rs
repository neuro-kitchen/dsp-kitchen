use cubecl::prelude::*;
use dsp_core::compute::{channel_position, sample_position};

/// `output[ch, t] = input[ch, (t + shift[ch]) mod samples]`, `shift[ch] ≥ 0`.
#[cube(launch)]
pub fn apply_channel_delays_kernel<F: Float>(
    input: &[F],
    output: &mut [F],
    shifts: &[u32],
    channels: u32,
    samples: u32,
) {
    let t = sample_position();
    let ch = channel_position();
    if ch < channels && t < samples {
        // shift < samples, so one subtraction wraps
        let mut src_t = t + shifts[ch as usize];
        if src_t >= samples {
            src_t -= samples;
        }
        output[(ch * samples + t) as usize] = input[(ch * samples + src_t) as usize];
    }
}

/// `env[ch, t] = |x[ch, t]| / std[ch]` (`0` for a flat channel).
#[cube(launch)]
pub fn delay_envelope_kernel<F: Float>(input: &[F], std: &[F], env: &mut [F], channels: u32, samples: u32) {
    let t = sample_position();
    let ch = channel_position();
    if ch < channels && t < samples {
        let sd = std[ch as usize];
        let idx = (ch * samples + t) as usize;
        if sd > F::new(0.0f32) {
            env[idx] = F::abs(input[idx]) / sd;
        } else {
            env[idx] = F::new(0.0f32);
        }
    }
}

/// Partial lagged products of one channel pair over one tile of the batch interior, one cube per
/// `(pair, tile)` (`CUBE_POS_X` = tile, `CUBE_POS_Y` = pair `a · channels + b`), one unit per lag
/// `li` (`lag = li − max_lag`): `partial[(pair · tiles + tile) · lags + li] = Σ_t env[a, t_a] ·
/// env[b, col_start + t]` over the tile's samples `t`, with `t_a = col_start + t + max_lag − li`
/// clamped to the row (lagged reads past the buffer repeat the edge sample, as upstream pads its
/// first and last batches). The cube loads channel
/// `b`'s tile and channel `a`'s tile with its lag margin into shared memory once; every lag then
/// reads them there.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn delay_cc_tile_kernel<F: Float>(
    env: &[F],
    partial: &mut [F],
    channels: u32,
    row_len: u32,
    col_start: u32,
    samples: u32,
    max_lag: u32,
    tiles: u32,
    #[comptime] tile: u32,
    #[comptime] margin: u32,
    #[comptime] units: u32,
) {
    let (t_index, pair) = (CUBE_POS_X, CUBE_POS_Y);
    let (a, b) = (pair / channels, pair % channels);
    let t0 = t_index * tile;
    let count = u32::min(tile, samples - t0);
    let mut env_b = Shared::<[F]>::new_slice(comptime!(tile as usize));
    let mut env_a = Shared::<[F]>::new_slice(comptime!((tile + 2 * margin) as usize));
    let lags = 2u32 * max_lag + 1u32;

    // Channel b: interior samples t0..t0 + count
    let mut k = UNIT_POS_X;
    while k < count {
        env_b[k as usize] = env[(b * row_len + col_start + t0 + k) as usize];
        k += units;
    }
    // Channel a: positions col_start + t0 − max_lag + k (k < count + 2 · max_lag), clamped to the row
    let mut k = UNIT_POS_X;
    while k < count + 2u32 * max_lag {
        let pos = col_start + t0 + k;
        let mut ta = 0u32;
        if pos >= max_lag {
            ta = u32::min(pos - max_lag, row_len - 1u32);
        }
        env_a[k as usize] = env[(a * row_len + ta) as usize];
        k += units;
    }
    sync_cube();

    let li = UNIT_POS_X;
    if li < lags {
        // Sample t of the tile pairs env_b[t] with env_a at offset t + (2 · max_lag − li)
        let shift = 2u32 * max_lag - li;
        let mut acc = F::new(0.0f32);
        let mut t = 0u32;
        while t < count {
            acc += env_a[(t + shift) as usize] * env_b[t as usize];
            t += 1u32;
        }
        partial[((pair * tiles + t_index) * lags + li) as usize] = acc;
    }
}

/// `cc[q] += Σ_tile partial[(pair · tiles + tile) · lags + li] / samples` for `q = pair · lags +
/// li` (tiles summed in order). One unit per `q`.
#[cube(launch)]
pub fn delay_cc_accumulate_kernel<F: Float>(partial: &[F], cc: &mut [F], triples: u32, lags: u32, tiles: u32, samples: u32) {
    let q = ABSOLUTE_POS as u32;
    if q < triples {
        let (pair, li) = (q / lags, q % lags);
        let mut acc = F::new(0.0f32);
        let mut tile = 0u32;
        while tile < tiles {
            acc += partial[((pair * tiles + tile) * lags + li) as usize];
            tile += 1u32;
        }
        cc[q as usize] += acc / F::cast_from(u32::max(samples, 1u32));
    }
}
