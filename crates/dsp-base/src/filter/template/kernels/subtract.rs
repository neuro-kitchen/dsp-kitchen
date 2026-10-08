use cubecl::prelude::*;
use dsp_core::compute::negative_infinity;

/// Joint cross-correlation of the template with the signal for every `(event, lag)` of a layer:
/// `scores[e · lags + l] = Σ_c Σ_t x[c, s + t] · T[c, t]` with `s = starts[e] + l − max_lag`, or `−∞`
/// when the window leaves the signal. One unit per `(event, lag)`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn template_scores_kernel<F: Float>(
    signal: &[F],
    template: &[F],
    starts: &[i32],
    scores: &mut [F],
    channels: u32,
    samples: u32,
    template_len: u32,
    num_events: u32,
    lags: u32,
    max_lag: u32,
) {
    let unit = ABSOLUTE_POS as u32;
    if unit < num_events * lags {
        let e = unit / lags;
        let l = unit - e * lags;
        let s = starts[e as usize] + i32::cast_from(l) - i32::cast_from(max_lag);
        let mut score = negative_infinity::<F>();
        if s >= 0i32 && s + i32::cast_from(template_len) <= i32::cast_from(samples) {
            let s = u32::cast_from(s);
            score = F::new(0.0f32);
            let mut c = 0u32;
            while c < channels {
                let (sig, tpl) = ((c * samples + s) as usize, (c * template_len) as usize);
                let mut t = 0u32;
                while t < template_len {
                    score += signal[sig + t as usize] * template[tpl + t as usize];
                    t += 1u32;
                }
                c += 1u32;
            }
        }
        scores[unit as usize] = score;
    }
}

/// Subtracts the aligned, scaled template of every event of a layer from one channel: the best lag
/// is the first maximum of the event's scores (as the host search), `α = max(⟨x, T_c⟩ / ‖T_c‖², 0)`
/// with dynamic scaling, else 1. Channels whose template energy is below `min_energy` are skipped.
/// One unit per `(event, channel)`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn template_subtract_kernel<F: Float + CubeElement + LaunchArg>(
    signal: &mut [F],
    template: &[F],
    energies: &[F],
    starts: &[i32],
    scores: &[F],
    channels: u32,
    samples: u32,
    template_len: u32,
    num_events: u32,
    lags: u32,
    max_lag: u32,
    dynamic_scaling: u32,
    min_energy: F,
) {
    let unit = ABSOLUTE_POS as u32;
    if unit < num_events * channels {
        let e = unit / channels;
        let c = unit - e * channels;

        let mut best = 0u32;
        let mut best_score = negative_infinity::<F>();
        let mut l = 0u32;
        while l < lags {
            let score = scores[(e * lags + l) as usize];
            if score > best_score {
                best_score = score;
                best = l;
            }
            l += 1u32;
        }
        let energy = energies[c as usize];
        if best_score > negative_infinity::<F>() && energy > min_energy {
            let s = u32::cast_from(starts[e as usize] + i32::cast_from(best) - i32::cast_from(max_lag));
            let (sig, tpl) = ((c * samples + s) as usize, (c * template_len) as usize);
            let mut alpha = F::new(1.0f32);
            if dynamic_scaling != 0u32 {
                let mut dot = F::new(0.0f32);
                let mut t = 0u32;
                while t < template_len {
                    dot += signal[sig + t as usize] * template[tpl + t as usize];
                    t += 1u32;
                }
                alpha = F::max(dot / energy, F::new(0.0f32));
            }
            let mut t = 0u32;
            while t < template_len {
                signal[sig + t as usize] -= alpha * template[tpl + t as usize];
                t += 1u32;
            }
        }
    }
}
