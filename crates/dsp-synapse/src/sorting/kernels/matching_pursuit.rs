use cubecl::prelude::*;

/// Scores every template start against the multi-channel residual for one matching-pursuit pass.
///
/// One unit per start sample `s` (`ABSOLUTE_POS`). For each unit `u` it correlates the residual
/// window `[s, s + t_len)` with the template's rows (`row_offsets[u]..row_offsets[u + 1]` into
/// `row_channels` / `row_data`), fits the amplitude `a = dot / ‖W_u‖²` and, when `a` lies in
/// `[min_scale, max_scale]`, the energy reduction `2·a·dot − a²·‖W_u‖²`. Writes the best unit,
/// amplitude and gain per start (gain 0 when nothing fits).
#[cube(launch)]
pub fn mp_score_kernel(
    residual: &Array<f32>,
    row_offsets: &Array<u32>,
    row_channels: &Array<u32>,
    row_data: &Array<f32>,
    energies: &Array<f32>,
    best_unit: &mut Array<u32>,
    best_scale: &mut Array<f32>,
    best_gain: &mut Array<f32>,
    num_samples: u32,
    num_starts: u32,
    num_units: u32,
    t_len: u32,
    min_scale: f32,
    max_scale: f32,
) {
    let s = ABSOLUTE_POS as u32;
    if s < num_starts {
        let mut gain_best = 0.0f32;
        let mut unit_best = 0u32;
        let mut scale_best = 0.0f32;
        let mut u = 0u32;
        while u < num_units {
            let mut dot = 0.0f32;
            let mut k = row_offsets[u as usize];
            let k_end = row_offsets[(u + 1u32) as usize];
            while k < k_end {
                let base = row_channels[k as usize] * num_samples + s;
                let tbase = k * t_len;
                let mut i = 0u32;
                while i < t_len {
                    dot += residual[(base + i) as usize] * row_data[(tbase + i) as usize];
                    i += 1u32;
                }
                k += 1u32;
            }
            let energy = energies[u as usize];
            let a = dot / energy;
            if a >= min_scale && a <= max_scale {
                let gain = 2.0f32 * a * dot - a * a * energy;
                if gain > gain_best {
                    gain_best = gain;
                    unit_best = u;
                    scale_best = a;
                }
            }
            u += 1u32;
        }
        best_unit[s as usize] = unit_best;
        best_scale[s as usize] = scale_best;
        best_gain[s as usize] = gain_best;
    }
}

/// Subtracts the picked, scaled templates from the residual. One unit per
/// `(pick, row slot, sample)` (`ABSOLUTE_POS`, `max_rows` slots per pick; slots beyond a unit's
/// row count do nothing). Picks of one pass are at least `t_len` apart, so writes never overlap.
#[cube(launch)]
pub fn mp_subtract_kernel(
    residual: &mut Array<f32>,
    row_offsets: &Array<u32>,
    row_channels: &Array<u32>,
    row_data: &Array<f32>,
    pick_units: &Array<u32>,
    pick_starts: &Array<u32>,
    pick_scales: &Array<f32>,
    num_samples: u32,
    num_picks: u32,
    max_rows: u32,
    t_len: u32,
) {
    let id = ABSOLUTE_POS as u32;
    let per_pick = max_rows * t_len;
    if id < num_picks * per_pick {
        let p = id / per_pick;
        let rem = id - p * per_pick;
        let slot = rem / t_len;
        let i = rem - slot * t_len;
        let u = pick_units[p as usize];
        let k = row_offsets[u as usize] + slot;
        if k < row_offsets[(u + 1u32) as usize] {
            let at = row_channels[k as usize] * num_samples + pick_starts[p as usize] + i;
            residual[at as usize] -= pick_scales[p as usize] * row_data[(k * t_len + i) as usize];
        }
    }
}

/// One unit per pick: copies the best unit and amplitude scale at start `starts[i]` and the energy
/// reductions at `starts[i] ± 1` (`starts` are interior: `1 ≤ s < len − 1`).
#[cube(launch)]
pub fn mp_gather_picks_kernel(
    best_unit: &Array<u32>,
    best_scale: &Array<f32>,
    best_gain: &Array<f32>,
    starts: &Array<u32>,
    out_units: &mut Array<u32>,
    out_scales: &mut Array<f32>,
    out_gain_prev: &mut Array<f32>,
    out_gain_next: &mut Array<f32>,
    picks: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < picks {
        let s = starts[i as usize];
        out_units[i as usize] = best_unit[s as usize];
        out_scales[i as usize] = best_scale[s as usize];
        out_gain_prev[i as usize] = best_gain[(s - 1u32) as usize];
        out_gain_next[i as usize] = best_gain[(s + 1u32) as usize];
    }
}
