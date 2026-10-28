//! Kernels of [`crate::sorters::mountainsort5::snippets`].

use cubecl::prelude::*;

use super::detect::NO_CHANNEL;

/// One unit per element `(j, s, k)` of `out` (`[events, width, max_neighbours]`): sample
/// `ev_samples[j] − n_before + s` of the `k`-th neighbourhood channel of `ev_channels[j]` (`table`,
/// padded with [`NO_CHANNEL`], which gives 0). The caller keeps every window inside the trace.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gather_snippets_kernel<F: Float>(
    trace: &[F],
    ev_samples: &[u32],
    ev_channels: &[u32],
    table: &[u32],
    out: &mut [F],
    total: u32,
    samples: u32,
    width: u32,
    max_neighbours: u32,
    n_before: u32,
) {
    let e = ABSOLUTE_POS as u32;
    if e < total {
        let k = e % max_neighbours;
        let s = (e / max_neighbours) % width;
        let j = e / (max_neighbours * width);
        let nb = table[(ev_channels[j as usize] * max_neighbours + k) as usize];
        let mut v = F::new(0.0f32);
        if nb != NO_CHANNEL {
            v = trace[(nb * samples + ev_samples[j as usize] + s - n_before) as usize];
        }
        out[e as usize] = v;
    }
}

/// One unit per element `(i, s, k)` of a batch of masked snippets: row `rows[i]` of `compact`
/// (`[_, width, max_neighbours]`, channel row `row_channels[i]` of `table`) written to `dense`
/// (`[batch, width, channels]`, zeroed by the caller) at its channel. Padding slots write nothing.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn scatter_snippets_kernel<F: Float>(
    compact: &[F],
    rows: &[u32],
    row_channels: &[u32],
    table: &[u32],
    dense: &mut [F],
    total: u32,
    width: u32,
    max_neighbours: u32,
    channels: u32,
) {
    let e = ABSOLUTE_POS as u32;
    if e < total {
        let k = e % max_neighbours;
        let s = (e / max_neighbours) % width;
        let i = e / (max_neighbours * width);
        let nb = table[(row_channels[i as usize] * max_neighbours + k) as usize];
        if nb != NO_CHANNEL {
            let src = (rows[i as usize] * width + s) * max_neighbours + k;
            dense[((i * width + s) * channels + nb) as usize] = compact[src as usize];
        }
    }
}
