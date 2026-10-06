//! Envelopes of samples already on the device (for example the output of a processing
//! pipeline): only the `[min, max]` columns are downloaded, never the samples.
//!
//! One cube per `(channel, column)` ([`LaunchGeometry::per_row`]). Each unit folds a strided share
//! of the column (neighbouring units read neighbouring samples), then the cube merges the partial
//! results pairwise in shared memory. Column edges come from the host as sample offsets, so any
//! [`Columns`] (even pixel columns or pyramid buckets) runs the same kernel, with 32-bit indices
//! only (WebGPU has no 64-bit integers). NaN samples are skipped, as on the host.

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_core::compute::{row_position, LaunchGeometry};
use dsp_core::{DspError, DspResult};

use super::fold::{finish, Columns};

/// Values stored per column: `min`, then `max`.
const PAIR: u32 = 2;

/// Min/max of samples `edges[x]..edges[x + 1]` of every channel row (rows `row_stride` apart),
/// into `out[PAIR · (channel · columns + x)]` (min) and the value after it (max). Columns with no
/// sample other than NaN stay `[+∞, −∞]`. `units` is the cube's x size (a power of two).
#[cube(launch)]
pub fn envelope_kernel<F: Float>(
    input: &Array<F>,
    edges: &Array<u32>,
    out: &mut Array<F>,
    columns: u32,
    rows: u32,
    row_stride: u32,
    #[comptime] units: u32,
) {
    let row = row_position();
    // `row` is uniform across the cube, so every unit of a cube takes the same branch
    if row < rows {
        let unit = UNIT_POS_X;
        let channel = row / columns;
        let column = row % columns;
        let base = channel * row_stride;
        let end = edges[(column + 1u32) as usize];

        // Comparisons are false for NaN, so NaN samples never replace a bound
        let mut lo = F::new(f32::INFINITY);
        let mut hi = F::new(f32::NEG_INFINITY);
        let mut s = edges[column as usize] + unit;
        while s < end {
            let x = input[(base + s) as usize];
            if x < lo {
                lo = x;
            }
            if x > hi {
                hi = x;
            }
            s += units;
        }

        let mut lo_s = SharedMemory::<F>::new(comptime!(units as usize));
        let mut hi_s = SharedMemory::<F>::new(comptime!(units as usize));
        lo_s[unit as usize] = lo;
        hi_s[unit as usize] = hi;
        sync_cube();

        let stride = RuntimeCell::<u32>::new(units / 2u32);
        while stride.read() > 0u32 {
            let s = stride.read();
            if unit < s {
                let other = (unit + s) as usize;
                let (lo_o, hi_o) = (lo_s[other], hi_s[other]);
                if lo_o < lo_s[unit as usize] {
                    lo_s[unit as usize] = lo_o;
                }
                if hi_o > hi_s[unit as usize] {
                    hi_s[unit as usize] = hi_o;
                }
            }
            sync_cube();
            stride.store(s / 2u32);
        }

        if unit == 0u32 {
            out[(row * PAIR) as usize] = lo_s[0];
            out[(row * PAIR + 1u32) as usize] = hi_s[0];
        }
    }
}

/// `[min, max]` per column of every channel of a `[channels, samples]` buffer on the device
/// (channel-major; it holds recording samples `first..first + samples`), as
/// `out[channel · columns.count() + x]`. Columns are clipped to the buffer; columns with no
/// sample (or only NaN) are NaN, as on the host.
pub fn envelope_on_device<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    input: &Handle,
    channels: usize,
    samples: usize,
    first: u64,
    columns: Columns,
) -> DspResult<Vec<[f32; 2]>> {
    let count = columns.count();
    if channels == 0 || count == 0 {
        return Ok(Vec::new());
    }
    let values = channels.checked_mul(samples).filter(|&v| u32::try_from(v).is_ok());
    let rows = channels.checked_mul(count).filter(|&r| u32::try_from(r * PAIR as usize).is_ok());
    let (Some(values), Some(rows)) = (values, rows) else {
        return Err(DspError::InvalidConfig(format!(
            "{channels} channels × {samples} samples / {count} columns exceed 32-bit device indexing"
        )));
    };

    // Column edges as offsets into the buffer, clipped to it
    let last = first + samples as u64;
    let offset = |s: u64| (s.clamp(first, last) - first) as u32;
    let edges: Vec<u32> = (0..count).map(|x| offset(columns.start(x))).chain([offset(columns.end())]).collect();
    let longest = edges.windows(2).map(|w| w[1].saturating_sub(w[0])).max().unwrap_or(0) as usize;

    let edges_handle = client.create_from_slice(u32::as_bytes(&edges));
    let out = client.empty(rows * PAIR as usize * std::mem::size_of::<F>());
    let geom = LaunchGeometry::per_row(client, rows, longest);
    unsafe {
        envelope_kernel::launch::<F, R>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ArrayArg::from_raw_parts(input.clone(), values),
            ArrayArg::from_raw_parts(edges_handle, edges.len()),
            ArrayArg::from_raw_parts(out.clone(), rows * PAIR as usize),
            count as u32,
            rows as u32,
            samples as u32,
            geom.cube_dim.x,
        );
    }
    let bytes = client.read_one_unchecked(out);
    let mut env: Vec<[f32; 2]> = F::from_bytes(&bytes)
        .as_chunks::<{ PAIR as usize }>()
        .0
        .iter()
        .map(|p| [p[0].to_f32().unwrap_or(f32::NAN), p[1].to_f32().unwrap_or(f32::NAN)])
        .collect();
    finish(&mut env);
    Ok(env)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envelope::fold::{fold_row, EMPTY};
    use dsp_core::compute::{ComputeTarget, ComputeTask};

    fn host(row: &[f32], columns: Columns) -> Vec<[f32; 2]> {
        let mut acc = vec![EMPTY; columns.count()];
        fold_row(row, columns.start(0), columns, &mut acc);
        finish(&mut acc);
        acc
    }

    struct Matches;

    impl ComputeTask for Matches {
        type Output = ();
        fn run<R: Runtime>(self, client: ComputeClient<R>) {
            let (channels, samples, first) = (3usize, 70_001usize, 1_000u64);
            let mut data: Vec<f32> = (0..channels * samples).map(|i| ((i * 7919) % 1013) as f32 - 500.0).collect();
            data[samples + 12_345] = f32::NAN;
            data[2 * samples..2 * samples + 40].fill(f32::NAN);
            let input = client.create_from_slice(f32::as_bytes(&data));
            for columns in [
                Columns::Even { start: first, len: samples as u64, width: 97 },
                Columns::Even { start: first + 5, len: 20, width: 64 },
                Columns::Buckets { origin: first, size: 40, count: 1_751 },
                Columns::Even { start: first, len: samples as u64, width: 1 },
            ] {
                let env = envelope_on_device::<R, f32>(&client, &input, channels, samples, first, columns).unwrap();
                for c in 0..channels {
                    let row = &data[c * samples..(c + 1) * samples];
                    let (from, to) = (columns.start(0) - first, columns.end().min(first + samples as u64) - first);
                    let expected = host(&row[from as usize..to as usize], columns);
                    let got = &env[c * columns.count()..(c + 1) * columns.count()];
                    for (x, (g, e)) in got.iter().zip(&expected).enumerate() {
                        let same = |a: f32, b: f32| a == b || (a.is_nan() && b.is_nan());
                        assert!(same(g[0], e[0]) && same(g[1], e[1]), "{} channel {c} column {x}: {g:?} vs {e:?} ({columns:?})", R::name(&client));
                    }
                }
            }
        }
    }

    #[test]
    fn device_envelope_matches_host_fold() {
        let targets = ComputeTarget::available();
        assert!(!targets.is_empty(), "no CubeCL runtime compiled in");
        for target in targets {
            target.run(Matches).expect("compiled-in runtime");
        }
    }
}
