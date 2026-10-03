//! The min/max reduction behind every envelope: decimation, the min/max cache and summary, and
//! the renderers' raw reads all fold samples into `[min, max]` columns here.
//!
//! Folds are written so the compiler vectorizes them (plain `<`/`>` selects over fixed lanes;
//! `f32::min`/`max` carry NaN rules that block it) and NaN samples are skipped. Blocks are split
//! across the global rayon pool (the one decoders such as zarrs already use): channel-major
//! blocks by channel, time-major blocks by column ranges.

use std::ops::Range;

use dsp_core::MemoryOrder;
use rayon::prelude::*;

/// The empty accumulator: folding any sample into it gives that sample.
pub const EMPTY: [f32; 2] = [f32::INFINITY, f32::NEG_INFINITY];

/// Independent accumulators per fold, so the loop maps onto vector registers.
const LANES: usize = 8;

/// Folds `samples` into `acc` (NaN samples are skipped).
#[inline]
pub fn fold(samples: &[f32], acc: [f32; 2]) -> [f32; 2] {
    let mut lo = [acc[0]; LANES];
    let mut hi = [acc[1]; LANES];
    let mut chunks = samples.chunks_exact(LANES);
    for c in &mut chunks {
        for i in 0..LANES {
            lo[i] = if c[i] < lo[i] { c[i] } else { lo[i] };
            hi[i] = if c[i] > hi[i] { c[i] } else { hi[i] };
        }
    }
    let mut out = acc;
    for i in 0..LANES {
        out = merge(out, [lo[i], hi[i]]);
    }
    for &v in chunks.remainder() {
        out = merge(out, [v, v]);
    }
    out
}

/// `[min, max]` of two accumulators.
#[inline]
pub fn merge(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [if b[0] < a[0] { b[0] } else { a[0] }, if b[1] > a[1] { b[1] } else { a[1] }]
}

/// Turns accumulators no sample reached (still [`EMPTY`]) into NaN, which renderers draw empty.
pub fn finish(acc: &mut [[f32; 2]]) {
    for v in acc.iter_mut().filter(|v| v[0] > v[1]) {
        *v = [f32::NAN, f32::NAN];
    }
}

/// `max − min` of `samples` (NaN skipped; 0 when there is nothing to measure).
pub fn peak_to_peak(samples: &[f32]) -> f32 {
    let [lo, hi] = fold(samples, EMPTY);
    if hi >= lo { hi - lo } else { 0.0 }
}

/// Mean `max − min` per column over `rows` envelope rows of `width` columns (row-major), the
/// activity of each column across channels. Columns no row covers (NaN) are NaN.
pub fn mean_range(env: &[[f32; 2]], rows: usize, width: usize) -> Vec<f32> {
    (0..width)
        .map(|x| {
            let (sum, n) = (0..rows).map(|r| env[r * width + x]).filter(|v| v[0].is_finite() && v[1].is_finite()).fold((0.0f64, 0usize), |(s, n), v| (s + (v[1] - v[0]) as f64, n + 1));
            if n == 0 { f32::NAN } else { (sum / n as f64) as f32 }
        })
        .collect()
}

/// How samples map to output columns (pixel columns or pyramid buckets).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Columns {
    /// `width` columns splitting `start..start + len`: column `x` covers
    /// `start + x·len/width .. start + (x+1)·len/width` (empty when `len < width`).
    Even { start: u64, len: u64, width: usize },
    /// `count` buckets of `size` samples from `origin` (the last may be partial).
    Buckets { origin: u64, size: u64, count: usize },
}

impl Columns {
    pub fn count(&self) -> usize {
        match *self {
            Columns::Even { width, .. } => width,
            Columns::Buckets { count, .. } => count,
        }
    }

    /// First sample of column `x`.
    pub fn start(&self, x: usize) -> u64 {
        match *self {
            Columns::Even { start, len, width } => start + x as u64 * len / width as u64,
            Columns::Buckets { origin, size, .. } => origin + x as u64 * size,
        }
    }

    /// The column holding sample `s` (samples outside the range go to the nearest end column).
    pub fn column_of(&self, s: u64) -> usize {
        match *self {
            Columns::Even { start, len, width } => {
                if s < start || len == 0 {
                    0
                } else {
                    (((s - start + 1) * width as u64 - 1) / len).min(width as u64 - 1) as usize
                }
            }
            Columns::Buckets { origin, size, count } => (s.saturating_sub(origin) / size).min(count as u64 - 1) as usize,
        }
    }

    /// First sample after column `x` (`u64::MAX` for the last one).
    fn next_start(&self, x: usize) -> u64 {
        if x + 1 < self.count() { self.start(x + 1) } else { u64::MAX }
    }
}

/// Samples of every channel, as a source read them.
#[derive(Debug, Clone, Copy)]
pub struct Block<'a> {
    pub data: &'a [f32],
    /// `ChannelMajor`: `data[c * samples + t]`; `TimeMajor`: `data[t * channels + c]`.
    pub order: MemoryOrder,
    pub channels: usize,
    pub samples: usize,
    /// Recording sample index of the block's first sample.
    pub first: u64,
}

/// Folds samples `range` (block-relative) of every channel of `block` into that channel's
/// columns: `acc[c * columns.count() + x]`. Accumulators are merged into, not reset.
pub fn fold_block(block: &Block, range: Range<usize>, columns: Columns, acc: &mut [[f32; 2]]) {
    let count = columns.count();
    if block.channels == 0 || count == 0 || range.is_empty() {
        return;
    }
    debug_assert!(acc.len() >= block.channels * count && range.end <= block.samples);
    let first = block.first + range.start as u64;
    match block.order {
        MemoryOrder::ChannelMajor => {
            acc.par_chunks_mut(count).zip(block.data.par_chunks(block.samples)).take(block.channels).for_each(|(acc, row)| {
                fold_row(&row[range.clone()], first, columns, acc);
            });
        }
        MemoryOrder::TimeMajor => {
            let frames = &block.data[range.start * block.channels..range.end * block.channels];
            fold_frames(frames, block.channels, first, columns, acc);
        }
    }
}

/// Folds one channel's consecutive samples (the first is sample `first`) into its columns.
pub fn fold_row(row: &[f32], first: u64, columns: Columns, acc: &mut [[f32; 2]]) {
    let end = first + row.len() as u64;
    let (mut s, mut x) = (first, columns.column_of(first));
    while s < end {
        let next = columns.next_start(x).min(end);
        if next > s {
            acc[x] = fold(&row[(s - first) as usize..(next - first) as usize], acc[x]);
            s = next;
        }
        x += 1;
    }
}

/// Folds time-major frames (`channels` values each, the first is sample `first`). Column
/// ranges run in parallel, each into its own `[column][channel]` accumulators so the inner loop
/// runs over contiguous channels.
fn fold_frames(frames: &[f32], channels: usize, first: u64, columns: Columns, acc: &mut [[f32; 2]]) {
    let n = frames.len() / channels;
    let end = first + n as u64;
    let (x0, x1) = (columns.column_of(first), columns.column_of(end - 1) + 1);
    let tasks = rayon::current_num_threads().min(x1 - x0).max(1);
    let parts: Vec<(usize, Vec<f32>, Vec<f32>)> = (0..tasks)
        .into_par_iter()
        .map(|k| {
            let (xa, xb) = (x0 + k * (x1 - x0) / tasks, x0 + (k + 1) * (x1 - x0) / tasks);
            let s0 = if xa == x0 { first } else { columns.start(xa) };
            let s1 = if xb == x1 { end } else { columns.start(xb) };
            let mut lo = vec![f32::INFINITY; (xb - xa) * channels];
            let mut hi = vec![f32::NEG_INFINITY; (xb - xa) * channels];
            let (mut x, mut next) = (xa, columns.next_start(xa));
            for s in s0..s1 {
                while s >= next {
                    x += 1;
                    next = columns.next_start(x);
                }
                let frame = &frames[(s - first) as usize * channels..][..channels];
                let l = &mut lo[(x - xa) * channels..][..channels];
                let h = &mut hi[(x - xa) * channels..][..channels];
                for ((&v, l), h) in frame.iter().zip(l.iter_mut()).zip(h.iter_mut()) {
                    *l = if v < *l { v } else { *l };
                    *h = if v > *h { v } else { *h };
                }
            }
            (xa, lo, hi)
        })
        .collect();
    let count = columns.count();
    for (xa, lo, hi) in parts {
        for (i, (l, h)) in lo.chunks_exact(channels).zip(hi.chunks_exact(channels)).enumerate() {
            for c in 0..channels {
                let a = &mut acc[c * count + xa + i];
                *a = merge(*a, [l[c], h[c]]);
            }
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_peak_to_peak_and_mean_range() {
        assert_eq!(peak_to_peak(&[1.0, -2.0, f32::NAN, 3.5]), 5.5);
        assert_eq!(peak_to_peak(&[]), 0.0);
        // 2 rows × 3 columns; the last column is uncovered
        let nan = [f32::NAN, f32::NAN];
        let env = [[0.0, 2.0], [1.0, 2.0], nan, [0.0, 4.0], nan, nan];
        let m = mean_range(&env, 2, 3);
        assert_eq!(m[0], 3.0);
        assert_eq!(m[1], 1.0);
        assert!(m[2].is_nan());
    }

    use super::*;

    fn brute(row: &[f32]) -> [f32; 2] {
        row.iter().fold(EMPTY, |[a, b], &v| [a.min(v), b.max(v)])
    }

    #[test]
    fn test_fold_matches_brute_force_and_skips_nan() {
        let v: Vec<f32> = (0..1001).map(|i| ((i * 7919) % 2003) as f32 - 1000.0).collect();
        for len in [0, 1, 7, 8, 9, 100, 1001] {
            assert_eq!(fold(&v[..len], EMPTY), brute(&v[..len]), "len {len}");
        }
        assert_eq!(fold(&[3.0, f32::NAN, -2.0], EMPTY), [-2.0, 3.0]);
    }

    #[test]
    fn test_columns_map_samples_both_ways() {
        let even = Columns::Even { start: 100, len: 1000, width: 7 };
        for s in 100..1100u64 {
            let x = even.column_of(s);
            assert!(even.start(x) <= s && s < even.start(x + 1), "sample {s} in column {x}");
        }
        let buckets = Columns::Buckets { origin: 64, size: 10, count: 5 };
        assert_eq!((buckets.column_of(64), buckets.column_of(73), buckets.column_of(74), buckets.column_of(500)), (0, 0, 1, 4));
    }

    /// Both layouts, any block split, give the brute-force min/max of every column.
    #[test]
    fn test_fold_block_matches_brute_force_in_both_orders() {
        let (channels, samples) = (5usize, 3001usize);
        let value = |c: usize, t: usize| ((t * 7919 + c * 104_729) % 2003) as f32 - 1000.0;
        let cm: Vec<f32> = (0..channels).flat_map(|c| (0..samples).map(move |t| value(c, t))).collect();
        let tm: Vec<f32> = (0..samples).flat_map(|t| (0..channels).map(move |c| value(c, t))).collect();
        for columns in [Columns::Even { start: 1000, len: samples as u64, width: 97 }, Columns::Buckets { origin: 1000, size: 64, count: samples.div_ceil(64) }] {
            let count = columns.count();
            let mut want = vec![EMPTY; channels * count];
            for c in 0..channels {
                for x in 0..count {
                    let (a, b) = ((columns.start(x) - 1000) as usize, ((if x + 1 < count { columns.start(x + 1) } else { 1000 + samples as u64 }) - 1000) as usize);
                    want[c * count + x] = brute(&cm[c * samples + a..c * samples + b.min(samples)]);
                }
            }
            for (data, order) in [(&cm, MemoryOrder::ChannelMajor), (&tm, MemoryOrder::TimeMajor)] {
                let block = Block { data, order, channels, samples, first: 1000 };
                let mut acc = vec![EMPTY; channels * count];
                for range in [0..1234, 1234..1300, 1300..samples] {
                    fold_block(&block, range, columns, &mut acc);
                }
                assert_eq!(acc, want, "{order:?} {columns:?}");
            }
        }
    }
}
