//! Min / max columns of a live source, built as its blocks arrive: a view zoomed out over an hour
//! of a run is answered from them at once, instead of reading every sample (the ring holds
//! seconds, the Zarr would be read whole). Two levels, of [`LEVELS`] samples per bucket, so any
//! zoom-out finds buckets no wider than its columns.

use std::ops::Range;

/// Samples per bucket of each level, finest first. At 384 channels × 30 kHz the finer level keeps
/// ~22 kB per second of recording (~80 MB an hour), the coarser 1/64 of that.
pub const LEVELS: [u64; 2] = [4_096, 262_144];

struct Level {
    per: u64,
    /// Per channel: one (min, max) per finished bucket.
    buckets: Vec<Vec<(f32, f32)>>,
    /// The bucket being filled: per channel (min, max), and its samples so far.
    partial: Vec<(f32, f32)>,
    partial_samples: u64,
}

/// A source's overview from sample 0 on.
pub struct LiveOverview {
    channels: usize,
    levels: Vec<Level>,
    /// Samples taken in (blocks must come in order, as a source delivers them).
    samples: u64,
}

impl LiveOverview {
    pub fn new(channels: usize) -> Self {
        let level = |per| Level { per, buckets: vec![Vec::new(); channels], partial: vec![(f32::INFINITY, f32::NEG_INFINITY); channels], partial_samples: 0 };
        Self { channels, levels: LEVELS.iter().map(|&p| level(p)).collect(), samples: 0 }
    }

    /// Samples covered so far.
    pub fn samples(&self) -> u64 {
        self.samples
    }

    /// Takes in a block of `samples` per channel starting at `first_sample`, channel-major. A block
    /// that does not continue the last one (a gap) is counted as missing values (NaN columns).
    pub fn push(&mut self, first_sample: u64, values: &[f32], samples: usize) {
        if samples == 0 || values.len() < self.channels * samples {
            return;
        }
        // A gap: the missing samples count as nothing seen (NaN)
        while self.samples < first_sample {
            let n = (first_sample - self.samples).min(LEVELS[0]);
            for level in &mut self.levels {
                Self::take(level, self.channels, n as usize, |_, _| f32::NAN);
            }
            self.samples += n;
        }
        if first_sample + samples as u64 <= self.samples {
            return;
        }
        let skip = (self.samples - first_sample) as usize;
        let n = samples - skip;
        for level in &mut self.levels {
            Self::take(level, self.channels, n, |ch, k| values[ch * samples + skip + k]);
        }
        self.samples += n as u64;
    }

    fn take(level: &mut Level, channels: usize, n: usize, value: impl Fn(usize, usize) -> f32) {
        let mut k = 0;
        while k < n {
            let room = (level.per - level.partial_samples) as usize;
            let take = room.min(n - k);
            for ch in 0..channels {
                let (lo, hi) = &mut level.partial[ch];
                for i in k..k + take {
                    let v = value(ch, i);
                    if v.is_finite() {
                        *lo = lo.min(v);
                        *hi = hi.max(v);
                    }
                }
            }
            level.partial_samples += take as u64;
            k += take;
            if level.partial_samples == level.per {
                for ch in 0..channels {
                    let (lo, hi) = std::mem::replace(&mut level.partial[ch], (f32::INFINITY, f32::NEG_INFINITY));
                    level.buckets[ch].push(if lo <= hi { (lo, hi) } else { (f32::NAN, f32::NAN) });
                }
                level.partial_samples = 0;
            }
        }
    }

    /// `width` columns of `channels` over `range`, `[min, max]` per column, row by row
    /// (`min_max[2 · (row · width + column)]`), when a level's buckets are no wider than a column
    /// and cover the range; `None` otherwise (read the samples instead).
    pub fn columns(&self, channels: &[usize], range: Range<u64>, width: usize) -> Option<Vec<f32>> {
        if width == 0 || range.end <= range.start || channels.iter().any(|&c| c >= self.channels) {
            return None;
        }
        let per_column = (range.end - range.start) as f64 / width as f64;
        // The coarsest level whose buckets fit in a column, and that covers the range
        let level = self.levels.iter().rev().find(|l| (l.per as f64) <= per_column && l.buckets[0].len() as u64 * l.per >= range.end)?;
        let mut out = Vec::with_capacity(2 * channels.len() * width);
        for &ch in channels {
            let buckets = &level.buckets[ch];
            for col in 0..width {
                let a = (range.start as f64 + col as f64 * per_column) as u64 / level.per;
                let b = (((range.start as f64 + (col + 1) as f64 * per_column) as u64).div_ceil(level.per)).max(a + 1);
                let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
                for &(l, h) in &buckets[a as usize..(b as usize).min(buckets.len())] {
                    if l.is_finite() {
                        lo = lo.min(l);
                        hi = hi.max(h);
                    }
                }
                if lo <= hi {
                    out.extend([lo, hi]);
                } else {
                    out.extend([f32::NAN, f32::NAN]);
                }
            }
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Columns from the overview hold every sample's min and max, as reading them would; blocks
    /// of any size give the same buckets; a range not covered yet, or too fine, is left to reads.
    #[test]
    fn columns_hold_each_range_min_and_max() {
        let samples = 3 * LEVELS[0] as usize + 100;
        // Channel 0: a ramp; channel 1: its negative
        let signal: Vec<f32> = (0..samples).map(|i| i as f32).chain((0..samples).map(|i| -(i as f32))).collect();
        let mut o = LiveOverview::new(2);
        let block = 1000;
        let mut first = 0;
        while first < samples {
            let n = block.min(samples - first);
            let values: Vec<f32> = (0..2).flat_map(|ch| signal[ch * samples + first..ch * samples + first + n].to_vec()).collect();
            o.push(first as u64, &values, n);
            first += n;
        }
        assert_eq!(o.samples(), samples as u64);
        let range = 0..3 * LEVELS[0];
        let cols = o.columns(&[0, 1], range.clone(), 3).unwrap();
        let per = LEVELS[0] as f32;
        assert_eq!(&cols[0..2], &[0.0, per - 1.0]);
        assert_eq!(&cols[4..6], &[2.0 * per, 3.0 * per - 1.0]);
        assert_eq!(&cols[6..8], &[-(per - 1.0), 0.0], "the second channel's first column");
        assert!(o.columns(&[0], 0..4 * LEVELS[0], 4).is_none(), "not covered yet");
        assert!(o.columns(&[0], 0..LEVELS[0], 100).is_none(), "columns finer than the buckets: read");
        // A gap: the missing samples are nothing seen
        let mut g = LiveOverview::new(1);
        g.push(LEVELS[0], &vec![1.0; LEVELS[0] as usize], LEVELS[0] as usize);
        let c = g.columns(&[0], 0..2 * LEVELS[0], 2).unwrap();
        assert!(c[0].is_nan() && c[2] == 1.0);
    }
}
