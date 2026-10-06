//! What a viewer asks for ([`View`]) and what it gets ([`Envelope`]), the same whether the viewer
//! runs next to the recording or across the network (dsp-stream carries both).

use std::ops::Range;

use dsp_core::{DspError, DspResult, RecordingSource};

use crate::envelope::fold::{finish, fold_block, Columns, EMPTY};
use crate::pyramid::Pyramid;
use crate::read::read_channel_blocks;

/// Most values (channels × samples) one raw read of a view holds: bounds the memory of a raw
/// window, not the work (every sample of the window is still read once).
pub const RAW_BLOCK_VALUES: usize = 1 << 22;

/// A window of a recording drawn `width` columns wide.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct View {
    pub channels: Vec<usize>,
    /// First sample.
    pub start: u64,
    /// One past the last sample.
    pub end: u64,
    /// Pixel columns.
    pub width: usize,
}

/// The contents of a [`View`], row by row in the order of `View::channels`.
#[derive(Debug, Clone, PartialEq)]
pub enum Envelope {
    /// Fewer samples than columns: the samples themselves (`channels × samples`), drawn as lines.
    Samples(Vec<f32>),
    /// `[min, max]` per column (`channels × width`); NaN where nothing is known yet or every
    /// sample is NaN. `complete` is false while some columns wait for the pyramid to be built:
    /// ask again later.
    Columns { values: Vec<[f32; 2]>, complete: bool },
}

impl View {
    /// Samples in the window.
    pub fn samples(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }

    /// The window split evenly into `width` columns.
    pub fn columns(&self) -> Columns {
        Columns::Even { start: self.start, len: self.samples(), width: self.width }
    }

    /// Checks the view against a recording of `channels × samples`.
    pub fn validate(&self, channels: usize, samples: u64) -> DspResult<()> {
        if let Some(&channel) = self.channels.iter().find(|&&c| c >= channels) {
            return Err(DspError::InvalidChannel { channel, total: channels });
        }
        if self.start >= self.end || self.end > samples || self.width == 0 {
            return Err(DspError::SampleRange { start: self.start, end: self.end, total: samples });
        }
        Ok(())
    }

    /// The view's contents, as cheaply as is exact (see [`Self::read_into`]).
    pub fn read(&self, source: &dyn RecordingSource, pyramid: Option<&Pyramid>) -> DspResult<Envelope> {
        let mut out = Envelope::Samples(Vec::new());
        self.read_into(source, pyramid, &mut out)?;
        Ok(out)
    }

    /// The view's contents into `out`, reusing its buffer, as cheaply as is exact:
    /// - fewer samples than columns: the samples, read from `source`;
    /// - columns of at least the pyramid's base: from `pyramid` (no samples read);
    /// - otherwise (finer columns, or no pyramid): the samples read from `source` and folded into
    ///   columns, streamed in blocks of at most [`RAW_BLOCK_VALUES`] values aligned to the source's
    ///   storage chunks (each chunk decoded once), the next block read while the current one is
    ///   folded. Memory stays bounded however long the window.
    pub fn read_into(&self, source: &dyn RecordingSource, pyramid: Option<&Pyramid>, out: &mut Envelope) -> DspResult<()> {
        let info = source.info();
        self.validate(info.channel_count(), info.samples)?;
        let (rows, n) = (self.channels.len(), self.samples());
        let (mut samples, mut values) = match std::mem::replace(out, Envelope::Samples(Vec::new())) {
            Envelope::Samples(s) => (s, Vec::new()),
            Envelope::Columns { values, .. } => (Vec::new(), values),
        };
        if n <= self.width as u64 {
            samples.clear();
            samples.resize(rows * n as usize, 0.0);
            source.read(&self.channels, self.start..self.end, &mut samples)?;
            *out = Envelope::Samples(samples);
            return Ok(());
        }
        values.clear();
        values.resize(rows * self.width, EMPTY);
        if let Some(pyramid) = pyramid {
            if pyramid.envelope(&self.channels, self.start, self.end, self.width, &mut values)? {
                *out = Envelope::Columns { values, complete: pyramid.covers(self.start, self.end) };
                return Ok(());
            }
        }
        let columns = self.columns();
        let ranges = chunk_aligned(self.start, self.end, source.chunk_samples(), RAW_BLOCK_VALUES / rows.max(1));
        read_channel_blocks(source, &self.channels, &ranges, |block| fold_block(block, 0..block.samples, columns, &mut values))?;
        finish(&mut values);
        *out = Envelope::Columns { values, complete: true };
        Ok(())
    }
}

/// Blocks of `start..end` of at most `max_samples` samples (at least one storage chunk) whose
/// edges fall on the source's storage chunks (`chunk`), so no chunk is decoded twice.
fn chunk_aligned(start: u64, end: u64, chunk: Option<u64>, max_samples: usize) -> Vec<Range<u64>> {
    let chunk = chunk.filter(|&c| c > 0).unwrap_or(1);
    let step = ((max_samples as u64) / chunk).max(1) * chunk;
    let mut ranges = Vec::new();
    let mut b0 = start;
    while b0 < end {
        // The first block may start inside a chunk; every later one starts on a chunk edge
        let b1 = (b0 / chunk * chunk + step).min(end);
        ranges.push(b0..b1);
        b0 = b1;
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_cover_the_window_on_chunk_edges() {
        let ranges = chunk_aligned(10, 1000, Some(64), 200);
        assert_eq!(ranges.first().map(|r| r.start), Some(10));
        assert_eq!(ranges.last().map(|r| r.end), Some(1000));
        assert!(ranges.windows(2).all(|w| w[0].end == w[1].start && w[1].start % 64 == 0));
        assert!(ranges.iter().all(|r| r.end - r.start <= 192));
        // Blocks are at least one chunk even when the budget is smaller
        assert!(chunk_aligned(0, 1000, Some(512), 10).iter().all(|r| r.end - r.start <= 512));
    }

    #[test]
    fn streamed_blocks_fold_like_one_read() {
        let rec = recording();
        let (channels, start, end, width) = (vec![0usize, 2], 37u64, 19_000u64, 300usize);
        let columns = Columns::Even { start, len: end - start, width };
        let mut whole = vec![EMPTY; channels.len() * width];
        let mut data = vec![0.0; channels.len() * (end - start) as usize];
        rec.read(&channels, start..end, &mut data).unwrap();
        for (row, acc) in data.chunks_exact((end - start) as usize).zip(whole.chunks_exact_mut(width)) {
            crate::envelope::fold::fold_row(row, start, columns, acc);
        }
        let mut streamed = vec![EMPTY; channels.len() * width];
        let ranges = chunk_aligned(start, end, Some(1000), 2500);
        assert!(ranges.len() > 2);
        read_channel_blocks(&rec, &channels, &ranges, |b| fold_block(b, 0..b.samples, columns, &mut streamed)).unwrap();
        assert_eq!(streamed, whole);
    }
    use dsp_core::MemoryRecording;

    fn recording() -> MemoryRecording {
        let (channels, samples) = (3usize, 20_000usize);
        let data: Vec<f32> = (0..channels * samples).map(|i| ((i * 7919) % 1013) as f32 - 500.0).collect();
        MemoryRecording::new("v", data, channels, 1000.0).unwrap()
    }

    #[test]
    fn short_windows_return_samples() {
        let rec = recording();
        let view = View { channels: vec![2, 0], start: 100, end: 150, width: 64 };
        let Envelope::Samples(s) = view.read(&rec, None).unwrap() else { panic!("samples expected") };
        let mut expected = vec![0.0; 2 * 50];
        rec.read(&[2, 0], 100..150, &mut expected).unwrap();
        assert_eq!(s, expected);
    }

    #[test]
    fn pyramid_and_raw_reads_agree_on_every_peak() {
        let rec = recording();
        let pyramid = Pyramid::in_memory(&rec, 16).unwrap();
        let view = View { channels: vec![1], start: 0, end: 20_000, width: 50 };

        let Envelope::Columns { complete, .. } = view.read(&rec, Some(&pyramid)).unwrap() else { panic!() };
        assert!(!complete, "nothing built yet");

        assert!(pyramid.fill(&rec, 0, 20_000, 1 << 20, || false).unwrap());
        let Envelope::Columns { values: from_pyramid, complete } = view.read(&rec, Some(&pyramid)).unwrap() else { panic!() };
        let Envelope::Columns { values: raw, .. } = view.read(&rec, None).unwrap() else { panic!() };
        assert!(complete);
        let span = |v: &[[f32; 2]]| v.iter().fold(EMPTY, |a, c| [a[0].min(c[0]), a[1].max(c[1])]);
        assert_eq!(span(&from_pyramid), span(&raw), "the same extremes, column edges aside");
    }

    #[test]
    fn invalid_views_are_rejected() {
        let rec = recording();
        assert!(View { channels: vec![3], start: 0, end: 10, width: 4 }.read(&rec, None).is_err());
        assert!(View { channels: vec![0], start: 10, end: 10, width: 4 }.read(&rec, None).is_err());
        assert!(View { channels: vec![0], start: 0, end: 20_001, width: 4 }.read(&rec, None).is_err());
    }
}
