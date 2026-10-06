//! What a viewer asks for ([`View`]) and what it gets ([`Envelope`]), the same whether the viewer
//! runs next to the recording or across the network (dsp-stream carries both).

use dsp_core::{DspError, DspResult, RecordingSource};

use crate::envelope::fold::{finish, fold_row, Columns, EMPTY};
use crate::pyramid::Pyramid;

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

    /// The view's contents, as cheaply as is exact:
    /// - fewer samples than columns: the samples, read from `source`;
    /// - columns of at least the pyramid's base: from `pyramid` (no samples read);
    /// - otherwise (finer columns, or no pyramid): the samples read from `source` and folded
    ///   into columns. Without a pyramid this reads the whole window, however long.
    pub fn read(&self, source: &dyn RecordingSource, pyramid: Option<&Pyramid>) -> DspResult<Envelope> {
        let info = source.info();
        self.validate(info.channel_count(), info.samples)?;
        let n = self.samples();
        if n <= self.width as u64 {
            let mut samples = vec![0.0f32; self.channels.len() * n as usize];
            source.read(&self.channels, self.start..self.end, &mut samples)?;
            return Ok(Envelope::Samples(samples));
        }
        if let Some(pyramid) = pyramid {
            let mut values = vec![EMPTY; self.channels.len() * self.width];
            if pyramid.envelope(&self.channels, self.start, self.end, self.width, &mut values)? {
                return Ok(Envelope::Columns { values, complete: pyramid.covers(self.start, self.end) });
            }
        }
        let mut samples = vec![0.0f32; self.channels.len() * n as usize];
        source.read(&self.channels, self.start..self.end, &mut samples)?;
        let mut values = vec![EMPTY; self.channels.len() * self.width];
        for (row, acc) in samples.chunks_exact(n as usize).zip(values.chunks_exact_mut(self.width)) {
            fold_row(row, self.start, self.columns(), acc);
        }
        finish(&mut values);
        Ok(Envelope::Columns { values, complete: true })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
