//! Double-buffered out-of-core prefetch reader over [`dsp_core::RecordingSource`].
//!
//! Uses a scoped background producer thread and a two-slot buffer recycling ring so that
//! CPU-bound decompression (e.g. Zarr `gzip`/`zstd` or `mtscomp`) for window $i+1$ runs
//! concurrently while the GPU and spike detector process window $i$, with zero per-window
//! heap allocations.

use std::sync::mpsc::sync_channel;
use dsp_core::{ChunkSchedule, DspResult, HaloWindow, RecordingSource};

/// Double-buffered out-of-core reader that streams [`HaloWindow`]s from a [`RecordingSource`].
pub struct PrefetchReader<'a> {
    source: &'a dyn RecordingSource,
    channels: Vec<usize>,
    schedule: ChunkSchedule,
}

impl<'a> PrefetchReader<'a> {
    /// Creates a new prefetch reader over all channels of `source` for the given `schedule`.
    pub fn new(source: &'a dyn RecordingSource, schedule: ChunkSchedule) -> Self {
        let n_ch = source.info().channel_count();
        Self {
            source,
            channels: (0..n_ch).collect(),
            schedule,
        }
    }

    /// Creates a new prefetch reader over a specific subset of `channels`.
    pub fn with_channels(
        source: &'a dyn RecordingSource,
        channels: Vec<usize>,
        schedule: ChunkSchedule,
    ) -> Self {
        Self {
            source,
            channels,
            schedule,
        }
    }

    /// Streams all scheduled [`HaloWindow`]s with double-buffered background prefetching: calls
    /// `f(&window, &channel_major_samples)` for every window (scaled values, see
    /// [`RecordingSource::read`]), reading the next window on a background thread meanwhile.
    pub fn for_each_window<F>(&self, f: F) -> DspResult<()>
    where
        F: FnMut(&HaloWindow, &[f32]) -> DspResult<()>,
    {
        let source = self.source;
        self.stream(1, |ch, range, buf: &mut [f32]| source.read(ch, range, buf), f)
    }

    /// Like [`Self::for_each_window`] with the stored values ([`RecordingSource::read_stored`]):
    /// `info().format` little-endian, channel-major, before gain and offset.
    pub fn for_each_window_stored<F>(&self, f: F) -> DspResult<()>
    where
        F: FnMut(&HaloWindow, &[u8]) -> DspResult<()>,
    {
        let source = self.source;
        let bytes = source.info().format.bytes();
        self.stream(bytes, |ch, range, buf: &mut [u8]| source.read_stored(ch, range, buf), f)
    }

    /// Double-buffered window loop over `per_sample` elements per channel sample.
    fn stream<T, R, F>(&self, per_sample: usize, read: R, mut f: F) -> DspResult<()>
    where
        T: Copy + Default + Send,
        R: Fn(&[usize], std::ops::Range<u64>, &mut [T]) -> DspResult<()> + Sync,
        F: FnMut(&HaloWindow, &[T]) -> DspResult<()>,
    {
        if self.schedule.is_empty() || self.channels.is_empty() {
            return Ok(());
        }

        let n_ch = self.channels.len();
        let max_len = n_ch * self.schedule.max_read_samples() * per_sample;
        let windows = self.schedule.windows();
        let channels = &self.channels;
        let read = &read;

        std::thread::scope(|s| {
            let (ready_tx, ready_rx) = sync_channel::<DspResult<(HaloWindow, Vec<T>)>>(1);
            let (recycle_tx, recycle_rx) = sync_channel::<Vec<T>>(2);

            let _ = recycle_tx.send(vec![T::default(); max_len]);
            let _ = recycle_tx.send(vec![T::default(); max_len]);

            s.spawn(move || {
                for win in windows {
                    let Ok(mut buf) = recycle_rx.recv() else {
                        break;
                    };
                    let len = n_ch * win.read_len() * per_sample;
                    buf.resize(len, T::default());
                    match read(channels, win.read_global.clone(), &mut buf[..len]) {
                        Ok(()) => {
                            if ready_tx.send(Ok((win.clone(), buf))).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            let _ = ready_tx.send(Err(e));
                            break;
                        }
                    }
                }
            });

            for item in ready_rx {
                let (win, buf) = item?;
                f(&win, &buf)?;
                let _ = recycle_tx.send(buf);
            }

            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::MemoryRecording;

    #[test]
    fn test_prefetch_reader_streams_halo_windows() {
        let mut data = vec![0.0f32; 200];
        for c in 0..2 {
            for t in 0..100 {
                data[c * 100 + t] = (c * 1000 + t) as f32;
            }
        }
        let rec = MemoryRecording::new("test", data, 2, 1000.0).unwrap();
        let sched = ChunkSchedule::full_recording(100, 30, 5, 5);
        let reader = PrefetchReader::new(&rec, sched);

        let mut visited = Vec::new();
        reader
            .for_each_window(|win, buf| {
                let n = win.read_len();
                let ch0_first_interior = buf[win.valid_local.start];
                visited.push((win.index, win.valid_global.clone(), n, ch0_first_interior));
                Ok(())
            })
            .unwrap();

        assert_eq!(visited.len(), 4);
        assert_eq!(visited[0], (0, 0..30, 35, 0.0));
        assert_eq!(visited[1], (1, 30..60, 40, 30.0));
        assert_eq!(visited[2], (2, 60..90, 40, 60.0));
        assert_eq!(visited[3], (3, 90..100, 15, 90.0));
    }
}
