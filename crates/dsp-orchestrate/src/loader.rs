//! Out-of-core window loader supporting single-window extraction, strided subset batching,
//! and double-buffered concurrent prefetching over [`RecordingSource`].

use std::sync::mpsc::sync_channel;
use dsp_core::{ChunkSchedule, DspResult, HaloWindow, RecordingSource};

use crate::buffer::WindowBuffer;

/// Window data loader that reads [`HaloWindow`]s from a [`RecordingSource`].
pub struct WindowLoader<'a> {
    source: &'a dyn RecordingSource,
    channels: Vec<usize>,
}

impl<'a> WindowLoader<'a> {
    /// Creates a new loader over all channels of `source`.
    pub fn new(source: &'a dyn RecordingSource) -> Self {
        let n_ch = source.info().channel_count();
        Self {
            source,
            channels: (0..n_ch).collect(),
        }
    }

    /// Creates a new loader over a specific subset of `channels`.
    pub fn with_channels(source: &'a dyn RecordingSource, channels: Vec<usize>) -> Self {
        Self { source, channels }
    }

    /// The underlying recording source.
    #[inline]
    pub fn source(&self) -> &'a dyn RecordingSource {
        self.source
    }

    /// Channels selected for reading.
    #[inline]
    pub fn channels(&self) -> &[usize] {
        &self.channels
    }

    /// Loads a single window synchronously into an owned [`WindowBuffer`].
    pub fn load_window(&self, window: &HaloWindow) -> DspResult<WindowBuffer> {
        let n_ch = self.channels.len();
        let len = n_ch * window.read_len();
        let mut data = vec![0.0f32; len];
        self.source
            .read(&self.channels, window.read_global.clone(), &mut data)?;
        Ok(WindowBuffer::new(window.clone(), n_ch, data))
    }

    /// Iterates over an explicit subset of windows (e.g. strided learning batches),
    /// passing an owned [`WindowBuffer`] to `f` for each window.
    pub fn for_windows<F>(&self, windows: &[HaloWindow], mut f: F) -> DspResult<()>
    where
        F: FnMut(&WindowBuffer) -> DspResult<()>,
    {
        for window in windows {
            let buf = self.load_window(window)?;
            f(&buf)?;
        }
        Ok(())
    }

    /// Streams all scheduled [`HaloWindow`]s with double-buffered background prefetching.
    ///
    /// Runs a scoped background producer thread and a two-slot buffer recycling ring so that
    /// I/O read and de-compression for window $i+1$ overlap with processing window $i$,
    /// with zero steady-state heap allocations. Calls `f(&window, &raw_channel_major_samples)`.
    pub fn stream_schedule<F>(&self, schedule: &ChunkSchedule, f: F) -> DspResult<()>
    where
        F: FnMut(&HaloWindow, &[f32]) -> DspResult<()>,
    {
        let source = self.source;
        self.stream(
            schedule,
            1,
            |ch, range, buf: &mut [f32]| source.read(ch, range, buf),
            f,
        )
    }

    /// Like [`Self::stream_schedule`] with the stored raw bytes before gain and offset scaling
    /// ([`RecordingSource::read_stored`]).
    pub fn stream_stored<F>(&self, schedule: &ChunkSchedule, f: F) -> DspResult<()>
    where
        F: FnMut(&HaloWindow, &[u8]) -> DspResult<()>,
    {
        let source = self.source;
        let bytes = source.info().format.bytes();
        self.stream(
            schedule,
            bytes,
            |ch, range, buf: &mut [u8]| source.read_stored(ch, range, buf),
            f,
        )
    }

    /// Internal double-buffered streaming loop.
    fn stream<T, R, F>(
        &self,
        schedule: &ChunkSchedule,
        per_sample: usize,
        read: R,
        mut f: F,
    ) -> DspResult<()>
    where
        T: Copy + Default + Send,
        R: Fn(&[usize], std::ops::Range<u64>, &mut [T]) -> DspResult<()> + Sync,
        F: FnMut(&HaloWindow, &[T]) -> DspResult<()>,
    {
        if schedule.is_empty() || self.channels.is_empty() {
            return Ok(());
        }

        let n_ch = self.channels.len();
        let max_len = n_ch * schedule.max_read_samples() * per_sample;
        let windows = schedule.windows();
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
    fn test_window_loader_single_and_stream() {
        let mut data = vec![0.0f32; 200];
        for c in 0..2 {
            for t in 0..100 {
                data[c * 100 + t] = (c * 1000 + t) as f32;
            }
        }
        let rec = MemoryRecording::new("test", data, 2, 1000.0).unwrap();
        let sched = ChunkSchedule::full_recording(100, 30, 5, 5);
        let loader = WindowLoader::new(&rec);

        // 1. Single load
        let w0 = &sched.windows()[0];
        let buf0 = loader.load_window(w0).unwrap();
        assert_eq!(buf0.read_len(), 35);
        assert_eq!(buf0.channel(0)[0], 0.0);
        assert_eq!(buf0.channel(1)[0], 1000.0);

        // 2. Strided subset
        let subset: Vec<_> = sched.windows().iter().step_by(2).cloned().collect();
        let mut loaded_indices = Vec::new();
        loader
            .for_windows(&subset, |buf| {
                loaded_indices.push(buf.window().index);
                Ok(())
            })
            .unwrap();
        assert_eq!(loaded_indices, vec![0, 2]);

        // 3. Streaming
        let mut streamed = Vec::new();
        loader
            .stream_schedule(&sched, |win, raw| {
                streamed.push((win.index, raw.len()));
                Ok(())
            })
            .unwrap();
        assert_eq!(streamed.len(), 4);
    }
}
