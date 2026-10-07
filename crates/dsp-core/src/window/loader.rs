//! Double-buffered window reading over a [`RecordingSource`].
//!
//! A scoped background thread reads window `i + 1` (disk, decompression) while the caller
//! processes window `i`. Two buffers are recycled between the threads, so host memory stays at two
//! windows whatever the recording length, with no allocation per window.

use std::ops::Range;
use std::sync::mpsc::sync_channel;

use super::HaloWindow;
use crate::{DspResult, RecordingSource};

/// Buffers in flight between the reading thread and the caller.
const BUFFERS_IN_FLIGHT: usize = 2;

/// Streams [`HaloWindow`]s of a [`RecordingSource`], reading ahead on a background thread.
pub struct WindowLoader<'a> {
    source: &'a dyn RecordingSource,
    channels: Vec<usize>,
}

impl<'a> WindowLoader<'a> {
    /// Loader over all channels of `source`.
    pub fn new(source: &'a dyn RecordingSource) -> Self {
        let channels = (0..source.info().channel_count()).collect();
        Self { source, channels }
    }

    /// Loader over `channels` of `source`, in that order.
    pub fn with_channels(source: &'a dyn RecordingSource, channels: Vec<usize>) -> Self {
        Self { source, channels }
    }

    /// The recording read.
    #[inline]
    pub fn source(&self) -> &'a dyn RecordingSource {
        self.source
    }

    /// Channels read, in buffer order.
    #[inline]
    pub fn channels(&self) -> &[usize] {
        &self.channels
    }

    /// Calls `f(&window, samples)` for every window of `windows`, in order: `samples` is the
    /// channel-major `[channels, window.read_len()]` block of scaled values
    /// ([`RecordingSource::read`]). Any subset works (a full [`super::ChunkSchedule`], every
    /// n-th window, isolated windows); the next window is read while `f` runs.
    pub fn stream<F>(&self, windows: &[HaloWindow], mut f: F) -> DspResult<()>
    where
        F: FnMut(&HaloWindow, &[f32]) -> DspResult<()>,
    {
        self.stream_while(windows, |win, buf| f(win, buf).map(|()| true))
    }

    /// Like [`Self::stream`] until `f` returns `false` (e.g. once enough data is gathered): no
    /// window after that one is passed to `f`, and reading stops.
    pub fn stream_while<F>(&self, windows: &[HaloWindow], f: F) -> DspResult<()>
    where
        F: FnMut(&HaloWindow, &[f32]) -> DspResult<bool>,
    {
        let source = self.source;
        self.stream_with(windows, 1, |ch, range, buf: &mut [f32]| source.read(ch, range, buf), f)
    }

    /// Like [`Self::stream`] with the stored values ([`RecordingSource::read_stored`]):
    /// `info().format` little-endian bytes, channel-major, before gain and offset.
    pub fn stream_stored<F>(&self, windows: &[HaloWindow], mut f: F) -> DspResult<()>
    where
        F: FnMut(&HaloWindow, &[u8]) -> DspResult<()>,
    {
        self.stream_stored_while(windows, |win, buf| f(win, buf).map(|()| true))
    }

    /// Like [`Self::stream_stored`] until `f` returns `false`.
    pub fn stream_stored_while<F>(&self, windows: &[HaloWindow], f: F) -> DspResult<()>
    where
        F: FnMut(&HaloWindow, &[u8]) -> DspResult<bool>,
    {
        let source = self.source;
        let bytes = source.info().format.bytes();
        let read = |ch: &[usize], range, buf: &mut [u8]| source.read_stored(ch, range, buf);
        self.stream_with(windows, bytes, read, f)
    }

    /// Double-buffered loop over `per_sample` elements per channel sample, until `f` returns
    /// `false`.
    fn stream_with<T, R, F>(&self, windows: &[HaloWindow], per_sample: usize, read: R, mut f: F) -> DspResult<()>
    where
        T: Copy + Default + Send,
        R: Fn(&[usize], Range<u64>, &mut [T]) -> DspResult<()> + Sync,
        F: FnMut(&HaloWindow, &[T]) -> DspResult<bool>,
    {
        if windows.is_empty() || self.channels.is_empty() {
            return Ok(());
        }
        let n_ch = self.channels.len();
        let max_len = n_ch * windows.iter().map(HaloWindow::read_len).max().unwrap_or(0) * per_sample;
        let channels = &self.channels;
        let read = &read;

        std::thread::scope(|s| {
            let (ready_tx, ready_rx) = sync_channel::<DspResult<(usize, Vec<T>)>>(1);
            let (recycle_tx, recycle_rx) = sync_channel::<Vec<T>>(BUFFERS_IN_FLIGHT);
            for _ in 0..BUFFERS_IN_FLIGHT {
                let _ = recycle_tx.send(vec![T::default(); max_len]);
            }

            s.spawn(move || {
                for (i, win) in windows.iter().enumerate() {
                    let Ok(mut buf) = recycle_rx.recv() else { break };
                    let len = n_ch * win.read_len() * per_sample;
                    buf.resize(len, T::default());
                    let item = read(channels, win.read_global.clone(), &mut buf).map(|()| (i, buf));
                    let failed = item.is_err();
                    if ready_tx.send(item).is_err() || failed {
                        break;
                    }
                }
            });

            // Returning early drops both channel ends, which stops the reading thread
            for item in ready_rx {
                let (i, buf) = item?;
                if !f(&windows[i], &buf)? {
                    break;
                }
                let _ = recycle_tx.send(buf);
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::window::ChunkSchedule;
    use crate::MemoryRecording;

    fn recording() -> MemoryRecording {
        let data = (0..2).flat_map(|c| (0..100).map(move |t| (c * 1000 + t) as f32)).collect();
        MemoryRecording::new("test", data, 2, 1000.0).unwrap()
    }

    #[test]
    fn streams_a_full_schedule() {
        let rec = recording();
        let sched = ChunkSchedule::full_recording(100, 30, 5, 5);
        let mut visited = Vec::new();
        WindowLoader::new(&rec)
            .stream(sched.windows(), |win, buf| {
                let n = win.read_len();
                visited.push((win.index, win.valid_global.clone(), n, buf[win.valid_local.start], buf[n]));
                Ok(())
            })
            .unwrap();
        assert_eq!(visited.len(), 4);
        assert_eq!(visited[0], (0, 0..30, 35, 0.0, 1000.0));
        assert_eq!(visited[1], (1, 30..60, 40, 30.0, 1025.0));
        assert_eq!(visited[3], (3, 90..100, 15, 90.0, 1085.0));
    }

    #[test]
    fn streams_a_subset_and_stops_on_error() {
        let rec = recording();
        let sched = ChunkSchedule::full_recording(100, 30, 5, 5);
        let subset: Vec<HaloWindow> = sched.windows().iter().step_by(2).cloned().collect();
        let mut seen = Vec::new();
        WindowLoader::new(&rec)
            .stream(&subset, |win, _| {
                seen.push(win.index);
                Ok(())
            })
            .unwrap();
        assert_eq!(seen, vec![0, 2]);

        let mut calls = 0;
        let err = WindowLoader::new(&rec).stream(sched.windows(), |_, _| {
            calls += 1;
            Err(crate::DspError::InvalidConfig("stop".into()))
        });
        assert!(err.is_err());
        assert_eq!(calls, 1);

        let mut seen = Vec::new();
        WindowLoader::new(&rec)
            .stream_while(sched.windows(), |win, _| {
                seen.push(win.index);
                Ok(win.index < 1)
            })
            .unwrap();
        assert_eq!(seen, vec![0, 1]);
    }
}
