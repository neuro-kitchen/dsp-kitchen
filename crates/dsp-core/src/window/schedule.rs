//! Halo window scheduling: which samples each window owns and which it reads.
//!
//! When streaming a large recording through causal/non-causal processing, each non-overlapping
//! valid window `[s0 .. s1)` is padded with a `left_halo` (e.g. filter settling, lookback) and a
//! `right_halo` (e.g. lookahead), clamped to `[0 .. total_samples)`.

use std::ops::Range;

/// A single halo-padded window in a chunked streaming schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HaloWindow {
    /// 0-based window index in the schedule.
    pub index: usize,
    /// Non-overlapping global sample range `[s0 .. s1)` owned by this window.
    pub valid_global: Range<u64>,
    /// Halo-padded global sample range `[read_start .. read_end)` to read from `RecordingSource`.
    pub read_global: Range<u64>,
    /// Interior slice range `[left_pad .. left_pad + valid_len)` within the padded buffer.
    pub valid_local: Range<usize>,
}

impl HaloWindow {
    /// Total number of samples in the padded read buffer (`read_global`).
    #[inline]
    pub fn read_len(&self) -> usize {
        (self.read_global.end - self.read_global.start) as usize
    }

    /// Number of valid interior samples (`valid_global`).
    #[inline]
    pub fn valid_len(&self) -> usize {
        self.valid_local.len()
    }

    /// Returns `true` if a local sample index (relative to the padded buffer) lies inside
    /// the valid interior owned by this window.
    #[inline]
    pub fn is_interior_local(&self, local_sample: usize) -> bool {
        self.valid_local.contains(&local_sample)
    }

    /// Converts a local sample index (relative to the padded buffer) to a global recording sample index.
    #[inline]
    pub fn to_global_sample(&self, local_sample: usize) -> u64 {
        self.read_global.start + local_sample as u64
    }

    /// Global sample of a local event (e.g. a spike found in the padded buffer) if this window
    /// owns it (it lies in the valid interior), `None` if it belongs to a neighbouring window.
    #[inline]
    pub fn remap_event(&self, local_sample: usize) -> Option<u64> {
        self.is_interior_local(local_sample).then(|| self.to_global_sample(local_sample))
    }

    /// The window owning `valid` (clamped to `[0 .. total_samples)`), read with `left_halo` and
    /// `right_halo` samples of context, also clamped to the recording.
    pub fn around(index: usize, valid: Range<u64>, left_halo: u64, right_halo: u64, total_samples: u64) -> Self {
        let end = valid.end.min(total_samples);
        let start = valid.start.min(end);
        let read_start = start.saturating_sub(left_halo);
        let read_end = (end + right_halo).min(total_samples);
        let left_pad = (start - read_start) as usize;
        Self {
            index,
            valid_global: start..end,
            read_global: read_start..read_end,
            valid_local: left_pad..left_pad + (end - start) as usize,
        }
    }
}

/// Schedule of contiguous `HaloWindow`s covering a sample range `[start .. end)` of a recording.
#[derive(Debug, Clone)]
pub struct ChunkSchedule {
    windows: Vec<HaloWindow>,
    max_read_samples: usize,
}

impl ChunkSchedule {
    /// Builds a schedule over `target_range` within a recording of length `total_samples`.
    pub fn new(
        target_range: Range<u64>,
        batch_samples: u64,
        left_halo: u64,
        right_halo: u64,
        total_samples: u64,
    ) -> Self {
        let start = target_range.start.min(total_samples);
        let end = target_range.end.min(total_samples);
        let step = batch_samples.max(1);

        let mut windows = Vec::new();
        let mut max_read_samples = 0usize;
        let mut s0 = start;
        let mut index = 0usize;

        while s0 < end {
            let s1 = (s0 + step).min(end);
            let win = HaloWindow::around(index, s0..s1, left_halo, right_halo, total_samples);
            max_read_samples = max_read_samples.max(win.read_len());
            windows.push(win);

            s0 = s1;
            index += 1;
        }

        Self {
            windows,
            max_read_samples,
        }
    }

    /// Builds a schedule covering the entire recording `0 .. total_samples`.
    pub fn full_recording(
        total_samples: u64,
        batch_samples: u64,
        left_halo: u64,
        right_halo: u64,
    ) -> Self {
        Self::new(
            0..total_samples,
            batch_samples,
            left_halo,
            right_halo,
            total_samples,
        )
    }

    /// Slice of scheduled `HaloWindow`s.
    pub fn windows(&self) -> &[HaloWindow] {
        &self.windows
    }

    /// Consumes the schedule and returns the underlying vector of windows.
    pub fn into_windows(self) -> Vec<HaloWindow> {
        self.windows
    }

    /// Number of scheduled windows.
    pub fn len(&self) -> usize {
        self.windows.len()
    }

    /// Returns `true` if no windows are scheduled.
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    /// Maximum `read_len()` across all windows (used to pre-allocate persistent VRAM buffers).
    pub fn max_read_samples(&self) -> usize {
        self.max_read_samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_schedule_partitions_without_gaps_or_overlap() {
        let total = 10_000u64;
        let sched = ChunkSchedule::full_recording(total, 3_000, 500, 200);
        assert_eq!(sched.len(), 4);
        assert_eq!(sched.max_read_samples(), 3_700);

        let w = sched.windows();
        // Window 0: valid 0..3000, read 0..3200, local 0..3000
        assert_eq!(w[0].valid_global, 0..3_000);
        assert_eq!(w[0].read_global, 0..3_200);
        assert_eq!(w[0].valid_local, 0..3_000);
        assert_eq!(w[0].to_global_sample(w[0].valid_local.start), 0);

        // Window 1: valid 3000..6000, read 2500..6200, local 500..3500
        assert_eq!(w[1].valid_global, 3_000..6_000);
        assert_eq!(w[1].read_global, 2_500..6_200);
        assert_eq!(w[1].valid_local, 500..3_500);
        assert!(!w[1].is_interior_local(499));
        assert!(w[1].is_interior_local(500));
        assert_eq!(w[1].to_global_sample(500), 3_000);
        assert_eq!(w[1].to_global_sample(3_499), 5_999);

        // Window 3 (last): valid 9000..10000, read 8500..10000, local 500..1500
        assert_eq!(w[3].valid_global, 9_000..10_000);
        assert_eq!(w[3].read_global, 8_500..10_000);
        assert_eq!(w[3].valid_local, 500..1_500);
        assert_eq!(w[3].to_global_sample(1_499), 9_999);
    }

    #[test]
    fn around_clamps_and_remaps_events() {
        let w = HaloWindow::around(7, 30..60, 5, 5, 62);
        assert_eq!((w.index, w.read_global.clone(), w.valid_local.clone()), (7, 25..62, 5..35));
        assert_eq!(w.remap_event(4), None);
        assert_eq!(w.remap_event(5), Some(30));
        assert_eq!(w.remap_event(34), Some(59));
        assert_eq!(w.remap_event(35), None);
        assert_eq!(HaloWindow::around(0, 90..120, 5, 5, 100).valid_global, 90..100);
    }
}
