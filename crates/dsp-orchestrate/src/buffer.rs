//! Window buffer abstraction providing channel-major slicing, interior extraction,
//! and symmetric margin padding over halo-padded chunks.

use dsp_core::HaloWindow;

/// In-memory buffer of a halo-padded window for multiple channels in channel-major order.
///
/// Layout: `[channels, read_len]`, where channel $c$ occupies slice
/// `data[c * read_len .. (c + 1) * read_len]`.
#[derive(Debug, Clone)]
pub struct WindowBuffer {
    window: HaloWindow,
    channels: usize,
    data: Vec<f32>,
}

impl WindowBuffer {
    /// Creates a new window buffer.
    ///
    /// # Panics
    /// Panics in debug builds if `data.len() != channels * window.read_len()`.
    pub fn new(window: HaloWindow, channels: usize, data: Vec<f32>) -> Self {
        debug_assert_eq!(
            data.len(),
            channels * window.read_len(),
            "buffer size must equal channels * read_len"
        );
        Self {
            window,
            channels,
            data,
        }
    }

    /// The halo window metadata.
    #[inline]
    pub fn window(&self) -> &HaloWindow {
        &self.window
    }

    /// Number of channels in the buffer.
    #[inline]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Number of samples per channel in the read buffer (including left and right halos).
    #[inline]
    pub fn read_len(&self) -> usize {
        self.window.read_len()
    }

    /// Number of samples per channel in the valid interior.
    #[inline]
    pub fn valid_len(&self) -> usize {
        self.window.valid_len()
    }

    /// Returns the underlying flat channel-major slice.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Returns the underlying flat channel-major mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Consumes the buffer and returns the raw data vector.
    #[inline]
    pub fn into_data(self) -> Vec<f32> {
        self.data
    }

    /// Read-padded sample slice for a specific channel `ch`.
    #[inline]
    pub fn channel(&self, ch: usize) -> &[f32] {
        let n = self.read_len();
        &self.data[ch * n..(ch + 1) * n]
    }

    /// Read-padded mutable sample slice for a specific channel `ch`.
    #[inline]
    pub fn channel_mut(&mut self, ch: usize) -> &mut [f32] {
        let n = self.read_len();
        &mut self.data[ch * n..(ch + 1) * n]
    }

    /// Zero-copy slice of the valid interior for a specific channel `ch`.
    #[inline]
    pub fn channel_interior(&self, ch: usize) -> &[f32] {
        let ch_slice = self.channel(ch);
        &ch_slice[self.window.valid_local.clone()]
    }

    /// Extracts the compact `[channels, valid_len]` valid interior across all channels.
    ///
    /// Useful for learning batches (covariance estimation, whitening, template extraction)
    /// where halo settling margins must be omitted.
    pub fn interior(&self) -> Vec<f32> {
        let read_len = self.read_len();
        let valid_range = self.window.valid_local.clone();
        self.data
            .chunks_exact(read_len)
            .flat_map(|row| row[valid_range.clone()].iter().copied())
            .collect()
    }

    /// Crops the window to have a symmetric margin around its interior.
    ///
    /// Returns `(cropped_data, samples_per_channel, pad_samples)`, where `pad_samples`
    /// is the largest symmetric padding available on both edges.
    pub fn symmetric_pad(&self) -> (Vec<f32>, usize, usize) {
        let read_len = self.read_len();
        let left_pad = self.window.valid_local.start;
        let right_pad = read_len.saturating_sub(self.window.valid_local.end);
        let pad = left_pad.min(right_pad);
        let range = (self.window.valid_local.start - pad)..(self.window.valid_local.end + pad);
        let samples = range.len();

        let cropped = self
            .data
            .chunks_exact(read_len)
            .flat_map(|row| row[range.clone()].iter().copied())
            .collect();

        (cropped, samples, pad)
    }

    /// Checks if a local sample index lies inside the valid interior of this window.
    #[inline]
    pub fn is_interior_local(&self, local_sample: usize) -> bool {
        self.window.is_interior_local(local_sample)
    }

    /// Converts a local sample index to a global recording timestamp.
    #[inline]
    pub fn to_global_sample(&self, local_sample: usize) -> u64 {
        self.window.to_global_sample(local_sample)
    }

    /// Remaps a local sample index to a global timestamp if it falls within the valid interior.
    #[inline]
    pub fn remap_event(&self, local_sample: usize) -> Option<u64> {
        if self.is_interior_local(local_sample) {
            Some(self.to_global_sample(local_sample))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::ChunkSchedule;

    #[test]
    fn test_window_buffer_slicing_and_interior() {
        // Recording of 100 samples, batch 40, left 10, right 10
        let sched = ChunkSchedule::full_recording(100, 40, 10, 10);
        let w = &sched.windows()[1]; // Valid 40..80, read 30..90 (local 10..50)
        assert_eq!(w.read_len(), 60);
        assert_eq!(w.valid_len(), 40);

        // 2 channels
        let mut raw = vec![0.0f32; 2 * 60];
        for ch in 0..2 {
            for t in 0..60 {
                raw[ch * 60 + t] = (ch * 1000 + t) as f32;
            }
        }

        let buf = WindowBuffer::new(w.clone(), 2, raw);
        assert_eq!(buf.channel(0).len(), 60);
        assert_eq!(buf.channel(1).len(), 60);

        let ch0_interior = buf.channel_interior(0);
        assert_eq!(ch0_interior.len(), 40);
        assert_eq!(ch0_interior[0], 10.0);
        assert_eq!(ch0_interior[39], 49.0);

        let interior = buf.interior();
        assert_eq!(interior.len(), 2 * 40);
        assert_eq!(interior[0], 10.0);
        assert_eq!(interior[39], 49.0);
        assert_eq!(interior[40], 1010.0);
        assert_eq!(interior[79], 1049.0);

        let (sym, samples, pad) = buf.symmetric_pad();
        assert_eq!(pad, 10);
        assert_eq!(samples, 60);
        assert_eq!(sym.len(), 2 * 60);

        assert_eq!(buf.remap_event(9), None);
        assert_eq!(buf.remap_event(10), Some(40));
        assert_eq!(buf.remap_event(49), Some(79));
        assert_eq!(buf.remap_event(50), None);
    }
}
