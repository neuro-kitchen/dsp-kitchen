use std::sync::atomic::{AtomicUsize, Ordering};
use dsp_core::error::{DspError, DspResult};

/// Multi-channel circular ring buffer for real-time continuous stream ingestion.
/// Pre-allocates all memory at construction (zero allocation during streaming).
/// Layout: Channel-Major `[channels, capacity_samples]`.
pub struct MultiChannelRingBuffer {
    channels: usize,
    capacity_samples: usize,
    buffer: Vec<f32>,
    write_pos: AtomicUsize,
    read_pos: AtomicUsize,
    available_samples: AtomicUsize,
}

impl MultiChannelRingBuffer {
    pub fn new(channels: usize, capacity_samples: usize) -> Self {
        assert!(channels > 0, "Channels must be > 0");
        assert!(capacity_samples > 0, "Capacity must be > 0");

        let total_size = channels * capacity_samples;
        Self {
            channels,
            capacity_samples,
            buffer: vec![0.0f32; total_size],
            write_pos: AtomicUsize::new(0),
            read_pos: AtomicUsize::new(0),
            available_samples: AtomicUsize::new(0),
        }
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn capacity_samples(&self) -> usize {
        self.capacity_samples
    }

    pub fn available_samples(&self) -> usize {
        self.available_samples.load(Ordering::Acquire)
    }

    /// Pushes a chunk of shape [channels, chunk_samples] into the ring buffer.
    pub fn push_chunk(&mut self, data: &[f32], chunk_samples: usize) -> DspResult<()> {
        if data.len() != self.channels * chunk_samples {
            return Err(DspError::ShapeMismatch {
                expected: vec![self.channels * chunk_samples],
                actual: vec![data.len()],
            });
        }

        if chunk_samples > self.capacity_samples {
            return Err(DspError::BufferOverrun {
                capacity: self.capacity_samples,
                requested: chunk_samples,
            });
        }

        let w_pos = self.write_pos.load(Ordering::Relaxed);

        // Copy channel by channel wrapping around capacity
        for c in 0..self.channels {
            let src_offset = c * chunk_samples;
            let dst_channel_offset = c * self.capacity_samples;

            let first_part = (self.capacity_samples - w_pos).min(chunk_samples);
            let second_part = chunk_samples - first_part;

            // First segment
            self.buffer[dst_channel_offset + w_pos..dst_channel_offset + w_pos + first_part]
                .copy_from_slice(&data[src_offset..src_offset + first_part]);

            // Wrapped segment (if any)
            if second_part > 0 {
                self.buffer[dst_channel_offset..dst_channel_offset + second_part]
                    .copy_from_slice(&data[src_offset + first_part..src_offset + chunk_samples]);
            }
        }

        let next_w = (w_pos + chunk_samples) % self.capacity_samples;
        self.write_pos.store(next_w, Ordering::Release);

        let current_avail = self.available_samples.load(Ordering::Relaxed);
        let new_avail = (current_avail + chunk_samples).min(self.capacity_samples);
        self.available_samples.store(new_avail, Ordering::Release);

        Ok(())
    }

    /// Reads/pops a chunk of shape [channels, chunk_samples] from the ring buffer.
    pub fn pop_chunk(&mut self, output: &mut [f32], chunk_samples: usize) -> DspResult<()> {
        if output.len() != self.channels * chunk_samples {
            return Err(DspError::ShapeMismatch {
                expected: vec![self.channels * chunk_samples],
                actual: vec![output.len()],
            });
        }

        let current_avail = self.available_samples.load(Ordering::Acquire);
        if chunk_samples > current_avail {
            return Err(DspError::BufferUnderrun {
                available: current_avail,
                requested: chunk_samples,
            });
        }

        let r_pos = self.read_pos.load(Ordering::Relaxed);

        for c in 0..self.channels {
            let dst_offset = c * chunk_samples;
            let src_channel_offset = c * self.capacity_samples;

            let first_part = (self.capacity_samples - r_pos).min(chunk_samples);
            let second_part = chunk_samples - first_part;

            output[dst_offset..dst_offset + first_part]
                .copy_from_slice(&self.buffer[src_channel_offset + r_pos..src_channel_offset + r_pos + first_part]);

            if second_part > 0 {
                output[dst_offset + first_part..dst_offset + chunk_samples]
                    .copy_from_slice(&self.buffer[src_channel_offset..src_channel_offset + second_part]);
            }
        }

        let next_r = (r_pos + chunk_samples) % self.capacity_samples;
        self.read_pos.store(next_r, Ordering::Release);
        self.available_samples.fetch_sub(chunk_samples, Ordering::Release);

        Ok(())
    }

    pub fn clear(&mut self) {
        self.write_pos.store(0, Ordering::Relaxed);
        self.read_pos.store(0, Ordering::Relaxed);
        self.available_samples.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_push_pop_wrap() {
        let channels = 2;
        let capacity = 100;
        let mut ring = MultiChannelRingBuffer::new(channels, capacity);

        assert_eq!(ring.available_samples(), 0);

        // Push 60 samples
        let chunk1 = vec![1.0f32; channels * 60];
        ring.push_chunk(&chunk1, 60).unwrap();
        assert_eq!(ring.available_samples(), 60);

        // Pop 40 samples
        let mut out1 = vec![0.0f32; channels * 40];
        ring.pop_chunk(&mut out1, 40).unwrap();
        assert_eq!(ring.available_samples(), 20);
        assert_eq!(out1[0], 1.0);

        // Push 60 samples (wraps around 100)
        let chunk2 = vec![2.0f32; channels * 60];
        ring.push_chunk(&chunk2, 60).unwrap();
        assert_eq!(ring.available_samples(), 80);

        // Pop remaining 80 samples
        let mut out2 = vec![0.0f32; channels * 80];
        ring.pop_chunk(&mut out2, 80).unwrap();
        assert_eq!(ring.available_samples(), 0);

        // Verify values: first 20 should be 1.0, next 60 should be 2.0
        assert_eq!(out2[0], 1.0);
        assert_eq!(out2[19], 1.0);
        assert_eq!(out2[20], 2.0);
        assert_eq!(out2[79], 2.0);
    }
}
