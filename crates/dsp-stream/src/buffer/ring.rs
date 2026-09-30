use dsp_core::error::{DspError, DspResult};

/// What [`MultiChannelRingBuffer::push_chunk`] does when a chunk does not fit in the free space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OverrunPolicy {
    /// Refuse the chunk with [`DspError::BufferOverrun`]; nothing is overwritten.
    #[default]
    Reject,
    /// Overwrite the oldest unread samples and move the read position past them (counted in
    /// [`MultiChannelRingBuffer::dropped_samples`]). Live streams that must never block use this.
    DropOldest,
}

/// Multi-channel circular ring buffer for real-time continuous stream ingestion.
/// Pre-allocates all memory at construction (zero allocation during streaming).
/// Layout: Channel-Major `[channels, capacity_samples]`. Single owner (`&mut self`); share it across
/// threads behind a mutex or channel.
pub struct MultiChannelRingBuffer {
    channels: usize,
    capacity_samples: usize,
    buffer: Vec<f32>,
    write_pos: usize,
    read_pos: usize,
    available_samples: usize,
    policy: OverrunPolicy,
    dropped_samples: u64,
}

impl MultiChannelRingBuffer {
    pub fn new(channels: usize, capacity_samples: usize) -> Self {
        Self::with_policy(channels, capacity_samples, OverrunPolicy::Reject)
    }

    pub fn with_policy(channels: usize, capacity_samples: usize, policy: OverrunPolicy) -> Self {
        assert!(channels > 0, "Channels must be > 0");
        assert!(capacity_samples > 0, "Capacity must be > 0");
        Self {
            channels,
            capacity_samples,
            buffer: vec![0.0f32; channels * capacity_samples],
            write_pos: 0,
            read_pos: 0,
            available_samples: 0,
            policy,
            dropped_samples: 0,
        }
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn capacity_samples(&self) -> usize {
        self.capacity_samples
    }

    pub fn available_samples(&self) -> usize {
        self.available_samples
    }

    pub fn free_samples(&self) -> usize {
        self.capacity_samples - self.available_samples
    }

    /// Unread samples overwritten under [`OverrunPolicy::DropOldest`].
    pub fn dropped_samples(&self) -> u64 {
        self.dropped_samples
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
            return Err(DspError::BufferOverrun { capacity: self.capacity_samples, requested: chunk_samples });
        }
        let overflow = chunk_samples.saturating_sub(self.free_samples());
        if overflow > 0 {
            match self.policy {
                OverrunPolicy::Reject => {
                    return Err(DspError::BufferOverrun { capacity: self.free_samples(), requested: chunk_samples });
                }
                OverrunPolicy::DropOldest => {
                    self.read_pos = (self.read_pos + overflow) % self.capacity_samples;
                    self.available_samples -= overflow;
                    self.dropped_samples += overflow as u64;
                }
            }
        }

        let w_pos = self.write_pos;
        for c in 0..self.channels {
            let src = &data[c * chunk_samples..(c + 1) * chunk_samples];
            let dst = c * self.capacity_samples;
            let first = (self.capacity_samples - w_pos).min(chunk_samples);
            self.buffer[dst + w_pos..dst + w_pos + first].copy_from_slice(&src[..first]);
            self.buffer[dst..dst + chunk_samples - first].copy_from_slice(&src[first..]);
        }
        self.write_pos = (w_pos + chunk_samples) % self.capacity_samples;
        self.available_samples += chunk_samples;
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
        if chunk_samples > self.available_samples {
            return Err(DspError::BufferUnderrun { available: self.available_samples, requested: chunk_samples });
        }

        let r_pos = self.read_pos;
        for c in 0..self.channels {
            let dst = &mut output[c * chunk_samples..(c + 1) * chunk_samples];
            let src = c * self.capacity_samples;
            let first = (self.capacity_samples - r_pos).min(chunk_samples);
            dst[..first].copy_from_slice(&self.buffer[src + r_pos..src + r_pos + first]);
            dst[first..].copy_from_slice(&self.buffer[src..src + chunk_samples - first]);
        }
        self.read_pos = (r_pos + chunk_samples) % self.capacity_samples;
        self.available_samples -= chunk_samples;
        Ok(())
    }

    pub fn clear(&mut self) {
        self.write_pos = 0;
        self.read_pos = 0;
        self.available_samples = 0;
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

    fn ramp(start: usize, n: usize) -> Vec<f32> {
        // two channels: ch1 = -ch0
        let c0: Vec<f32> = (start..start + n).map(|v| v as f32).collect();
        c0.iter().copied().chain(c0.iter().map(|v| -v)).collect()
    }

    #[test]
    fn reject_policy_never_overwrites_unread_samples() {
        let mut ring = MultiChannelRingBuffer::new(2, 10);
        ring.push_chunk(&ramp(0, 8), 8).unwrap();
        assert!(matches!(ring.push_chunk(&ramp(8, 4), 4), Err(DspError::BufferOverrun { .. })));
        let mut out = vec![0.0; 16];
        ring.pop_chunk(&mut out, 8).unwrap();
        assert_eq!(out, ramp(0, 8));
    }

    #[test]
    fn drop_oldest_keeps_the_newest_samples_in_order() {
        let mut ring = MultiChannelRingBuffer::with_policy(2, 10, OverrunPolicy::DropOldest);
        ring.push_chunk(&ramp(0, 8), 8).unwrap();
        ring.push_chunk(&ramp(8, 6), 6).unwrap(); // 4 oldest dropped
        assert_eq!(ring.dropped_samples(), 4);
        assert_eq!(ring.available_samples(), 10);
        let mut out = vec![0.0; 20];
        ring.pop_chunk(&mut out, 10).unwrap();
        assert_eq!(out, ramp(4, 10));
    }
}
