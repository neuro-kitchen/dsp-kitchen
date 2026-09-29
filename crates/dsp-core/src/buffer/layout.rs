use serde::{Deserialize, Serialize};

/// Memory ordering convention for multi-channel neural signal data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryOrder {
    /// Channel-major: [channel_0_samples..., channel_1_samples...]
    /// Optimal for per-channel filtering and separable 1D convolution.
    ChannelMajor,

    /// Time-major (Interleaved): [c0_t0, c1_t0, ..., cN_t0, c0_t1, c1_t1, ...]
    /// Standard format written by acquisition hardware (SpikeGLX, Open Ephys, Intan).
    TimeMajor,
}

/// Striding and layout descriptor for multi-dimensional buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufferLayout {
    pub channels: usize,
    pub samples: usize,
    pub order: MemoryOrder,
}

impl BufferLayout {
    pub fn new(channels: usize, samples: usize, order: MemoryOrder) -> Self {
        Self {
            channels,
            samples,
            order,
        }
    }

    pub fn total_elements(&self) -> usize {
        self.channels * self.samples
    }

    /// Compute linear index into the underlying flat storage.
    #[inline(always)]
    pub fn linear_index(&self, channel: usize, sample: usize) -> usize {
        match self.order {
            MemoryOrder::ChannelMajor => channel * self.samples + sample,
            MemoryOrder::TimeMajor => sample * self.channels + channel,
        }
    }
}
