use crate::error::{DspError, DspResult};
use crate::time::{RationalTime, SampleRate};
use super::layout::{BufferLayout, MemoryOrder};

/// An authoritative multi-channel signal chunk in host memory.
#[derive(Debug, Clone, PartialEq)]
pub struct SignalChunk {
    layout: BufferLayout,
    sample_rate: SampleRate,
    start_time: RationalTime,
    data: Vec<f32>,
}

impl SignalChunk {
    pub fn new(
        channels: usize,
        samples: usize,
        sample_rate: SampleRate,
        start_time: RationalTime,
        order: MemoryOrder,
        data: Vec<f32>,
    ) -> DspResult<Self> {
        let expected_len = channels * samples;
        if data.len() != expected_len {
            return Err(DspError::ShapeMismatch {
                expected: vec![expected_len],
                actual: vec![data.len()],
            });
        }

        Ok(Self {
            layout: BufferLayout::new(channels, samples, order),
            sample_rate,
            start_time,
            data,
        })
    }

    pub fn zeros(
        channels: usize,
        samples: usize,
        sample_rate: SampleRate,
        start_time: RationalTime,
        order: MemoryOrder,
    ) -> Self {
        let data = vec![0.0f32; channels * samples];
        Self {
            layout: BufferLayout::new(channels, samples, order),
            sample_rate,
            start_time,
            data,
        }
    }

    pub fn layout(&self) -> &BufferLayout {
        &self.layout
    }

    pub fn channels(&self) -> usize {
        self.layout.channels
    }

    pub fn samples(&self) -> usize {
        self.layout.samples
    }

    pub fn sample_rate(&self) -> SampleRate {
        self.sample_rate
    }

    pub fn start_time(&self) -> RationalTime {
        self.start_time
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    #[inline(always)]
    pub fn get(&self, channel: usize, sample: usize) -> DspResult<f32> {
        if channel >= self.layout.channels || sample >= self.layout.samples {
            return Err(DspError::InvalidChannel {
                channel,
                total: self.layout.channels,
            });
        }
        let idx = self.layout.linear_index(channel, sample);
        Ok(self.data[idx])
    }

    #[inline(always)]
    pub fn set(&mut self, channel: usize, sample: usize, value: f32) -> DspResult<()> {
        if channel >= self.layout.channels || sample >= self.layout.samples {
            return Err(DspError::InvalidChannel {
                channel,
                total: self.layout.channels,
            });
        }
        let idx = self.layout.linear_index(channel, sample);
        self.data[idx] = value;
        Ok(())
    }
}
