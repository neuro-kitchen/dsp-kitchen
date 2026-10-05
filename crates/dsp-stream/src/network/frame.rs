use prost::Message;
use crate::purpose::StreamPurpose;

/// High-throughput Protobuf streaming frame for multi-channel continuous signals.
#[derive(Clone, PartialEq, Message)]
pub struct StreamFrame {
    /// Monotonically increasing frame sequence number.
    #[prost(uint64, tag = "1")]
    pub sequence_number: u64,

    /// Absolute sample index clock timestamp of the first sample in this chunk.
    #[prost(uint64, tag = "2")]
    pub timestamp_sample: u64,

    /// Number of recording channels.
    #[prost(uint32, tag = "3")]
    pub channels: u32,

    /// Number of time samples per channel.
    #[prost(uint32, tag = "4")]
    pub samples: u32,

    /// Hardware acquisition sample rate in Hz.
    #[prost(double, tag = "5")]
    pub sample_rate_hz: f64,

    /// Flattened continuous multi-channel float32 payload [channels x samples].
    #[prost(float, repeated, tag = "6")]
    pub data: Vec<f32>,

    /// Stream purpose discriminator: 0 = Processing, 1 = Visualization.
    #[prost(int32, tag = "7")]
    pub purpose: i32,

    /// Alternative compact payload: little-endian int16 `[channels x samples]` (half the bytes of
    /// `data`); value in µV = `raw * gain_uv`. Exactly one of `data` / `data_i16` is set.
    #[prost(bytes = "vec", tag = "8")]
    pub data_i16: Vec<u8>,

    /// µV per int16 step of `data_i16`.
    #[prost(float, tag = "9")]
    pub gain_uv: f32,
}

/// Default upper bound on an encoded frame (64 MiB); larger length prefixes are rejected before
/// allocating.
pub const DEFAULT_MAX_FRAME_BYTES: usize = 64 << 20;

/// A frame whose contents do not match its header.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FrameError {
    #[error("frame of {len} bytes exceeds the {max}-byte limit")]
    TooLarge { len: usize, max: usize },
    #[error("frame declares {channels} x {samples} samples but carries {actual} values")]
    PayloadMismatch { channels: u32, samples: u32, actual: usize },
    #[error("frame carries both a float32 and an int16 payload")]
    TwoPayloads,
}

fn purpose_code(purpose: StreamPurpose) -> i32 {
    match purpose {
        StreamPurpose::Processing => 0,
        StreamPurpose::Visualization { .. } => 1,
    }
}

impl StreamFrame {
    /// Creates a new streaming frame.
    pub fn new(
        sequence_number: u64,
        timestamp_sample: u64,
        channels: u32,
        samples: u32,
        sample_rate_hz: f64,
        data: Vec<f32>,
        purpose: StreamPurpose,
    ) -> Self {
        assert_eq!(
            data.len(),
            channels as usize * samples as usize,
            "Data payload length must match channels * samples"
        );
        Self {
            sequence_number,
            timestamp_sample,
            channels,
            samples,
            sample_rate_hz,
            data,
            purpose: purpose_code(purpose),
            data_i16: Vec::new(),
            gain_uv: 0.0,
        }
    }

    /// Creates a frame with the compact int16 payload (`raw[i] * gain_uv` µV).
    #[allow(clippy::too_many_arguments)]
    pub fn new_i16(
        sequence_number: u64,
        timestamp_sample: u64,
        channels: u32,
        samples: u32,
        sample_rate_hz: f64,
        raw: &[i16],
        gain_uv: f32,
        purpose: StreamPurpose,
    ) -> Self {
        assert_eq!(raw.len(), channels as usize * samples as usize, "payload length must match channels * samples");
        Self {
            sequence_number,
            timestamp_sample,
            channels,
            samples,
            sample_rate_hz,
            data: Vec::new(),
            purpose: purpose_code(purpose),
            data_i16: raw.iter().flat_map(|v| v.to_le_bytes()).collect(),
            gain_uv,
        }
    }

    /// Checks the payload against `channels x samples` (call on every decoded frame).
    pub fn validate(&self) -> Result<(), FrameError> {
        let expected = self.channels as usize * self.samples as usize;
        let mismatch = |actual| FrameError::PayloadMismatch { channels: self.channels, samples: self.samples, actual };
        match (self.data.is_empty(), self.data_i16.is_empty()) {
            (false, false) => Err(FrameError::TwoPayloads),
            (true, false) if self.data_i16.len() != expected * 2 => Err(mismatch(self.data_i16.len() / 2)),
            (_, true) if self.data.len() != expected => Err(mismatch(self.data.len())),
            _ => Ok(()),
        }
    }

    /// Samples in µV `[channels x samples]`, whichever payload the frame carries.
    pub fn values(&self) -> Vec<f32> {
        if self.data_i16.is_empty() {
            self.data.clone()
        } else {
            self.data_i16
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 * self.gain_uv)
                .collect()
        }
    }

    /// Encodes the frame into a byte vector with a 4-byte length prefix.
    pub fn encode_length_delimited(&self) -> Vec<u8> {
        let msg_len = self.encoded_len();
        let mut buf = Vec::with_capacity(4 + msg_len);
        buf.extend_from_slice(&(msg_len as u32).to_be_bytes());
        self.encode(&mut buf).expect("Protobuf encoding failed");
        buf
    }

    /// Decodes a frame from raw protobuf bytes.
    pub fn decode_from_slice(buf: &[u8]) -> Result<Self, prost::DecodeError> {
        <Self as Message>::decode(buf)
    }

    /// Returns total wire byte size including the 4-byte length prefix.
    pub fn total_wire_bytes(&self) -> usize {
        4 + self.encoded_len()
    }

    /// Returns the payload size in bytes (float32 or int16).
    pub fn payload_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>() + self.data_i16.len()
    }

    /// Resolves the typed StreamPurpose.
    pub fn stream_purpose(&self) -> StreamPurpose {
        match self.purpose {
            1 => StreamPurpose::Visualization {
                target_points_per_channel: self.samples as usize,
            },
            _ => StreamPurpose::Processing,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int16_payload_round_trips_and_validates() {
        let raw: Vec<i16> = (0..12).map(|v| v * 100 - 600).collect();
        let frame = StreamFrame::new_i16(0, 0, 3, 4, 30_000.0, &raw, 0.195, StreamPurpose::Processing);
        let decoded = StreamFrame::decode_from_slice(&frame.encode_length_delimited()[4..]).unwrap();
        decoded.validate().unwrap();
        let expected: Vec<f32> = raw.iter().map(|&v| v as f32 * 0.195).collect();
        assert_eq!(decoded.values(), expected);
        let f32_frame = StreamFrame::new(0, 0, 3, 4, 30_000.0, expected.clone(), StreamPurpose::Processing);
        assert!(frame.payload_bytes() * 2 == f32_frame.payload_bytes());
    }

    #[test]
    fn malformed_frames_are_rejected() {
        let mut frame = StreamFrame::new(0, 0, 2, 3, 1.0, vec![0.0; 6], StreamPurpose::Processing);
        frame.samples = 4;
        assert!(matches!(frame.validate(), Err(FrameError::PayloadMismatch { .. })));
        frame.samples = 3;
        frame.data_i16 = vec![0; 12];
        assert_eq!(frame.validate(), Err(FrameError::TwoPayloads));
        // Header multiplication does not overflow u32
        let big = StreamFrame { channels: u32::MAX, samples: 2, data: Vec::new(), data_i16: Vec::new(), ..frame };
        assert!(big.validate().is_err());
    }
}
