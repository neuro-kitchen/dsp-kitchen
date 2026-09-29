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
            (channels * samples) as usize,
            "Data payload length must match channels * samples"
        );
        let purpose_code = match purpose {
            StreamPurpose::Processing => 0,
            StreamPurpose::Visualization { .. } => 1,
        };
        Self {
            sequence_number,
            timestamp_sample,
            channels,
            samples,
            sample_rate_hz,
            data,
            purpose: purpose_code,
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
