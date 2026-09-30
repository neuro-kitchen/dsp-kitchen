use serde::{Deserialize, Serialize};

/// Numeric type of stored samples (little-endian on disk).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SampleType {
    I8,
    I16,
    U16,
    I32,
    I64,
    F32,
    F64,
}

impl SampleType {
    pub const fn bytes(self) -> usize {
        match self {
            SampleType::I8 => 1,
            SampleType::I16 | SampleType::U16 => 2,
            SampleType::I32 | SampleType::F32 => 4,
            SampleType::I64 | SampleType::F64 => 8,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            SampleType::I8 => "int8",
            SampleType::I16 => "int16",
            SampleType::U16 => "uint16",
            SampleType::I32 => "int32",
            SampleType::I64 => "int64",
            SampleType::F32 => "float32",
            SampleType::F64 => "float64",
        }
    }
}

/// Order of multi-channel samples in storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryOrder {
    /// All samples of channel 0, then channel 1, …
    ChannelMajor,
    /// Interleaved: every channel at t0, then every channel at t1, … (acquisition systems).
    TimeMajor,
    /// Fixed-size packets per channel scattered through a file (TDT TEV).
    Packets,
}
