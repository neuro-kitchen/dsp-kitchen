use serde::{Deserialize, Serialize};

/// Numeric type of the samples as stored on disk (always little-endian).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SampleFormat {
    I8,
    I16,
    U16,
    I32,
    F32,
    F64,
}

impl SampleFormat {
    pub const ALL: [SampleFormat; 6] =
        [SampleFormat::I8, SampleFormat::I16, SampleFormat::U16, SampleFormat::I32, SampleFormat::F32, SampleFormat::F64];

    pub const fn bytes(self) -> usize {
        match self {
            SampleFormat::I8 => 1,
            SampleFormat::I16 | SampleFormat::U16 => 2,
            SampleFormat::I32 | SampleFormat::F32 => 4,
            SampleFormat::F64 => 8,
        }
    }

    /// Canonical name (NumPy / Zarr spelling: `int16`, `float32`, …).
    pub const fn name(self) -> &'static str {
        match self {
            SampleFormat::I8 => "int8",
            SampleFormat::I16 => "int16",
            SampleFormat::U16 => "uint16",
            SampleFormat::I32 => "int32",
            SampleFormat::F32 => "float32",
            SampleFormat::F64 => "float64",
        }
    }

    /// Parses common spellings: `i16`/`int16`/`<i2`, `f32`/`float32`/`float32-le`, ….
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().trim_end_matches("-le") {
            "i8" | "int8" | "<i1" | "|i1" => Some(SampleFormat::I8),
            "i16" | "int16" | "<i2" => Some(SampleFormat::I16),
            "u16" | "uint16" | "<u2" => Some(SampleFormat::U16),
            "i32" | "int32" | "<i4" => Some(SampleFormat::I32),
            "f32" | "float32" | "float" | "<f4" => Some(SampleFormat::F32),
            "f64" | "float64" | "double" | "<f8" => Some(SampleFormat::F64),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_and_size() {
        assert_eq!(SampleFormat::parse("int16"), Some(SampleFormat::I16));
        assert_eq!(SampleFormat::parse("float32-le"), Some(SampleFormat::F32));
        assert_eq!(SampleFormat::parse("<u2"), Some(SampleFormat::U16));
        assert_eq!(SampleFormat::parse("<f8"), Some(SampleFormat::F64));
        assert_eq!(SampleFormat::parse("complex64"), None);
        assert_eq!(SampleFormat::I16.bytes(), 2);
        assert_eq!(SampleFormat::F64.bytes(), 8);
        for f in SampleFormat::ALL {
            assert_eq!(SampleFormat::parse(f.name()), Some(f));
        }
    }
}
