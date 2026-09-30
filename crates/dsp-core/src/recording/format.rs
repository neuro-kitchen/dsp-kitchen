use serde::{Deserialize, Serialize};

/// Numeric type of the samples as stored on disk (always little-endian).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SampleFormat {
    I16,
    U16,
    F32,
}

impl SampleFormat {
    pub const fn bytes(self) -> usize {
        match self {
            SampleFormat::I16 | SampleFormat::U16 => 2,
            SampleFormat::F32 => 4,
        }
    }

    /// Parses common spellings: `i16`/`int16`, `u16`/`uint16`, `f32`/`float32`/`float32-le`.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().trim_end_matches("-le") {
            "i16" | "int16" | "<i2" => Some(SampleFormat::I16),
            "u16" | "uint16" | "<u2" => Some(SampleFormat::U16),
            "f32" | "float32" | "float" | "<f4" => Some(SampleFormat::F32),
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
        assert_eq!(SampleFormat::parse("f64"), None);
        assert_eq!(SampleFormat::I16.bytes(), 2);
        assert_eq!(SampleFormat::F32.bytes(), 4);
    }
}
