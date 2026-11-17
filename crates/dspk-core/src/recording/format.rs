use serde::{Deserialize, Serialize};

/// Numeric type of the samples as stored on disk (always little-endian).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SampleFormat {
    I8,
    /// Unsigned 8-bit (8-bit WAV: 128 is zero).
    U8,
    I16,
    U16,
    /// Signed 24-bit, three bytes per sample (24-bit WAV, many acquisition systems).
    I24,
    I32,
    /// IEEE 754 half precision (binary16): 11 significant bits, range ±65 504.
    F16,
    F32,
    F64,
}

impl SampleFormat {
    pub const ALL: [SampleFormat; 9] = [
        SampleFormat::I8,
        SampleFormat::U8,
        SampleFormat::I16,
        SampleFormat::U16,
        SampleFormat::I24,
        SampleFormat::I32,
        SampleFormat::F16,
        SampleFormat::F32,
        SampleFormat::F64,
    ];

    pub const fn bytes(self) -> usize {
        match self {
            SampleFormat::I8 | SampleFormat::U8 => 1,
            SampleFormat::I16 | SampleFormat::U16 | SampleFormat::F16 => 2,
            SampleFormat::I24 => 3,
            SampleFormat::I32 | SampleFormat::F32 => 4,
            SampleFormat::F64 => 8,
        }
    }

    /// Canonical name (NumPy / Zarr spelling: `int16`, `float32`, …).
    pub const fn name(self) -> &'static str {
        match self {
            SampleFormat::I8 => "int8",
            SampleFormat::U8 => "uint8",
            SampleFormat::I16 => "int16",
            SampleFormat::U16 => "uint16",
            SampleFormat::I24 => "int24",
            SampleFormat::I32 => "int32",
            SampleFormat::F16 => "float16",
            SampleFormat::F32 => "float32",
            SampleFormat::F64 => "float64",
        }
    }

    /// Decodes `out.len()` contiguous little-endian samples of this format from `bytes`, scaled
    /// to `stored · gain + offset`.
    ///
    /// # Panics
    /// If `bytes` holds fewer than `out.len()` samples.
    pub fn decode(self, bytes: &[u8], out: &mut [f32], gain: f32, offset: f32) {
        fn run<const B: usize>(bytes: &[u8], out: &mut [f32], gain: f32, offset: f32, value: impl Fn([u8; B]) -> f32) {
            assert!(bytes.len() >= out.len() * B, "{} bytes hold fewer than {} samples", bytes.len(), out.len());
            for (o, b) in out.iter_mut().zip(bytes.as_chunks::<B>().0) {
                *o = value(*b) * gain + offset;
            }
        }
        match self {
            SampleFormat::I8 => run::<1>(bytes, out, gain, offset, |b| i8::from_le_bytes(b) as f32),
            SampleFormat::U8 => run::<1>(bytes, out, gain, offset, |b| b[0] as f32),
            // Sign-extended from the third byte's top bit
            SampleFormat::I24 => run::<3>(bytes, out, gain, offset, |b| (i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8) as f32),
            SampleFormat::I16 => run::<2>(bytes, out, gain, offset, |b| i16::from_le_bytes(b) as f32),
            SampleFormat::U16 => run::<2>(bytes, out, gain, offset, |b| u16::from_le_bytes(b) as f32),
            SampleFormat::I32 => run::<4>(bytes, out, gain, offset, |b| i32::from_le_bytes(b) as f32),
            SampleFormat::F16 => run::<2>(bytes, out, gain, offset, |b| f16_to_f32(u16::from_le_bytes(b))),
            SampleFormat::F32 => run::<4>(bytes, out, gain, offset, f32::from_le_bytes),
            SampleFormat::F64 => run::<8>(bytes, out, gain, offset, |b| f64::from_le_bytes(b) as f32),
        }
    }

    /// Parses common spellings: `i16`/`int16`/`<i2`, `f32`/`float32`/`float32-le`, ….
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().trim_end_matches("-le") {
            "i8" | "int8" | "<i1" | "|i1" => Some(SampleFormat::I8),
            "u8" | "uint8" | "<u1" | "|u1" => Some(SampleFormat::U8),
            "i24" | "int24" => Some(SampleFormat::I24),
            "i16" | "int16" | "<i2" => Some(SampleFormat::I16),
            "u16" | "uint16" | "<u2" => Some(SampleFormat::U16),
            "i32" | "int32" | "<i4" => Some(SampleFormat::I32),
            "f32" | "float32" | "float" | "<f4" => Some(SampleFormat::F32),
            "f16" | "float16" | "half" | "<f2" => Some(SampleFormat::F16),
            "f64" | "float64" | "double" | "<f8" => Some(SampleFormat::F64),
            _ => None,
        }
    }
}

/// A half-precision (IEEE 754 binary16) value's bits → f32, exactly (every f16 is an f32).
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let out = match (exp, mant) {
        (0, 0) => sign,
        // Subnormal: normalise
        (0, m) => {
            let shift = m.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | (((m << shift) & 0x3ff) << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, m) => sign | 0x7fc0_0000 | (m << 13),
        (e, m) => sign | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(out)
}

/// f32 → the nearest half-precision value's bits (round to nearest even; beyond ±65 504: ±∞).
pub fn f32_to_f16(value: f32) -> u16 {
    let x = value.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let mant = x & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        // Subnormal or zero
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rounded = (m + half - 1 + ((m >> shift) & 1)) >> shift;
        return sign | rounded as u16;
    }
    let rounded = mant + 0xfff + ((mant >> 13) & 1);
    let (e, m) = if rounded & 0x80_0000 != 0 { (e + 1, 0) } else { (e, rounded >> 13) };
    if e >= 0x1f { sign | 0x7c00 } else { sign | ((e as u16) << 10) | m as u16 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float16_conversions_are_exact_where_they_must_be() {
        // Every f16 survives f16 → f32 → f16
        for bits in 0u16..=u16::MAX {
            let v = f16_to_f32(bits);
            if v.is_nan() {
                continue;
            }
            assert_eq!(f32_to_f16(v), bits, "{bits:#06x} ({v})");
        }
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xfbff), -65504.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f32_to_f16(1e6), 0x7c00);
        // Ties to even: 1 + 2^-11 lies halfway between 1 and the next f16
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3c00);
        let mut out = [0.0f32; 2];
        SampleFormat::F16.decode(&[0x3c00u16.to_le_bytes(), 0xc000u16.to_le_bytes()].concat(), &mut out, 1.0, 0.0);
        assert_eq!(out, [1.0, -2.0]);
        assert_eq!(SampleFormat::parse("float16"), Some(SampleFormat::F16));
    }

    #[test]
    fn decode_scales_every_format() {
        let mut out = [0.0f32; 2];
        SampleFormat::I16.decode(&[0x10, 0x00, 0xF0, 0xFF], &mut out, 0.5, 1.0);
        assert_eq!(out, [9.0, -7.0]);
        SampleFormat::U16.decode(&[0xFF, 0xFF, 0x00, 0x00], &mut out, 1.0, 0.0);
        assert_eq!(out, [65535.0, 0.0]);
        SampleFormat::F64.decode(&[1.5f64.to_le_bytes(), (-2.0f64).to_le_bytes()].concat(), &mut out, 2.0, 0.0);
        assert_eq!(out, [3.0, -4.0]);
    }

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
