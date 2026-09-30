//! Little-endian sample decoding shared by the binary formats.

use crate::model::SampleType;

/// Decodes `out.len()` contiguous samples from `bytes`, applying `value * gain + offset`.
pub fn decode_into(ty: SampleType, bytes: &[u8], out: &mut [f32], gain: f64, offset: f64) {
    macro_rules! run {
        ($t:ty, $n:expr) => {
            for (o, b) in out.iter_mut().zip(bytes.chunks_exact($n)) {
                let v = <$t>::from_le_bytes(b.try_into().unwrap()) as f64;
                *o = (v * gain + offset) as f32;
            }
        };
    }
    // Unity scaling of float32 is the common case (TDT streams): skip the f64 round trip
    if ty == SampleType::F32 && gain == 1.0 && offset == 0.0 {
        for (o, b) in out.iter_mut().zip(bytes.chunks_exact(4)) {
            *o = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        }
        return;
    }
    match ty {
        SampleType::I8 => run!(i8, 1),
        SampleType::I16 => run!(i16, 2),
        SampleType::U16 => run!(u16, 2),
        SampleType::I32 => run!(i32, 4),
        SampleType::I64 => run!(i64, 8),
        SampleType::F32 => run!(f32, 4),
        SampleType::F64 => run!(f64, 8),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_types_and_scaling() {
        let bytes: Vec<u8> = [-2i16, 3].iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut out = [0.0f32; 2];
        decode_into(SampleType::I16, &bytes, &mut out, 0.5, 1.0);
        assert_eq!(out, [0.0, 2.5]);

        let bytes: Vec<u8> = [1.5f32, -4.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        decode_into(SampleType::F32, &bytes, &mut out, 1.0, 0.0);
        assert_eq!(out, [1.5, -4.0]);

        let bytes: Vec<u8> = [7.25f64].iter().flat_map(|v| v.to_le_bytes()).collect();
        decode_into(SampleType::F64, &bytes, &mut out[..1], 2.0, 0.0);
        assert_eq!(out[0], 14.5);
    }
}
