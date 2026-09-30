//! Little-endian sample decoding / encoding shared by the binary formats.

use dsp_core::SampleFormat;

/// Stored value of sample `idx` in `bytes` (not scaled).
#[inline(always)]
pub(crate) fn decode(format: SampleFormat, bytes: &[u8], idx: usize) -> f32 {
    match format {
        SampleFormat::I16 => i16::from_le_bytes([bytes[2 * idx], bytes[2 * idx + 1]]) as f32,
        SampleFormat::U16 => u16::from_le_bytes([bytes[2 * idx], bytes[2 * idx + 1]]) as f32,
        SampleFormat::F32 => {
            let b = &bytes[4 * idx..4 * idx + 4];
            f32::from_le_bytes([b[0], b[1], b[2], b[3]])
        }
    }
}

/// Decodes `out.len()` contiguous samples from `bytes` into µV.
pub(crate) fn decode_run(format: SampleFormat, bytes: &[u8], out: &mut [f32], gain: f32, offset: f32) {
    match format {
        SampleFormat::I16 => {
            for (o, b) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *o = i16::from_le_bytes([b[0], b[1]]) as f32 * gain + offset;
            }
        }
        SampleFormat::U16 => {
            for (o, b) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *o = u16::from_le_bytes([b[0], b[1]]) as f32 * gain + offset;
            }
        }
        SampleFormat::F32 => {
            for (o, b) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                *o = f32::from_le_bytes([b[0], b[1], b[2], b[3]]) * gain + offset;
            }
        }
    }
}

/// Appends `value_uv` to `dst` as a stored sample (integers are rounded and saturated).
#[inline(always)]
pub(crate) fn encode(format: SampleFormat, value_uv: f32, gain: f32, offset: f32, dst: &mut Vec<u8>) {
    let stored = (value_uv - offset) / gain;
    match format {
        SampleFormat::I16 => dst.extend_from_slice(&(stored.round() as i16).to_le_bytes()),
        SampleFormat::U16 => dst.extend_from_slice(&(stored.round() as u16).to_le_bytes()),
        SampleFormat::F32 => dst.extend_from_slice(&stored.to_le_bytes()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_and_saturation() {
        let mut bytes = Vec::new();
        for v in [-5.0f32, 0.0, 7.4, 1e9] {
            encode(SampleFormat::I16, v, 0.5, 0.0, &mut bytes);
        }
        let mut out = [0.0f32; 4];
        decode_run(SampleFormat::I16, &bytes, &mut out, 0.5, 0.0);
        assert_eq!(out, [-5.0, 0.0, 7.5, i16::MAX as f32 * 0.5]);
        assert_eq!(decode(SampleFormat::I16, &bytes, 1), 0.0);

        let mut f = Vec::new();
        encode(SampleFormat::F32, 3.25, 1.0, 0.0, &mut f);
        assert_eq!(decode(SampleFormat::F32, &f, 0), 3.25);
    }
}
