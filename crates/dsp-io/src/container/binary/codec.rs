//! Little-endian sample decoding / encoding shared by the binary formats.

use dsp_core::{RecordingInfo, SampleFormat};

/// Calls `$f::<B, T>(… , value)` for the stored type of `$format` (`B` bytes, little-endian).
macro_rules! with_format {
    ($format:expr, $f:ident($($arg:expr),*)) => {
        match $format {
            SampleFormat::I8 => $f::<1>($($arg,)* |b| i8::from_le_bytes(b) as f32),
            SampleFormat::I16 => $f::<2>($($arg,)* |b| i16::from_le_bytes(b) as f32),
            SampleFormat::U16 => $f::<2>($($arg,)* |b| u16::from_le_bytes(b) as f32),
            SampleFormat::I32 => $f::<4>($($arg,)* |b| i32::from_le_bytes(b) as f32),
            SampleFormat::F32 => $f::<4>($($arg,)* f32::from_le_bytes),
            SampleFormat::F64 => $f::<8>($($arg,)* |b| f64::from_le_bytes(b) as f32),
        }
    };
}

/// Applies each channel's gain and offset to time-major stored values (`out[t · channels + c]`)
/// of `info`'s channels, giving µV.
pub(crate) fn scale_frames(info: &RecordingInfo, out: &mut [f32]) {
    let ch = &info.channels;
    if ch.is_empty() {
        return;
    }
    let (g, o) = (ch[0].gain, ch[0].offset);
    if ch.iter().all(|c| c.gain == g && c.offset == o) {
        if (g, o) != (1.0, 0.0) {
            out.iter_mut().for_each(|v| *v = *v * g + o);
        }
        return;
    }
    let (gains, offsets): (Vec<f32>, Vec<f32>) = ch.iter().map(|c| (c.gain, c.offset)).unzip();
    for frame in out.chunks_exact_mut(ch.len()) {
        for ((v, g), o) in frame.iter_mut().zip(&gains).zip(&offsets) {
            *v = *v * g + o;
        }
    }
}

/// Decodes `channels` (`(stored channel, gain, offset)`) of `n` interleaved frames of `frame_bytes`
/// each into channel-major µV rows of `out` (`out[i · n + s]`). Every frame is read once for all
/// requested channels.
pub(crate) fn decode_frames(
    format: SampleFormat,
    frames: &[u8],
    frame_bytes: usize,
    channels: &[(usize, f32, f32)],
    n: usize,
    out: &mut [f32],
) {
    fn run<const B: usize>(
        frames: &[u8],
        frame_bytes: usize,
        channels: &[(usize, f32, f32)],
        n: usize,
        out: &mut [f32],
        value: impl Fn([u8; B]) -> f32,
    ) {
        for (s, frame) in frames.chunks_exact(frame_bytes).take(n).enumerate() {
            for (i, &(ch, gain, offset)) in channels.iter().enumerate() {
                let b: [u8; B] = frame[ch * B..ch * B + B].try_into().expect("sample inside frame");
                out[i * n + s] = value(b) * gain + offset;
            }
        }
    }
    with_format!(format, run(frames, frame_bytes, channels, n, out))
}

/// Copies the stored values of block columns `cols` into channel-major `out` (`bytes` per value).
/// The block is `[n][width]` values when `time_major`, else `[width][n]`.
pub(crate) fn select_stored(block: &[u8], time_major: bool, width: usize, n: usize, cols: &[usize], bytes: usize, out: &mut [u8]) {
    fn frames<const B: usize>(block: &[u8], width: usize, n: usize, cols: &[usize], out: &mut [u8]) {
        let (block, _) = block.as_chunks::<B>();
        let (out, _) = out.as_chunks_mut::<B>();
        for (t, frame) in block.chunks_exact(width).take(n).enumerate() {
            for (i, &c) in cols.iter().enumerate() {
                out[i * n + t] = frame[c];
            }
        }
    }
    if time_major {
        match bytes {
            1 => frames::<1>(block, width, n, cols, out),
            2 => frames::<2>(block, width, n, cols, out),
            4 => frames::<4>(block, width, n, cols, out),
            8 => frames::<8>(block, width, n, cols, out),
            other => unreachable!("{other}-byte samples"),
        }
    } else {
        for (i, &c) in cols.iter().enumerate() {
            out[i * n * bytes..(i + 1) * n * bytes].copy_from_slice(&block[c * n * bytes..(c + 1) * n * bytes]);
        }
    }
}

/// Decoded values in memory order to little-endian (a no-op on little-endian hosts).
pub(crate) fn native_to_le(values: &mut [u8], bytes: usize) {
    if cfg!(target_endian = "big") {
        for v in values.chunks_exact_mut(bytes) {
            v.reverse();
        }
    }
}

/// Appends `value_uv` to `dst` as a stored sample (integers are rounded and saturated).
#[inline(always)]
pub(crate) fn encode(format: SampleFormat, value_uv: f32, gain: f32, offset: f32, dst: &mut Vec<u8>) {
    let stored = (value_uv - offset) / gain;
    match format {
        SampleFormat::I8 => dst.extend_from_slice(&(stored.round() as i8).to_le_bytes()),
        SampleFormat::I16 => dst.extend_from_slice(&(stored.round() as i16).to_le_bytes()),
        SampleFormat::U16 => dst.extend_from_slice(&(stored.round() as u16).to_le_bytes()),
        SampleFormat::I32 => dst.extend_from_slice(&(stored.round() as i32).to_le_bytes()),
        SampleFormat::F32 => dst.extend_from_slice(&stored.to_le_bytes()),
        SampleFormat::F64 => dst.extend_from_slice(&(stored as f64).to_le_bytes()),
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
        SampleFormat::I16.decode(&bytes, &mut out, 0.5, 0.0);
        assert_eq!(out, [-5.0, 0.0, 7.5, i16::MAX as f32 * 0.5]);

        // Two interleaved frames of three channels; channels 2 and 0 requested.
        let mut frames = Vec::new();
        for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            encode(SampleFormat::I16, v, 1.0, 0.0, &mut frames);
        }
        let mut rows = [0.0f32; 4];
        decode_frames(SampleFormat::I16, &frames, 6, &[(2, 2.0, 1.0), (0, 1.0, 0.0)], 2, &mut rows);
        assert_eq!(rows, [7.0, 13.0, 1.0, 4.0]);

        // Every format round-trips through encode / decode
        for format in SampleFormat::ALL {
            let mut b = Vec::new();
            // Values every format holds (uint16 has no negatives, int8 stops at 127)
            for v in [0.0f32, 42.0, 100.0] {
                encode(format, v, 1.0, 0.0, &mut b);
            }
            let mut back = [0.0f32; 3];
            format.decode(&b, &mut back, 1.0, 0.0);
            assert_eq!(back, [0.0, 42.0, 100.0], "{format:?}");
        }

        let mut f = Vec::new();
        encode(SampleFormat::F32, 3.25, 1.0, 0.0, &mut f);
        let mut v = [0.0f32];
        SampleFormat::F32.decode(&f, &mut v, 1.0, 0.0);
        assert_eq!(v, [3.25]);
    }
}
