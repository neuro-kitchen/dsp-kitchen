//! Stored samples to µV on the device.
//!
//! Recordings stored as integers upload their stored values (half the bytes of f32 for int16) and
//! are scaled here with each channel's gain and offset. Values arrive as 32-bit words because
//! WGSL has no 8- or 16-bit integer types; the kernel extracts and sign-extends them.

use cubecl::prelude::*;
use dsp_core::compute::LaunchGeometry;
use dsp_core::{DspError, DspResult, SampleFormat};

/// How a stored value's bits are read.
const SIGNED: u32 = 0;
const UNSIGNED: u32 = 1;
const FLOAT: u32 = 2;

/// Converts `total` stored values (`bytes` each, packed little-endian into `words`, channel-major
/// rows of `num_samples`) to `output[i] = value · gains[ch] + offsets[ch]`. One unit per value
/// (`ABSOLUTE_POS`).
#[cube(launch)]
pub fn unpack_stored_kernel(
    words: &Array<u32>,
    gains: &Array<f32>,
    offsets: &Array<f32>,
    output: &mut Array<f32>,
    num_samples: u32,
    total: u32,
    #[comptime] bytes: u32,
    #[comptime] kind: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total {
        let ch = idx / num_samples;
        let per_word = comptime!(4 / bytes);
        let bits = comptime!(bytes * 8);
        let raw = if comptime!(bytes == 4) {
            words[idx as usize]
        } else {
            let word = words[(idx / per_word) as usize];
            let shift = (idx % per_word) * bits;
            (word >> shift) & comptime!((1u32 << (bytes * 8)) - 1)
        };
        let value = if comptime!(kind == FLOAT) {
            f32::reinterpret(raw)
        } else if comptime!(kind == UNSIGNED) {
            f32::cast_from(raw)
        } else if comptime!(bytes == 4) {
            // Two's complement: negate the magnitude so small negative values stay exact
            if raw >= 0x8000_0000u32 { -f32::cast_from((!raw) + 1u32) } else { f32::cast_from(raw) }
        } else {
            let half = comptime!(1u32 << (bytes * 8 - 1));
            if raw >= half { f32::cast_from(raw) - comptime!((1u64 << (bytes * 8)) as f32) } else { f32::cast_from(raw) }
        };
        output[idx as usize] = value * gains[ch as usize] + offsets[ch as usize];
    }
}

/// Uploads-ready words for `stored` little-endian values (padded to whole 32-bit words).
pub fn stored_words(stored: &[u8]) -> Vec<u32> {
    stored.chunks(4).map(|c| {
        let mut w = [0u8; 4];
        w[..c.len()].copy_from_slice(c);
        u32::from_le_bytes(w)
    }).collect()
}

/// Converts a device buffer of stored `format` values (`[channels, samples]`, as uploaded from
/// [`stored_words`]) into f32 µV in `output` using per-channel `gains` / `offsets` (device buffers
/// of `channels` floats). `float64` has no device representation; convert those on the host.
#[allow(clippy::too_many_arguments)]
pub fn execute_unpack_stored<R: Runtime>(
    client: &ComputeClient<R>,
    words: &cubecl::server::Handle,
    format: SampleFormat,
    gains: &cubecl::server::Handle,
    offsets: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    channels: usize,
    samples: usize,
) -> DspResult<()> {
    let (bytes, kind) = match format {
        SampleFormat::I8 => (1, SIGNED),
        SampleFormat::I16 => (2, SIGNED),
        SampleFormat::U16 => (2, UNSIGNED),
        SampleFormat::I32 => (4, SIGNED),
        SampleFormat::F32 => (4, FLOAT),
        SampleFormat::F64 => return Err(DspError::UnsupportedFormat("float64 samples on the device".into())),
    };
    let total = channels * samples;
    if total == 0 {
        return Ok(());
    }
    let geom = LaunchGeometry::elementwise(client, total);
    unsafe {
        unpack_stored_kernel::launch::<R>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ArrayArg::from_raw_parts(words.clone(), (total * bytes as usize).div_ceil(4)),
            ArrayArg::from_raw_parts(gains.clone(), channels),
            ArrayArg::from_raw_parts(offsets.clone(), channels),
            ArrayArg::from_raw_parts(output.clone(), total),
            samples as u32,
            total as u32,
            bytes,
            kind,
        );
    }
    Ok(())
}
