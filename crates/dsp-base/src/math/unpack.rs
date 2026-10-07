//! Stored samples to scaled values on the device.
//!
//! Recordings stored as integers upload their stored values (half the bytes of f32 for int16) and
//! are scaled here with each channel's gain and offset (into the channel's unit). Values arrive as 32-bit words because
//! WGSL has no 8- or 16-bit integer types; the kernel extracts and sign-extends them.

use cubecl::prelude::*;
use dsp_core::compute::LaunchGeometry;
use dsp_core::{DspError, DspResult, SampleFormat};

use crate::core::DspFloat;

/// Bytes per upload word (WGSL has no 8- or 16-bit integers).
const WORD_BYTES: usize = size_of::<u32>();

/// How a stored value's bits are read.
const SIGNED: u32 = 0;
const UNSIGNED: u32 = 1;
const FLOAT: u32 = 2;

/// Converts `total` stored values (`bytes` each, packed little-endian into `words`, channel-major
/// rows of `num_samples`) to `output[i] = value · gains[ch] + offsets[ch]`. One unit per value
/// (`ABSOLUTE_POS`).
#[cube(launch)]
pub fn unpack_stored_kernel<F: Float>(
    words: &[u32],
    gains: &[F],
    offsets: &[F],
    output: &mut [F],
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
            F::cast_from(f32::reinterpret(raw))
        } else if comptime!(kind == UNSIGNED) {
            F::cast_from(raw)
        } else if comptime!(bytes == 4) {
            // Two's complement: negate the magnitude so small negative values stay exact
            if raw >= 0x8000_0000u32 { -F::cast_from((!raw) + 1u32) } else { F::cast_from(raw) }
        } else {
            let half = comptime!(1u32 << (bytes * 8 - 1));
            if raw >= half { F::cast_from(raw) - F::cast_from(comptime!(1u32 << (bytes * 8))) } else { F::cast_from(raw) }
        };
        output[idx as usize] = value * gains[ch as usize] + offsets[ch as usize];
    }
}

/// Device buffer of `stored` little-endian values as 32-bit words for [`execute_unpack_stored`].
/// Uploaded as is when it fills whole words (no host copy); otherwise padded through
/// [`stored_words`]. Words are read little-endian on the device, which every CubeCL target is.
pub fn upload_stored(client: &Client, stored: &[u8]) -> cubecl::server::Handle {
    if stored.len() % WORD_BYTES == 0 && !stored.is_empty() {
        client.create_from_slice(stored)
    } else {
        client.create_from_slice(u32::as_bytes(&stored_words(stored)))
    }
}

/// Uploads-ready words for `stored` little-endian values (padded to whole 32-bit words).
pub fn stored_words(stored: &[u8]) -> Vec<u32> {
    stored.chunks(WORD_BYTES).map(|c| {
        let mut w = [0u8; WORD_BYTES];
        w[..c.len()].copy_from_slice(c);
        u32::from_le_bytes(w)
    }).collect()
}

/// Converts a device buffer of stored `format` values (`[channels, samples]`, as uploaded from
/// [`stored_words`]) into scaled `F` values in `output` using per-channel `gains` / `offsets` (device
/// buffers of `channels` values of `F`). `float64` storage is not unpacked on the device; convert
/// those on the host.
#[allow(clippy::too_many_arguments)]
pub fn execute_unpack_stored<F: DspFloat>(
    client: &Client,
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
        unpack_stored_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(words.clone(), (total * bytes as usize).div_ceil(WORD_BYTES)),
            BufferArg::from_raw_parts(gains.clone(), channels),
            BufferArg::from_raw_parts(offsets.clone(), channels),
            BufferArg::from_raw_parts(output.clone(), total),
            samples as u32,
            total as u32,
            bytes,
            kind,
        );
    }
    Ok(())
}
