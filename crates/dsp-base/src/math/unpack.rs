//! Stored samples to scaled values on the device.
//!
//! Recordings stored as integers upload their stored values (half the bytes of f32 for int16) and
//! are scaled here with each channel's gain and offset (into the channel's unit). Values arrive as 32-bit words because
//! WGSL has no 8- or 16-bit integer types; the kernel extracts and sign-extends them.

use cubecl::prelude::*;
use dsp_core::compute::LaunchGeometry;
use dsp_core::{DspError, DspResult, SampleFormat};

use super::kernels::unpack::{FLOAT, SIGNED, UNSIGNED};
use super::kernels::unpack_stored_kernel;
use crate::core::DspFloat;

/// Bytes per upload word (WGSL has no 8- or 16-bit integers).
const WORD_BYTES: usize = size_of::<u32>();


/// Device buffer of `stored` little-endian values as 32-bit words for [`execute_unpack_stored`].
/// Uploaded as is when it fills whole words; otherwise padded through [`stored_words`]. Words are
/// read little-endian on the device, which every CubeCL target is.
pub fn upload_stored(client: &Client, stored: &[u8]) -> cubecl::server::Handle {
    client.create(cubecl::bytes::Bytes::from_bytes_vec(stored_bytes(stored)))
}

/// Bytes needed for `stored` as whole 32-bit words.
pub fn stored_word_bytes(stored_len: usize) -> usize {
    stored_len.div_ceil(WORD_BYTES).max(1) * WORD_BYTES
}

/// Writes `stored` (as [`upload_stored`] lays it out) to the start of the existing device buffer
/// `handle` (at least [`stored_word_bytes`] bytes): no allocation, see [`crate::core::buffer::write`].
pub fn write_stored(client: &Client, handle: &cubecl::server::Handle, stored: &[u8]) {
    client.write(handle, cubecl::bytes::Bytes::from_bytes_vec(stored_bytes(stored)));
}

/// [`write_stored`] taking ownership of `stored` (padded in place to whole words): no copy when it
/// already fills whole words.
pub fn write_stored_owned(client: &Client, handle: &cubecl::server::Handle, mut stored: Vec<u8>) {
    let words = stored_word_bytes(stored.len());
    stored.resize(words, 0);
    client.write(handle, cubecl::bytes::Bytes::from_elems(stored));
}

/// `stored` padded to whole 32-bit words (copied once).
fn stored_bytes(stored: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(stored_word_bytes(stored.len()));
    bytes.extend_from_slice(stored);
    bytes.resize(stored_word_bytes(stored.len()), 0);
    bytes
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
