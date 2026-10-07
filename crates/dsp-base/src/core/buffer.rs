//! Typed device buffers: sizes come from the element type, never from a literal byte count.

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_core::compute::MAX_DEVICE_ELEMENTS;

/// Bytes of `len` elements of `E`, at least one element (runtimes reject empty allocations).
///
/// # Panics
/// If `len` exceeds [`MAX_DEVICE_ELEMENTS`] (kernels index with 32-bit integers): sizes chosen by
/// users are checked earlier with [`dsp_core::compute::device_elements`], so this is a backstop
/// against a silent wrap.
pub fn bytes<E: CubeElement>(len: usize) -> usize {
    assert!(len <= MAX_DEVICE_ELEMENTS, "device buffer of {len} elements exceeds the 32-bit index limit ({MAX_DEVICE_ELEMENTS})");
    len.max(1) * size_of::<E>()
}

/// Uninitialized device buffer of `len` elements of `E`.
pub fn empty<E: CubeElement>(client: &Client, len: usize) -> Handle {
    client.empty(bytes::<E>(len))
}

/// Device copy of `data`.
///
/// # Panics
/// If `data` holds more than [`MAX_DEVICE_ELEMENTS`] elements (see [`bytes`]).
pub fn upload<E: CubeElement>(client: &Client, data: &[E]) -> Handle {
    bytes::<E>(data.len());
    client.create_from_slice(E::as_bytes(data))
}

/// Device buffer of `len` zeros.
pub fn zeros<E: CubeElement + Default>(client: &Client, len: usize) -> Handle {
    upload(client, &vec![E::default(); len.max(1)])
}

/// Host copy of a device buffer of `E` (waits for the queued work that writes it).
pub fn download<E: CubeElement>(client: &Client, handle: Handle) -> Vec<E> {
    E::from_bytes(&client.read_one_unchecked(handle)).to_vec()
}

/// Host copy of the first `len` elements of `E` of a device buffer: only those bytes are read, so a
/// buffer sized for the largest batch costs only the current batch's transfer.
pub fn download_prefix<E: CubeElement>(client: &Client, handle: Handle, len: usize) -> Vec<E> {
    let used = handle.size_in_used();
    let want = (len * size_of::<E>()) as u64;
    assert!(want <= used, "{len} elements exceed the buffer's {used} bytes");
    let mut out = download::<E>(client, handle.offset_end(used - want));
    out.truncate(len);
    out
}

/// Host copy of elements `start .. start + len` of `E` of a device buffer: only those bytes are
/// read.
pub fn download_range<E: CubeElement>(client: &Client, handle: Handle, start: usize, len: usize) -> Vec<E> {
    let used = handle.size_in_used();
    let (from, want) = ((start * size_of::<E>()) as u64, (len * size_of::<E>()) as u64);
    assert!(from + want <= used, "elements {start}..{} exceed the buffer's {used} bytes", start + len);
    let mut out = download::<E>(client, handle.offset_start(from).offset_end(used - from - want));
    out.truncate(len);
    out
}

/// `handle` without its last `unused` elements of `E` (a view of a buffer larger than the data).
pub fn truncate<E: CubeElement>(handle: Handle, unused: usize) -> Handle {
    handle.offset_end((unused * size_of::<E>()) as u64)
}
