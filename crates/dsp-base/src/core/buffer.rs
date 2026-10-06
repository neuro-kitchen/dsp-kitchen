//! Typed device buffers: sizes come from the element type, never from a literal byte count.

use cubecl::prelude::*;
use cubecl::server::Handle;

/// Bytes of `len` elements of `E`, at least one element (runtimes reject empty allocations).
pub fn bytes<E: CubeElement>(len: usize) -> usize {
    len.max(1) * size_of::<E>()
}

/// Uninitialized device buffer of `len` elements of `E`.
pub fn empty<R: Runtime, E: CubeElement>(client: &ComputeClient<R>, len: usize) -> Handle {
    client.empty(bytes::<E>(len))
}

/// Device copy of `data`.
pub fn upload<R: Runtime, E: CubeElement>(client: &ComputeClient<R>, data: &[E]) -> Handle {
    client.create_from_slice(E::as_bytes(data))
}

/// Device buffer of `len` zeros.
pub fn zeros<R: Runtime, E: CubeElement + Default>(client: &ComputeClient<R>, len: usize) -> Handle {
    upload(client, &vec![E::default(); len.max(1)])
}

/// Host copy of a device buffer of `E` (waits for the queued work that writes it).
pub fn download<R: Runtime, E: CubeElement>(client: &ComputeClient<R>, handle: Handle) -> Vec<E> {
    E::from_bytes(&client.read_one_unchecked(handle)).to_vec()
}

/// Host copy of the first `len` elements of `E` of a device buffer: only those bytes are read, so a
/// buffer sized for the largest batch costs only the current batch's transfer.
pub fn download_prefix<R: Runtime, E: CubeElement>(client: &ComputeClient<R>, handle: Handle, len: usize) -> Vec<E> {
    let used = handle.size_in_used();
    let want = (len * size_of::<E>()) as u64;
    assert!(want <= used, "{len} elements exceed the buffer's {used} bytes");
    let mut out = download::<R, E>(client, handle.offset_end(used - want));
    out.truncate(len);
    out
}

/// Host copy of elements `start .. start + len` of `E` of a device buffer: only those bytes are
/// read.
pub fn download_range<R: Runtime, E: CubeElement>(client: &ComputeClient<R>, handle: Handle, start: usize, len: usize) -> Vec<E> {
    let used = handle.size_in_used();
    let (from, want) = ((start * size_of::<E>()) as u64, (len * size_of::<E>()) as u64);
    assert!(from + want <= used, "elements {start}..{} exceed the buffer's {used} bytes", start + len);
    let mut out = download::<R, E>(client, handle.offset_start(from).offset_end(used - from - want));
    out.truncate(len);
    out
}

/// `handle` without its last `unused` elements of `E` (a view of a buffer larger than the data).
pub fn truncate<E: CubeElement>(handle: Handle, unused: usize) -> Handle {
    handle.offset_end((unused * size_of::<E>()) as u64)
}
