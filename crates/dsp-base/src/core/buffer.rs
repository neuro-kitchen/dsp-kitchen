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

/// `handle` without its last `unused` elements of `E` (a view of a buffer larger than the data).
pub fn truncate<E: CubeElement>(handle: Handle, unused: usize) -> Handle {
    handle.offset_end((unused * size_of::<E>()) as u64)
}
