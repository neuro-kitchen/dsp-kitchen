use cubecl::prelude::*;
use cubecl::server::Handle;

use super::buffer;

/// A device buffer reused across calls, reallocated only when a call needs more than it holds.
#[derive(Debug, Clone, Default)]
pub struct Scratch {
    handle: Option<Handle>,
    capacity_bytes: usize,
}

impl Scratch {
    pub fn new() -> Self {
        Self::default()
    }

    /// A buffer of at least `len` elements of `E` (contents unspecified).
    pub fn get<R: Runtime, E: CubeElement>(&mut self, client: &ComputeClient<R>, len: usize) -> Handle {
        let need = buffer::bytes::<E>(len);
        match &self.handle {
            Some(handle) if self.capacity_bytes >= need => handle.clone(),
            _ => {
                let handle = client.empty(need);
                self.handle = Some(handle.clone());
                self.capacity_bytes = need;
                handle
            }
        }
    }
}
