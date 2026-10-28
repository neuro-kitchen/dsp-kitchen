//! Filling a buffer with one value on the device.

use cubecl::prelude::*;
use dspk_core::compute::element_position;

use crate::core::DspFloat;

/// `out[i] = value` for `i < len`. One unit per element.
#[cube(launch)]
pub fn fill_kernel<F: DspFloat>(out: &mut [F], value: F, len: u32) {
    let i = element_position() as u32;
    if i < len {
        out[i as usize] = value;
    }
}
