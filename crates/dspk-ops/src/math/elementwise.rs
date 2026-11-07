//! One function of each sample: rectify (`|x|`), square, square root, base-10 logarithm. With
//! filters they make the usual envelopes: rectify then low-pass (EMG envelope); square, moving
//! average, square root (moving RMS); log of an energy (dB-like scales).

use cubecl::prelude::*;
use cubecl::server::Handle;
use dspk_core::compute::LaunchGeometry;

use super::kernels::elementwise_kernel;
use crate::core::{cast, DspFloat};

/// Smallest magnitude [`Elementwise::Log10`] takes the logarithm of (`log₁₀` of it: −12): zeros
/// and silence give a finite floor instead of −∞.
pub const LOG10_FLOOR: f32 = 1e-12;

/// The function [`execute_elementwise`] applies to each sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Elementwise {
    /// `|x|` (full-wave rectification).
    Abs,
    /// `x²` (instantaneous power).
    Square,
    /// `√x`, negative values taken as 0.
    Sqrt,
    /// `log₁₀ |x|`, magnitudes below [`LOG10_FLOOR`] taken as it.
    Log10,
}

impl Elementwise {
    fn code(self) -> u32 {
        match self {
            Elementwise::Abs => 0,
            Elementwise::Square => 1,
            Elementwise::Sqrt => 2,
            Elementwise::Log10 => 3,
        }
    }

    /// The function on the host (references and tests).
    pub fn apply(self, x: f64) -> f64 {
        match self {
            Elementwise::Abs => x.abs(),
            Elementwise::Square => x * x,
            Elementwise::Sqrt => x.max(0.0).sqrt(),
            Elementwise::Log10 => x.abs().max(LOG10_FLOOR as f64).log10(),
        }
    }
}

/// `output = f(input)` over `len` values (any shape: element-wise).
pub fn execute_elementwise<F: DspFloat>(client: &Client, input: &Handle, output: &Handle, len: usize, function: Elementwise) {
    let geom = LaunchGeometry::elementwise(client, len);
    unsafe {
        elementwise_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(input.clone(), len),
            BufferArg::from_raw_parts(output.clone(), len),
            cast::<F>(LOG10_FLOOR as f64),
            function.code(),
        );
    }
}
