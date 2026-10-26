use cubecl::prelude::*;
use num_traits::NumCast;

/// Element type of dsp-base kernels: a CubeCL float that also crosses to and from the host.
///
/// Kernels take `F: Float`; host dispatchers take `F: DspFloat` so they can size, upload and read
/// buffers of `F`. `f32` runs on every runtime; `f64` and `f16` only where the runtime supports them.
pub trait DspFloat: Float + CubeElement + LaunchArg<RuntimeArg = Self> {}

impl<F: Float + CubeElement + LaunchArg<RuntimeArg = F>> DspFloat for F {}

/// `value` as `F` (rounded to its precision).
pub fn cast<F: DspFloat>(value: f64) -> F {
    <F as NumCast>::from(value).expect("finite value representable in the kernel float type")
}

/// `values` as `F`.
pub fn cast_all<F: DspFloat>(values: &[f64]) -> Vec<F> {
    values.iter().map(|&v| cast(v)).collect()
}

/// `values` as `F` from `f32` host data.
pub fn cast_f32<F: DspFloat>(values: &[f32]) -> Vec<F> {
    values.iter().map(|&v| cast(v as f64)).collect()
}

/// `value` as `f64`.
pub fn to_f64<F: DspFloat>(value: F) -> f64 {
    value.to_f64().expect("float converts to f64")
}
