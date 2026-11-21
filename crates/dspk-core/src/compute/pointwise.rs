//! A run of pointwise ops applied to one value: the one definition the fused pointwise kernel
//! (`dspk-ops`) and the fused reduction (`dspk-view`, a zoomed-out view's last ops folded into its
//! min/max) both call, so a fused chain computes exactly what its ops do one kernel each.
//!
//! A run is two device arrays: `codes[k]` (one of the `POINTWISE_*` codes) and `params[2k]`,
//! `params[2k + 1]` (that op's two numbers). The formulas are those of the single-op kernels
//! (`dspk-ops` `scale_samples_kernel`, `clamp_samples_kernel`, `elementwise_kernel`), operation
//! for operation.

use cubecl::prelude::*;

/// `x · a + b` (scale, offset; a baseline is `1, −baseline`).
pub const POINTWISE_AFFINE: u32 = 0;
/// `clamp(x, a, b)`.
pub const POINTWISE_CLAMP: u32 = 1;
/// `|x|`.
pub const POINTWISE_ABS: u32 = 2;
/// `x²`.
pub const POINTWISE_SQUARE: u32 = 3;
/// `√max(x, 0)`.
pub const POINTWISE_SQRT: u32 = 4;
/// `log₁₀ max(|x|, a)` (`a`: the floor).
pub const POINTWISE_LOG10: u32 = 5;

/// `x` through the first `count` ops of a run.
#[cube]
pub fn apply_pointwise<F: Float>(x: F, codes: &[u32], params: &[F], count: u32) -> F {
    let mut v = x;
    for k in 0..count {
        let code = codes[k as usize];
        let a = params[(2u32 * k) as usize];
        let b = params[(2u32 * k + 1u32) as usize];
        if code == 0u32 {
            v = v * a + b;
        } else if code == 1u32 {
            v = F::clamp(v, a, b);
        } else if code == 2u32 {
            v = F::abs(v);
        } else if code == 3u32 {
            v = v * v;
        } else if code == 4u32 {
            v = F::sqrt(F::max(v, F::new(0.0f32)));
        } else {
            v = F::ln(F::max(F::abs(v), a)) * F::new(core::f32::consts::LOG10_E);
        }
    }
    v
}
