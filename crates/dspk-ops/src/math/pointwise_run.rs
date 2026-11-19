//! Runs of consecutive pointwise stages (scale, baseline, clamp, rectify, square, sqrt, log₁₀)
//! fused into one kernel: each sample is read once and written once for the whole run, instead of
//! once per stage. The ops are those of the single-op kernels, operation for operation
//! (`dspk_core::compute::apply_pointwise`), so a fused run gives what its stages give one by one.

use cubecl::prelude::*;
use cubecl::server::Handle;
use dspk_core::compute::{
    apply_pointwise, element_position, LaunchGeometry, POINTWISE_ABS, POINTWISE_AFFINE, POINTWISE_CLAMP, POINTWISE_LOG10, POINTWISE_SQRT, POINTWISE_SQUARE,
};

use crate::core::{buffer, cast, DspFloat};
use crate::math::{Elementwise, LOG10_FLOOR};
use crate::pipeline::PipelineStage;

/// A pointwise stage as `(code, a, b)` (`dspk_core::compute::POINTWISE_*`), or `None` for any
/// other stage.
pub fn pointwise_op(stage: &PipelineStage) -> Option<(u32, f64, f64)> {
    Some(match stage {
        PipelineStage::Scale { alpha, beta } => (POINTWISE_AFFINE, *alpha as f64, *beta as f64),
        PipelineStage::SubtractBaseline { baseline } => (POINTWISE_AFFINE, 1.0, -*baseline as f64),
        PipelineStage::Clamp { min, max } => (POINTWISE_CLAMP, *min as f64, *max as f64),
        PipelineStage::Elementwise(f) => match f {
            Elementwise::Abs => (POINTWISE_ABS, 0.0, 0.0),
            Elementwise::Square => (POINTWISE_SQUARE, 0.0, 0.0),
            Elementwise::Sqrt => (POINTWISE_SQRT, 0.0, 0.0),
            Elementwise::Log10 => (POINTWISE_LOG10, LOG10_FLOOR as f64, 0.0),
        },
        _ => return None,
    })
}

/// A run of pointwise ops on the device: its codes and parameters, uploaded once.
pub struct PointwiseRun {
    codes: Handle,
    params: Handle,
    count: usize,
}

impl PointwiseRun {
    /// The run of `ops` (`(code, a, b)`, see [`pointwise_op`]), uploaded for element type `F`.
    pub fn upload<F: DspFloat>(client: &Client, ops: &[(u32, f64, f64)]) -> Self {
        let mut codes: Vec<u32> = ops.iter().map(|o| o.0).collect();
        let mut params: Vec<F> = ops.iter().flat_map(|o| [cast::<F>(o.1), cast::<F>(o.2)]).collect();
        // A binding may not be empty: an empty run still binds one op's worth (never applied)
        if codes.is_empty() {
            codes.push(POINTWISE_AFFINE);
            params.extend([cast::<F>(1.0), cast::<F>(0.0)]);
        }
        Self { codes: client.create_from_slice(u32::as_bytes(&codes)), params: buffer::upload(client, &params), count: ops.len() }
    }

    /// Ops in the run.
    pub fn count(&self) -> usize {
        self.count
    }

    /// The device array of codes (`count` values), for a kernel that applies the run itself.
    pub fn codes(&self) -> &Handle {
        &self.codes
    }

    /// The device array of parameters (`2 · count` values of `F`).
    pub fn params(&self) -> &Handle {
        &self.params
    }

    /// `output = run(input)` over `len` values.
    pub fn apply<F: DspFloat>(&self, client: &Client, input: &Handle, output: &Handle, len: usize) {
        if len == 0 {
            return;
        }
        let geom = LaunchGeometry::elementwise(client, len);
        // SAFETY: input and output hold `len` values of `F`; codes `count` (≥ 1) u32, params
        // `2 · count` (≥ 2) values of `F`
        unsafe {
            pointwise_run_kernel::launch::<F>(
                client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(input.clone(), len),
                BufferArg::from_raw_parts(output.clone(), len),
                BufferArg::from_raw_parts(self.codes.clone(), self.count.max(1)),
                BufferArg::from_raw_parts(self.params.clone(), 2 * self.count.max(1)),
                self.count as u32,
            );
        }
    }
}

#[cube(launch)]
fn pointwise_run_kernel<F: Float + CubeElement + LaunchArg>(input: &[F], output: &mut [F], codes: &[u32], params: &[F], count: u32) {
    let i = element_position();
    if i < input.len() {
        output[i] = apply_pointwise::<F>(input[i], codes, params, count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{execute_clamp, execute_elementwise, execute_scaling};
    use dspk_core::compute::{ComputeTarget, ComputeTask};

    struct SameAsOneByOne;

    impl ComputeTask for SameAsOneByOne {
        type Output = ();
        fn run(self, client: Client) {
            let n = 10_007;
            let x: Vec<f32> = (0..n).map(|i| ((i * 7919) % 2003) as f32 * 0.37 - 300.0).collect();
            let stages = [
                PipelineStage::Scale { alpha: 1.7, beta: -0.25 },
                PipelineStage::Elementwise(Elementwise::Abs),
                PipelineStage::SubtractBaseline { baseline: 12.5 },
                PipelineStage::Clamp { min: -40.0, max: 400.0 },
                PipelineStage::Elementwise(Elementwise::Square),
                PipelineStage::Elementwise(Elementwise::Sqrt),
                PipelineStage::Elementwise(Elementwise::Log10),
            ];
            // One kernel per stage, as the pipeline ran them
            let mut a = buffer::upload(&client, &x);
            for stage in &stages {
                let out = buffer::empty::<f32>(&client, n);
                match stage {
                    PipelineStage::Scale { alpha, beta } => execute_scaling::<f32>(&client, &a, &out, n, *alpha, *beta),
                    PipelineStage::SubtractBaseline { baseline } => execute_scaling::<f32>(&client, &a, &out, n, 1.0, -*baseline),
                    PipelineStage::Clamp { min, max } => execute_clamp::<f32>(&client, &a, &out, n, *min, *max),
                    PipelineStage::Elementwise(f) => execute_elementwise::<f32>(&client, &a, &out, n, *f),
                    _ => unreachable!(),
                }
                a = out;
            }
            let one_by_one: Vec<f32> = buffer::download(&client, a);
            // The run, fused
            let ops: Vec<_> = stages.iter().map(|s| pointwise_op(s).unwrap()).collect();
            let run = PointwiseRun::upload::<f32>(&client, &ops);
            let (input, out) = (buffer::upload(&client, &x), buffer::empty::<f32>(&client, n));
            run.apply::<f32>(&client, &input, &out, n);
            let fused: Vec<f32> = buffer::download(&client, out);
            for (i, (f, o)) in fused.iter().zip(&one_by_one).enumerate() {
                // The same operations; a compiler may contract `x · a + b` differently in one
                // kernel than in another: at most one rounding step
                let tol = 2.0 * f32::EPSILON * o.abs().max(1.0);
                assert!((f - o).abs() <= tol, "{} sample {i}: fused {f}, one by one {o}", client.name());
            }
        }
    }

    #[test]
    fn a_fused_run_equals_its_stages_one_by_one() {
        for target in ComputeTarget::for_tests() {
            target.run(SameAsOneByOne).expect("compiled-in runtime");
        }
    }
}
