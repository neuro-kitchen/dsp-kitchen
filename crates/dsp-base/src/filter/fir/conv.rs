//! FIR dispatch: a direct kernel and a shared-memory tiled kernel per filter shape, chosen per device
//! and problem size by CubeCL's autotuner (the tiled kernel only where its tile fits the runtime's
//! shared memory).

use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, local_tuner};
use dsp_core::compute::LaunchGeometry;
use dsp_core::compute::tune::{size_class, tune_id};

use super::kernels::{fir_centered_filter_kernel, fir_centered_tiled_kernel, fir_filter_kernel, fir_tiled_kernel};
use crate::core::{buffer, DspFloat, EdgeMode};

/// Edge handling of [`execute_fir`] matching `scipy.signal.lfilter` (zero history).
pub const FIR_DEFAULT_EDGE: EdgeMode = EdgeMode::Zeros;

/// Which FIR kernel runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FirKernel {
    /// One unit per output reading its taps' samples straight from the input.
    Direct,
    /// Cubes stage their samples and halo in shared memory first (falls back to `Direct` when the
    /// tile does not fit the runtime's shared memory).
    Tiled,
}

/// Taps reaching back only (`y[t] = Σ taps[k] · x[t − k]`) or centred on the output sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    Causal { num_taps: usize },
    Centered { radius: usize },
}

impl Shape {
    fn num_taps(self) -> usize {
        match self {
            Shape::Causal { num_taps } => num_taps,
            Shape::Centered { radius } => 2 * radius + 1,
        }
    }

    /// Samples a tile row needs beyond its own `tile_x`.
    fn halo(self) -> usize {
        self.num_taps() - 1
    }
}

/// One FIR call (cloned per autotune candidate).
#[derive(Clone)]
struct FirInputs<R: Runtime> {
    client: ComputeClient<R>,
    input: Handle,
    output: Handle,
    taps: Handle,
    channels: usize,
    samples: usize,
    shape: Shape,
    edge: EdgeMode,
}

impl<R: Runtime> FirInputs<R> {
    fn run<F: DspFloat>(&self, kernel: FirKernel) {
        let (client, total) = (&self.client, self.channels * self.samples);
        let geom = LaunchGeometry::channels_samples(client, self.channels, self.samples);
        let (tile_x, tile_y) = (geom.cube_dim.x, geom.cube_dim.y);
        let tile_bytes = (tile_x as usize + self.shape.halo()) * tile_y as usize * size_of::<F>();
        let tiled = kernel == FirKernel::Tiled && tile_bytes <= client.properties().hardware.max_shared_memory_size;
        // SAFETY: the handles hold `total` samples and `num_taps` taps of `F`
        let (input, output, taps) = unsafe {
            (
                ArrayArg::from_raw_parts(self.input.clone(), total),
                ArrayArg::from_raw_parts(self.output.clone(), total),
                ArrayArg::from_raw_parts(self.taps.clone(), self.shape.num_taps()),
            )
        };
        let (channels, samples, edge) = (self.channels as u32, self.samples as u32, self.edge.id());
        match (self.shape, tiled) {
            (Shape::Causal { num_taps }, false) => fir_filter_kernel::launch::<F, R>(
                client, geom.cube_count, geom.cube_dim, input, output, taps, channels, samples, num_taps as u32, edge,
            ),
            (Shape::Causal { num_taps }, true) => fir_tiled_kernel::launch::<F, R>(
                client, geom.cube_count, geom.cube_dim, input, output, taps, channels, samples, num_taps as u32, tile_x, tile_y, edge,
            ),
            (Shape::Centered { radius }, false) => fir_centered_filter_kernel::launch::<F, R>(
                client, geom.cube_count, geom.cube_dim, input, output, taps, channels, samples, radius as u32, edge,
            ),
            (Shape::Centered { radius }, true) => fir_centered_tiled_kernel::launch::<F, R>(
                client, geom.cube_count, geom.cube_dim, input, output, taps, channels, samples, radius as u32, tile_x, tile_y, edge,
            ),
        }
    }
}

/// [`FirInputs::run`] with the kernel CubeCL's autotuner found fastest for this device, element type,
/// filter shape and problem size. Benchmarks write to a scratch output.
fn tuned<R: Runtime, F: DspFloat>(inputs: FirInputs<R>) {
    static TUNER: LocalTuner<String, String> = local_tuner!("fir-kernel");
    let set = TUNER.init(|| {
        let key = |p: &FirInputs<R>| {
            format!("{}-{:?}-c{}-t{}", F::type_name(), p.shape, size_class(p.channels), size_class(p.samples))
        };
        let scratch = |_: &String, p: &FirInputs<R>| FirInputs { output: buffer::empty::<R, F>(&p.client, p.channels * p.samples), ..p.clone() };
        let set: TunableSet<String, FirInputs<R>, ()> = TunableSet::new(key, scratch);
        [FirKernel::Direct, FirKernel::Tiled].into_iter().fold(set, |set, kernel| {
            set.with(Tunable::new(&format!("{kernel:?}"), move |p: FirInputs<R>| {
                p.run::<F>(kernel);
                Ok::<_, String>(())
            }))
        })
    });
    let client = inputs.client.clone();
    TUNER.execute(&tune_id(&client), &client, set, inputs)
}

fn dispatch<R: Runtime, F: DspFloat>(inputs: FirInputs<R>, kernel: Option<FirKernel>) {
    if inputs.channels == 0 || inputs.samples == 0 {
        return;
    }
    match kernel {
        Some(kernel) => inputs.run::<F>(kernel),
        None => tuned::<R, F>(inputs),
    }
}

/// Causal multi-channel FIR filtering of a `[channels, samples]` buffer with `num_taps` taps.
#[allow(clippy::too_many_arguments)]
pub fn execute_fir<R: Runtime, F: DspFloat>(
    client: &ComputeClient<R>,
    input: &Handle,
    output: &Handle,
    taps: &Handle,
    channels: usize,
    samples: usize,
    num_taps: usize,
    edge: EdgeMode,
) {
    execute_fir_with::<R, F>(client, input, output, taps, channels, samples, num_taps, edge, None);
}

/// [`execute_fir`] with a fixed kernel (`None` = autotuned); for tests and benchmarks.
#[allow(clippy::too_many_arguments)]
pub fn execute_fir_with<R: Runtime, F: DspFloat>(
    client: &ComputeClient<R>,
    input: &Handle,
    output: &Handle,
    taps: &Handle,
    channels: usize,
    samples: usize,
    num_taps: usize,
    edge: EdgeMode,
    kernel: Option<FirKernel>,
) {
    let inputs = FirInputs {
        client: client.clone(),
        input: input.clone(),
        output: output.clone(),
        taps: taps.clone(),
        channels,
        samples,
        shape: Shape::Causal { num_taps: num_taps.max(1) },
        edge,
    };
    dispatch::<R, F>(inputs, kernel);
}

/// Centered (zero-phase for symmetric taps) FIR filtering with `2 · radius + 1` taps.
#[allow(clippy::too_many_arguments)]
pub fn execute_fir_centered<R: Runtime, F: DspFloat>(
    client: &ComputeClient<R>,
    input: &Handle,
    output: &Handle,
    taps: &Handle,
    channels: usize,
    samples: usize,
    radius: usize,
    edge: EdgeMode,
) {
    execute_fir_centered_with::<R, F>(client, input, output, taps, channels, samples, radius, edge, None);
}

/// [`execute_fir_centered`] with a fixed kernel (`None` = autotuned); for tests and benchmarks.
#[allow(clippy::too_many_arguments)]
pub fn execute_fir_centered_with<R: Runtime, F: DspFloat>(
    client: &ComputeClient<R>,
    input: &Handle,
    output: &Handle,
    taps: &Handle,
    channels: usize,
    samples: usize,
    radius: usize,
    edge: EdgeMode,
    kernel: Option<FirKernel>,
) {
    let inputs = FirInputs {
        client: client.clone(),
        input: input.clone(),
        output: output.clone(),
        taps: taps.clone(),
        channels,
        samples,
        shape: Shape::Centered { radius },
        edge,
    };
    dispatch::<R, F>(inputs, kernel);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiled_matches_direct<R: Runtime>(client: &ComputeClient<R>) {
        let (channels, samples) = (5usize, 1_003usize);
        let x: Vec<f32> = (0..channels * samples).map(|i| ((i * 7919) % 211) as f32 - 100.0).collect();
        let input = buffer::upload(client, &x);
        for edge in [EdgeMode::Zeros, EdgeMode::Odd, EdgeMode::Reflect, EdgeMode::Nearest] {
            for num_taps in [1usize, 4, 33] {
                let taps: Vec<f32> = (0..num_taps).map(|k| 1.0 / (k as f32 + 1.0)).collect();
                let taps_h = buffer::upload(client, &taps);
                let run = |kernel| {
                    let out = buffer::empty::<R, f32>(client, x.len());
                    execute_fir_with::<R, f32>(client, &input, &out, &taps_h, channels, samples, num_taps, edge, Some(kernel));
                    buffer::download::<R, f32>(client, out)
                };
                let (direct, tiled) = (run(FirKernel::Direct), run(FirKernel::Tiled));
                assert!(direct.iter().zip(&tiled).all(|(a, b)| (a - b).abs() < 1e-3), "{} causal {num_taps} taps {edge:?}", R::name(client));
            }
            for radius in [0usize, 3, 20] {
                let taps: Vec<f32> = (0..2 * radius + 1).map(|k| 1.0 / (k as f32 + 1.0)).collect();
                let taps_h = buffer::upload(client, &taps);
                let run = |kernel| {
                    let out = buffer::empty::<R, f32>(client, x.len());
                    execute_fir_centered_with::<R, f32>(client, &input, &out, &taps_h, channels, samples, radius, edge, Some(kernel));
                    buffer::download::<R, f32>(client, out)
                };
                let (direct, tiled) = (run(FirKernel::Direct), run(FirKernel::Tiled));
                assert!(direct.iter().zip(&tiled).all(|(a, b)| (a - b).abs() < 1e-3), "{} centered radius {radius} {edge:?}", R::name(client));
            }
        }
    }
    runtime_test!(test_fir_tiled_matches_direct, tiled_matches_direct);
}
