//! FIR dispatch: a direct kernel, a shared-memory tiled kernel and vectorized kernels (`N` outputs
//! per unit, for each width the runtime offers) per filter shape, chosen per device and problem size
//! by CubeCL's autotuner (the tiled kernel only where its tile fits the runtime's shared memory, a
//! vector width only where it divides the row). Every choice gives the same output.

use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, local_tuner};
use dspk_core::compute::LaunchGeometry;
use dspk_core::compute::tune::{size_class, tune_id};

use super::kernels::{
    fir_centered_filter_kernel, fir_centered_tiled_kernel, fir_centered_vector_kernel, fir_filter_kernel, fir_tiled_kernel, fir_vector_kernel,
    fir_vector_table_kernel,
};
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
    /// `width` consecutive outputs per unit (`Vector<F, width>`; SIMD on the CPU runtime). Runs the
    /// widest width up to `width` that the runtime offers and that divides the row, else `Direct`.
    Vectorized { width: usize },
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
struct FirInputs {
    client: Client,
    input: Handle,
    output: Handle,
    taps: Handle,
    channels: usize,
    samples: usize,
    shape: Shape,
    edge: EdgeMode,
}

impl FirInputs {
    fn run<F: DspFloat>(&self, kernel: FirKernel) {
        if let FirKernel::Vectorized { width } = kernel {
            if let Some(width) = vector_width::<F>(&self.client, self.samples, width) {
                return self.run_vectorized::<F>(width);
            }
        }
        let (client, total) = (&self.client, self.channels * self.samples);
        let geom = LaunchGeometry::channels_samples(client, self.channels, self.samples);
        let (tile_x, tile_y) = (geom.cube_dim.x, geom.cube_dim.y);
        let tile_bytes = (tile_x as usize + self.shape.halo()) * tile_y as usize * size_of::<F>();
        let tiled = kernel == FirKernel::Tiled && tile_bytes <= client.properties().hardware.max_shared_memory_size;
        // SAFETY: the handles hold `total` samples and `num_taps` taps of `F`
        let (input, output, taps) = unsafe {
            (
                BufferArg::from_raw_parts(self.input.clone(), total),
                BufferArg::from_raw_parts(self.output.clone(), total),
                BufferArg::from_raw_parts(self.taps.clone(), self.shape.num_taps()),
            )
        };
        let (channels, samples, edge) = (self.channels as u32, self.samples as u32, self.edge.id());
        match (self.shape, tiled) {
            (Shape::Causal { num_taps }, false) => fir_filter_kernel::launch::<F>(
                client, geom.cube_count, geom.cube_dim, input, output, taps, channels, samples, num_taps as u32, edge,
            ),
            (Shape::Causal { num_taps }, true) => fir_tiled_kernel::launch::<F>(
                client, geom.cube_count, geom.cube_dim, input, output, taps, channels, samples, num_taps as u32, tile_x, tile_y, edge,
            ),
            (Shape::Centered { radius }, false) => fir_centered_filter_kernel::launch::<F>(
                client, geom.cube_count, geom.cube_dim, input, output, taps, channels, samples, radius as u32, edge,
            ),
            (Shape::Centered { radius }, true) => fir_centered_tiled_kernel::launch::<F>(
                client, geom.cube_count, geom.cube_dim, input, output, taps, channels, samples, radius as u32, tile_x, tile_y, edge,
            ),
        }
    }
}

impl FirInputs {
    /// The vectorized kernels with `width` outputs per unit (`samples` a multiple of `width`).
    fn run_vectorized<F: DspFloat>(&self, width: usize) {
        let (client, total, num_taps) = (&self.client, self.channels * self.samples, self.shape.num_taps());
        let table_len = (num_taps + width - 1) * width;
        let table = buffer::empty::<F>(client, table_len);
        let centered = matches!(self.shape, Shape::Centered { .. });
        let geom = LaunchGeometry::elementwise(client, table_len);
        // SAFETY: `taps` holds `num_taps` values of `F`, `table` `table_len`
        unsafe {
            fir_vector_table_kernel::launch::<F>(
                client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(self.taps.clone(), num_taps),
                BufferArg::from_raw_parts(table.clone(), table_len),
                num_taps as u32,
                width as u32,
                centered,
            );
        }
        let geom = LaunchGeometry::elementwise(client, total / width);
        // SAFETY: input and output hold `total` samples of `F` (`total / width` vectors), `table`
        // `table_len` values
        let (input, output, table) = unsafe {
            (
                BufferArg::from_raw_parts(self.input.clone(), total),
                BufferArg::from_raw_parts(self.output.clone(), total),
                BufferArg::from_raw_parts(table, table_len),
            )
        };
        let (samples, edge) = (self.samples as u32, self.edge.id());
        match self.shape {
            Shape::Causal { num_taps } => fir_vector_kernel::launch::<F>(
                client, geom.cube_count, geom.cube_dim, width, input, output, table, samples, num_taps as u32, edge,
            ),
            Shape::Centered { radius } => fir_centered_vector_kernel::launch::<F>(
                client, geom.cube_count, geom.cube_dim, width, input, output, table, samples, radius as u32, edge,
            ),
        }
    }
}

/// The widest vector width up to `at_most` (above 1) that `client` offers for `F` and that divides
/// `samples`.
fn vector_width<F: DspFloat>(client: &Client, samples: usize, at_most: usize) -> Option<usize> {
    client.io_optimized_vector_sizes(size_of::<F>()).filter(|&w| w > 1 && w <= at_most && samples % w == 0).max()
}

/// [`FirInputs::run`] with the kernel CubeCL's autotuner found fastest for this device, element type,
/// filter shape and problem size. Benchmarks write to a scratch output.
fn tuned<F: DspFloat>(inputs: FirInputs) {
    static TUNER: LocalTuner<String, String> = local_tuner!("fir-kernel-v2");
    let client = inputs.client.clone();
    let id = tune_id(&client);
    let widths: Vec<usize> = client.io_optimized_vector_sizes(size_of::<F>()).filter(|&w| w > 1).collect();
    let set = TUNER.init(&id, move || {
        let key = |p: &FirInputs| {
            // `a`: the power of two dividing the row (up to 16), which decides the vector widths that run
            format!("{}-{:?}-c{}-t{}-a{}", F::type_name(), p.shape, size_class(p.channels), size_class(p.samples), p.samples.trailing_zeros().min(4))
        };
        let scratch = |_: &String, p: &FirInputs| FirInputs { output: buffer::empty::<F>(&p.client, p.channels * p.samples), ..p.clone() };
        let set: TunableSet<String, FirInputs, ()> = TunableSet::new(key, scratch);
        let kernels = [FirKernel::Direct, FirKernel::Tiled].into_iter().chain(widths.iter().map(|&width| FirKernel::Vectorized { width }));
        kernels.fold(set, |set, kernel| {
            set.with(Tunable::new(&format!("{kernel:?}"), move |p: FirInputs| {
                p.run::<F>(kernel);
                Ok::<_, String>(())
            }))
        })
    });
    TUNER.execute(&id, &client, set, inputs)
}

fn dispatch<F: DspFloat>(inputs: FirInputs, kernel: Option<FirKernel>) {
    if inputs.channels == 0 || inputs.samples == 0 {
        return;
    }
    match kernel {
        Some(kernel) => inputs.run::<F>(kernel),
        None => tuned::<F>(inputs),
    }
}

/// Causal multi-channel FIR filtering of a `[channels, samples]` buffer with `num_taps` taps.
#[allow(clippy::too_many_arguments)]
pub fn execute_fir<F: DspFloat>(
    client: &Client,
    input: &Handle,
    output: &Handle,
    taps: &Handle,
    channels: usize,
    samples: usize,
    num_taps: usize,
    edge: EdgeMode,
) {
    execute_fir_with::<F>(client, input, output, taps, channels, samples, num_taps, edge, None);
}

/// [`execute_fir`] with a fixed kernel (`None` = autotuned); for tests and benchmarks.
#[allow(clippy::too_many_arguments)]
pub fn execute_fir_with<F: DspFloat>(
    client: &Client,
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
    dispatch::<F>(inputs, kernel);
}

/// Centered (zero-phase for symmetric taps) FIR filtering with `2 · radius + 1` taps.
#[allow(clippy::too_many_arguments)]
pub fn execute_fir_centered<F: DspFloat>(
    client: &Client,
    input: &Handle,
    output: &Handle,
    taps: &Handle,
    channels: usize,
    samples: usize,
    radius: usize,
    edge: EdgeMode,
) {
    execute_fir_centered_with::<F>(client, input, output, taps, channels, samples, radius, edge, None);
}

/// [`execute_fir_centered`] with a fixed kernel (`None` = autotuned); for tests and benchmarks.
#[allow(clippy::too_many_arguments)]
pub fn execute_fir_centered_with<F: DspFloat>(
    client: &Client,
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
    dispatch::<F>(inputs, kernel);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiled_matches_direct(client: &Client) {
        let (channels, samples) = (5usize, 1_003usize);
        let x: Vec<f32> = (0..channels * samples).map(|i| ((i * 7919) % 211) as f32 - 100.0).collect();
        let input = buffer::upload(client, &x);
        for edge in [EdgeMode::Zeros, EdgeMode::Odd, EdgeMode::Reflect, EdgeMode::Nearest] {
            for num_taps in [1usize, 4, 33] {
                let taps: Vec<f32> = (0..num_taps).map(|k| 1.0 / (k as f32 + 1.0)).collect();
                let taps_h = buffer::upload(client, &taps);
                let run = |kernel| {
                    let out = buffer::empty::<f32>(client, x.len());
                    execute_fir_with::<f32>(client, &input, &out, &taps_h, channels, samples, num_taps, edge, Some(kernel));
                    buffer::download::<f32>(client, out)
                };
                let (direct, tiled) = (run(FirKernel::Direct), run(FirKernel::Tiled));
                assert!(direct.iter().zip(&tiled).all(|(a, b)| (a - b).abs() < 1e-3), "{} causal {num_taps} taps {edge:?}", client.name());
            }
            for radius in [0usize, 3, 20] {
                let taps: Vec<f32> = (0..2 * radius + 1).map(|k| 1.0 / (k as f32 + 1.0)).collect();
                let taps_h = buffer::upload(client, &taps);
                let run = |kernel| {
                    let out = buffer::empty::<f32>(client, x.len());
                    execute_fir_centered_with::<f32>(client, &input, &out, &taps_h, channels, samples, radius, edge, Some(kernel));
                    buffer::download::<f32>(client, out)
                };
                let (direct, tiled) = (run(FirKernel::Direct), run(FirKernel::Tiled));
                assert!(direct.iter().zip(&tiled).all(|(a, b)| (a - b).abs() < 1e-3), "{} centered radius {radius} {edge:?}", client.name());
            }
        }
    }
    runtime_test!(test_fir_tiled_matches_direct, tiled_matches_direct);

    /// Vectorized kernels are bit-identical to `Direct` for every edge and shape, for `f32` and
    /// `f64`, rows divisible by the width or not (then `Direct` runs).
    fn vectorized_matches_direct_of<F: DspFloat + std::fmt::Debug>(client: &Client) {
        let channels = 3usize;
        for samples in [64usize, 1_000, 1_003] {
            let x: Vec<F> = (0..channels * samples).map(|i| crate::core::cast::<F>(((i * 7919) % 211) as f64 * 0.37 - 40.0)).collect();
            let input = buffer::upload(client, &x);
            for edge in [EdgeMode::Zeros, EdgeMode::Odd, EdgeMode::Reflect, EdgeMode::Nearest] {
                for (centered, n) in [(false, 1usize), (false, 4), (false, 33), (true, 0), (true, 3), (true, 20)] {
                    let num_taps = if centered { 2 * n + 1 } else { n };
                    let taps: Vec<F> = (0..num_taps).map(|k| crate::core::cast::<F>(1.0 / (k as f64 + 1.3))).collect();
                    let taps_h = buffer::upload(client, &taps);
                    let run = |kernel| {
                        let out = buffer::empty::<F>(client, x.len());
                        if centered {
                            execute_fir_centered_with::<F>(client, &input, &out, &taps_h, channels, samples, n, edge, Some(kernel));
                        } else {
                            execute_fir_with::<F>(client, &input, &out, &taps_h, channels, samples, n, edge, Some(kernel));
                        }
                        buffer::download::<F>(client, out)
                    };
                    let direct = run(FirKernel::Direct);
                    for width in [2usize, 4, 8, 16] {
                        assert_eq!(run(FirKernel::Vectorized { width }), direct, "{} {} samples {samples} {edge:?} centered {centered} n {n} width {width}", client.name(), F::type_name());
                    }
                }
            }
        }
    }

    fn vectorized_matches_direct(client: &Client) {
        vectorized_matches_direct_of::<f32>(client);
        if client.properties().supports_type(f64::elem_type_native()) {
            vectorized_matches_direct_of::<f64>(client);
        }
    }
    runtime_test!(test_fir_vectorized_matches_direct, vectorized_matches_direct);
}
