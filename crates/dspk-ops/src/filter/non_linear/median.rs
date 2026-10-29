use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, local_tuner};
use dspk_core::compute::LaunchGeometry;
use dspk_core::compute::tune::{size_class, tune_id};

use super::kernels::{median_filter_kernel, median_filter_vector_kernel};
use crate::core::{buffer, DspFloat, EdgeMode};

/// Samples on each side of the centre in the 9-point median window.
pub const MEDIAN9_RADIUS: usize = 4;

/// Widest running-median window: the window is held in registers and ranked in `width²` steps.
pub const MAX_MEDIAN_WIDTH: usize = 31;

/// Which running-median kernel runs (both give the same output).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MedianKernel {
    /// One unit per output.
    Direct,
    /// `width` consecutive outputs per unit (`Vector<F, width>`; SIMD on the CPU runtime). Runs the
    /// widest vector width up to `width` that the runtime offers and that divides the row, else
    /// `Direct`.
    Vectorized { width: usize },
}

/// Edge handling of the running median, matching `scipy.signal.medfilt` (zero padding).
pub const MEDIAN_DEFAULT_EDGE: EdgeMode = EdgeMode::Zeros;

/// Running median over an odd `width` (≤ [`MAX_MEDIAN_WIDTH`]) along time of a `[channels, samples]`
/// buffer.
///
/// # Panics
/// If `width` is even, zero, or larger than [`MAX_MEDIAN_WIDTH`].
pub fn execute_median<F: DspFloat>(
    client: &Client,
    input: &Handle,
    output: &Handle,
    channels: usize,
    samples: usize,
    width: usize,
    edge: EdgeMode,
) {
    execute_median_with::<F>(client, input, output, channels, samples, width, edge, None);
}

/// [`execute_median`] with a fixed kernel (`None` = autotuned); for tests and benchmarks.
#[allow(clippy::too_many_arguments)]
pub fn execute_median_with<F: DspFloat>(
    client: &Client,
    input: &Handle,
    output: &Handle,
    channels: usize,
    samples: usize,
    width: usize,
    edge: EdgeMode,
    kernel: Option<MedianKernel>,
) {
    assert!(width % 2 == 1 && width <= MAX_MEDIAN_WIDTH, "median width {width} must be odd and at most {MAX_MEDIAN_WIDTH}");
    if channels == 0 || samples == 0 {
        return;
    }
    let inputs = MedianInputs { client: client.clone(), input: input.clone(), output: output.clone(), channels, samples, width, edge };
    match kernel {
        Some(kernel) => inputs.run::<F>(kernel),
        None => tuned::<F>(inputs),
    }
}

/// One running-median call (cloned per autotune candidate).
#[derive(Clone)]
struct MedianInputs {
    client: Client,
    input: Handle,
    output: Handle,
    channels: usize,
    samples: usize,
    width: usize,
    edge: EdgeMode,
}

impl MedianInputs {
    fn run<F: DspFloat>(&self, kernel: MedianKernel) {
        let client = &self.client;
        let total = self.channels * self.samples;
        let vector = match kernel {
            MedianKernel::Vectorized { width } => {
                client.io_optimized_vector_sizes(size_of::<F>()).filter(|&w| w > 1 && w <= width && self.samples % w == 0).max()
            }
            MedianKernel::Direct => None,
        };
        // SAFETY: both buffers hold `channels · samples` values of `F`
        let (input, output) = unsafe { (BufferArg::from_raw_parts(self.input.clone(), total), BufferArg::from_raw_parts(self.output.clone(), total)) };
        match vector {
            None => {
                let geom = LaunchGeometry::channels_samples(client, self.channels, self.samples);
                median_filter_kernel::launch::<F>(
                    client, geom.cube_count, geom.cube_dim, input, output, self.channels as u32, self.samples as u32, self.width as u32, self.edge.id(),
                )
            }
            Some(n) => {
                let geom = LaunchGeometry::elementwise(client, total / n);
                median_filter_vector_kernel::launch::<F>(
                    client, geom.cube_count, geom.cube_dim, n, input, output, self.samples as u32, self.width as u32, self.edge.id(),
                )
            }
        }
    }
}

/// [`MedianInputs::run`] with the kernel CubeCL's autotuner found fastest for this device, element
/// type, window width and size. Benchmarks write to a scratch output.
fn tuned<F: DspFloat>(inputs: MedianInputs) {
    static TUNER: LocalTuner<String, String> = local_tuner!("median-filter-v1");
    let client = inputs.client.clone();
    let id = tune_id(&client);
    let widths: Vec<usize> = client.io_optimized_vector_sizes(size_of::<F>()).filter(|&w| w > 1).collect();
    let set = TUNER.init(&id, move || {
        // `a`: the power of two dividing the row (up to 16), which decides the vector widths that run
        let key = |p: &MedianInputs| {
            format!("{}-w{}-c{}-t{}-a{}", F::type_name(), p.width, size_class(p.channels), size_class(p.samples), p.samples.trailing_zeros().min(4))
        };
        let scratch = |_: &String, p: &MedianInputs| MedianInputs { output: buffer::empty::<F>(&p.client, p.channels * p.samples), ..p.clone() };
        let set: TunableSet<String, MedianInputs, ()> = TunableSet::new(key, scratch);
        let kernels = std::iter::once(MedianKernel::Direct).chain(widths.iter().map(|&width| MedianKernel::Vectorized { width }));
        kernels.fold(set, |set, kernel| {
            set.with(Tunable::new(&format!("{kernel:?}"), move |p: MedianInputs| {
                p.run::<F>(kernel);
                Ok::<_, String>(())
            }))
        })
    });
    TUNER.execute(&id, &client, set, inputs)
}

/// 9-point running median with [`MEDIAN_DEFAULT_EDGE`] (the branch-free `med9` network).
pub fn execute_median_9p<F: DspFloat>(
    client: &Client,
    input: &Handle,
    output: &Handle,
    channels: usize,
    samples: usize,
) {
    execute_median::<F>(client, input, output, channels, samples, 2 * MEDIAN9_RADIUS + 1, MEDIAN_DEFAULT_EDGE);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The vectorized kernel is bit-identical to `Direct` (f32 / f64, the med9 network and rank
    /// widths, every edge, rows divisible by the vector width or not).
    fn vectorized_matches_direct_of<F: DspFloat + std::fmt::Debug>(client: &Client) {
        let channels = 3;
        for samples in [64usize, 1_000, 1_003] {
            let x: Vec<F> = (0..channels * samples).map(|i| crate::core::cast::<F>(((i * 7919) % 211) as f64 * 0.37 - 40.0)).collect();
            let input = buffer::upload(client, &x);
            // Every edge for the med9 network and a short rank window; zeros for the others (each
            // case is its own compiled kernel)
            let all = [EdgeMode::Zeros, EdgeMode::Odd, EdgeMode::Reflect, EdgeMode::Nearest];
            for (width, edges) in [(3usize, &all[..]), (9, &all[..]), (1, &all[..1]), (5, &all[..1]), (31, &all[..1])] {
                for &edge in edges {
                    let run = |kernel| {
                        let out = buffer::empty::<F>(client, x.len());
                        execute_median_with::<F>(client, &input, &out, channels, samples, width, edge, Some(kernel));
                        buffer::download::<F>(client, out)
                    };
                    let direct = run(MedianKernel::Direct);
                    for vector in [2usize, 16] {
                        assert_eq!(run(MedianKernel::Vectorized { width: vector }), direct, "{} {} {samples} samples width {width} {edge:?} vector {vector}", client.name(), F::type_name());
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
    runtime_test!(test_median_vectorized_matches_direct, vectorized_matches_direct);
}
