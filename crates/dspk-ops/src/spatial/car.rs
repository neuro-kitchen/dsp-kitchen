use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, local_tuner};
use dspk_core::compute::tune::{size_class, tune_id};
use dspk_core::compute::LaunchGeometry;

use crate::core::{buffer, DspFloat};
pub use super::kernels::{common_median_bitonic_kernel, common_median_kernel, common_median_vector_kernel, direct_car_kernel, direct_car_vector_kernel};

/// High-level host dispatcher for direct, single-pass Common Average Referencing (CAR).
/// Computes and subtracts the common average directly in VRAM without auxiliary buffers, with the
/// faster of [`CarKernel`]'s choices for this device (same output).
pub fn execute_direct_car<F: DspFloat>(
    client: &Client,
    input: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    channels: usize,
    samples: usize,
) {
    execute_direct_car_with::<F>(client, input, output, channels, samples, None);
}

/// Which common-average kernel runs (both give the same output).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CarKernel {
    /// One unit per sample ([`direct_car_kernel`]).
    Direct,
    /// `width` consecutive samples per unit ([`direct_car_vector_kernel`]; SIMD on the CPU
    /// runtime). Runs the widest width up to `width` that the runtime offers and that divides the
    /// sample count, else `Direct`.
    Vectorized { width: usize },
}

/// [`execute_direct_car`] with a fixed kernel (`None` = autotuned); for tests and benchmarks.
pub fn execute_direct_car_with<F: DspFloat>(
    client: &Client,
    input: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    channels: usize,
    samples: usize,
    kernel: Option<CarKernel>,
) {
    if channels == 0 || samples == 0 {
        return;
    }
    let inputs = CmrInputs { client: client.clone(), input: input.clone(), output: output.clone(), channels, samples };
    match kernel {
        Some(kernel) => inputs.run_car::<F>(kernel),
        None => tuned_car::<F>(inputs),
    }
}

impl CmrInputs {
    fn run_car<F: DspFloat>(&self, kernel: CarKernel) {
        let client = &self.client;
        let total = self.channels * self.samples;
        let width = match kernel {
            CarKernel::Vectorized { width } => client.io_optimized_vector_sizes(size_of::<F>()).filter(|&w| w > 1 && w <= width && self.samples % w == 0).max(),
            CarKernel::Direct => None,
        };
        // SAFETY: both buffers hold `channels · samples` values of `F`
        let (input, output) = unsafe { (BufferArg::from_raw_parts(self.input.clone(), total), BufferArg::from_raw_parts(self.output.clone(), total)) };
        match width {
            None => {
                let geom = LaunchGeometry::per_sample(client, self.samples);
                direct_car_kernel::launch::<F>(client, geom.cube_count, geom.cube_dim, input, output, self.channels as u32, self.samples as u32)
            }
            Some(width) => {
                let vectors = self.samples / width;
                let geom = LaunchGeometry::elementwise(client, vectors);
                direct_car_vector_kernel::launch::<F>(client, geom.cube_count, geom.cube_dim, width, input, output, self.channels as u32, vectors as u32)
            }
        }
    }
}

/// [`CmrInputs::run_car`] with the kernel CubeCL's autotuner found fastest for this device,
/// element type and size. Benchmarks write to a scratch output.
fn tuned_car<F: DspFloat>(inputs: CmrInputs) {
    static TUNER: LocalTuner<String, String> = local_tuner!("common-average-v1");
    let client = inputs.client.clone();
    let id = tune_id(&client);
    let widths: Vec<usize> = client.io_optimized_vector_sizes(size_of::<F>()).filter(|&w| w > 1).collect();
    let set = TUNER.init(&id, move || {
        let key = |p: &CmrInputs| format!("{}-c{}-t{}-a{}", F::type_name(), size_class(p.channels), size_class(p.samples), p.samples.trailing_zeros().min(4));
        let scratch = |_: &String, p: &CmrInputs| CmrInputs { output: buffer::empty::<F>(&p.client, p.channels * p.samples), ..p.clone() };
        let set: TunableSet<String, CmrInputs, ()> = TunableSet::new(key, scratch);
        let kernels = std::iter::once(CarKernel::Direct).chain(widths.iter().map(|&width| CarKernel::Vectorized { width }));
        kernels.fold(set, |set, kernel| {
            set.with(Tunable::new(&format!("{kernel:?}"), move |p: CmrInputs| {
                p.run_car::<F>(kernel);
                Ok::<_, String>(())
            }))
        })
    });
    TUNER.execute(&id, &client, set, inputs)
}

/// How [`execute_common_median`] finds each sample's median over channels. All are exact and give
/// the same output; which is faster depends on the device, so the autotuner picks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmrKernel {
    /// Rank counting ([`common_median_kernel`]): each value compared with every channel, no barriers.
    Rank,
    /// Bitonic sort in shared memory ([`common_median_bitonic_kernel`]): about `log₂² C / 2` steps
    /// per sample, one barrier each.
    Bitonic,
    /// Rank counting on `width` consecutive samples per unit ([`common_median_vector_kernel`];
    /// SIMD on the CPU runtime). Runs the widest width up to `width` that the runtime offers and
    /// that divides the sample count, else `Rank`.
    Vectorized { width: usize },
}

/// One common-median call (cloned per autotune candidate).
#[derive(Clone)]
struct CmrInputs {
    client: Client,
    input: Handle,
    output: Handle,
    channels: usize,
    samples: usize,
}

impl CmrInputs {
    fn run<F: DspFloat>(&self, kernel: CmrKernel) {
        let client = &self.client;
        if let CmrKernel::Vectorized { width } = kernel {
            let width = client.io_optimized_vector_sizes(size_of::<F>()).filter(|&w| w > 1 && w <= width && self.samples % w == 0).max();
            if let Some(width) = width {
                let total = self.channels * self.samples;
                let vectors = self.samples / width;
                let geom = LaunchGeometry::elementwise(client, vectors);
                // SAFETY: both buffers hold `channels · samples` values (`total / width` vectors)
                unsafe {
                    common_median_vector_kernel::launch::<F>(
                        client,
                        geom.cube_count,
                        geom.cube_dim,
                        width,
                        BufferArg::from_raw_parts(self.input.clone(), total),
                        BufferArg::from_raw_parts(self.output.clone(), total),
                        self.channels as u32,
                        vectors as u32,
                    );
                }
                return;
            }
            return self.run::<F>(CmrKernel::Rank);
        }
        let geom = LaunchGeometry::per_row(client, self.samples, self.channels);
        let total = self.channels * self.samples;
        let (c, s, units) = (self.channels as u32, self.samples as u32, geom.cube_dim.x);
        // SAFETY: both buffers hold `channels · samples` values
        let (input, output) = unsafe {
            (BufferArg::from_raw_parts(self.input.clone(), total), BufferArg::from_raw_parts(self.output.clone(), total))
        };
        match kernel {
            CmrKernel::Rank | CmrKernel::Vectorized { .. } => common_median_kernel::launch::<F>(client, geom.cube_count, geom.cube_dim, input, output, c, s, c, units),
            CmrKernel::Bitonic => {
                let padded = self.channels.next_power_of_two() as u32;
                common_median_bitonic_kernel::launch::<F>(client, geom.cube_count, geom.cube_dim, input, output, c, s, padded, units)
            }
        }
    }
}

/// [`CmrInputs::run`] with the kernel CubeCL's autotuner found fastest for this device, element type
/// and problem size. Benchmarks write to a scratch output.
fn tuned<F: DspFloat>(inputs: CmrInputs) {
    static TUNER: LocalTuner<String, String> = local_tuner!("common-median-v2");
    let client = inputs.client.clone();
    let id = tune_id(&client);
    let widths: Vec<usize> = client.io_optimized_vector_sizes(size_of::<F>()).filter(|&w| w > 1).collect();
    // CubeCL's CPU runtime emulates barriers: Bitonic takes seconds there, even just to time it
    let cpu = client.name() == "cpu";
    let set = TUNER.init(&id, move || {
        // `a`: the power of two dividing the sample count (up to 16), which decides the widths that run
        let key = |p: &CmrInputs| format!("{}-c{}-t{}-a{}", F::type_name(), size_class(p.channels), size_class(p.samples), p.samples.trailing_zeros().min(4));
        let scratch = |_: &String, p: &CmrInputs| CmrInputs { output: buffer::empty::<F>(&p.client, p.channels * p.samples), ..p.clone() };
        let set: TunableSet<String, CmrInputs, ()> = TunableSet::new(key, scratch);
        let kernels = [CmrKernel::Rank, CmrKernel::Bitonic].into_iter().filter(|&k| !(k == CmrKernel::Bitonic && cpu)).chain(widths.iter().map(|&width| CmrKernel::Vectorized { width }));
        kernels.fold(set, |set, kernel| {
            set.with(Tunable::new(&format!("{kernel:?}"), move |p: CmrInputs| {
                p.run::<F>(kernel);
                Ok::<_, String>(())
            }))
        })
    });
    TUNER.execute(&id, &client, set, inputs)
}

/// Common median reference: subtracts the median over channels at every sample (SpikeInterface's
/// `common_reference(operator="median")`), one cube per sample on the device, with the faster of
/// [`CmrKernel`]'s exact methods for this device.
pub fn execute_common_median<F: DspFloat>(
    client: &Client,
    input: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    channels: usize,
    samples: usize,
) {
    execute_common_median_with::<F>(client, input, output, channels, samples, None);
}

/// [`execute_common_median`] with a fixed kernel (`None` = autotuned); for tests and benchmarks.
pub fn execute_common_median_with<F: DspFloat>(
    client: &Client,
    input: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    channels: usize,
    samples: usize,
    kernel: Option<CmrKernel>,
) {
    if channels == 0 || samples == 0 {
        return;
    }
    let inputs = CmrInputs { client: client.clone(), input: input.clone(), output: output.clone(), channels, samples };
    match kernel {
        Some(kernel) => inputs.run::<F>(kernel),
        None => tuned::<F>(inputs),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::buffer;

    /// Exact against a sorted reference, in `F` (signed fractional values, so the key's sign
    /// handling and both middle values of an even count are exercised).
    fn medians_of<F: DspFloat + std::fmt::Debug>(client: &Client) {
        // 304 samples: every vector width divides; 300: widths above 4 fall back; 301: none divides
        for (channels, samples) in [(1usize, 300usize), (4, 304), (5, 301), (37, 300), (384, 304)] {
            let x: Vec<F> = (0..channels * samples).map(|i| crate::core::cast::<F>(((i * 7919) % 211) as f64 * 0.37 - 40.0)).collect();
            let half = crate::core::cast::<F>(0.5);
            let expected: Vec<F> = (0..channels * samples)
                .map(|i| {
                    let (c, s) = (i / samples, i % samples);
                    let mut col: Vec<F> = (0..channels).map(|c| x[c * samples + s]).collect();
                    col.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
                    let m = if channels % 2 == 1 { col[channels / 2] } else { (col[channels / 2 - 1] + col[channels / 2]) * half };
                    x[c * samples + s] - m
                })
                .collect();
            let vectorized = [2, 4, 8, 16].map(|width| Some(CmrKernel::Vectorized { width }));
            for kernel in [Some(CmrKernel::Rank), Some(CmrKernel::Bitonic), None].into_iter().chain(vectorized) {
                let out = buffer::empty::<F>(client, x.len());
                execute_common_median_with::<F>(client, &buffer::upload(client, &x), &out, channels, samples, kernel);
                let got = buffer::download::<F>(client, out);
                assert_eq!(got, expected, "{channels} channels, {kernel:?}");
            }
        }
    }

    fn medians(client: &Client) {
        medians_of::<f32>(client);
        // float64 where the runtime has it (not WebGPU)
        if client.properties().supports_type(f64::elem_type_native()) {
            medians_of::<f64>(client);
        }
    }
    runtime_test!(test_common_median_reference, medians);

    /// The vectorized common average is bit-identical to `Direct` (sample counts every width
    /// divides, some, none).
    fn averages(client: &Client) {
        for (channels, samples) in [(1usize, 300usize), (5, 304), (37, 301), (384, 304)] {
            let x: Vec<f32> = (0..channels * samples).map(|i| ((i * 7919) % 211) as f32 * 0.37 - 40.0).collect();
            let input = buffer::upload(client, &x);
            let run = |kernel| {
                let out = buffer::empty::<f32>(client, x.len());
                execute_direct_car_with::<f32>(client, &input, &out, channels, samples, Some(kernel));
                buffer::download::<f32>(client, out)
            };
            let direct = run(CarKernel::Direct);
            for width in [2usize, 4, 16] {
                assert_eq!(run(CarKernel::Vectorized { width }), direct, "{} {channels} ch {samples} samples width {width}", client.name());
            }
        }
    }
    runtime_test!(test_common_average_vectorized_matches_direct, averages);
}
