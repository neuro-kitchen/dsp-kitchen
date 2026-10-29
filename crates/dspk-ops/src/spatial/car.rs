use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, local_tuner};
use dspk_core::compute::tune::{size_class, tune_id};
use dspk_core::compute::LaunchGeometry;

use crate::core::{buffer, DspFloat};
pub use super::kernels::{common_median_bitonic_kernel, common_median_kernel, direct_car_kernel};

/// High-level host dispatcher for direct, single-pass Common Average Referencing (CAR).
/// Computes and subtracts the common average directly in VRAM without auxiliary buffers.
pub fn execute_direct_car<F: DspFloat>(
    client: &Client,
    input: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    channels: usize,
    samples: usize,
) {
    let geom = LaunchGeometry::per_sample(client, samples);
    let total_elements = channels * samples;

    unsafe {
        direct_car_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(input.clone(), total_elements),
            BufferArg::from_raw_parts(output.clone(), total_elements),
            channels as u32,
            samples as u32,
        );
    }
}

/// How [`execute_common_median`] finds each sample's median over channels. Both are exact and give
/// the same output; which is faster depends on the device, so the autotuner picks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmrKernel {
    /// Rank counting ([`common_median_kernel`]): each value compared with every channel, no barriers.
    Rank,
    /// Bitonic sort in shared memory ([`common_median_bitonic_kernel`]): about `log₂² C / 2` steps
    /// per sample, one barrier each.
    Bitonic,
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
        let geom = LaunchGeometry::per_row(client, self.samples, self.channels);
        let total = self.channels * self.samples;
        let (c, s, units) = (self.channels as u32, self.samples as u32, geom.cube_dim.x);
        // SAFETY: both buffers hold `channels · samples` values
        let (input, output) = unsafe {
            (BufferArg::from_raw_parts(self.input.clone(), total), BufferArg::from_raw_parts(self.output.clone(), total))
        };
        match kernel {
            CmrKernel::Rank => common_median_kernel::launch::<F>(client, geom.cube_count, geom.cube_dim, input, output, c, s, c, units),
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
    static TUNER: LocalTuner<String, String> = local_tuner!("common-median-v1");
    let client = inputs.client.clone();
    let id = tune_id(&client);
    let set = TUNER.init(&id, || {
        let key = |p: &CmrInputs| format!("{}-c{}-t{}", F::type_name(), size_class(p.channels), size_class(p.samples));
        let scratch = |_: &String, p: &CmrInputs| CmrInputs { output: buffer::empty::<F>(&p.client, p.channels * p.samples), ..p.clone() };
        let set: TunableSet<String, CmrInputs, ()> = TunableSet::new(key, scratch);
        [CmrKernel::Rank, CmrKernel::Bitonic].into_iter().fold(set, |set, kernel| {
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
/// [`CmrKernel`]'s two exact methods for this device.
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
        for channels in [1usize, 4, 5, 37, 384] {
            let samples = 300;
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
            for kernel in [Some(CmrKernel::Rank), Some(CmrKernel::Bitonic), None] {
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
}
