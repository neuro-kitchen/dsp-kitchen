//! Rational resampling (`scipy.signal.resample_poly`).

use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, local_tuner};
use dspk_core::compute::LaunchGeometry;
use dspk_core::compute::tune::{size_class, tune_id};

use super::design::{firwin, FirWindow};
use super::kernels::{upfirdn_kernel, upfirdn_vector_kernel};
use crate::core::{buffer, cast_all, DspFloat, EdgeMode};

/// `resample_poly`'s padding (`padtype="constant"`, zeros).
pub const RESAMPLE_POLY_DEFAULT_EDGE: EdgeMode = EdgeMode::Zeros;

/// Window `resample_poly` designs its filter with by default.
pub const RESAMPLE_POLY_WINDOW: FirWindow = FirWindow::Kaiser { beta: 5.0 };

/// Half-length of the designed filter per unit of the larger rate (`half_len = 10 · max(up, down)`).
pub const RESAMPLE_POLY_HALF_LEN_PER_RATE: usize = 10;

/// The anti-aliasing filter of [`resample_poly`].
#[derive(Debug, Clone, PartialEq)]
pub enum ResampleFilter {
    /// Designed with `firwin(2 · half_len + 1, 1 / max(up, down), window)`.
    Window(FirWindow),
    /// Explicit taps at the up-sampled rate (odd length; centred).
    Taps(Vec<f64>),
}

/// Which polyphase kernel runs (every choice gives the same output).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleKernel {
    /// One unit per output.
    Direct,
    /// `width` consecutive outputs per unit (`Vector<F, width>`; SIMD on the CPU runtime). Runs the
    /// widest width up to `width` that the runtime offers and that divides the output row, else
    /// `Direct`.
    Vectorized { width: usize },
}

impl Default for ResampleFilter {
    fn default() -> Self {
        ResampleFilter::Window(RESAMPLE_POLY_WINDOW)
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Samples out of `samples` resampled by `up / down` (`ceil(samples · up / down)`).
pub fn resample_poly_len(samples: usize, up: usize, down: usize) -> usize {
    let g = gcd(up, down).max(1);
    (samples * (up / g)).div_ceil(down / g)
}

/// Resamples every channel of a `[channels, samples]` buffer of `F` by `up / down` into `output`
/// (`[channels, resample_poly_len(samples, up, down)]`), as `scipy.signal.resample_poly`: the filter is
/// scaled by `up` and aligned so output sample 0 sits on input sample 0. Returns the output length.
///
/// # Panics
/// If `up` or `down` is zero.
#[allow(clippy::too_many_arguments)]
pub fn resample_poly<F: DspFloat>(
    client: &Client,
    input: &Handle,
    output: &Handle,
    channels: usize,
    samples: usize,
    up: usize,
    down: usize,
    filter: &ResampleFilter,
    edge: EdgeMode,
) -> usize {
    resample_poly_with::<F>(client, input, output, channels, samples, up, down, filter, edge, None)
}

/// [`resample_poly`] with a fixed kernel (`None` = autotuned); for tests and benchmarks.
#[allow(clippy::too_many_arguments)]
pub fn resample_poly_with<F: DspFloat>(
    client: &Client,
    input: &Handle,
    output: &Handle,
    channels: usize,
    samples: usize,
    up: usize,
    down: usize,
    filter: &ResampleFilter,
    edge: EdgeMode,
    kernel: Option<ResampleKernel>,
) -> usize {
    assert!(up > 0 && down > 0, "resample_poly factors must be positive");
    let g = gcd(up, down);
    let (up, down) = (up / g, down / g);
    let out_len = resample_poly_len(samples, up, down);
    if channels == 0 || out_len == 0 {
        return out_len;
    }

    let (mut taps, half_len) = match filter {
        ResampleFilter::Window(window) => {
            let half_len = RESAMPLE_POLY_HALF_LEN_PER_RATE * up.max(down);
            (firwin(2 * half_len + 1, 1.0 / up.max(down) as f64, *window), half_len)
        }
        ResampleFilter::Taps(taps) => (taps.clone(), (taps.len().max(1) - 1) / 2),
    };
    for t in &mut taps {
        *t *= up as f64;
    }
    // scipy's alignment: pad the filter by `pre_pad` zeros and drop the first `pre_remove` outputs
    let pre_pad = down - half_len % down;
    let pre_remove = (half_len + pre_pad) / down;
    let start = (pre_remove * down) as i64 - pre_pad as i64;
    // Farthest the taps reach outside the input, for non-zero edge modes
    let s_min = (start - (taps.len() as i64 - 1)).div_euclid(up as i64);
    let s_max = ((out_len as i64 - 1) * down as i64 + start).div_euclid(up as i64);
    let ext_pad = (-s_min).max(s_max - (samples as i64 - 1)).max(0) as usize;

    let inputs = PolyInputs {
        client: client.clone(),
        input: input.clone(),
        output: output.clone(),
        taps,
        channels,
        samples,
        out_len,
        up,
        down,
        start,
        ext_pad,
        edge,
    };
    match kernel {
        Some(kernel) => inputs.run::<F>(kernel),
        None => tuned::<F>(inputs),
    }
    out_len
}

/// One `resample_poly` call after design and alignment (cloned per autotune candidate).
#[derive(Clone)]
struct PolyInputs {
    client: Client,
    input: Handle,
    output: Handle,
    /// Scaled by `up`.
    taps: Vec<f64>,
    channels: usize,
    samples: usize,
    out_len: usize,
    up: usize,
    down: usize,
    start: i64,
    ext_pad: usize,
    edge: EdgeMode,
}

impl PolyInputs {
    fn run<F: DspFloat>(&self, kernel: ResampleKernel) {
        let client = &self.client;
        let width = match kernel {
            ResampleKernel::Vectorized { width } => client
                .io_optimized_vector_sizes(size_of::<F>())
                .filter(|&w| w > 1 && w <= width && self.out_len % w == 0)
                .max(),
            ResampleKernel::Direct => None,
        };
        let (in_total, out_total) = (self.channels * self.samples, self.channels * self.out_len);
        let args = (self.samples as u32, self.out_len as u32, self.taps.len() as u32, self.up as u32, self.down as u32, self.start as i32, self.ext_pad as u32, self.edge.id());
        match width {
            None => {
                let taps_h = buffer::upload(client, &cast_all::<F>(&self.taps));
                let geom = LaunchGeometry::channels_samples(client, self.channels, self.out_len);
                // SAFETY: input holds `in_total`, output `out_total` values of `F`
                unsafe {
                    upfirdn_kernel::launch::<F>(
                        client,
                        geom.cube_count,
                        geom.cube_dim,
                        BufferArg::from_raw_parts(self.input.clone(), in_total),
                        BufferArg::from_raw_parts(taps_h, self.taps.len()),
                        BufferArg::from_raw_parts(self.output.clone(), out_total),
                        self.channels as u32,
                        args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7,
                    );
                }
            }
            Some(width) => {
                let table = buffer::upload(client, &cast_all::<F>(&poly_table(&self.taps, self.down, width)));
                let table_len = (self.taps.len() + (width - 1) * self.down) * width;
                let geom = LaunchGeometry::elementwise(client, out_total / width);
                // SAFETY: as above; `table` holds `table_len` values (`table_len / width` vectors)
                unsafe {
                    upfirdn_vector_kernel::launch::<F>(
                        client,
                        geom.cube_count,
                        geom.cube_dim,
                        width,
                        BufferArg::from_raw_parts(self.input.clone(), in_total),
                        BufferArg::from_raw_parts(table, table_len),
                        BufferArg::from_raw_parts(self.output.clone(), out_total),
                        args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7,
                    );
                }
            }
        }
    }
}

/// The tap table of [`upfirdn_vector_kernel`]: `K + (width − 1) · down` vectors, lane `l` of vector
/// `d + reach` holding `taps[d + l · down]` (zero outside), `reach = (width − 1) · down`.
fn poly_table(taps: &[f64], down: usize, width: usize) -> Vec<f64> {
    let reach = (width - 1) * down;
    let mut table = Vec::with_capacity((taps.len() + reach) * width);
    for e in 0..taps.len() + reach {
        for l in 0..width {
            let j = e as isize - reach as isize + (l * down) as isize;
            table.push(if (0..taps.len() as isize).contains(&j) { taps[j as usize] } else { 0.0 });
        }
    }
    table
}

/// [`PolyInputs::run`] with the kernel CubeCL's autotuner found fastest for this device, element
/// type, rates, filter length and size. Benchmarks write to a scratch output.
fn tuned<F: DspFloat>(inputs: PolyInputs) {
    static TUNER: LocalTuner<String, String> = local_tuner!("resample-poly-v1");
    let client = inputs.client.clone();
    let id = tune_id(&client);
    let widths: Vec<usize> = client.io_optimized_vector_sizes(size_of::<F>()).filter(|&w| w > 1).collect();
    let set = TUNER.init(&id, move || {
        let key = |p: &PolyInputs| {
            // `a`: the power of two dividing the output row (up to 16), which decides the widths that run
            format!(
                "{}-u{}-d{}-k{}-c{}-t{}-a{}",
                F::type_name(),
                p.up,
                p.down,
                p.taps.len(),
                size_class(p.channels),
                size_class(p.out_len),
                p.out_len.trailing_zeros().min(4)
            )
        };
        let scratch = |_: &String, p: &PolyInputs| PolyInputs { output: buffer::empty::<F>(&p.client, p.channels * p.out_len), ..p.clone() };
        let set: TunableSet<String, PolyInputs, ()> = TunableSet::new(key, scratch);
        let kernels = std::iter::once(ResampleKernel::Direct).chain(widths.iter().map(|&width| ResampleKernel::Vectorized { width }));
        kernels.fold(set, |set, kernel| {
            set.with(Tunable::new(&format!("{kernel:?}"), move |p: PolyInputs| {
                p.run::<F>(kernel);
                Ok::<_, String>(())
            }))
        })
    });
    TUNER.execute(&id, &client, set, inputs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Host `resample_poly` straight from its definition: up-sample with zeros, convolve with the
    /// padded filter, down-sample, keep `[pre_remove, pre_remove + out_len)`.
    fn host_resample_poly(x: &[f64], up: usize, down: usize) -> Vec<f64> {
        let half_len = RESAMPLE_POLY_HALF_LEN_PER_RATE * up.max(down);
        let mut h = firwin(2 * half_len + 1, 1.0 / up.max(down) as f64, RESAMPLE_POLY_WINDOW);
        h.iter_mut().for_each(|t| *t *= up as f64);
        let pre_pad = down - half_len % down;
        let pre_remove = (half_len + pre_pad) / down;
        let h: Vec<f64> = std::iter::repeat_n(0.0, pre_pad).chain(h).collect();
        let xu: Vec<f64> = (0..x.len() * up).map(|i| if i % up == 0 { x[i / up] } else { 0.0 }).collect();
        let out_len = resample_poly_len(x.len(), up, down);
        (0..out_len)
            .map(|m| {
                let n = (m + pre_remove) * down;
                (0..h.len()).filter(|&k| k <= n && n - k < xu.len()).map(|k| h[k] * xu[n - k]).sum()
            })
            .collect()
    }

    fn matches_host(client: &Client) {
        let samples = 997;
        let x: Vec<f64> = (0..samples).map(|i| (i as f64 * 0.05).sin() * 10.0 + ((i * 7919) % 31) as f64 * 0.1).collect();
        let input = buffer::upload(client, &x.iter().map(|v| *v as f32).collect::<Vec<_>>());
        for (up, down) in [(1usize, 3usize), (3, 1), (2, 3), (160, 147)] {
            let out_len = resample_poly_len(samples, up, down);
            let output = buffer::empty::<f32>(client, out_len);
            let n = resample_poly::<f32>(client, &input, &output, 1, samples, up, down, &ResampleFilter::default(), RESAMPLE_POLY_DEFAULT_EDGE);
            assert_eq!(n, out_len);
            let got = buffer::download::<f32>(client, output);
            let want = host_resample_poly(&x, up, down);
            for (m, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!((*g as f64 - w).abs() < 1e-3, "{} {up}/{down} sample {m}: {g} vs {w}", client.name());
            }
        }
    }
    runtime_test!(test_resample_poly_matches_definition, matches_host);

    /// The vectorized kernel is bit-identical to `Direct` (f32 / f64, rates up and down, every
    /// edge, output rows divisible by the width or not).
    fn vectorized_matches_direct_of<F: DspFloat + std::fmt::Debug>(client: &Client) {
        let channels = 3;
        for samples in [96usize, 997, 1_024] {
            let x: Vec<F> = (0..channels * samples).map(|i| crate::core::cast::<F>(((i * 7919) % 211) as f64 * 0.37 - 40.0)).collect();
            let input = buffer::upload(client, &x);
            for (up, down) in [(1usize, 3usize), (3, 1), (2, 3), (1, 2), (160, 147)] {
                for edge in [EdgeMode::Zeros, EdgeMode::Odd, EdgeMode::Reflect, EdgeMode::Nearest] {
                    let out_len = resample_poly_len(samples, up, down);
                    let run = |kernel| {
                        let out = buffer::empty::<F>(client, channels * out_len);
                        resample_poly_with::<F>(client, &input, &out, channels, samples, up, down, &ResampleFilter::default(), edge, Some(kernel));
                        buffer::download::<F>(client, out)
                    };
                    let direct = run(ResampleKernel::Direct);
                    for width in [2usize, 4, 8, 16] {
                        assert_eq!(run(ResampleKernel::Vectorized { width }), direct, "{} {} {samples} samples {up}/{down} {edge:?} width {width}", client.name(), F::type_name());
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
    runtime_test!(test_resample_poly_vectorized_matches_direct, vectorized_matches_direct);
}
