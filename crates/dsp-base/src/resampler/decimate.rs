//! Integer down-sampling behind an anti-aliasing filter (`scipy.signal.decimate`, zero phase).

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_core::compute::LaunchGeometry;

use super::design::{firwin, FirWindow};
use super::kernels::downsample_kernel;
use super::poly::{resample_poly, ResampleFilter, RESAMPLE_POLY_DEFAULT_EDGE};
use crate::core::{buffer, DspFloat};
use crate::filter::iir::DeviceFilter;
use crate::filter::{FilterBand, FilterError, FilterMode, FilterSpec};

/// Anti-aliasing cutoff as a fraction of the new Nyquist frequency (`decimate`'s IIR design).
pub const DECIMATE_IIR_CUTOFF: f64 = 0.8;

/// FIR taps per unit of the factor (`decimate`'s FIR design: `20 · q + 1` taps).
pub const DECIMATE_FIR_TAPS_PER_FACTOR: usize = 20;

/// Anti-aliasing filter of [`decimate`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecimateFilter {
    /// Chebyshev type I low-pass at [`DECIMATE_IIR_CUTOFF`] of the new Nyquist, run forward-backward,
    /// then every `q`-th sample (`ftype="iir"`).
    Iir { order: usize, ripple_db: f64 },
    /// Hamming `firwin` of `taps_per_factor · q + 1` taps through [`resample_poly`] (`ftype="fir"`).
    Fir { taps_per_factor: usize },
}

/// `decimate`'s default: an order-8 Chebyshev type I with 0.05 dB ripple.
pub const DECIMATE_DEFAULT: DecimateFilter = DecimateFilter::Iir { order: 8, ripple_db: 0.05 };

/// Samples out of `samples` decimated by `q` (`ceil(samples / q)`).
pub fn decimate_len(samples: usize, q: usize) -> usize {
    samples.div_ceil(q.max(1))
}

/// Down-samples every channel of a `[channels, samples]` buffer of `F` by `q` into `output`
/// (`[channels, decimate_len(samples, q)]`) behind `filter`. Returns the output length.
///
/// The IIR path pads with odd reflection over the filter's settling length (where `sosfiltfilt`
/// pads a fixed `3·(2·sections + 1)`), so samples near the ends can differ slightly from scipy.
///
/// # Panics
/// If `q` is zero.
#[allow(clippy::too_many_arguments)]
pub fn decimate<R: Runtime, F: DspFloat>(
    client: &ComputeClient<R>,
    input: &Handle,
    output: &Handle,
    channels: usize,
    samples: usize,
    q: usize,
    filter: DecimateFilter,
) -> Result<usize, FilterError> {
    assert!(q > 0, "decimation factor must be positive");
    let out_len = decimate_len(samples, q);
    if channels == 0 || samples == 0 {
        return Ok(out_len);
    }
    match filter {
        DecimateFilter::Iir { order, ripple_db } => {
            // Designed on scipy's normalized grid: fs = 2, so the new Nyquist is 1 / q
            let spec = FilterSpec::chebyshev1(order, ripple_db, FilterBand::Lowpass(DECIMATE_IIR_CUTOFF / q as f64))
                .with_mode(FilterMode::ForwardBackward);
            let device = DeviceFilter::<F>::new(client, &spec, 2.0)?;
            let filtered = buffer::empty::<R, F>(client, channels * samples);
            let scratch = buffer::empty::<R, F>(client, device.scratch_len(channels, samples));
            let state = buffer::empty::<R, F>(client, channels * device.state_len());
            device.apply(client, input, &filtered, &scratch, &state, channels, samples);

            let geom = LaunchGeometry::channels_samples(client, channels, out_len);
            unsafe {
                downsample_kernel::launch::<F, R>(
                    client,
                    geom.cube_count,
                    geom.cube_dim,
                    ArrayArg::from_raw_parts(filtered, channels * samples),
                    ArrayArg::from_raw_parts(output.clone(), channels * out_len),
                    channels as u32,
                    samples as u32,
                    out_len as u32,
                    q as u32,
                    0u32,
                );
            }
            Ok(out_len)
        }
        DecimateFilter::Fir { taps_per_factor } => {
            let taps = firwin(taps_per_factor * q + 1, 1.0 / q as f64, FirWindow::Hamming);
            let filter = ResampleFilter::Taps(taps);
            Ok(resample_poly::<R, F>(client, input, output, channels, samples, 1, q, &filter, RESAMPLE_POLY_DEFAULT_EDGE))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iir_matches_host<R: Runtime>(client: &ComputeClient<R>) {
        let (samples, q) = (5_000usize, 4usize);
        let x: Vec<f64> = (0..samples).map(|i| (i as f64 * 0.01).sin() * 50.0 + (i as f64 * 1.9).sin() * 5.0 - 20.0).collect();
        let input = buffer::upload(client, &x.iter().map(|v| *v as f32).collect::<Vec<_>>());
        let out_len = decimate_len(samples, q);
        let output = buffer::empty::<R, f32>(client, out_len);
        assert_eq!(decimate::<R, f32>(client, &input, &output, 1, samples, q, DECIMATE_DEFAULT).unwrap(), out_len);
        let got = buffer::download::<R, f32>(client, output);

        let DecimateFilter::Iir { order, ripple_db } = DECIMATE_DEFAULT else { unreachable!() };
        let sos = crate::filter::design::chebyshev1_sos(order, ripple_db, FilterBand::Lowpass(DECIMATE_IIR_CUTOFF / q as f64), 2.0).unwrap();
        let filtered = sos.filtfilt(&x, sos.settling_samples(crate::filter::design::DEFAULT_SETTLING_TOLERANCE).min(samples - 1));
        for (m, g) in got.iter().enumerate() {
            let w = filtered[m * q];
            assert!((*g as f64 - w).abs() < 2e-3 * 70.0, "{} sample {m}: {g} vs {w}", R::name(client));
        }
    }
    runtime_test!(test_decimate_iir_matches_host, iir_matches_host);
}
