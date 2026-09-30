//! The SOS kernel gives the same result on every CubeCL runtime (one launch geometry, no
//! device-specific paths). Run with `--features cpu` to include the CPU runtime.

mod common;

use common::*;
use cubecl::prelude::*;
use dsp_base::filter::{DeviceFilter, FilterBand, FilterMode, FilterSpec};

fn run<R: Runtime>(client: &ComputeClient<R>, spec: &FilterSpec, x: &[f64], channels: usize) -> Vec<f64> {
    let n = x.len();
    let data: Vec<f32> = (0..channels).flat_map(|_| x.iter().map(|v| *v as f32)).collect();
    let input = client.create_from_slice(f32::as_bytes(&data));
    let output = client.empty(data.len() * 4);
    let filter = DeviceFilter::new(client, spec, 30_000.0).unwrap();
    let scratch = client.empty((filter.scratch_len(channels, n) * 4).max(4));
    let state = client.empty(channels * filter.state_len() * 4);
    filter.apply(client, &input, &output, &scratch, &state, channels, n);
    f32::from_bytes(&client.read_one_unchecked(output)).iter().map(|v| *v as f64).collect()
}

fn check<R: Runtime>(client: &ComputeClient<R>) {
    let x = test_signal(20_000, 30_000.0);
    let scale = max_abs(&x);
    // 130 channels: more than one cube of 64 channels, not a multiple of it.
    let channels = 130;
    for mode in [FilterMode::Forward, FilterMode::ForwardBackward] {
        for spec in [
            FilterSpec::butterworth(5, FilterBand::Bandpass(300.0, 6000.0)),
            FilterSpec::notch(60.0, 30.0),
        ] {
            let spec = spec.with_mode(mode);
            let sos = spec.design(30_000.0).unwrap();
            let expected = match mode {
                FilterMode::Forward => sos.filter(&x, true),
                FilterMode::ForwardBackward => sos.filtfilt(&x, sos.settling_samples(1e-3).min(x.len() - 1)),
            };
            let got = run(client, &spec, &x, channels);
            for c in 0..channels {
                let err = max_abs_diff(&got[c * x.len()..(c + 1) * x.len()], &expected);
                assert!(err < 2e-5 * scale, "{} {spec:?} channel {c}: max error {err}", R::name(client));
            }
        }
    }
}

#[test]
fn wgpu_runtime() {
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    check(&WgpuRuntime::client(&WgpuDevice::default()));
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_runtime() {
    use cubecl::cpu::{CpuDevice, CpuRuntime};
    check(&CpuRuntime::client(&CpuDevice));
}
