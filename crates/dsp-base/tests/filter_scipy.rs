//! Filter designs and filtering against `scipy.signal` (fixtures from `scipy_reference.py`).

mod common;

use common::*;
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use dsp_base::filter::{DeviceFilter, FilterBand, FilterMode, FilterSpec, Sos};

#[test]
fn designs_match_scipy() {
    let fx = fixtures();
    let fs = fx["fs"].as_f64().unwrap();
    for case in fx["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let expected = sos_from(&case["sos"]);
        let got = spec_from(&case["spec"]).design(fs).unwrap();
        assert_eq!(got.len(), expected.len(), "{name}: section count");
        for (i, (g, e)) in got.sections.iter().zip(&expected.sections).enumerate() {
            let scale = e.b.iter().chain(&e.a).fold(0.0f64, |m, c| m.max(c.abs()));
            for (gc, ec) in g.b.iter().chain(&g.a).zip(e.b.iter().chain(&e.a)) {
                assert!(
                    (gc - ec).abs() <= 1e-9 * scale.max(1e-300) + 1e-15,
                    "{name} section {i}: got {:?} / {:?}, scipy {:?} / {:?}",
                    g.b, g.a, e.b, e.a
                );
            }
        }
        assert_eq!(
            got.settling_samples(fx["tol"].as_f64().unwrap()),
            case["settling"].as_u64().unwrap() as usize,
            "{name}: settling"
        );
    }
}

#[test]
fn host_reference_matches_sosfilt_and_sosfiltfilt() {
    let fx = fixtures();
    let (fs, n) = (fx["fs"].as_f64().unwrap(), fx["n"].as_u64().unwrap() as usize);
    let x = test_signal(n, fs);
    for case in fx["cases"].as_array().unwrap().iter().filter(|c| c.get("forward").is_some()) {
        let name = case["name"].as_str().unwrap();
        let sos = sos_from(&case["sos"]);
        let fwd = floats(&case["forward"]);
        let fb = floats(&case["forward_backward"]);
        let pad = case["padlen"].as_u64().unwrap() as usize;
        let scale = max_abs(&x);
        assert!(max_abs_diff(&sos.filter(&x, true), &fwd) < 1e-8 * scale, "{name}: sosfilt");
        assert!(max_abs_diff(&sos.filtfilt(&x, pad), &fb) < 1e-8 * scale, "{name}: sosfiltfilt");
    }
}

fn run_device(sos: &Sos, mode: FilterMode, x: &[f64], channels: usize) -> Vec<Vec<f64>> {
    let client = WgpuRuntime::client(&WgpuDevice::default());
    let n = x.len();
    let data: Vec<f32> = (0..channels).flat_map(|c| x.iter().map(move |v| (*v as f32) * (1.0 + c as f32))).collect();
    let input = client.create_from_slice(f32::as_bytes(&data));
    let output = client.empty(data.len() * 4);
    let filter = DeviceFilter::from_sos(&client, sos.clone(), mode);
    let scratch = client.empty((filter.scratch_len(channels, n) * 4).max(4));
    let state = client.empty(channels * filter.state_len() * 4);
    filter.apply(&client, &input, &output, &scratch, &state, channels, n);
    let out = f32::from_bytes(&client.read_one_unchecked(output)).to_vec();
    out.chunks(n)
        .enumerate()
        .map(|(c, row)| row.iter().map(|v| *v as f64 / (1.0 + c as f64)).collect())
        .collect()
}

#[test]
fn device_kernel_matches_scipy() {
    let fx = fixtures();
    let (fs, n) = (fx["fs"].as_f64().unwrap(), fx["n"].as_u64().unwrap() as usize);
    let x = test_signal(n, fs);
    let scale = max_abs(&x);
    for case in fx["cases"].as_array().unwrap().iter().filter(|c| c.get("forward").is_some()) {
        let name = case["name"].as_str().unwrap();
        let sos = sos_from(&case["sos"]);
        assert_eq!(sos.settling_samples(1e-3).min(n - 1), case["padlen"].as_u64().unwrap() as usize);
        for (mode, key) in [(FilterMode::Forward, "forward"), (FilterMode::ForwardBackward, "forward_backward")] {
            let expected = floats(&case[key]);
            for (c, got) in run_device(&sos, mode, &x, 3).iter().enumerate() {
                let err = max_abs_diff(got, &expected);
                // f32 data and state: error relative to the input amplitude (500 µV DC + signal).
                assert!(err < 2e-5 * scale, "{name} {key} channel {c}: max error {err}");
            }
        }
    }
}

#[test]
fn device_low_cutoffs_match_f64_reference() {
    // Poles next to z = 1: plain f32 direct form II is off by tens of µV here.
    let fs = 30_000.0;
    let x = test_signal(60_000, fs);
    let scale = max_abs(&x);
    let specs = [
        ("highpass 0.5 Hz order 2", FilterSpec::butterworth(2, FilterBand::Highpass(0.5))),
        ("bandpass 0.5-300 Hz order 4", FilterSpec::butterworth(4, FilterBand::Bandpass(0.5, 300.0))),
        ("notch 60 Hz Q30", FilterSpec::notch(60.0, 30.0)),
    ];
    for (name, spec) in specs {
        let sos = spec.design(fs).unwrap();
        let pad = sos.settling_samples(1e-3).min(x.len() - 1);
        let expected_fwd = sos.filter(&x, true);
        let expected_fb = sos.filtfilt(&x, pad);
        for (mode, expected) in [(FilterMode::Forward, &expected_fwd), (FilterMode::ForwardBackward, &expected_fb)] {
            let got = &run_device(&sos, mode, &x, 1)[0];
            let err = max_abs_diff(got, expected);
            assert!(err < 2e-5 * scale, "{name} {mode:?}: max error {err}");
        }
    }
}
