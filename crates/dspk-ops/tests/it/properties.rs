//! Properties the maths guarantees, checked on the device in float32 (references are not our
//! own kernels): linearity and time invariance of a recursive filter, zero phase of a
//! forward-backward pass, Parseval's identity for the real FFT the overlap-save FIR runs on.

#![cfg(feature = "wgpu")]

use crate::common::*;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use dspk_ops::filter::{FilterBand, FilterMode, FilterSpec, FilterStart};

const FS: f64 = 24_414.0625;

/// A deterministic signal with broadband content (no `rand` here).
fn noise(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 200.0
        })
        .collect()
}

#[test]
fn a_recursive_filter_is_linear() {
    let sos = FilterSpec::butterworth(4, FilterBand::Bandpass(300.0, 6000.0)).design(FS).unwrap();
    let (x1, x2) = (noise(40_000, 1), noise(40_000, 2));
    let (a, b) = (1.7, -0.6);
    let mix: Vec<f64> = x1.iter().zip(&x2).map(|(p, q)| a * p + b * q).collect();
    for client in &test_clients() {
        let run = |x: &[f64]| run_device(client, &sos, FilterMode::ForwardBackward, FilterStart::Rest, x, 1).remove(0);
        let (y1, y2, y) = (run(&x1), run(&x2), run(&mix));
        let sum: Vec<f64> = y1.iter().zip(&y2).map(|(p, q)| a * p + b * q).collect();
        let err = max_abs_diff(&y, &sum) / max_abs(&y);
        assert!(err < 1e-4, "{}: f(a·x1 + b·x2) vs a·f(x1) + b·f(x2): {err:.2e} of the peak", client.name());
    }
}

#[test]
fn a_recursive_filter_is_time_invariant() {
    let sos = FilterSpec::butterworth(4, FilterBand::Highpass(300.0)).design(FS).unwrap();
    let x = noise(30_000, 3);
    let shift = 1_237;
    let mut delayed = vec![0.0; shift];
    delayed.extend_from_slice(&x);
    for client in &test_clients() {
        let y = run_device(client, &sos, FilterMode::Forward, FilterStart::Rest, &x, 1).remove(0);
        let yd = run_device(client, &sos, FilterMode::Forward, FilterStart::Rest, &delayed, 1).remove(0);
        let err = max_abs_diff(&yd[shift..], &y) / max_abs(&y);
        assert!(err < 1e-4, "{}: a delayed input gives the delayed output: {err:.2e} of the peak", client.name());
        assert!(yd[..shift].iter().all(|v| v.abs() < 1e-6), "{}: nothing before the input starts", client.name());
    }
}

#[test]
fn a_forward_backward_filter_has_zero_phase() {
    // A symmetric burst: a zero-phase filter keeps it symmetric (away from the ends, which a pass
    // from rest treats unlike each other)
    let n = 60_001;
    let mid = (n / 2) as f64;
    let x: Vec<f64> = (0..n)
        .map(|i| {
            let t = (i as f64 - mid) / FS;
            (-(t * 200.0).powi(2)).exp() * (2.0 * std::f64::consts::PI * 900.0 * t).cos() * 100.0
        })
        .collect();
    let sos = FilterSpec::butterworth(4, FilterBand::Bandpass(300.0, 3000.0)).design(FS).unwrap();
    let edge = sos.settling_samples(1e-6);
    for client in &test_clients() {
        let y = run_device(client, &sos, FilterMode::ForwardBackward, FilterStart::Rest, &x, 1).remove(0);
        let peak = max_abs(&y);
        let worst = (edge..n - edge).map(|i| (y[i] - y[n - 1 - i]).abs()).fold(0.0, f64::max) / peak;
        assert!(worst < 1e-4, "{}: y(t) vs y(−t) around the burst: {worst:.2e} of the peak", client.name());
    }
}

#[test]
fn the_real_fft_keeps_energy() {
    // Parseval for a real FFT of n points: Σ x² = (|X₀|² + |X_{n/2}|² + 2 Σ_{0<k<n/2} |X_k|²) / n
    let n = 4096usize;
    let x: Vec<f32> = noise(n, 4).iter().map(|v| *v as f32).collect();
    let dtype = ElemType::Float(FloatKind::F32);
    for client in &test_clients() {
        let input = client.create_from_slice(f32::as_bytes(&x));
        let (re, im) = cubek_fft::rfft(client, TensorHandle::new_contiguous(vec![1, n], input, dtype), 1, dtype);
        let re = f32::from_bytes(&client.read_one_unchecked(re.handle.clone())).to_vec();
        let im = f32::from_bytes(&client.read_one_unchecked(im.handle.clone())).to_vec();
        let bins = n / 2 + 1;
        let power = |k: usize| (re[k] as f64).powi(2) + (im[k] as f64).powi(2);
        let spectral = (power(0) + power(bins - 1) + 2.0 * (1..bins - 1).map(power).sum::<f64>()) / n as f64;
        let temporal: f64 = x.iter().map(|v| (*v as f64).powi(2)).sum();
        let err = (spectral - temporal).abs() / temporal;
        assert!(err < 1e-5, "{}: Parseval: {spectral} vs {temporal} ({err:.2e})", client.name());
    }
}
