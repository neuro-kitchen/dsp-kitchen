//! Timings and accuracy of the symmetric eigensolver at whitening sizes, on the discrete and the
//! integrated GPU (wgpu): one 384 × 384 channel covariance (the global path) and 384 local 32 × 32
//! neighbourhoods (the batched shared-memory path). Ignored by default:
//!
//! ```text
//! cargo test -p dsp-base --release --test bench_eigen -- --ignored --nocapture
//! ```

use std::time::Instant;

use cubecl::device::{WgpuDevice, WgpuDeviceKind};
use cubecl::prelude::*;
use cubecl::Device;
use dsp_base::core::buffer;
use dsp_base::linalg::{symmetric_eigen_batched, EigenOptions, SymmetricEigen};

/// Channels of a Neuropixels probe, and a local whitening neighbourhood.
const CHANNELS: usize = 384;
const NEIGHBOURHOOD: usize = 32;

/// A covariance-like SPD matrix with eigenvalues spread over `decades` orders of magnitude
/// (`Q diag(λ) Qᵀ`, `Q` from Householder reflections), row-major `[n, n]`.
fn spd(n: usize, decades: f64, seed: usize) -> Vec<f64> {
    let lambda: Vec<f64> = (0..n).map(|i| 10f64.powf(-decades * i as f64 / (n - 1).max(1) as f64)).collect();
    let mut a: Vec<f64> = (0..n * n).map(|e| if e / n == e % n { lambda[e / n] } else { 0.0 }).collect();
    for r in 0..3 {
        let v: Vec<f64> = (0..n).map(|i| (((i + 31 * r + seed) * 7919) % 211) as f64 / 211.0 - 0.5).collect();
        let vv: f64 = v.iter().map(|x| x * x).sum();
        // A ← H A H with H = I − 2 v vᵀ / vᵀv
        let h = |i: usize, j: usize| (if i == j { 1.0 } else { 0.0 }) - 2.0 * v[i] * v[j] / vv;
        let ha: Vec<f64> = (0..n * n).map(|e| (0..n).map(|k| h(e / n, k) * a[k * n + e % n]).sum()).collect();
        a = (0..n * n).map(|e| (0..n).map(|k| ha[(e / n) * n + k] * h(k, e % n)).sum()).collect();
    }
    a
}

/// Largest `‖A v − λ v‖ / λ` over the eigenpairs (relative residual, so small eigenvalues count).
fn worst_relative_residual(a: &[f64], e: &SymmetricEigen) -> f64 {
    let n = e.n;
    (0..n)
        .map(|k| {
            let r: f64 = (0..n)
                .map(|i| {
                    let av: f64 = (0..n).map(|j| a[i * n + j] * e.vectors[j * n + k]).sum();
                    (av - e.values[k] * e.vectors[i * n + k]).powi(2)
                })
                .sum::<f64>()
                .sqrt();
            r / e.values[k].abs().max(f64::MIN_POSITIVE)
        })
        .fold(0.0, f64::max)
}

fn bench(device: &str, client: &Client) {
    for (n, batch) in [(CHANNELS, 1usize), (NEIGHBOURHOOD, CHANNELS)] {
        let mats: Vec<f64> = (0..batch).flat_map(|b| spd(n, 3.0, b)).collect();
        let handle = buffer::upload(client, &mats.iter().map(|&x| x as f32).collect::<Vec<_>>());
        // Warm-up (compilation), then one timed solve: the readback inside waits for the device
        symmetric_eigen_batched::<f32>(client, &handle, batch, n, EigenOptions::default());
        let start = Instant::now();
        let eig = symmetric_eigen_batched::<f32>(client, &handle, batch, n, EigenOptions::default());
        let ms = start.elapsed().as_secs_f64() * 1e3;
        let worst = worst_relative_residual(&mats[..n * n], &eig[0]);
        println!("{device:>6} | {batch:>3} x {n:>3}x{n:<3} | {ms:>9.2} ms | worst relative residual {worst:.2e}");
    }
}

#[test]
#[ignore = "benchmark: run with --ignored --nocapture"]
fn bench_eigensolver() {
    for (name, kind) in [("dGPU", WgpuDeviceKind::DiscreteGpu(0)), ("iGPU", WgpuDeviceKind::IntegratedGpu(0))] {
        bench(name, &Device::Wgpu(WgpuDevice::new(kind)).client());
    }
}
