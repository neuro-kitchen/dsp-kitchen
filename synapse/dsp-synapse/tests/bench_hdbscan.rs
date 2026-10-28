//! HDBSCAN on clip-like data (61-sample waveforms: clusters of different spreads, scattered
//! outliers, exact duplicates for distance ties), timed, with the labels kept as a reference so a
//! rewrite can be checked for identical output. Ignored by default:
//!
//! ```text
//! HDBSCAN_POINTS=40000 HDBSCAN_REFERENCE=/tmp/hdbscan_ref.bin \
//!   cargo test -p dsp-synapse --release --test bench_hdbscan -- --ignored --nocapture
//! ```
//!
//! The first run writes `HDBSCAN_REFERENCE`; later runs compare against it.

use std::time::Instant;

use dsp_core::compute::{ComputeTarget, ComputeTask};
use dsp_synapse::sorting::hdbscan;

/// Samples per clip (Kilosort4 / EMUsort `nt`).
const DIMS: usize = 61;
/// EMUsort `hdbscan_min_cluster_size`.
const MIN_CLUSTER_SIZE: usize = 20;
/// Points when `HDBSCAN_POINTS` is not set.
const DEFAULT_POINTS: usize = 40_000;
/// Waveform clusters, and every how many points one is a scattered outlier / an exact duplicate.
const CLUSTERS: usize = 9;
const OUTLIER_EVERY: usize = 23;
const DUPLICATE_EVERY: usize = 97;

/// Deterministic clip-like points (`[n, DIMS]` row-major).
fn clips(n: usize) -> Vec<f32> {
    let mut state = 0x2545_f491u32;
    let mut uniform = move || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
    };
    let mut x = vec![0.0f32; n * DIMS];
    for i in 0..n {
        let row = i * DIMS;
        if i % DUPLICATE_EVERY == DUPLICATE_EVERY - 1 {
            x.copy_within(row - DIMS..row, row);
            continue;
        }
        let (k, spread, amp) = if i % OUTLIER_EVERY == 0 {
            (0, 6.0, 0.0)
        } else {
            let k = i % CLUSTERS;
            (k, 0.1 + 0.08 * k as f32, 1.0 + 0.3 * k as f32)
        };
        for t in 0..DIMS {
            // A trough of cluster-specific width and latency
            let tau = (t as f32 - 20.0 - k as f32) / (2.0 + k as f32 * 0.5);
            x[row + t] = -amp * (-tau * tau).exp() + spread * uniform();
        }
    }
    x
}

struct Run(usize);

impl ComputeTask for Run {
    type Output = Vec<i32>;
    fn run(self, client: cubecl::prelude::Client) -> Vec<i32> {
        let x = clips(self.0);
        let start = Instant::now();
        let labels = hdbscan(&client, &x, self.0, DIMS, MIN_CLUSTER_SIZE);
        let clusters = labels.iter().copied().max().map_or(0, |m| m + 1);
        let noise = labels.iter().filter(|&&l| l < 0).count();
        println!("{} | {} points x {DIMS} | {:.2} s | {clusters} clusters, {noise} noise", client.name(), self.0, start.elapsed().as_secs_f64());
        labels
    }
}

#[test]
#[ignore = "benchmark: run with --ignored --nocapture"]
fn bench_and_compare_hdbscan() {
    let n = std::env::var("HDBSCAN_POINTS").ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_POINTS);
    let labels = ComputeTarget::from_env().expect("a runtime").run(Run(n)).expect("runtime");
    let Ok(path) = std::env::var("HDBSCAN_REFERENCE") else { return };
    let bytes: Vec<u8> = labels.iter().flat_map(|l| l.to_le_bytes()).collect();
    match std::fs::read(&path) {
        Ok(reference) => {
            let want: Vec<i32> = reference.chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            let differ = labels.iter().zip(&want).filter(|(a, b)| a != b).count();
            assert_eq!(labels.len(), want.len(), "point counts differ from the reference");
            assert_eq!(differ, 0, "{differ} labels differ from the reference {path}");
            println!("labels identical to {path}");
        }
        Err(_) => {
            std::fs::write(&path, bytes).expect("write reference");
            println!("reference written to {path}");
        }
    }
}
