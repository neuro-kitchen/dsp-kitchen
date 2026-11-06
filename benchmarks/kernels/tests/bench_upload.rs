//! Host → device upload strategies for one detection window (384 channels × 60 000 `f32`,
//! 92 MB), each ending in a device sync. Ignored by default:
//!
//! ```text
//! cargo test -p dspk-benchmarks --profile validate --test bench_upload -- --ignored --nocapture
//! ```

use std::time::Instant;

use cubecl::bytes::Bytes;
use cubecl::prelude::*;
use dspk_benchmarks::Report;
use dspk_core::compute::bench::sync;

const CHANNELS: usize = 384;
const WINDOW: usize = 60_000;
/// Timed uploads per strategy (after one warm-up).
const ITERATIONS: usize = 5;

fn time(client: &Client, mut f: impl FnMut()) -> f64 {
    f();
    sync(client);
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        f();
        sync(client);
    }
    start.elapsed().as_secs_f64() * 1e3 / ITERATIONS as f64
}

#[test]
#[ignore = "benchmark: run with --ignored --nocapture"]
fn bench_uploads() {
    // Every runtime the tests run on (`DSPK_TEST_RUNTIMES`), opened through dspk
    for target in dspk_core::ComputeTarget::for_tests() {
        println!("{}:", target.name());
        let mut report = Report::new("kernels/upload", target.name(), &format!("one {CHANNELS} × {WINDOW} float32 window per upload, each ending in a device sync; mean of {ITERATIONS} after a warm-up"));
        uploads(&target.client().expect("compiled-in runtime"), &mut report);
        report.write().expect("writing the record");
    }
}

fn uploads(client: &Client, record: &mut Report) {
    let client = client.clone();
    let x: Vec<f32> = (0..CHANNELS * WINDOW).map(|i| (i % 1013) as f32).collect();
    let bytes = x.len() * size_of::<f32>();
    let mut report = |name: &str, ms: f64| {
        println!("{name:<38} | {ms:>8.2} ms | {:>6.2} GB/s", bytes as f64 / ms / 1e6);
        record.case(name, &[CHANNELS, WINDOW], ms, &[("gb_per_s", bytes as f64 / ms / 1e6)]);
    };

    report("(a) create_from_slice (new buffer)", time(&client, || drop(client.create_from_slice(f32::as_bytes(&x)))));
    report("(b) create(Bytes) (new buffer, owned)", time(&client, || drop(client.create(Bytes::from_elems(x.clone())))));
    let persistent = client.empty(bytes);
    report("(c) write into a persistent buffer", time(&client, || client.write(&persistent, Bytes::from_elems(x.clone()))));
}
