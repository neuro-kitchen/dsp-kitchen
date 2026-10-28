//! Host → device upload strategies for one detection window (384 channels × 60 000 `f32`,
//! 92 MB), each ending in a device sync. Ignored by default:
//!
//! ```text
//! cargo test -p dspk-ops --profile validate --test bench_upload -- --ignored --nocapture
//! ```

use std::time::Instant;

use cubecl::bytes::Bytes;
use cubecl::prelude::*;
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
        uploads(&target.client().expect("compiled-in runtime"));
    }
}

fn uploads(client: &Client) {
    let client = client.clone();
    let x: Vec<f32> = (0..CHANNELS * WINDOW).map(|i| (i % 1013) as f32).collect();
    let bytes = x.len() * size_of::<f32>();
    let report = |name: &str, ms: f64| println!("{name:<38} | {ms:>8.2} ms | {:>6.2} GB/s", bytes as f64 / ms / 1e6);

    report("(a) create_from_slice (new buffer)", time(&client, || drop(client.create_from_slice(f32::as_bytes(&x)))));
    report("(b) create(Bytes) (new buffer, owned)", time(&client, || drop(client.create(Bytes::from_elems(x.clone())))));
    let persistent = client.empty(bytes);
    report("(c) write into a persistent buffer", time(&client, || client.write(&persistent, Bytes::from_elems(x.clone()))));
}
