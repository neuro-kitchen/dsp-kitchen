//! The fold a browser page's overview helpers run in JS (`engine/dspk-engine-web/overview-fold.js`)
//! must be this crate's fold. This test folds a small chunked recording with `fold_row` (as the
//! pyramid folds a page) and checks `tests/fixtures/overview-fold.json` still holds that answer;
//! `engine/dspk-engine-web/tests/overview-fold.test.mjs` runs the JS fold on the same fixture.
//! After a deliberate change to the fold: `DSPK_UPDATE_FIXTURES=1 cargo test -p dspk-view --test
//! overview_fold_fixture`, then run the JS test.

use dspk_view::envelope::fold::finish;
use dspk_view::envelope::{fold_row, Columns, EMPTY};
use serde_json::{json, Value};

const CHANNELS: usize = 3;
const SAMPLES: u64 = 1000;
const BASE: u64 = 16;
const PAGE: u64 = 128;
const CHUNK_SAMPLES: u64 = 100;
const CHUNK_CHANNELS: usize = 2;
const GAIN: f64 = 0.5;
const OFFSET: f64 = -1.0;
/// Pages 2..5: samples 256..640, chunks 2..=6 along time.
const PAGES: (u64, u64) = (2, 5);
/// The chunk not stored (all fill value): time chunk 4, channel chunk 1.
const MISSING: (u64, usize) = (4, 1);

fn stored(t: u64, c: usize) -> i16 {
    ((t * 37 + c as u64 * 101) % 211) as i16 - 105
}

fn fixture() -> Value {
    let (s0, s1) = (PAGES.0 * PAGE, (PAGES.1 * PAGE).min(SAMPLES));
    // Chunks covering the unit, stored [CHUNK_SAMPLES, CHUNK_CHANNELS] in C order
    let mut chunks = Vec::new();
    for i in s0 / CHUNK_SAMPLES..=(s1 - 1) / CHUNK_SAMPLES {
        for j in 0..CHANNELS.div_ceil(CHUNK_CHANNELS) {
            let (t0, c0) = (i * CHUNK_SAMPLES, j * CHUNK_CHANNELS);
            let values: Option<Vec<i16>> = ((i, j) != MISSING).then(|| {
                (0..CHUNK_SAMPLES).flat_map(|dt| (0..CHUNK_CHANNELS).map(move |dc| if c0 + dc < CHANNELS { stored(t0 + dt, c0 + dc) } else { 0 })).collect()
            });
            chunks.push(json!({"t0": t0, "c0": c0, "values": values}));
        }
    }
    // The pyramid's fold of each page: scaled values into buckets of BASE from the page's start
    let value = |t: u64, c: usize| -> f32 {
        let missing = t / CHUNK_SAMPLES == MISSING.0 && c / CHUNK_CHANNELS == MISSING.1;
        let s = if missing { 0.0 } else { stored(t, c) as f32 };
        s * GAIN as f32 + OFFSET as f32
    };
    let mut expected = Vec::new();
    for c in 0..CHANNELS {
        for p in PAGES.0..PAGES.1 {
            let (a, b) = (p * PAGE, ((p + 1) * PAGE).min(SAMPLES));
            let count = (b - a).div_ceil(BASE) as usize;
            let row: Vec<f32> = (a..b).map(|t| value(t, c)).collect();
            let mut acc = vec![EMPTY; count];
            fold_row(&row, a, Columns::Buckets { origin: a, size: BASE, count }, &mut acc);
            finish(&mut acc);
            expected.extend(acc.iter().flat_map(|v| [v[0], v[1]]));
        }
    }
    json!({
        "meta": {"data_type": "int16", "compression": "none", "fill": 0.0, "gain": GAIN, "offset": OFFSET,
                 "channels": CHANNELS, "samples": SAMPLES, "base": BASE, "page_samples": PAGE,
                 "chunk_samples": CHUNK_SAMPLES, "chunk_channels": CHUNK_CHANNELS},
        "unit": {"generation": 1, "pages": [PAGES.0, PAGES.1], "samples": [s0, s1], "chunks": chunks},
        "expected": expected,
    })
}

#[test]
fn the_page_fold_fixture_is_this_crates_fold() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/overview-fold.json");
    let made = fixture();
    if std::env::var_os("DSPK_UPDATE_FIXTURES").is_some() {
        std::fs::write(&path, serde_json::to_string(&made).unwrap() + "\n").unwrap();
    }
    let kept: Value = serde_json::from_str(&std::fs::read_to_string(&path).expect("the fixture (DSPK_UPDATE_FIXTURES=1 writes it)")).unwrap();
    assert_eq!(kept, made, "the fold changed: write the fixture again and run the JS test");
    assert!(made["expected"].as_array().unwrap().iter().all(|v| v.as_f64().is_some_and(f64::is_finite)), "every bucket holds samples");
}
