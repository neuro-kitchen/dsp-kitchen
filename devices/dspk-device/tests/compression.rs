//! What compression costs and saves on a device's raw recording: bytes on disk against the raw
//! bytes, and the time to store one chunk, for `none` and zstd levels, on generator data shaped
//! like the two first tests (Neuropixels-like int16, a PPG-like int32 pair).
//!
//! `cargo test -p dspk-device --profile validate --test compression -- --ignored --nocapture`

use std::path::Path;
use std::time::Instant;

use dspk_core::SampleFormat;
use dspk_io::live::{Generator, GeneratorParams, LiveSource, LiveZarrWriter};
use dspk_io::{ZarrCompression, ZarrOptions};

fn disk_bytes(dir: &Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let p = e.path();
            total += if p.is_dir() { disk_bytes(&p) } else { e.metadata().map(|m| m.len()).unwrap_or(0) };
        }
    }
    total
}

fn measure(label: &str, params: GeneratorParams, chunks: u64) {
    let generator = Generator::new("gen", &params).unwrap();
    let layout = generator.layout().clone();
    let chunk = params.sample_rate_hz as u64; // 1 s
    let blocks: Vec<_> = (0..chunks).map(|k| generator.block_at(k * chunk, chunk as usize).unwrap()).collect();
    let raw = blocks.iter().map(|b| b.data.len() as u64).sum::<u64>();
    println!("\n{label}: {} channels × {} Hz {}, {chunks} chunks of 1 s ({:.1} MB raw)", params.channels, params.sample_rate_hz, layout.format.name(), raw as f64 / 1e6);
    println!("{:<10} {:>10} {:>8} {:>14}", "setting", "on disk", "ratio", "ms per chunk");
    for (name, compression) in [("none", ZarrCompression::None), ("zstd-1", ZarrCompression::Zstd { level: 1 }), ("zstd-3", ZarrCompression::Zstd { level: 3 }), ("zstd-9", ZarrCompression::Zstd { level: 9 })] {
        let dir = std::env::temp_dir().join(format!("dspk-compress-{}-{label}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut writer = LiveZarrWriter::create(&dir, &layout, &ZarrOptions { chunk_samples: chunk as usize, compression, parallel: false }).unwrap();
        let t0 = Instant::now();
        for b in &blocks {
            writer.push(b).unwrap();
        }
        writer.finish().unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / chunks as f64;
        let on_disk = disk_bytes(&dir);
        println!("{name:<10} {:>8.2} MB {:>7.2}× {ms:>14.2}", on_disk as f64 / 1e6, raw as f64 / on_disk as f64);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
#[ignore = "a measurement: run with --ignored --nocapture"]
fn compression_ratio_and_time() {
    measure("neuropixels-like", GeneratorParams { channels: 32, sample_rate_hz: 30_000.0, format: SampleFormat::I16, ..Default::default() }, 10);
    measure(
        "ppg-like",
        GeneratorParams { channels: 2, sample_rate_hz: 400.0, format: SampleFormat::I32, gain_uv: 1.0, noise_uv: 2.0, units: 0, sine_hz: 1.2, sine_uv: 5_000.0, ..Default::default() },
        60,
    );
}
