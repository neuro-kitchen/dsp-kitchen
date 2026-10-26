use std::path::PathBuf;
use std::time::Instant;

use clap::Args;

/// Seconds read by the throughput test unless told otherwise.
const DEFAULT_READ_SEC: f64 = 10.0;
/// Channels listed by name before the last one.
const LISTED_CHANNELS: usize = 3;

const MS_PER_S: f64 = 1e3;
const S_PER_HOUR: f64 = 3600.0;
const BYTES_PER_GB: f64 = 1e9;
const PER_MEGA: f64 = 1e6;
/// Bytes of one value as reads return it (`f32`).
const F32_BYTES: f64 = std::mem::size_of::<f32>() as f64;

#[derive(Args, Debug)]
pub struct OpenArgs {
    /// Recording (SpikeGLX .bin / .cbin, raw .bin + JSON .meta, NWB, Zarr)
    path: PathBuf,
    /// Seconds of every channel to read for the throughput test
    #[arg(long, default_value_t = DEFAULT_READ_SEC)]
    read_sec: f64,
}

/// Prints what `dsp-io` detects for a recording and times chunked reads.
pub fn run(args: &OpenArgs) -> anyhow::Result<()> {
    let (path, read_sec) = (args.path.as_path(), args.read_sec);
    let t = Instant::now();
    let rec = dsp_io::open(path)?;
    let open_ms = t.elapsed().as_secs_f64() * MS_PER_S;
    let info = rec.info();

    println!("=== Recording: {} ===", info.name);
    println!("Opened in:     {open_ms:.1} ms");
    println!("Channels:      {}", info.channel_count());
    println!("Samples:       {} per channel", info.samples);
    println!("Sample rate:   {} Hz", info.sample_rate_hz());
    println!("Duration:      {:.1} s ({:.2} h)", info.duration_sec(), info.duration_sec() / S_PER_HOUR);
    println!("Stored as:     {:?}, {:?} ({:.2} GB uncompressed)", info.format, info.order, info.data_bytes() as f64 / BYTES_PER_GB);
    let first: Vec<String> = info.channels.iter().take(LISTED_CHANNELS).map(|c| format!("{} ({:.4} {} per step)", c.name, c.gain, c.unit.symbol())).collect();
    println!("Channels:      {} ... {}", first.join(", "), info.channels.last().map_or("", |c| c.name.as_str()));
    let source = dsp_io::sources(path)?;
    let probe = match dsp_io::default_source(&source) {
        Some(entry) => dsp_io::probe_of(path, &entry.id)?,
        None => None,
    };
    if let Some(layout) = &probe {
        let ys = layout.sites().iter().map(|s| s.position.y_um);
        let span = ys.clone().fold(f32::MIN, f32::max) - ys.fold(f32::MAX, f32::min);
        println!("Probe:         {} ({} sites, {span:.0} µm span)", layout.name, layout.total_channels());
    }
    for (k, v) in &info.metadata {
        println!("  {k}: {v}");
    }

    // Read `read_sec` of every channel from the middle of the recording, in 1 s chunks
    let sr = info.sample_rate_hz();
    let n = (sr as u64).min(info.samples);
    let chunks = ((read_sec * sr) as u64 / n.max(1)).max(1);
    let start = (info.samples / 2).saturating_sub(chunks * n / 2);
    let channels: Vec<usize> = (0..info.channel_count()).collect();
    let mut buf = vec![0.0f32; channels.len() * n as usize];
    let t = Instant::now();
    let mut s0 = start;
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for _ in 0..chunks {
        let s1 = (s0 + n).min(info.samples);
        let len = channels.len() * (s1 - s0) as usize;
        rec.read(&channels, s0..s1, &mut buf[..len])?;
        // Neural range only (the first channel is never a sync word)
        for &v in &buf[..(s1 - s0) as usize] {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        s0 = s1;
    }
    let secs = t.elapsed().as_secs_f64();
    let samples = (s0 - start) as f64 * channels.len() as f64;
    println!(
        "Read:          {:.1} s of all channels in {:.2} s ({:.0} Msamples/s, {:.1}x real time)",
        (s0 - start) as f64 / sr,
        secs,
        samples / secs / PER_MEGA,
        (s0 - start) as f64 / sr / secs
    );
    println!("{} range:     {lo:.1} .. {hi:.1} {}", info.channels[0].name, info.channels[0].unit.symbol());

    // The same window as stored values (what pipelines upload for integer recordings)
    let bytes = info.format.bytes();
    let mut stored = vec![0u8; channels.len() * n as usize * bytes];
    let t = Instant::now();
    let mut s0 = start;
    let mut supported = true;
    for _ in 0..chunks {
        let s1 = (s0 + n).min(info.samples);
        let len = channels.len() * (s1 - s0) as usize * bytes;
        if rec.read_stored(&channels, s0..s1, &mut stored[..len]).is_err() {
            supported = false;
            break;
        }
        s0 = s1;
    }
    if supported {
        let secs = t.elapsed().as_secs_f64();
        println!(
            "Read stored:   {:.0} Msamples/s as {} ({:.2} GB moved instead of {:.2} GB as f32)",
            samples / secs / PER_MEGA,
            info.format.name(),
            samples * bytes as f64 / BYTES_PER_GB,
            samples * F32_BYTES / BYTES_PER_GB
        );
    } else {
        println!("Read stored:   not supported by this reader");
    }
    Ok(())
}
