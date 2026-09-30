use std::path::Path;
use std::time::Instant;

/// Prints what `dsp-io` detects for a recording and times chunked reads.
pub fn run_info(path: &Path, read_sec: f64) -> anyhow::Result<()> {
    let t = Instant::now();
    let rec = dsp_io::open(path)?;
    let open_ms = t.elapsed().as_secs_f64() * 1000.0;
    let info = rec.info();

    println!("=== Recording: {} ===", info.name);
    println!("Opened in:     {open_ms:.1} ms");
    println!("Channels:      {}", info.channel_count());
    println!("Samples:       {} per channel", info.samples);
    println!("Sample rate:   {} Hz", info.sample_rate_hz());
    println!("Duration:      {:.1} s ({:.2} h)", info.duration_sec(), info.duration_sec() / 3600.0);
    println!("Stored as:     {:?}, {:?} ({:.2} GB uncompressed)", info.format, info.order, info.data_bytes() as f64 / 1e9);
    let first: Vec<String> = info.channels.iter().take(3).map(|c| format!("{} ({:.4} uV/bit)", c.name, c.gain_uv)).collect();
    println!("Channels:      {} ... {}", first.join(", "), info.channels.last().map_or("", |c| c.name.as_str()));
    if let Some(layout) = &info.layout {
        let ys = layout.sites().iter().map(|s| s.position.y_um);
        let span = ys.clone().fold(f32::MIN, f32::max) - ys.fold(f32::MAX, f32::min);
        println!("Probe:         {} ({} sites, {span:.0} um span)", layout.name, layout.total_channels());
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
        samples / secs / 1e6,
        (s0 - start) as f64 / sr / secs
    );
    println!("{} range:     {lo:.1} .. {hi:.1} uV", info.channels[0].name);
    Ok(())
}
