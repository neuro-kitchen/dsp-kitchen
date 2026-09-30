use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use dsp_core::{MemoryOrder, RecordingSource, SampleFormat};
use dsp_io::{SyntheticParams, SyntheticRecording};

#[derive(Args, Debug)]
pub struct GenerateArgs {
    /// Number of sensor channels (e.g. 1, 4, 32, 64, 384, 1024)
    #[arg(short, long, default_value_t = 384)]
    channels: usize,

    /// Number of time samples per channel (e.g. 30,000 = 1 sec @ 30kHz); ignored with --duration
    #[arg(short, long, default_value_t = 30000)]
    samples: u64,

    /// Recording length, e.g. 90s, 10m, 2h (overrides --samples)
    #[arg(short, long)]
    duration: Option<String>,

    /// Sampling rate in Hz
    #[arg(short = 'r', long, default_value_t = 30000.0)]
    sample_rate: f64,

    /// Amplitude of white Gaussian noise (microvolts RMS)
    #[arg(short, long, default_value_t = 15.0)]
    noise_uv: f32,

    /// Amplitude of 60 Hz power-line interference (microvolts)
    #[arg(short, long, default_value_t = 25.0)]
    line_noise_uv: f32,

    /// Inject biological action potentials (spikes) across channels
    #[arg(long, default_value_t = true)]
    spikes: bool,

    /// Number of simulated units (default: one per 4 channels, 1..=64)
    #[arg(long)]
    units: Option<usize>,

    /// Storage format: 'bin' (flat binary + JSON .meta) or 'zarr' (chunked Zarr v3)
    #[arg(short = 'f', long, default_value = "bin")]
    format: String,

    /// Sample type for 'bin': float32 or int16
    #[arg(long, default_value = "float32")]
    dtype: String,

    /// Memory order for 'bin': channel-major or time-major (interleaved, as acquisition systems write)
    #[arg(long, default_value = "channel-major")]
    order: String,

    /// µV per stored integer step for int16 output
    #[arg(long, default_value_t = 0.195)]
    gain_uv: f32,

    /// Output file/directory path
    #[arg(short, long, default_value = "playground/data/mock_signal_384ch.bin")]
    output: PathBuf,
}

/// Parses `90`, `90s`, `10m`, `2h`, `1.5h` into seconds.
fn parse_duration(s: &str) -> anyhow::Result<f64> {
    let s = s.trim();
    let (num, mult) = match s.chars().last() {
        Some('h') => (&s[..s.len() - 1], 3600.0),
        Some('m') => (&s[..s.len() - 1], 60.0),
        Some('s') => (&s[..s.len() - 1], 1.0),
        _ => (s, 1.0),
    };
    Ok(num.parse::<f64>()? * mult)
}

pub fn run_generate(a: &GenerateArgs) -> anyhow::Result<()> {
    let duration_sec = match &a.duration {
        Some(d) => parse_duration(d)?,
        None => a.samples as f64 / a.sample_rate,
    };
    let units = if a.spikes { a.units.unwrap_or((a.channels / 4).clamp(1, 64)) } else { 0 };
    let source = SyntheticRecording::new(SyntheticParams {
        channels: a.channels,
        sample_rate_hz: a.sample_rate,
        duration_sec,
        noise_uv: a.noise_uv,
        line_noise_uv: a.line_noise_uv,
        units,
        ..Default::default()
    })?;
    let info = source.info();

    let dtype = SampleFormat::parse(&a.dtype).ok_or_else(|| anyhow::anyhow!("unknown --dtype {}", a.dtype))?;
    let order = match a.order.as_str() {
        "channel-major" => MemoryOrder::ChannelMajor,
        "time-major" | "interleaved" => MemoryOrder::TimeMajor,
        other => anyhow::bail!("unknown --order {other} (channel-major or time-major)"),
    };
    let zarr = a.format == "zarr";
    let (gain, bytes) = if zarr {
        (1.0, info.samples * a.channels as u64 * 4)
    } else {
        let gain = if dtype == SampleFormat::F32 { 1.0 } else { a.gain_uv };
        (gain, info.samples * a.channels as u64 * dtype.bytes() as u64)
    };

    println!("=== Synthetic Multi-Channel Signal Generator (dsp-io) ===");
    println!("Channels:            {}", a.channels);
    println!("Samples per channel: {}", info.samples);
    println!("Sample Rate:         {:.1} Hz", a.sample_rate);
    println!("Duration:            {:.3} seconds", info.duration_sec());
    println!("Gaussian Noise RMS:  {:.1} uV", a.noise_uv);
    println!("60 Hz Line Noise:    {:.1} uV", a.line_noise_uv);
    println!("Units (drifting):    {units}");
    if zarr {
        println!("Storage Format:      zarr (float32, [channels, samples])");
    } else {
        println!("Storage Format:      bin ({dtype:?}, {order:?}, gain {gain} uV)");
    }
    println!("Size:                {:.2} GB", bytes as f64 / 1e9);
    println!("Destination:         {}", a.output.display());

    let started = Instant::now();
    let mut last_pct = u64::MAX;
    let progress = |done: u64, total: u64| {
        let pct = done * 100 / total.max(1);
        if pct / 5 != last_pct / 5 || done == total {
            last_pct = pct;
            let mbps = (bytes as f64 * done as f64 / total.max(1) as f64) / 1e6 / started.elapsed().as_secs_f64().max(1e-9);
            println!("  {pct:3}%  ({mbps:.0} MB/s)");
        }
    };

    // Bounded chunks (~1 s of data) so hours-long files never sit in memory
    let chunk = (a.sample_rate as usize).max(1);
    if zarr {
        let path = if a.output.extension().is_some_and(|e| e == "bin") { a.output.with_extension("zarr") } else { a.output.clone() };
        dsp_io::write_zarr(&source, &path, chunk, progress)?;
        println!("\n[Zarr store written] {} in {:.1}s", path.display(), started.elapsed().as_secs_f64());
    } else {
        dsp_io::write_raw(&source, &a.output, dtype, order, gain, chunk, progress)?;
        println!(
            "\n[Binary written] {} + {} in {:.1}s",
            a.output.display(),
            dsp_io::RawParams::sidecar_path(&a.output).display(),
            started.elapsed().as_secs_f64()
        );
    }

    // Verify by reopening through format detection
    let path = if zarr && a.output.extension().is_some_and(|e| e == "bin") { a.output.with_extension("zarr") } else { a.output.clone() };
    let reopened = dsp_io::open(&path)?;
    let mut first = [0.0f32; 1];
    reopened.read(&[0], 0..1.min(reopened.info().samples), &mut first[..1.min(reopened.info().samples as usize)])?;
    println!(
        "[Verified] {} channels x {} samples @ {:.1} Hz, ch0[0] = {:.2} uV",
        reopened.info().channel_count(),
        reopened.info().samples,
        reopened.info().sample_rate_hz(),
        first[0]
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_duration;

    #[test]
    fn test_parse_duration() {
        assert_eq!(parse_duration("90").unwrap(), 90.0);
        assert_eq!(parse_duration("10m").unwrap(), 600.0);
        assert_eq!(parse_duration("1.5h").unwrap(), 5400.0);
        assert!(parse_duration("abc").is_err());
    }
}
