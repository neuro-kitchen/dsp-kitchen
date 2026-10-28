//! `generate`: a synthetic recording (dsp-io `SyntheticRecording`) written as raw binary (with its
//! JSON sidecar) or a Zarr store, in bounded chunks, then reopened through format detection.

use std::path::PathBuf;
use std::time::Instant;

use clap::{Args, ValueEnum};
use dsp_core::{MemoryOrder, RecordingSource, SampleFormat};
use dsp_io::{SyntheticParams, SyntheticRecording};

/// Defaults: a Neuropixels-sized probe for one second at 30 kHz.
const DEFAULT_CHANNELS: usize = 384;
const DEFAULT_SAMPLE_RATE_HZ: f64 = 30_000.0;
const DEFAULT_DURATION_SEC: f64 = 1.0;
/// Synthetic noise and line interference (µV), and the int16 step (µV) of a typical
/// extracellular headstage.
const DEFAULT_NOISE_UV: f32 = 15.0;
const DEFAULT_LINE_NOISE_UV: f32 = 25.0;
const DEFAULT_INT16_STEP_UV: f32 = 0.195;
/// One simulated unit per this many channels, at most [`MAX_UNITS`].
const CHANNELS_PER_UNIT: usize = 4;
const MAX_UNITS: usize = 64;
/// Seconds written per chunk (bounds memory whatever the length).
const CHUNK_SEC: f64 = 1.0;
/// Progress is printed every this many percent.
const PROGRESS_STEP_PERCENT: u64 = 5;
const PERCENT: u64 = 100;
const BYTES_PER_GB: f64 = 1e9;
const BYTES_PER_MB: f64 = 1e6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Container {
    /// Flat binary + JSON `.meta` sidecar
    Bin,
    /// Chunked Zarr v3 store (float32)
    Zarr,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Dtype {
    Float32,
    Int16,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Order {
    ChannelMajor,
    /// Interleaved frames, as acquisition systems write
    TimeMajor,
}

#[derive(Args, Debug)]
pub struct GenerateArgs {
    /// Output file (bin) or folder (zarr)
    output: PathBuf,
    #[arg(short, long, default_value_t = DEFAULT_CHANNELS)]
    channels: usize,
    /// Length: seconds, or with a unit (90s, 10m, 2h)
    #[arg(short, long, default_value_t = DEFAULT_DURATION_SEC.to_string())]
    duration: String,
    /// Sample rate (Hz)
    #[arg(short = 'r', long, default_value_t = DEFAULT_SAMPLE_RATE_HZ)]
    sample_rate: f64,
    /// White Gaussian noise, RMS (µV)
    #[arg(long, default_value_t = DEFAULT_NOISE_UV)]
    noise: f32,
    /// Power-line interference amplitude (µV)
    #[arg(long, default_value_t = DEFAULT_LINE_NOISE_UV)]
    line_noise: f32,
    /// Simulated units (default: one per 4 channels, at most 64); 0 for noise only
    #[arg(long)]
    units: Option<usize>,
    #[arg(short = 'f', long, value_enum, default_value_t = Container::Bin)]
    format: Container,
    /// Stored sample type (bin)
    #[arg(long, value_enum, default_value_t = Dtype::Float32)]
    dtype: Dtype,
    /// Memory order (bin)
    #[arg(long, value_enum, default_value_t = Order::ChannelMajor)]
    order: Order,
    /// µV per stored step (bin, int16)
    #[arg(long, default_value_t = DEFAULT_INT16_STEP_UV)]
    step: f32,
}

/// Seconds per unit suffix of `--duration`.
const DURATION_UNITS: [(char, f64); 3] = [('h', 3600.0), ('m', 60.0), ('s', 1.0)];

/// Parses `90`, `90s`, `10m`, `2h`, `1.5h` into seconds.
fn parse_duration(s: &str) -> anyhow::Result<f64> {
    let s = s.trim();
    let (number, seconds) = match DURATION_UNITS.iter().find(|(suffix, _)| s.ends_with(*suffix)) {
        Some(&(_, per)) => (&s[..s.len() - 1], per),
        None => (s, 1.0),
    };
    Ok(number.parse::<f64>()? * seconds)
}

pub fn run(a: &GenerateArgs) -> anyhow::Result<()> {
    let duration_sec = parse_duration(&a.duration)?;
    let units = a.units.unwrap_or((a.channels / CHANNELS_PER_UNIT).clamp(1, MAX_UNITS));
    let source = SyntheticRecording::new(SyntheticParams {
        channels: a.channels,
        sample_rate_hz: a.sample_rate,
        duration_sec,
        noise_uv: a.noise,
        line_noise_uv: a.line_noise,
        units,
        ..Default::default()
    })?;
    let info = source.info();
    let (format, order) = match a.format {
        Container::Zarr => (SampleFormat::F32, MemoryOrder::ChannelMajor),
        Container::Bin => (
            match a.dtype {
                Dtype::Float32 => SampleFormat::F32,
                Dtype::Int16 => SampleFormat::I16,
            },
            match a.order {
                Order::ChannelMajor => MemoryOrder::ChannelMajor,
                Order::TimeMajor => MemoryOrder::TimeMajor,
            },
        ),
    };
    let step = if format == SampleFormat::F32 { 1.0 } else { a.step };
    let bytes = info.samples * a.channels as u64 * format.bytes() as u64;
    let path = match a.format {
        Container::Zarr if a.output.extension().is_some_and(|e| e == "bin") => a.output.with_extension("zarr"),
        _ => a.output.clone(),
    };

    println!("Synthetic recording: {} channels × {} samples at {} Hz ({:.3} s)", a.channels, info.samples, a.sample_rate, info.duration_sec());
    println!("Noise {} µV RMS, line noise {} µV, {units} drifting units", a.noise, a.line_noise);
    println!("Writing {} ({format:?}, {order:?}, {:.2} GB) to {}", if a.format == Container::Zarr { "Zarr" } else { "raw binary" }, bytes as f64 / BYTES_PER_GB, path.display());

    let started = Instant::now();
    let mut last_step = u64::MAX;
    let progress = |done: u64, total: u64| {
        let pct = done * PERCENT / total.max(1);
        if pct / PROGRESS_STEP_PERCENT != last_step || done == total {
            last_step = pct / PROGRESS_STEP_PERCENT;
            let mb_s = bytes as f64 * done as f64 / total.max(1) as f64 / BYTES_PER_MB / started.elapsed().as_secs_f64().max(f64::EPSILON);
            println!("  {pct:3}%  ({mb_s:.0} MB/s)");
        }
    };
    let chunk = ((a.sample_rate * CHUNK_SEC) as usize).max(1);
    match a.format {
        Container::Zarr => dsp_io::write_zarr(&source, &path, chunk, progress)?,
        Container::Bin => {
            dsp_io::write_raw(&source, &path, format, order, step, chunk, progress)?;
        }
    }
    println!("Written in {:.1} s", started.elapsed().as_secs_f64());

    // Reopen through format detection
    let reopened = dsp_io::open(&path)?;
    let i = reopened.info();
    println!("Reopened: {} channels × {} samples at {} Hz ({:?})", i.channel_count(), i.samples, i.sample_rate_hz(), i.format);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_duration;

    #[test]
    fn durations_parse_with_or_without_units() {
        assert_eq!(parse_duration("90").unwrap(), 90.0);
        assert_eq!(parse_duration("10m").unwrap(), 600.0);
        assert_eq!(parse_duration("1.5h").unwrap(), 5400.0);
        assert!(parse_duration("abc").is_err());
    }
}
