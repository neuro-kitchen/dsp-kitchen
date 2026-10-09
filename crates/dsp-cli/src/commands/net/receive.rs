//! `receive`: a session with a `serve`r: the recording's description, then its signal, with
//! throughput, real-time factor, inter-arrival times and latency; optionally saved as raw binary
//! with a dsp-io sidecar (written as frames arrive, so memory stays bounded).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::Context;
use clap::Args;
use dsp_core::{MemoryOrder, RecordingInfo, SampleFormat};
use dsp_io::RawParams;
use dsp_stream::protocol::wire::SignalFrame;
use dsp_stream::{client_config, ServerTrust, Session};

const DEFAULT_ADDR: &str = "127.0.0.1:50051";
const DEFAULT_SERVER_NAME: &str = "localhost";
/// Seconds between live reports.
const REPORT_EVERY: Duration = Duration::from_secs(1);
/// Inter-arrival and latency percentiles reported.
const PERCENTILES: [f64; 3] = [50.0, 95.0, 99.0];
const PERCENT: f64 = 100.0;
const MS_PER_S: f64 = 1e3;
const NS_PER_MS: f64 = 1e6;
const BYTES_PER_MB: f64 = 1e6;
const PER_MEGA: f64 = 1e6;

#[derive(Args, Debug)]
pub struct ReceiveArgs {
    /// Server address
    #[arg(default_value = DEFAULT_ADDR)]
    addr: SocketAddr,
    /// Name the server's certificate must carry
    #[arg(long, default_value = DEFAULT_SERVER_NAME)]
    server_name: String,
    /// Certificate(s) to trust: PEM, or the DER written by `serve --certificate-out`
    #[arg(long, required_unless_present = "insecure")]
    certificate: Option<PathBuf>,
    /// Accept any server certificate (encrypted, not authenticated): trusted networks only
    #[arg(long, conflicts_with = "certificate")]
    insecure: bool,
    /// First sample to receive
    #[arg(long, default_value_t = 0)]
    from_sample: u64,
    /// Stop after this many seconds (default: at the end of the recording or Ctrl+C)
    #[arg(long)]
    duration: Option<f64>,
    /// Stop after this many frames
    #[arg(long)]
    max_frames: Option<u64>,
    /// Save the received signal (raw binary + JSON sidecar, readable by `dsp-cli open`)
    #[arg(long)]
    save: Option<PathBuf>,
}

/// Nearest-rank percentile of sorted `values`.
fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = ((pct / PERCENT) * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

fn describe(label: &str, mut values: Vec<f64>) {
    values.sort_by(f64::total_cmp);
    let mean = values.iter().sum::<f64>() / values.len().max(1) as f64;
    let ps: Vec<String> = PERCENTILES.iter().map(|&p| format!("p{p:.0} {:.3}", percentile(&values, p))).collect();
    println!("  {label:<22} mean {mean:.3} ms, {}", ps.join(", "));
}

/// Writes frames time-major as they arrive: stored words when every channel shares one gain
/// and offset in µV (what the raw sidecar can describe), else scaled `f32`.
struct Saver {
    out: BufWriter<File>,
    path: PathBuf,
    params: RawParams,
    stored: bool,
    samples: u64,
}

impl Saver {
    fn new(path: &Path, info: &RecordingInfo) -> anyhow::Result<Self> {
        let first = &info.channels[0];
        let uniform = info.channels.iter().all(|c| c.gain == first.gain && c.offset == first.offset && c.unit == first.unit);
        let stored = uniform;
        let format = if stored { info.format } else { SampleFormat::F32 };
        let mut params = RawParams::new(info.channel_count(), info.sample_rate_hz(), format, MemoryOrder::TimeMajor);
        params.unit = first.unit.clone();
        if stored {
            (params.gain, params.offset) = (first.gain, first.offset);
        } else if info.channels.iter().any(|c| c.unit != first.unit) {
            println!("Note: the raw sidecar records one unit; this stream's channels differ, saved as {}", first.unit.symbol());
        }
        Ok(Self { out: BufWriter::new(File::create(path).with_context(|| format!("creating {}", path.display()))?), path: path.to_path_buf(), params, stored, samples: 0 })
    }

    fn write(&mut self, frame: &SignalFrame, info: &RecordingInfo) -> anyhow::Result<()> {
        let (channels, n) = (info.channel_count(), frame.samples as usize);
        if self.stored {
            let b = info.format.bytes();
            for t in 0..n {
                for c in 0..channels {
                    let at = (c * n + t) * b;
                    self.out.write_all(&frame.data[at..at + b])?;
                }
            }
        } else {
            let values = dsp_stream::protocol::decode_signal(frame, info)?;
            for t in 0..n {
                for c in 0..channels {
                    self.out.write_all(&values[c * n + t].to_le_bytes())?;
                }
            }
        }
        self.samples += n as u64;
        Ok(())
    }

    fn finish(mut self) -> anyhow::Result<()> {
        self.out.flush()?;
        self.params.samples = Some(self.samples);
        self.params.write_sidecar(&self.path)?;
        println!("Saved {} samples per channel to {} (+ {})", self.samples, self.path.display(), RawParams::sidecar_path(&self.path).display());
        Ok(())
    }
}

pub async fn run(args: &ReceiveArgs) -> anyhow::Result<()> {
    let trust = match &args.certificate {
        Some(path) => ServerTrust::from_file(path)?,
        None => {
            println!("Server certificate not verified (--insecure)");
            ServerTrust::DangerAcceptAnyCertificate
        }
    };
    let connect_start = Instant::now();
    let mut session = Session::connect(args.addr, &args.server_name, client_config(&trust)?).await?;
    let info = session.info().clone();
    println!("Connected to {} in {:.1} ms", args.addr, connect_start.elapsed().as_secs_f64() * MS_PER_S);
    println!("{}: {} channels at {} Hz, {} samples, stored {:?}", info.name, info.channel_count(), info.sample_rate_hz(), info.samples, info.format);

    let mut saver = args.save.as_deref().map(|p| Saver::new(p, &info)).transpose()?;
    let mut signal = session.subscribe(args.from_sample).await?;
    let started = Instant::now();
    let (mut frames, mut bytes, mut samples) = (0u64, 0usize, 0u64);
    let (mut gaps_ms, mut latency_ms) = (Vec::new(), Vec::new());
    let (mut last_arrival, mut last_report) = (None::<Instant>, Instant::now());
    let deadline = args.duration.map(Duration::from_secs_f64);

    let received = async {
        while let Some(frame) = signal.next_frame().await? {
            let arrival = Instant::now();
            let now_ns = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
            latency_ms.push(now_ns.saturating_sub(frame.sent_unix_ns) as f64 / NS_PER_MS);
            if let Some(previous) = last_arrival {
                gaps_ms.push((arrival - previous).as_secs_f64() * MS_PER_S);
            }
            last_arrival = Some(arrival);
            frames += 1;
            bytes += frame.data.len();
            samples += frame.samples as u64;
            if let Some(saver) = saver.as_mut() {
                saver.write(&frame, &info)?;
            }
            if last_report.elapsed() >= REPORT_EVERY {
                let secs = started.elapsed().as_secs_f64();
                println!("  {secs:6.1} s  {frames} frames, {:.1} MB/s, {:.2}× real time", bytes as f64 / BYTES_PER_MB / secs, samples as f64 / info.sample_rate_hz() / secs);
                last_report = Instant::now();
            }
            if deadline.is_some_and(|d| started.elapsed() >= d) || args.max_frames.is_some_and(|m| frames >= m) {
                break;
            }
        }
        anyhow::Ok(())
    };
    tokio::select! {
        result = received => result?,
        _ = tokio::signal::ctrl_c() => println!("Interrupted"),
    }
    session.close().await;

    let secs = started.elapsed().as_secs_f64();
    println!("Received {frames} frames, {samples} samples per channel in {secs:.2} s");
    if frames > 0 {
        println!("  payload                {:.2} MB, {:.1} MB/s, {:.1} Msamples/s", bytes as f64 / BYTES_PER_MB, bytes as f64 / BYTES_PER_MB / secs, (samples * info.channel_count() as u64) as f64 / PER_MEGA / secs);
        println!("  real-time factor       {:.2}×", samples as f64 / info.sample_rate_hz() / secs);
        describe("inter-arrival", gaps_ms);
        describe("latency (sent→read)", latency_ms);
        println!("  (latency across hosts includes their clock offset)");
    }
    if let Some(saver) = saver {
        saver.finish()?;
    }
    Ok(())
}
