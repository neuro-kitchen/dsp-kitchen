//! `device`: run a dspk device (as the `dspk-device` program does), or work with one as a client.
//!
//! ```bash
//! dspk device run --generator --channels 32 --rate 30000 --pair      # a device, here
//! dspk pair 127.0.0.1:50060 42-1234-5678 --out me.pem                 # once, with its code
//! dspk device status    --identity me.pem 127.0.0.1:50060
//! dspk device configure --identity me.pem 127.0.0.1:50060 chain.json  # takes control first
//! dspk device start     --identity me.pem 127.0.0.1:50060 --label test
//! dspk device watch     --identity me.pem 127.0.0.1:50060 gen --seconds 5
//! dspk device stop      --identity me.pem 127.0.0.1:50060
//! ```

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Context;
use clap::{Args, Subcommand};
use dspk_device::{serve, DeviceConfig, LiveDevice, DEFAULT_PORT};
use dspk_io::live::GeneratorParams;
use dspk_stream::live::quic::DeviceClient;
use dspk_stream::live::wire;
use dspk_stream::transport::{live_client_config, ClientIdentity, ServerTrust};

#[derive(Args, Debug)]
pub struct DeviceArgs {
    #[command(subcommand)]
    command: DeviceCommand,
}

#[derive(Subcommand, Debug)]
enum DeviceCommand {
    /// Run a device here, from its device.toml or with a generator
    Run(RunArgs),
    /// The device's state, its run, its sources' health
    Status(ClientArgs),
    /// Send a chain (a pipeline spec, JSON or TOML); takes control first
    Configure {
        #[command(flatten)]
        client: ClientArgs,
        spec: PathBuf,
        /// Only check it: what the device would refuse, and why
        #[arg(long)]
        dry_run: bool,
    },
    /// Start a run: a recording, or a preview that stores nothing
    Start {
        #[command(flatten)]
        client: ClientArgs,
        #[arg(long)]
        preview: bool,
        #[arg(long, default_value = "")]
        label: String,
        #[arg(long, default_value = "")]
        notes: String,
    },
    /// Stop the run
    Stop(ClientArgs),
    /// The device's taps
    Taps(ClientArgs),
    /// Receive a tap for a while: frames, values per second, gaps, latency
    Watch {
        #[command(flatten)]
        client: ClientArgs,
        tap: String,
        #[arg(long, default_value_t = 5.0)]
        seconds: f64,
    },
    /// The device's last log entries
    Log(ClientArgs),
}

#[derive(Args, Debug)]
pub struct RunArgs {
    #[arg(long, conflicts_with = "generator")]
    config: Option<PathBuf>,
    #[arg(long)]
    generator: bool,
    #[arg(long, default_value_t = 32)]
    channels: usize,
    #[arg(long, default_value_t = 30_000.0)]
    rate: f64,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    #[arg(long, default_value = "dspk-device-data")]
    data: PathBuf,
    #[arg(long)]
    listen: Option<SocketAddr>,
    /// Print a pairing code for a director
    #[arg(long)]
    pair: bool,
}

#[derive(Args, Debug)]
pub struct ClientArgs {
    /// This computer's identity, from `dspk pair` with the device
    #[arg(long)]
    identity: PathBuf,
    /// The device: host:port (UDP)
    device: String,
    /// The name in the device's certificate
    #[arg(long, default_value = "localhost")]
    name: String,
}

async fn connect(args: &ClientArgs) -> anyhow::Result<DeviceClient> {
    let identity = ClientIdentity::from_file(&args.identity).with_context(|| format!("reading {}", args.identity.display()))?;
    let config = live_client_config(&ServerTrust::from_identity(&identity), Some(&identity))?;
    let addr = tokio::net::lookup_host(&args.device).await.with_context(|| format!("resolving {}", args.device))?.next().with_context(|| format!("{}: no address", args.device))?;
    Ok(DeviceClient::connect(addr, &args.name, config).await?)
}

fn print_status(s: &wire::DeviceStatus) {
    let state = wire::DeviceState::try_from(s.state).map(|s| s.as_str_name()).unwrap_or("?");
    println!("state     {state}{}", if s.message.is_empty() { String::new() } else { format!(" ({})", s.message) });
    println!("director  {}", if s.director.is_empty() { "none" } else { &s.director });
    if let Some(run) = &s.run {
        let what = if run.recording { format!("recording {}", run.path) } else { "preview".into() };
        println!("run       {what}: {} samples{}, {} lost", run.head_sample, if run.running { " so far" } else { "" }, run.lost_samples);
    }
    for src in &s.sources {
        println!("source    {}: {} at {:.1} Hz, {} checksum errors", src.name, if src.connected { "connected" } else { "SILENT" }, src.measured_rate_hz, src.checksum_errors);
    }
    if let Some(h) = &s.health {
        println!("health    block latency {} µs median, {} µs max; ring {:.0} %; {} frames dropped", h.block_latency_median_us, h.block_latency_max_us, h.ring_use * 100.0, h.frames_dropped);
    }
}

pub async fn run(args: &DeviceArgs) -> anyhow::Result<()> {
    match &args.command {
        DeviceCommand::Run(a) => {
            let config = match (&a.config, a.generator) {
                (Some(path), _) => DeviceConfig::from_file(path)?,
                (None, true) => {
                    let params = GeneratorParams { channels: a.channels, sample_rate_hz: a.rate, seed: a.seed, ..Default::default() };
                    DeviceConfig::generator(a.data.clone(), a.listen.unwrap_or(SocketAddr::from(([127, 0, 0, 1], DEFAULT_PORT))), params)
                }
                (None, false) => anyhow::bail!("give --config device.toml, or --generator"),
            };
            let device = tokio::task::spawn_blocking(move || LiveDevice::open(config)).await??;
            serve(device, a.pair, |l| {
                println!("Device listening on {} (QUIC)", l.addr);
                println!("Authority: {} (SHA-256)", l.authority_fingerprint);
                if let Some(code) = &l.pairing_code {
                    println!("Pairing code (director): {code}   →   dspk pair {} {code}", l.addr);
                }
            })
            .await
        }
        DeviceCommand::Status(c) => {
            print_status(&connect(c).await?.status().await?);
            Ok(())
        }
        DeviceCommand::Configure { client, spec, dry_run } => {
            let text = std::fs::read_to_string(spec).with_context(|| format!("reading {}", spec.display()))?;
            let device = connect(client).await?;
            if !dry_run {
                device.request_control().await?;
            }
            let result = device.configure(&text, *dry_run).await?;
            if result.refusals.is_empty() {
                let chain = result.chain.unwrap_or_default();
                println!("{}: {} taps ({}), blocks of {} samples, delay {} samples", if *dry_run { "Accepted" } else { "Configured" }, chain.taps.len(), chain.taps.iter().map(|t| t.name.as_str()).collect::<Vec<_>>().join(", "), chain.block_samples, chain.delay_samples);
            } else {
                println!("Refused:");
                for r in &result.refusals {
                    println!("  {} ({}): {}", if r.node.is_empty() { "spec" } else { &r.node }, r.op, r.reason);
                }
                anyhow::bail!("{} refusal(s)", result.refusals.len());
            }
            Ok(())
        }
        DeviceCommand::Start { client, preview, label, notes } => {
            let device = connect(client).await?;
            device.request_control().await?;
            print_status(&device.start(!preview, label, notes).await?);
            Ok(())
        }
        DeviceCommand::Stop(c) => {
            let device = connect(c).await?;
            device.request_control().await?;
            print_status(&device.stop().await?);
            Ok(())
        }
        DeviceCommand::Taps(c) => {
            for t in connect(c).await?.list_taps().await? {
                let kind = wire::TapKind::try_from(t.kind).map(|k| k.as_str_name()).unwrap_or("?");
                println!("{:<16} {kind:<18} {} channels  node {}", t.name, t.channels.len(), if t.node.is_empty() { "(raw)" } else { &t.node });
            }
            Ok(())
        }
        DeviceCommand::Watch { client, tap, seconds } => {
            let device = connect(client).await?;
            let mut rx = device.subscribe(tap, &[]).await?;
            let (mut frames, mut values, mut gaps, mut last_seq, mut latencies) = (0u64, 0u64, 0u64, None::<u64>, Vec::new());
            let start = Instant::now();
            let channels = rx.info().channels.len().max(1) as u64;
            while start.elapsed() < Duration::from_secs_f64(*seconds) {
                let Ok(next) = tokio::time::timeout(Duration::from_secs(2), rx.next()).await else {
                    println!("no frame for 2 s (is a run going?)");
                    continue;
                };
                let Some(f) = next? else { break };
                if let Some(prev) = last_seq
                    && f.sequence != prev + 1
                {
                    gaps += f.sequence - prev - 1;
                }
                last_seq = Some(f.sequence);
                frames += 1;
                values += f.count as u64 * channels;
                latencies.push(dspk_stream::live::unix_now_ns().saturating_sub(f.sent_unix_ns) / 1000);
            }
            latencies.sort_unstable();
            let p = |q: f64| latencies.get(((latencies.len() as f64 - 1.0) * q).round().max(0.0) as usize).copied().unwrap_or(0);
            let secs = start.elapsed().as_secs_f64();
            println!("{frames} frames, {:.0} values/s, {gaps} frames dropped for this client; delivery latency median {} µs, p99 {} µs", values as f64 / secs, p(0.5), p(0.99));
            Ok(())
        }
        DeviceCommand::Log(c) => {
            for e in connect(c).await?.log(50, wire::LogLevel::Info).await? {
                let level = wire::LogLevel::try_from(e.level).map(|l| l.as_str_name()).unwrap_or("?");
                println!("{level:<16} {:<12} {}", e.source, e.message);
            }
            Ok(())
        }
    }
}
