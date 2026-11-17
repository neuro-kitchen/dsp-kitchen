//! `device`: run a dspk device (as the `dspk-device` program does), or work with one as a client.
//!
//! ```bash
//! dspk device run --generator --channels 32 --rate 30000 --pair      # a device, here
//! dspk pair 127.0.0.1:50060 42-1234-5678 --out me.pem                 # once, with its code
//! dspk device status    --identity me.pem 127.0.0.1:50060
//! dspk device configure --identity me.pem 127.0.0.1:50060 chain.json  # takes control first
//! dspk device start     --identity me.pem 127.0.0.1:50060 --label test
//! dspk device watch     --identity me.pem 127.0.0.1:50060 gen --seconds 5
//! dspk device set       --identity me.pem 127.0.0.1:50060 spikes level -80
//! dspk device pair-code --identity me.pem 127.0.0.1:50060             # a viewer pairs with it
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
    /// Change one parameter of the running chain (its filters start again), e.g. `spikes level -80`
    Set {
        #[command(flatten)]
        client: ClientArgs,
        node: String,
        parameter: String,
        /// The new value as JSON: a number, a list `[300, 3000]`, a string `"forward"`
        value: String,
    },
    /// Put a marker in the run ("sensor moved")
    Mark {
        #[command(flatten)]
        client: ClientArgs,
        label: String,
    },
    /// A one-time pairing code for another computer (director only): a viewer, or with
    /// --control one that may direct
    PairCode {
        #[command(flatten)]
        client: ClientArgs,
        #[arg(long)]
        control: bool,
    },
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

/// `stopped`, `running`, … (the protocol's names without their prefix).
fn state_name(state: i32) -> String {
    wire::DeviceState::try_from(state).map(|s| s.as_str_name().trim_start_matches("DEVICE_STATE_").to_ascii_lowercase()).unwrap_or_else(|_| "?".into())
}

fn print_status(s: &wire::DeviceStatus) {
    let state = state_name(s.state);
    println!("state     {state}{}", if s.message.is_empty() { String::new() } else { format!(" ({})", s.message) });
    println!("director  {}", if s.director.is_empty() { "none" } else { &s.director });
    let waiting = s.run.as_ref().is_some_and(|r| r.running && r.head_sample == 0);
    if let Some(run) = &s.run {
        let what = if run.recording { format!("recording {}", run.path) } else { "preview".into() };
        if waiting {
            println!("run       {what}: starting, waiting for the first data");
        } else {
            println!("run       {what}: {} samples{}, {} lost", run.head_sample, if run.running { " so far" } else { "" }, run.lost_samples);
        }
    }
    for src in &s.sources {
        let health = if waiting { "starting".into() } else if src.connected { format!("connected at {:.1} Hz", src.measured_rate_hz) } else { "SILENT (no data for over a second)".into() };
        println!("source    {}: {health}, {} checksum errors, {} lost", src.name, src.checksum_errors, src.lost_samples);
    }
    if let Some(h) = &s.health
        && !waiting
    {
        println!("health    block latency {} µs median, {} µs max; ring {:.0} %; {} frames dropped", h.block_latency_median_us, h.block_latency_max_us, h.ring_use * 100.0, h.frames_dropped);
    }
}

/// Runs `f` with a connection, then closes it so the device sees this client leave at once.
async fn with_device<T>(args: &ClientArgs, f: impl AsyncFnOnce(&DeviceClient) -> anyhow::Result<T>) -> anyhow::Result<T> {
    let device = connect(args).await?;
    let result = f(&device).await;
    device.finish().await;
    result
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
                if let Some(web) = l.web_addr {
                    println!("Pages: http://{web}/live");
                }
                if let Some(code) = &l.pairing_code {
                    println!("Pairing code (director): {code}   →   dspk pair {} {code}", l.addr);
                }
            })
            .await
        }
        DeviceCommand::Status(c) => with_device(c, async |d| Ok(print_status(&d.status().await?))).await,
        DeviceCommand::Configure { client, spec, dry_run } => {
            let text = std::fs::read_to_string(spec).with_context(|| format!("reading {}", spec.display()))?;
            with_device(client, async |d| {
                if !dry_run {
                    d.request_control().await?;
                }
                let result = d.configure(&text, *dry_run).await?;
                if result.refusals.is_empty() {
                    let chain = result.chain.unwrap_or_default();
                    let taps = chain.taps.iter().map(|t| t.name.as_str()).collect::<Vec<_>>().join(", ");
                    println!("{}: {} taps ({taps}), blocks of {} samples, delay {} samples", if *dry_run { "Accepted" } else { "Configured" }, chain.taps.len(), chain.block_samples, chain.delay_samples);
                    Ok(())
                } else {
                    println!("Refused:");
                    for r in &result.refusals {
                        println!("  {} ({}): {}", if r.node.is_empty() { "spec" } else { &r.node }, r.op, r.reason);
                    }
                    anyhow::bail!("{} refusal(s)", result.refusals.len())
                }
            })
            .await
        }
        DeviceCommand::Start { client, preview, label, notes } => with_device(client, async |d| {
            d.request_control().await?;
            print_status(&d.start(!preview, label, notes).await?);
            Ok(())
        })
        .await,
        DeviceCommand::Stop(c) => with_device(c, async |d| {
            d.request_control().await?;
            print_status(&d.stop().await?);
            Ok(())
        })
        .await,
        DeviceCommand::Set { client, node, parameter, value } => with_device(client, async |d| {
            d.request_control().await?;
            d.set_parameter(node, parameter, value).await?;
            println!("{node}.{parameter} = {value} (the chain's filters start again from the next block)");
            Ok(())
        })
        .await,
        DeviceCommand::Mark { client, label } => with_device(client, async |d| {
            let s = d.mark(0, label).await?;
            println!("Marked {label:?} at sample {}", s.run.map_or(0, |r| r.head_sample));
            Ok(())
        })
        .await,
        DeviceCommand::PairCode { client, control } => with_device(client, async |d| {
            d.request_control().await?;
            let code = d.pairing_code(*control).await?;
            let what = if *control { "may direct" } else { "watch only" };
            println!("Pairing code ({what}): {}   →   dspk pair {} {} --name <its name>", code.code, client.device, code.code);
            Ok(())
        })
        .await,
        DeviceCommand::Taps(c) => with_device(c, async |d| {
            for t in d.list_taps().await? {
                let kind = wire::TapKind::try_from(t.kind).map(|k| k.as_str_name().trim_start_matches("TAP_KIND_").to_ascii_lowercase()).unwrap_or_else(|_| "?".into());
                println!("{:<16} {kind:<8} {:>3} channels  {}", t.name, t.channels.len(), if t.node.is_empty() { "raw data".to_string() } else { format!("node {}", t.node) });
            }
            Ok(())
        })
        .await,
        DeviceCommand::Watch { client, tap, seconds } => with_device(client, async |d| {
            let mut rx = d.subscribe(tap, &[]).await?;
            let (mut frames, mut values, mut gaps, mut last_seq, mut latencies) = (0u64, 0u64, 0u64, None::<u64>, Vec::new());
            let start = Instant::now();
            let channels = rx.info().channels.len().max(1) as u64;
            let events = rx.info().kind == wire::TapKind::Events as i32;
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
                values += if events { f.count as u64 } else { f.count as u64 * channels };
                latencies.push(dspk_stream::live::unix_now_ns().saturating_sub(f.sent_unix_ns) / 1000);
            }
            latencies.sort_unstable();
            let p = |q: f64| latencies.get(((latencies.len() as f64 - 1.0) * q).round().max(0.0) as usize).copied().unwrap_or(0);
            let secs = start.elapsed().as_secs_f64();
            let what = if events { "events" } else { "values" };
            println!("{frames} frames, {:.0} {what}/s, {gaps} frames dropped for this client; delivery latency median {} µs, p99 {} µs", values as f64 / secs, p(0.5), p(0.99));
            Ok(())
        })
        .await,
        DeviceCommand::Log(c) => with_device(c, async |d| {
            for e in d.log(50, wire::LogLevel::Info).await? {
                let level = wire::LogLevel::try_from(e.level).map(|l| l.as_str_name().trim_start_matches("LOG_LEVEL_").to_ascii_lowercase()).unwrap_or_else(|_| "?".into());
                println!("{level:<6} {:<12} {}", e.source, e.message);
            }
            Ok(())
        })
        .await,
    }
}
