//! `dspk-device`: runs a device from its `device.toml`, or a generator for a quick test.
//!
//! ```bash
//! dspk-device --config device.toml --pair
//! dspk-device --generator --channels 32 --rate 30000 --data ./device-data --pair
//! ```

use std::net::SocketAddr;
use std::path::PathBuf;

use clap::Parser;
use dspk_device::{serve, DeviceConfig, LiveDevice, DEFAULT_PORT};
use dspk_io::live::GeneratorParams;

#[derive(Parser)]
#[command(name = "dspk-device", version, about = "A dspk device: live source, causal chain, taps, recording")]
struct Args {
    /// The device's configuration.
    #[arg(long, conflicts_with = "generator")]
    config: Option<PathBuf>,
    /// No configuration: one generator (noise, spikes and a sine).
    #[arg(long)]
    generator: bool,
    #[arg(long, default_value_t = 32)]
    channels: usize,
    /// Sample rate (Hz).
    #[arg(long, default_value_t = 30_000.0)]
    rate: f64,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// With --generator: where recordings and the device's authority go.
    #[arg(long, default_value = "dspk-device-data")]
    data: PathBuf,
    /// With --generator: the address to listen on (UDP).
    #[arg(long)]
    listen: Option<SocketAddr>,
    /// Print a pairing code for a director (valid for a few minutes).
    #[arg(long)]
    pair: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).init();
    let args = Args::parse();
    let config = match (&args.config, args.generator) {
        (Some(path), _) => DeviceConfig::from_file(path)?,
        (None, true) => {
            let params = GeneratorParams { channels: args.channels, sample_rate_hz: args.rate, seed: args.seed, ..Default::default() };
            DeviceConfig::generator(args.data.clone(), args.listen.unwrap_or(SocketAddr::from(([127, 0, 0, 1], DEFAULT_PORT))), params)
        }
        (None, false) => anyhow::bail!("give --config device.toml, or --generator"),
    };
    let device = tokio::task::spawn_blocking(move || LiveDevice::open(config)).await??;
    serve(device, args.pair, |l| {
        println!("dspk-device listening on {} (QUIC)", l.addr);
        println!("authority fingerprint: {}", l.authority_fingerprint);
        if let Some(web) = l.web_addr {
            println!("pages: http://{web}/live (the live protocol over a WebSocket)");
        }
        if let Some(code) = &l.pairing_code {
            println!("pairing code (director): {code}");
        }
    })
    .await
}
