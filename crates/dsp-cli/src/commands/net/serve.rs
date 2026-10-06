//! `serve`: a recording (or a synthetic one) over QUIC with dsp-stream. Clients get its exact
//! description, the stored samples, and views answered from its min/max pyramid (built in the
//! background, or reused from the file next to the recording).

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use anyhow::Context;
use clap::Args;
use dsp_core::{ProgressEvent, ProgressSink, RecordingSource};
use dsp_io::{SyntheticParams, SyntheticRecording};
use dsp_stream::{serve_recording, server_config, Pacing, ServeOptions, Server, ServerIdentity};
use dsp_view::{Pyramid, PyramidBuilder, MEMORY_BASE};

use crate::progress::TerminalProgress;

const DEFAULT_BIND: &str = "127.0.0.1:50051";
const LOCALHOST: &str = "localhost";
/// Synthetic source when no recording is given: a Neuropixels-sized probe for an hour at 30 kHz.
const SYNTHETIC_CHANNELS: usize = 384;
const SYNTHETIC_SAMPLE_RATE_HZ: f64 = 30_000.0;
const SYNTHETIC_DURATION_SEC: f64 = 3_600.0;
/// Progress label of the pyramid build.
const PYRAMID_STAGE: &str = "Building pyramid";

#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Recording to serve (any format dsp-io reads); a synthetic one when omitted
    recording: Option<PathBuf>,
    /// UDP address to listen on
    #[arg(short, long, default_value = DEFAULT_BIND)]
    bind: SocketAddr,
    /// Channels of the synthetic recording
    #[arg(long, default_value_t = SYNTHETIC_CHANNELS)]
    channels: usize,
    /// Sample rate of the synthetic recording (Hz)
    #[arg(long, default_value_t = SYNTHETIC_SAMPLE_RATE_HZ)]
    sample_rate: f64,
    /// Seconds of signal per frame
    #[arg(long, default_value_t = dsp_stream::transport::DEFAULT_FRAME_SEC)]
    frame_sec: f64,
    /// Start over at the end of the recording
    #[arg(long)]
    r#loop: bool,
    /// Send as fast as clients take it, not at the recording's rate
    #[arg(long)]
    unpaced: bool,
    /// Certificate chain (PEM); a self-signed certificate is generated when omitted
    #[arg(long, requires = "private_key")]
    certificate: Option<PathBuf>,
    /// Private key (PEM) of --certificate
    #[arg(long, requires = "certificate")]
    private_key: Option<PathBuf>,
    /// Where to write the generated self-signed certificate (DER), for clients to trust
    #[arg(long, conflicts_with = "certificate")]
    certificate_out: Option<PathBuf>,
}

pub async fn run(args: &ServeArgs) -> anyhow::Result<()> {
    let (source, pyramid): (Arc<dyn RecordingSource>, Pyramid) = match &args.recording {
        Some(path) => {
            let source: Arc<dyn RecordingSource> = Arc::from(dsp_io::open(path).with_context(|| format!("opening {}", path.display()))?);
            let entries = dsp_io::sources(path)?;
            let id = dsp_io::default_source(&entries).map(|e| e.id.clone()).unwrap_or_default();
            let (pyramid, _) = Pyramid::open_or_create(source.as_ref(), Some((path.as_path(), id.as_str())))?;
            (source, pyramid)
        }
        None => {
            let params = SyntheticParams { channels: args.channels, sample_rate_hz: args.sample_rate, duration_sec: SYNTHETIC_DURATION_SEC, ..Default::default() };
            let source: Arc<dyn RecordingSource> = Arc::new(SyntheticRecording::new(params)?);
            let pyramid = Pyramid::in_memory(source.as_ref(), MEMORY_BASE)?;
            (source, pyramid)
        }
    };
    let pyramid = Arc::new(pyramid);
    let info = source.info();
    println!("Serving {}: {} channels × {} samples at {} Hz ({:?})", info.name, info.channel_count(), info.samples, info.sample_rate_hz(), info.format);
    match pyramid.path() {
        Some(p) if pyramid.is_complete() => println!("Views from {}", p.display()),
        _ => {
            let label = pyramid.path().map_or_else(|| "memory".to_string(), |p| p.display().to_string());
            println!("Views from a pyramid being built in {label}");
            let bar = TerminalProgress::default();
            let on_progress: dsp_view::OnProgress = Arc::new(move |p: dsp_view::Progress| {
                bar.report(&ProgressEvent { stage: PYRAMID_STAGE, step: 1, steps: 1, done: p.done, total: p.total, unit: "samples" });
            });
            PyramidBuilder::default().run(source.clone(), pyramid.clone(), 0, Arc::new(AtomicBool::new(false)), on_progress);
        }
    }

    let identity = match (&args.certificate, &args.private_key) {
        (Some(certificate_chain), Some(private_key)) => ServerIdentity::Files { certificate_chain: certificate_chain.clone(), private_key: private_key.clone() },
        _ => ServerIdentity::SelfSigned { names: vec![LOCALHOST.to_string(), args.bind.ip().to_string()] },
    };
    let tls = server_config(&identity)?;
    if let Some(out) = &args.certificate_out {
        std::fs::write(out, &tls.certificate).with_context(|| format!("writing {}", out.display()))?;
        println!("Certificate written to {} (clients: --certificate {})", out.display(), out.display());
    }
    let server = Server::bind(args.bind, tls.config)?;
    let options = ServeOptions {
        frame_sec: args.frame_sec,
        pacing: if args.unpaced { Pacing::Unpaced } else { Pacing::RealTime },
        loop_playback: args.r#loop,
        ..Default::default()
    };
    println!("Listening on {} (Ctrl+C to stop)", server.local_addr()?);

    tokio::select! {
        _ = tokio::signal::ctrl_c() => println!("Stopping"),
        _ = async {
            while let Some(connection) = server.accept().await {
                let remote = connection.remote_address();
                println!("Client {remote} connected");
                let (source, pyramid, options) = (source.clone(), pyramid.clone(), options.clone());
                tokio::spawn(async move {
                    match serve_recording(connection, source, Some(pyramid), options).await {
                        Ok(()) => println!("Client {remote} left"),
                        Err(e) => println!("Client {remote}: {e}"),
                    }
                });
            }
        } => {}
    }
    server.close();
    Ok(())
}
