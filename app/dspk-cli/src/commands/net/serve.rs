//! `serve`: a recording (or a synthetic one) over QUIC with dspk-stream. Clients get its exact
//! description, the stored samples, and views answered from its min/max pyramid (built in the
//! background, or reused from the file next to the recording).

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use anyhow::Context;
use clap::Args;
use dspk_core::{ProgressEvent, ProgressSink, RecordingSource};
use dspk_synapse::io::{SyntheticParams, SyntheticRecording};
use dspk_stream::files::DataRoot;
use dspk_stream::{serve_session, server_config_with_clients, Authority, Device, OpenRecording, Pacing, RecordingOpener, ServeOptions, Server, ServerIdentity, StreamError, StreamResult};
use dspk_view::{Pyramid, PyramidBuilder, MEMORY_BASE};

use crate::progress::TerminalProgress;

const DEFAULT_BIND: &str = "127.0.0.1:50051";
const LOCALHOST: &str = "localhost";
/// Synthetic source when no recording is given: a Neuropixels-sized probe for an hour at 30 kHz.
const SYNTHETIC_CHANNELS: usize = 384;
const SYNTHETIC_SAMPLE_RATE_HZ: f64 = 30_000.0;
const SYNTHETIC_DURATION_SEC: f64 = 3_600.0;
/// The audit log of control requests, in the authority folder (JSON lines, mode 0600).
const AUDIT_FILE: &str = "audit.jsonl";
/// Progress label of the pyramid build.
const PYRAMID_STAGE: &str = "Building pyramid";

#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Recording every client starts with (any format dspk-io reads); a synthetic one when
    /// neither this nor --data is given
    recording: Option<PathBuf>,
    /// Data root: clients list and open the recordings under it (paths relative to it; hidden
    /// files and links out of it are refused)
    #[arg(long, env = "DSPK_SERVE_DATA")]
    data: Option<PathBuf>,
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
    #[arg(long, default_value_t = dspk_stream::transport::DEFAULT_FRAME_SEC)]
    frame_sec: f64,
    /// Start over at the end of the recording
    #[arg(long)]
    r#loop: bool,
    /// Send as fast as clients take it, not at the recording's rate
    #[arg(long)]
    unpaced: bool,
    /// Certificate chain (PEM); when omitted, the server's authority issues one for the --name
    /// names at every start
    #[arg(long, requires = "private_key")]
    certificate: Option<PathBuf>,
    /// Private key (PEM) of --certificate
    #[arg(long, requires = "certificate")]
    private_key: Option<PathBuf>,
    /// Where to write the certificate clients trust (DER): the server's authority. Clients
    /// enrolled with `dspk clients enroll` have it in their identity file already
    #[arg(long, conflicts_with = "certificate")]
    certificate_out: Option<PathBuf>,
    /// The server's authority folder: its CA and the enrolled clients (default: <data dir>/serve,
    /// e.g. ~/.local/share/dspk/serve; created on first use)
    #[arg(long, env = super::clients::AUTH_DIR_ENV)]
    auth_dir: Option<PathBuf>,
    /// The compute runtime jobs (sorts) run on: wgpu, cuda, cpu (default: $DSPK_RUNTIME, else
    /// the first compiled in, as `dspk doctor` shows)
    #[arg(long)]
    runtime: Option<String>,
    /// Also let in clients without a certificate, read only. Anyone who reaches the port can
    /// then read the recording: trusted networks only
    #[arg(long)]
    allow_anonymous: bool,
    /// Clients connected at once; more are refused until one leaves
    #[arg(long, default_value_t = dspk_stream::DEFAULT_MAX_CONNECTIONS)]
    max_clients: usize,
    /// Host names or addresses clients reach the server by, for the self-signed certificate
    /// (besides localhost and the --bind address); repeat for several
    #[arg(long = "name", conflicts_with = "certificate")]
    names: Vec<String>,
}

/// The names of the self-signed certificate: localhost, the --bind address unless it is a
/// wildcard (0.0.0.0, ::), then each --name, without repeats.
fn self_signed_names(args: &ServeArgs) -> Vec<String> {
    let mut names = vec![LOCALHOST.to_string()];
    if !args.bind.ip().is_unspecified() {
        names.push(args.bind.ip().to_string());
    }
    for n in &args.names {
        if !names.contains(n) {
            names.push(n.clone());
        }
    }
    names
}

/// Ctrl+C (SIGINT), or SIGTERM on Unix: what `podman stop`, systemd and Kubernetes send.
async fn shutdown() {
    #[cfg(unix)]
    {
        let Ok(mut term) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) else {
            let _ = tokio::signal::ctrl_c().await;
            return;
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        }
    }
    #[cfg(not(unix))]
    let _ = tokio::signal::ctrl_c().await;
}

/// Opens a recording and its pyramid (the file next to it, built when missing).
fn open_with_pyramid(path: &Path, source: Option<&str>) -> anyhow::Result<(Arc<dyn RecordingSource>, Pyramid)> {
    let entries = dspk_synapse::io::sources(path).with_context(|| format!("reading {}", path.display()))?;
    let id = match source {
        Some(id) => id.to_string(),
        None => dspk_io::default_source(&entries).map(|e| e.id.clone()).unwrap_or_default(),
    };
    let source: Arc<dyn RecordingSource> = Arc::from(if id.is_empty() { dspk_synapse::io::open(path) } else { dspk_synapse::io::open_source(path, &id) }.with_context(|| format!("opening {}", path.display()))?);
    let (pyramid, _) = Pyramid::open_or_create(source.as_ref(), Some((path, id.as_str())))?;
    Ok((source, pyramid))
}

/// Opens the recordings clients ask for (through dspk-synapse's formats), building a missing
/// pyramid in the background without a progress bar.
struct FileOpener;

impl RecordingOpener for FileOpener {
    fn open(&self, path: &Path, source: Option<&str>) -> StreamResult<OpenRecording> {
        let (source, pyramid) = open_with_pyramid(path, source).map_err(|e| StreamError::Protocol(format!("{e:#}")))?;
        let pyramid = Arc::new(pyramid);
        if !pyramid.is_complete() {
            PyramidBuilder::default().run(source.clone(), pyramid.clone(), 0, Arc::new(AtomicBool::new(false)), Arc::new(|_| {}));
        }
        Ok(OpenRecording::new(source, Some(pyramid)))
    }
}

/// The recording every client starts with: says what it is and builds its pyramid in the
/// background (with a progress bar) when the file next to it is missing or incomplete.
fn start_recording(source: Arc<dyn RecordingSource>, pyramid: Pyramid) -> OpenRecording {
    let pyramid = Arc::new(pyramid);
    let info = source.info();
    println!("Serving {}: {} channels × {} samples at {} Hz ({:?})", info.name, info.channel_count(), info.samples, info.sample_rate_hz(), info.format);
    match pyramid.path() {
        Some(p) if pyramid.is_complete() => println!("Views from {}", p.display()),
        _ => {
            let label = pyramid.path().map_or_else(|| "memory".to_string(), |p| p.display().to_string());
            println!("Views from a pyramid being built in {label}");
            let bar = TerminalProgress::default();
            let on_progress: dspk_view::OnProgress = Arc::new(move |p: dspk_view::Progress| {
                bar.report(&ProgressEvent { stage: PYRAMID_STAGE, step: 1, steps: 1, done: p.done, total: p.total, unit: "samples" });
            });
            PyramidBuilder::default().run(source.clone(), pyramid.clone(), 0, Arc::new(AtomicBool::new(false)), on_progress);
        }
    }

    OpenRecording::new(source, Some(pyramid))
}

pub async fn run(args: &ServeArgs) -> anyhow::Result<()> {
    let initial: Option<(Arc<dyn RecordingSource>, Pyramid)> = match (&args.recording, &args.data) {
        (Some(path), _) => Some(open_with_pyramid(path, None)?),
        (None, Some(_)) => None,
        (None, None) => {
            let params = SyntheticParams { channels: args.channels, sample_rate_hz: args.sample_rate, duration_sec: SYNTHETIC_DURATION_SEC, ..Default::default() };
            let source: Arc<dyn RecordingSource> = Arc::new(SyntheticRecording::new(params)?);
            let pyramid = Pyramid::in_memory(source.as_ref(), MEMORY_BASE)?;
            Some((source, pyramid))
        }
    };
    let files = match &args.data {
        Some(dir) => {
            let root = DataRoot::new(dir)?;
            println!("Files: recordings under {} (clients list and open them)", root.path().display());
            Some((root, Arc::new(FileOpener) as Arc<dyn RecordingOpener>))
        }
        None => None,
    };
    let recording = match initial {
        Some((source, pyramid)) => Some(Arc::new(start_recording(source, pyramid))),
        None => None,
    };

    let identity = match (&args.certificate, &args.private_key) {
        (Some(certificate_chain), Some(private_key)) => ServerIdentity::Files { certificate_chain: certificate_chain.clone(), private_key: private_key.clone() },
        _ => ServerIdentity::SelfSigned { names: self_signed_names(args) },
    };
    let dir = super::clients::auth_dir(args.auth_dir.as_deref());
    let authority = Arc::new(Authority::open_or_create(&dir).with_context(|| format!("opening the authority in {}", dir.display()))?);
    let tls = server_config_with_clients(&authority, &identity, args.allow_anonymous)?;
    if let Some(out) = &args.certificate_out {
        std::fs::write(out, &tls.certificate).with_context(|| format!("writing {}", out.display()))?;
        println!("Certificate clients trust written to {} (`dspk receive --certificate {}`)", out.display(), out.display());
    }
    let enrolled = authority.clients()?.iter().filter(|c| c.active()).count();
    println!("Clients: certificates enrolled in {} ({enrolled} active; `dspk clients enroll <name>`)", dir.display());
    if args.allow_anonymous {
        println!("Warning: --allow-anonymous: anyone who reaches this port can read the recording");
    } else if enrolled == 0 {
        println!("No client is enrolled yet: nobody can connect until one is");
    }
    let server = Server::bind(args.bind, tls.config)?.with_max_connections(args.max_clients).with_authority(authority);
    let options = ServeOptions {
        frame_sec: args.frame_sec,
        pacing: if args.unpaced { Pacing::Unpaced } else { Pacing::RealTime },
        loop_playback: args.r#loop,
        ..Default::default()
    };
    // The runtime jobs run on: --runtime, else $DSPK_RUNTIME, else the first compiled in
    let runtime = match &args.runtime {
        Some(name) => dspk_core::ComputeTarget::parse(name)?.checked()?.name().to_string(),
        None => dspk_core::ComputeTarget::from_env()?.name().to_string(),
    };
    // Jobs (sorts) on the files, when there are files and the sorter program is here
    let jobs = match &files {
        Some((root, _)) => match crate::sorters_program::locate(env!("CARGO_PKG_VERSION"), std::env::current_exe().ok().as_deref()) {
            Some(program) => {
                let audit = dir.join(AUDIT_FILE);
                println!("Jobs: sorts with {} on {runtime} (control clients); audit log {}", program.display(), audit.display());
                let runner = Arc::new(super::jobs::SorterRunner { program, runtime: Some(runtime.clone()) });
                Some(Arc::new(dspk_stream::jobs::Jobs::new(runner, root.clone(), Some(audit))))
            }
            None => {
                println!("Jobs: none (dspk-cli-sorter not found: `dspk sorters install`)");
                None
            }
        },
        None => None,
    };
    let device = Arc::new(Device {
        name: args.names.first().cloned().unwrap_or_else(|| LOCALHOST.to_string()),
        version: env!("CARGO_PKG_VERSION").to_string(),
        files,
        recording,
        jobs: jobs.clone(),
        runtime: if jobs.is_some() { runtime } else { String::new() },
        options,
    });
    println!("Listening on {} (Ctrl+C to stop)", server.local_addr()?);

    tokio::select! {
        _ = shutdown() => {
            println!("Stopping");
            // No job outlives the server: a running sort is stopped (its program killed, its
            // partial output removed)
            if let Some(jobs) = &jobs
                && jobs.state() == dspk_stream::protocol::wire::DeviceState::Running
            {
                println!("Stopping the running job");
                let jobs = jobs.clone();
                let _ = tokio::task::spawn_blocking(move || jobs.shutdown(std::time::Duration::from_secs(10))).await;
            }
        }
        _ = async {
            while let Some((connection, peer)) = server.accept().await {
                let remote = format!("{} ({}, {})", connection.remote_address(), peer.name, peer.permission);
                println!("Client {remote} connected");
                let device = device.clone();
                tokio::spawn(async move {
                    match serve_session(connection, peer, device).await {
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
