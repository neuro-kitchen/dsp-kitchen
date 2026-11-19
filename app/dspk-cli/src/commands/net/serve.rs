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
use dspk_stream::transport::server_config_with_pairing;
use dspk_stream::{serve_session, Authority, Device, OpenRecording, Pacing, RecordingOpener, ServeOptions, Server, ServerIdentity, StreamError, StreamResult};
use dspk_pipeline::view::ProcessedChains;
use dspk_view::{Pyramid, PyramidBuilder, MEMORY_BASE};

use crate::progress::TerminalProgress;

const DEFAULT_BIND: &str = "127.0.0.1:50051";
/// Default of --web-bind: this computer only, as --bind.
const DEFAULT_WEB_BIND: &str = "127.0.0.1:50052";
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
    /// Show a one-time pairing code at start: a computer runs `dspk pair <host:port> <code>` and
    /// gets its certificate, with no file to hand over (10 minutes, 3 tries; more codes:
    /// `dspk clients pair`)
    #[arg(long)]
    pair: bool,
    /// The computer paired with --pair may also configure, start and stop jobs
    #[arg(long, requires = "pair")]
    pair_control: bool,
    /// Also serve browsers: the folder of the web app (its index.html, .js and .wasm) on
    /// --web-bind, with a WebSocket for the same requests. Plain HTTP when that address is
    /// loopback (a browser counts it as secure), HTTPS otherwise: with --certificate, else with
    /// the server's own certificate, which a browser warns about. A page signs in with a pairing
    /// code (`--pair`, `dspk clients pair`) and keeps a token
    #[arg(long)]
    web: Option<PathBuf>,
    /// TCP address of the web endpoint
    #[arg(long, default_value = DEFAULT_WEB_BIND, requires = "web")]
    web_bind: SocketAddr,
    /// Plain HTTP on --web-bind even when it is not loopback: for a container whose port is
    /// published only to this computer (`-p 127.0.0.1:50052:50052`), or behind a proxy that holds
    /// the certificate. Anyone who reaches the port sees the pairing code and the page's token
    #[arg(long, requires = "web", env = "DSPK_SERVE_WEB_PLAIN_HTTP")]
    web_plain_http: bool,
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
/// Curation of the sortings under the data root, as a server plugin: dspk-synapse computes, the
/// stream carries (the one place that knows both).
struct SynapsePlugin(dspk_synapse::curation::service::Service, std::sync::Mutex<Option<ProcessedKept>>);

/// The processed signal a page's views were last computed from, as a recording (one object per
/// chain, so the service's windows of it are read once).
struct ProcessedKept {
    signal: Arc<dyn dspk_view::SignalBackend>,
    source: Arc<dyn RecordingSource>,
}

/// A server's processed signal read as a recording.
struct ProcessedSource(Arc<dyn dspk_view::SignalBackend>);

impl RecordingSource for ProcessedSource {
    fn info(&self) -> &dspk_core::RecordingInfo {
        self.0.info()
    }

    fn read(&self, channels: &[usize], samples: std::ops::Range<u64>, out: &mut [f32]) -> dspk_core::DspResult<()> {
        self.0.read(channels, samples, out)
    }
}

impl SynapsePlugin {
    /// The recording a `views` call reads: the open one with the page's processing applied when
    /// it sends some (and keeps every channel: a chain that remaps channels would renumber the
    /// units' channels, so it is read as stored), else as stored.
    fn views_source(&self, body: &[u8], context: &dspk_stream::PluginContext) -> StreamResult<Option<Arc<dyn RecordingSource>>> {
        #[derive(serde::Deserialize)]
        struct Processing {
            #[serde(default)]
            processing: Vec<String>,
        }
        let stored = context.recording.clone();
        let ops = serde_json::from_slice::<Processing>(body).map(|p| p.processing).unwrap_or_default();
        let (Some(make), Some(raw)) = (context.processed.as_ref(), stored.as_ref()) else { return Ok(stored) };
        if ops.is_empty() {
            return Ok(stored);
        }
        let signal = make(&ops)?;
        if signal.info().channel_count() != raw.info().channel_count() {
            return Ok(stored);
        }
        let mut kept = self.1.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(k) = kept.as_ref().filter(|k| Arc::ptr_eq(&k.signal, &signal)) {
            return Ok(Some(k.source.clone()));
        }
        let source: Arc<dyn RecordingSource> = Arc::new(ProcessedSource(signal.clone()));
        *kept = Some(ProcessedKept { signal, source: source.clone() });
        Ok(Some(source))
    }
}

impl dspk_stream::ServerPlugin for SynapsePlugin {
    fn id(&self) -> &'static str {
        dspk_synapse::curation::service::PLUGIN
    }

    /// A curation saved: the sorting and how many edits.
    fn writes(&self, method: &str, body: &[u8]) -> Option<String> {
        let call: dspk_synapse::curation::service::SaveCall = (method == "save").then(|| serde_json::from_slice(body).ok()).flatten()?;
        Some(format!("{} ({} edits)", call.path, call.ops.len()))
    }

    fn call(&self, method: &str, body: &[u8], context: &dspk_stream::PluginContext) -> StreamResult<Vec<u8>> {
        // The sorters this server runs, as its dspk-cli-sorter describes them (a page lists them)
        if method == "sorters" {
            let program = crate::sorters_program::locate(env!("CARGO_PKG_VERSION"), std::env::current_exe().ok().as_deref())
                .ok_or_else(|| StreamError::Server("this server has no dspk-cli-sorter: `dspk sorters install` on it".into()))?;
            let out = std::process::Command::new(&program).arg("describe").output().map_err(|e| StreamError::Server(format!("{}: {e}", program.display())))?;
            if !out.status.success() {
                return Err(StreamError::Server(format!("dspk-cli-sorter describe: {}", String::from_utf8_lossy(&out.stderr).trim())));
            }
            return Ok(out.stdout);
        }
        let resolve = |path: &str| -> dspk_core::DspResult<PathBuf> {
            let resolve = context.resolve.as_ref().ok_or_else(|| dspk_core::DspError::InvalidConfig("this server serves no files".into()))?;
            resolve(path).map_err(|e| dspk_core::DspError::InvalidConfig(e.to_string()))
        };
        let caller = dspk_synapse::curation::service::Caller { name: &context.peer, may_write: context.may_write };
        // Views read the signal on the GPU once (then from memory): unless a sort holds the GPU, and
        // then on the CPU
        let gpu = dspk_core::lease::interactive();
        let run = || -> StreamResult<Vec<u8>> {
            let recording = if method == "views" { self.views_source(body, context)? } else { context.recording.clone() };
            self.0.call(method, body, &resolve, recording.as_ref(), caller).map_err(|e| StreamError::Server(e.to_string()))
        };
        let answer = if gpu.is_none() { dspk_core::device::with_target(dspk_core::ComputeTarget::Cpu, run) } else { run() };
        drop(gpu);
        // What the read held on the GPU is not needed again (the windows are in memory): back to
        // the driver, so another program (a sort) finds it free
        if method == "views" {
            dspk_core::compute::release_device_memory();
        }
        answer
    }
}

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
        let recording = OpenRecording::new(source.clone(), Some(pyramid.clone())).with_processor(processor_of(path));
        if pyramid.is_complete() {
            return Ok(recording);
        }
        Ok(recording.with_refocus(build_pyramid(source, pyramid, Arc::new(|_| {}))))
    }
}

/// The chains a recording's views are processed with, for clients that cannot compute (pages):
/// dspk-pipeline's, as the desktop app's (`dspk_stream` answers views with them). Files the ops
/// name (a remap's channel map) are read from the recording's folder.
fn processor_of(recording: &Path) -> Arc<ProcessedChains> {
    let chains = Arc::new(ProcessedChains::new(recording.parent().map(Path::to_path_buf).unwrap_or_default()));
    // A sort takes the GPU: the chains kept for views go (the next view makes its own)
    let weak = Arc::downgrade(&chains);
    dspk_core::lease::on_release(move || {
        if let Some(chains) = weak.upgrade() {
            chains.clear();
        }
    });
    chains
}

/// The recording every client starts with: says what it is and builds its pyramid in the
/// background (with a progress bar) when the file next to it is missing or incomplete.
fn start_recording(path: &Path, source: Arc<dyn RecordingSource>, pyramid: Pyramid) -> OpenRecording {
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
            let refocus = build_pyramid(source.clone(), pyramid.clone(), on_progress);
            return OpenRecording::new(source, Some(pyramid)).with_processor(processor_of(path)).with_refocus(refocus);
        }
    }

    OpenRecording::new(source, Some(pyramid)).with_processor(processor_of(path))
}

/// Starts building `pyramid` in the background from the start, and returns what moves its work
/// to a sample clients look at (nearest it first, then outward).
fn build_pyramid(source: Arc<dyn RecordingSource>, pyramid: Arc<Pyramid>, on_progress: dspk_view::OnProgress) -> Arc<dyn Fn(u64) + Send + Sync> {
    let builder = Arc::new(PyramidBuilder::default());
    let cancel = Arc::new(AtomicBool::new(false));
    builder.run(source.clone(), pyramid.clone(), 0, cancel.clone(), on_progress.clone());
    // `run` starts once; later calls only move the focus
    Arc::new(move |sample| builder.run(source.clone(), pyramid.clone(), sample, cancel.clone(), on_progress.clone()))
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
        Some((source, pyramid)) => Some(Arc::new(start_recording(args.recording.as_deref().unwrap_or(Path::new("")), source, pyramid))),
        None => None,
    };

    let identity = match (&args.certificate, &args.private_key) {
        (Some(certificate_chain), Some(private_key)) => ServerIdentity::Files { certificate_chain: certificate_chain.clone(), private_key: private_key.clone() },
        _ => ServerIdentity::SelfSigned { names: self_signed_names(args) },
    };
    let dir = super::clients::auth_dir(args.auth_dir.as_deref());
    let authority = Arc::new(Authority::open_or_create(&dir).with_context(|| format!("opening the authority in {}", dir.display()))?);
    // Computers without a certificate may connect to pair with a code, and for nothing else
    // unless --allow-anonymous
    let tls = server_config_with_pairing(&authority, &identity, args.allow_anonymous)?;
    if let Some(out) = &args.certificate_out {
        std::fs::write(out, &tls.certificate).with_context(|| format!("writing {}", out.display()))?;
        println!("Certificate clients trust written to {} (`dspk receive --certificate {}`)", out.display(), out.display());
    }
    let enrolled = authority.clients()?.iter().filter(|c| c.active()).count();
    println!("Clients: certificates enrolled in {} ({enrolled} active; `dspk clients enroll <name>`)", dir.display());
    if args.allow_anonymous {
        println!("Warning: --allow-anonymous: anyone who reaches this port can read the recording");
    } else if enrolled == 0 && !args.pair {
        println!("No client is enrolled yet: nobody can connect until one is (`dspk clients pair` makes a pairing code)");
    }
    if args.pair {
        let permission = if args.pair_control { dspk_stream::Permission::Control } else { dspk_stream::Permission::Read };
        let code = authority.new_pairing_code(permission)?;
        println!("Pairing code: {}   ({permission}; once, {} minutes)", code.display(), dspk_stream::transport::pairing::CODE_MINUTES);
        println!("On the other computer: dspk pair <this host>:{} {}", args.bind.port(), code.display());
    }
    let server = Server::bind(args.bind, tls.config)?.with_max_connections(args.max_clients).with_authority(authority.clone()).with_pairing(self_signed_names(args), args.allow_anonymous);
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
        // Curation of the sortings under the data root, computed here for the pages it serves
        plugins: vec![Arc::new(SynapsePlugin(dspk_synapse::curation::service::Service::default(), Default::default()))],
    });
    println!("Listening on {} (Ctrl+C to stop)", server.local_addr()?);
    if let Some(folder) = &args.web {
        anyhow::ensure!(folder.join("index.html").is_file(), "{}: no index.html there (the web app's folder)", folder.display());
        use dspk_stream::transport::web::WebTls;
        // A real certificate when one is given; on this computer alone, plain HTTP (a browser
        // counts loopback as secure, and a proxy with a real certificate can stand in front);
        // else the server's own authority, which a browser warns about
        let (tls, note) = match &identity {
            ServerIdentity::Files { .. } => (WebTls::Files(identity.clone()), "the certificate given with --certificate"),
            _ if args.web_bind.ip().is_loopback() => (WebTls::Plain, "this computer only; for other computers put `tailscale serve` or a reverse proxy in front, or give --certificate"),
            _ if args.web_plain_http => (WebTls::PlainOffLoopback, "PLAIN HTTP off loopback (--web-plain-http): publish this port only to 127.0.0.1, or behind a proxy that holds a certificate"),
            _ => (WebTls::Authority, "the certificate is this server's own, so a browser warns until its authority is trusted; give --certificate for a real one"),
        };
        let web = dspk_stream::transport::web::WebServer::bind(args.web_bind, authority.clone(), device.clone(), self_signed_names(args), Some(folder.clone()), tls).await?;
        let scheme = if web.is_tls() { "https" } else { "http" };
        println!("Web app: {scheme}://{} (browsers: a pairing code, then a token; {note})", web.local_addr()?);
        tokio::spawn(web.run());
    }

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
