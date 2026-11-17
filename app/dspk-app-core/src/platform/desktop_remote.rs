//! Servers on a computer: QUIC sessions on one network runtime, an identity file per server.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::Context as _;
use dspk_stream::{client_config_with, ClientIdentity, RemoteSignal, ServerTrust, Session};

use crate::data::jobs::{Cancel, JobMessage};
use crate::data::remote::{parse_uri, profile_named, remember, results_dir, wire, ServerProfile};

/// Threads of the network runtime (connections are I/O-bound).
const NETWORK_THREADS: usize = 2;
/// How often a remote job's status is asked for.
const STATUS_EVERY: Duration = Duration::from_secs(1);

/// Opens the `dspk://` recording `path` (blocking: from a background thread, never on the
/// network runtime).
pub fn open_uri(path: &Path) -> anyhow::Result<Arc<RemoteSignal>> {
    let (server, remote) = parse_uri(path).with_context(|| format!("{} is not a dspk:// path", dspk_session::paths::short_path(&path)))?;
    let profile = profile_named(&server).with_context(|| format!("no server named {server:?} in Settings"))?;
    let signal = runtime().block_on(open_recording(profile, remote))?;
    remember(&signal);
    Ok(signal)
}

/// The network runtime every connection runs on (started on first use).
pub fn runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(NETWORK_THREADS)
            .thread_name("dspk-net")
            .enable_all()
            .build()
            .expect("the network runtime starts")
    })
}

/// A new session with `profile`'s server (its identity presented, the server trusted through it).
pub async fn connect(profile: &ServerProfile) -> anyhow::Result<Session> {
    profile.check().map_err(anyhow::Error::msg)?;
    let identity = ClientIdentity::from_file(&profile.identity).with_context(|| format!("reading {}", dspk_session::paths::short_path(&profile.identity)))?;
    let config = client_config_with(&ServerTrust::from_identity(&identity), Some(&identity))?;
    let addr: SocketAddr = profile.address.parse()?;
    Session::connect(addr, &profile.server_name, config).await.with_context(|| format!("connecting to {} ({})", profile.name, profile.address))
}

/// Who the server is and who this app is to it (Settings: "Test").
pub async fn test(profile: ServerProfile) -> anyhow::Result<wire::ServerInfo> {
    let session = connect(&profile).await?;
    let info = session.server().clone();
    session.close().await;
    Ok(info)
}

/// The entries of `dir` under the server's data root.
pub async fn list(profile: ServerProfile, dir: String) -> anyhow::Result<wire::FileList> {
    let session = connect(&profile).await?;
    let list = session.list_files(&dir).await;
    session.close().await;
    Ok(list?)
}

/// The recording at `path` on the server, as a signal the app draws (a session of its own).
pub async fn open_recording(profile: ServerProfile, path: String) -> anyhow::Result<Arc<RemoteSignal>> {
    let mut session = connect(&profile).await?;
    session.open(&path, None).await?;
    Ok(Arc::new(RemoteSignal::new(Arc::new(session), runtime().handle().clone())?))
}

/// Fetches `path` (a file, or a directory such as a .sorting.zarr) from the server into
/// `<results>/<server>/`, under a new name when one is there already; returns the local path.
pub async fn fetch(profile: ServerProfile, path: String, results: PathBuf) -> anyhow::Result<PathBuf> {
    let session = connect(&profile).await?;
    let name = path.trim_end_matches('/').rsplit('/').next().unwrap_or("fetched").to_string();
    let dir = results_dir(&results, &profile);
    tokio::fs::create_dir_all(&dir).await?;
    let mut local = dir.join(&name);
    let mut n = 1;
    while local.exists() {
        n += 1;
        local = dir.join(format!("{n}-{name}"));
    }
    session.fetch(path.trim_end_matches('/'), &local).await?;
    session.close().await;
    Ok(local)
}

/// A sort on the server: configured and started, followed until it ends (progress as
/// [`JobMessage::Progress`]), its sorting fetched into `results` ([`JobMessage::Done`] with the
/// local path). Cancelling stops the job on the server.
pub fn remote_sort(profile: ServerProfile, job: wire::Configure, results: PathBuf, send: impl Fn(JobMessage) + Send + Sync + 'static) -> Cancel {
    let cancel = Cancel::default();
    let c = cancel.clone();
    runtime().spawn(async move {
        let outcome = async {
            let session = connect(&profile).await?;
            let configured = session.configure(job).await?;
            session.start().await?;
            send(JobMessage::Progress(format!("{} on {}", configured.job, profile.name), None));
            let mut stop_sent = false;
            let status = loop {
                tokio::time::sleep(STATUS_EVERY).await;
                if c.cancelled() && !stop_sent {
                    let _ = session.stop().await;
                    stop_sent = true;
                }
                let s = session.status().await?;
                if s.state != wire::DeviceState::Running as i32 {
                    break s;
                }
                let fraction = (s.total > 0).then(|| s.done as f32 / s.total as f32);
                let step = if s.steps > 0 { format!("[{}/{}] ", s.step, s.steps) } else { String::new() };
                send(JobMessage::Progress(format!("{} on {}: {step}{} {}/{} {}", configured.job, profile.name, s.stage, s.done, s.total, s.unit), fraction));
            };
            if status.state == wire::DeviceState::Error as i32 {
                anyhow::bail!("the sort failed on {}: {}", profile.name, status.message);
            }
            if c.cancelled() {
                anyhow::bail!("stopped");
            }
            // The result, fetched next to the others from this server (never overwritten)
            let name = status.output.rsplit('/').next().unwrap_or("result.sorting.zarr").to_string();
            let dir = results_dir(&results, &profile);
            tokio::fs::create_dir_all(&dir).await?;
            let mut local = dir.join(&name);
            let mut n = 1;
            while local.exists() {
                n += 1;
                local = dir.join(format!("{n}-{name}"));
            }
            send(JobMessage::Progress(format!("Fetching {} from {}", status.output, profile.name), None));
            session.fetch(&status.output, &local).await?;
            session.close().await;
            anyhow::Ok(local)
        };
        send(match outcome.await {
            Ok(local) => JobMessage::Done(local),
            Err(e) => JobMessage::Failed(format!("{e:#}")),
        });
    });
    cancel
}


/// Where a page's processing happens; the desktop names its own device instead.
pub fn processing_label() -> Option<String> {
    None
}

/// A page that runs alone opens the files the user picks through its engine; the desktop has
/// its file dialog.
pub fn runs_alone() -> bool {
    false
}

/// See [`runs_alone`]: never asked on the desktop.
pub async fn pick_files(_folder: bool) -> anyhow::Result<(Vec<String>, Option<String>)> {
    anyhow::bail!("the desktop app opens files with its file dialog")
}

/// The server a page talks to; the desktop has none of its own.
pub fn page_profile() -> ServerProfile {
    ServerProfile { name: "this computer".into(), address: String::new(), server_name: crate::data::remote::DEFAULT_SERVER_NAME.into(), identity: PathBuf::new() }
}
