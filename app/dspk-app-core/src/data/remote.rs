//! Servers the app works with (`dspk serve`): saved profiles (Settings), connections on one network
//! runtime, recordings read on the server ([`RemoteSignal`]: views come from the server's pyramid),
//! and jobs run there (a sort: configured, started, followed, its result fetched), reported as
//! the app's other jobs ([`JobMessage`]).

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::Context as _;
use dspk_stream::protocol::wire;
use dspk_stream::{client_config_with, ClientIdentity, RemoteSignal, ServerTrust, Session};
use serde::{Deserialize, Serialize};

use super::jobs::{Cancel, JobMessage};

/// Threads of the network runtime (connections are I/O-bound).
const NETWORK_THREADS: usize = 2;
/// How often a remote job's status is asked for.
const STATUS_EVERY: Duration = Duration::from_secs(1);
/// The certificate name of a server when the profile names none.
pub const DEFAULT_SERVER_NAME: &str = "localhost";

/// A server the user works with, as saved in Settings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerProfile {
    /// What the user calls it ("lab GPU").
    pub name: String,
    /// `IP:port` (UDP), e.g. `192.168.1.20:50051`.
    pub address: String,
    /// The name its certificate carries (one of `serve --name`, or localhost).
    #[serde(default = "default_server_name")]
    pub server_name: String,
    /// This app's identity on it: the file `dspk clients enroll` wrote.
    pub identity: PathBuf,
}

fn default_server_name() -> String {
    DEFAULT_SERVER_NAME.to_string()
}

impl ServerProfile {
    /// What is wrong with the profile before connecting (address, identity file), if anything.
    pub fn check(&self) -> Result<(), String> {
        if self.name.trim().is_empty() {
            return Err("give the server a name".into());
        }
        self.address.parse::<SocketAddr>().map_err(|_| format!("address {:?}: an IP and port, e.g. 192.168.1.20:50051", self.address))?;
        if !self.identity.is_file() {
            return Err(format!("identity file {} not found", self.identity.display()));
        }
        Ok(())
    }
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
    let identity = ClientIdentity::from_file(&profile.identity).with_context(|| format!("reading {}", profile.identity.display()))?;
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

/// Where a server's results are fetched to: `<results>/<server name>/`.
pub fn results_dir(base: &Path, profile: &ServerProfile) -> PathBuf {
    let safe: String = profile.name.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') { c } else { '_' }).collect();
    base.join(if safe.is_empty() { "server".into() } else { safe })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_are_checked_before_connecting() {
        let mut p = ServerProfile { name: "lab".into(), address: "192.168.1.20:50051".into(), server_name: DEFAULT_SERVER_NAME.into(), identity: PathBuf::from("/nonexistent.pem") };
        assert!(p.check().unwrap_err().contains("identity"));
        p.address = "lab-gpu".into();
        assert!(p.check().unwrap_err().contains("address"));
        assert_eq!(results_dir(Path::new("/r"), &ServerProfile { name: "lab GPU/1".into(), ..p }), PathBuf::from("/r/lab_GPU_1"));
    }
}
