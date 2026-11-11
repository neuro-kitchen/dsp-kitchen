//! Servers the app works with (`dspk serve`): saved profiles (Settings), connections on one network
//! runtime, recordings read on the server ([`RemoteSignal`]: views come from the server's pyramid),
//! and jobs run there (a sort: configured, started, followed, its result fetched), reported as
//! the app's other jobs ([`JobMessage`]).

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Duration;

use anyhow::Context as _;
/// The protocol's messages (a remote job is a `wire::Configure`), for the app's plugins.
pub use dspk_stream::protocol::wire;
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
        if self.name.contains('/') {
            return Err("a server name cannot contain '/' (it is part of dspk:// paths)".into());
        }
        self.address.parse::<SocketAddr>().map_err(|_| format!("address {:?}: an IP and port, e.g. 192.168.1.20:50051", self.address))?;
        if !self.identity.is_file() {
            return Err(format!("identity file {} not found", crate::domain::paths::short_path(&self.identity)));
        }
        Ok(())
    }
}

/// How a recording on a server is named wherever the app names recordings by path (the session
/// document, recent files, sortings' links): `dspk://<server name>/<path on the server>`.
pub const SCHEME: &str = "dspk://";

/// The path naming `path` on the server `server`.
pub fn uri(server: &str, path: &str) -> PathBuf {
    PathBuf::from(format!("{SCHEME}{server}/{}", path.trim_start_matches('/')))
}

/// The server and the path on it, for a `dspk://` path.
pub fn parse_uri(path: &Path) -> Option<(String, String)> {
    let rest = path.to_str()?.strip_prefix(SCHEME)?;
    let (server, path) = rest.split_once('/')?;
    (!server.is_empty() && !path.is_empty()).then(|| (server.to_string(), path.to_string()))
}

/// The servers saved in Settings, for code that only has a path (opening a recording).
fn profiles() -> &'static RwLock<Vec<ServerProfile>> {
    static PROFILES: OnceLock<RwLock<Vec<ServerProfile>>> = OnceLock::new();
    PROFILES.get_or_init(|| RwLock::new(Vec::new()))
}

/// Updates the servers known to [`profile_named`] (the store calls it when Settings change).
pub fn set_profiles(servers: &[ServerProfile]) {
    *profiles().write().unwrap_or_else(|e| e.into_inner()) = servers.to_vec();
}

/// The saved server called `name`.
pub fn profile_named(name: &str) -> Option<ServerProfile> {
    profiles().read().unwrap_or_else(|e| e.into_inner()).iter().find(|p| p.name == name).cloned()
}

/// Opens the `dspk://` recording `path` (blocking: from a background thread, never on the
/// network runtime).
pub fn open_uri(path: &Path) -> anyhow::Result<Arc<RemoteSignal>> {
    let (server, remote) = parse_uri(path).with_context(|| format!("{} is not a dspk:// path", crate::domain::paths::short_path(&path)))?;
    let profile = profile_named(&server).with_context(|| format!("no server named {server:?} in Settings"))?;
    runtime().block_on(open_recording(profile, remote))
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
    let identity = ClientIdentity::from_file(&profile.identity).with_context(|| format!("reading {}", crate::domain::paths::short_path(&profile.identity)))?;
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
    use dspk_core::{MemoryRecording, RecordingSource};
    use dspk_stream::files::DataRoot;
    use dspk_stream::{serve_session, server_config_with_clients, Authority, Device, OpenRecording, Permission, RecordingOpener, ServeOptions, Server, ServerIdentity};
    use dspk_view::{Envelope, SignalBackend, View};

    /// Opens every recording as a 3-channel test signal.
    struct TestOpener;
    impl RecordingOpener for TestOpener {
        fn open(&self, _: &Path, _: Option<&str>) -> dspk_stream::StreamResult<OpenRecording> {
            let data: Vec<f32> = (0..3 * 4_000).map(|i| (i % 50) as f32).collect();
            Ok(OpenRecording::new(Arc::new(MemoryRecording::new("t", data, 3, 1000.0).unwrap()) as Arc<dyn RecordingSource>, None))
        }
    }

    /// The app's data layer against a real server: list, open (RemoteSignal), fetch.
    #[test]
    fn list_open_and_fetch_from_a_server() {
        let base = std::env::temp_dir().join(format!("dspk-app-remote-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let root = base.join("data");
        std::fs::create_dir_all(root.join("m1")).unwrap();
        std::fs::write(root.join("m1/rec.bin"), [0u8; 8]).unwrap();
        std::fs::write(root.join("m1/notes.txt"), "hello").unwrap();
        let authority = Arc::new(Authority::open_or_create(&base.join("auth")).unwrap());
        authority.enroll("app", Permission::Read).unwrap().write(&base.join("app.pem")).unwrap();
        let tls = server_config_with_clients(&authority, &ServerIdentity::SelfSigned { names: vec!["localhost".into()] }, false).unwrap();
        let device = Arc::new(Device {
            name: "test".into(),
            version: "0".into(),
            files: Some((DataRoot::new(&root).unwrap(), Arc::new(TestOpener))),
            recording: None,
            jobs: None,
            runtime: String::new(),
            options: ServeOptions::default(),
        });
        let addr = runtime().block_on(async {
            let server = Server::bind("127.0.0.1:0".parse().unwrap(), tls.config).unwrap().with_authority(authority);
            let addr = server.local_addr().unwrap();
            tokio::spawn(async move {
                while let Some((c, peer)) = server.accept().await {
                    tokio::spawn(serve_session(c, peer, device.clone()));
                }
            });
            addr
        });
        let profile = ServerProfile { name: "test server".into(), address: addr.to_string(), server_name: "localhost".into(), identity: base.join("app.pem") };

        let info = runtime().block_on(test(profile.clone())).unwrap();
        assert_eq!(info.peer_name, "app");
        let list = runtime().block_on(list(profile.clone(), "m1".into())).unwrap();
        assert_eq!(list.entries.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(), ["notes.txt", "rec.bin"]);

        let signal = runtime().block_on(open_recording(profile.clone(), "m1/rec.bin".into())).unwrap();
        assert_eq!(signal.info().channel_count(), 3);
        let mut env = Envelope::Samples(Vec::new());
        signal.view(&View { channels: vec![2], start: 0, end: 10, width: 10 }, &mut env).unwrap();
        assert!(matches!(env, Envelope::Samples(v) if v.len() == 10));

        // The dspk:// path: what opening from the browser, recent files or the session goes through
        set_profiles(std::slice::from_ref(&profile));
        let set = crate::data::recording::SourceSet::open(&uri(&profile.name, "m1/rec.bin")).unwrap();
        let ds = set.default_dataset();
        assert_eq!(ds.total_channels, 3);
        let mut window = vec![0.0f32; 2 * 100];
        ds.source.as_ref().expect("sample reads through RemoteSource").read(&[0, 1], 10..110, &mut window).unwrap();
        // MemoryRecording is channel-major: channel c, sample t = data[c · 4000 + t] = (c · 4000 + t) % 50
        assert_eq!((window[0], window[100], window[99]), (10.0, 10.0, (109 % 50) as f32));
        assert!(crate::data::recording::SourceSet::open(&uri("unknown", "m1/rec.bin")).is_err(), "a server not in Settings");

        let local = runtime().block_on(fetch(profile.clone(), "m1/notes.txt".into(), base.join("results"))).unwrap();
        assert_eq!(local, base.join("results/test_server/notes.txt"));
        assert_eq!(std::fs::read_to_string(&local).unwrap(), "hello");
        let again = runtime().block_on(fetch(profile, "m1/notes.txt".into(), base.join("results"))).unwrap();
        assert_eq!(again, base.join("results/test_server/2-notes.txt"), "never overwritten");
        let _ = std::fs::remove_dir_all(&base);
    }

    /// Writes a small sorting folder after reporting progress (stands in for dspk-cli-sorter).
    struct FakeSorter;
    impl dspk_stream::jobs::JobRunner for FakeSorter {
        fn sorters(&self) -> Vec<String> {
            vec!["kilosort4".into()]
        }
        fn run(&self, job: &dspk_stream::jobs::JobSpec, report: &dyn dspk_stream::jobs::JobReport, _: &std::sync::atomic::AtomicBool) -> Result<String, String> {
            for done in 0..3 {
                report.progress(dspk_stream::jobs::JobProgress { stage: "Sorting".into(), step: 1, steps: 1, done, total: 3, unit: "steps".into() });
                std::thread::sleep(Duration::from_millis(400));
            }
            std::fs::create_dir_all(job.output.join("spikes")).map_err(|e| e.to_string())?;
            std::fs::write(job.output.join("zarr.json"), "{}").map_err(|e| e.to_string())?;
            Ok("2 units".into())
        }
    }

    /// A sort on a server as the app runs it: configured, followed (progress), fetched.
    #[test]
    fn a_remote_sort_is_followed_and_fetched() {
        let base = std::env::temp_dir().join(format!("dspk-app-remote-sort-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let root = base.join("data");
        std::fs::create_dir_all(root.join("m1")).unwrap();
        std::fs::write(root.join("m1/rec.bin"), [0u8; 8]).unwrap();
        let authority = Arc::new(Authority::open_or_create(&base.join("auth")).unwrap());
        authority.enroll("lab", Permission::Control).unwrap().write(&base.join("lab.pem")).unwrap();
        let tls = server_config_with_clients(&authority, &ServerIdentity::SelfSigned { names: vec!["localhost".into()] }, false).unwrap();
        let data_root = DataRoot::new(&root).unwrap();
        let jobs = Arc::new(dspk_stream::jobs::Jobs::new(Arc::new(FakeSorter), data_root.clone(), None));
        let device = Arc::new(Device { name: "t".into(), version: "0".into(), files: Some((data_root, Arc::new(TestOpener))), recording: None, jobs: Some(jobs), runtime: "cpu".into(), options: ServeOptions::default() });
        let addr = runtime().block_on(async {
            let server = Server::bind("127.0.0.1:0".parse().unwrap(), tls.config).unwrap().with_authority(authority);
            let addr = server.local_addr().unwrap();
            tokio::spawn(async move {
                while let Some((c, peer)) = server.accept().await {
                    tokio::spawn(serve_session(c, peer, device.clone()));
                }
            });
            addr
        });
        let profile = ServerProfile { name: "gpu".into(), address: addr.to_string(), server_name: "localhost".into(), identity: base.join("lab.pem") };
        let job = wire::Configure { sorter: "kilosort4".into(), recording: "m1/rec.bin".into(), ..Default::default() };
        let (tx, rx) = std::sync::mpsc::channel();
        let _cancel = remote_sort(profile, job, base.join("results"), move |m| {
            let _ = tx.send(m);
        });
        let (mut progress, mut end) = (0, None);
        while let Ok(m) = rx.recv_timeout(Duration::from_secs(20)) {
            match m {
                JobMessage::Progress(..) => progress += 1,
                done => {
                    end = Some(done);
                    break;
                }
            }
        }
        let Some(JobMessage::Done(local)) = end else { panic!("the sort did not finish: {end:?}") };
        assert!(progress >= 2, "progress reported while it ran ({progress})");
        assert_eq!(local, base.join("results/gpu/rec.kilosort4.sorting.zarr"));
        assert!(local.join("zarr.json").is_file() && local.join("spikes").is_dir(), "the sorting fetched whole");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn remote_paths_round_trip() {
        let p = uri("lab GPU", "/m1/rec.ap.bin");
        assert_eq!(p, PathBuf::from("dspk://lab GPU/m1/rec.ap.bin"));
        assert_eq!(parse_uri(&p), Some(("lab GPU".into(), "m1/rec.ap.bin".into())));
        assert_eq!(p.file_name().unwrap(), "rec.ap.bin");
        assert_eq!(parse_uri(Path::new("/data/rec.bin")), None);
        assert_eq!(parse_uri(Path::new("dspk://lab")), None);
    }

    #[test]
    fn profiles_are_checked_before_connecting() {
        let mut p = ServerProfile { name: "lab".into(), address: "192.168.1.20:50051".into(), server_name: DEFAULT_SERVER_NAME.into(), identity: PathBuf::from("/nonexistent.pem") };
        assert!(p.check().unwrap_err().contains("identity"));
        p.address = "lab-gpu".into();
        assert!(p.check().unwrap_err().contains("address"));
        assert_eq!(results_dir(Path::new("/r"), &ServerProfile { name: "lab GPU/1".into(), ..p }), PathBuf::from("/r/lab_GPU_1"));
    }
}
