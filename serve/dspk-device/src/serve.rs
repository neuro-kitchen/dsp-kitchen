//! The device on the network: QUIC + TLS, clients pair once with a code, then connect with their
//! certificate. No certificate, no session; a lost connection never stops the device.
//!
//! One endpoint, three protocols (by the TLS application protocol each connection asks for):
//! - the live protocol (`dspk-live/1`): the device itself ([`crate::LiveDevice`]);
//! - the session protocol (`dspk-stream/2`): its finished recordings, listed, opened, viewed and
//!   fetched as `dspk serve` serves files (a recording still running is refused there);
//! - pairing (`dspk-pair/1`): a computer gets its certificate with a one-time code.

use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use dspk_core::RecordingSource;
use dspk_io::live::SessionLog;
use dspk_stream::live::quic::{is_live, serve_device};
use dspk_stream::live::Device;
use dspk_stream::transport::files::DataRoot;
use dspk_stream::transport::{server_config_live, Authority, Permission, Server, ServerIdentity};
use dspk_stream::{serve_session, OpenRecording, RecordingOpener, ServeOptions, StreamError, StreamResult};
use dspk_view::{Pyramid, PyramidBuilder};

use crate::device::LiveDevice;

/// What `serve` reports once listening.
pub struct Listening {
    pub addr: std::net::SocketAddr,
    /// A pairing code for a director (control permission), valid for a few minutes.
    pub pairing_code: Option<String>,
    /// SHA-256 of the device's authority: what a paired client pins.
    pub authority_fingerprint: String,
    /// Where pages reach the device (`/live`), if it serves them.
    pub web_addr: Option<std::net::SocketAddr>,
}

/// Opens the device's finished recordings for the session protocol: the raw data (or a derived
/// tap's store) with a min / max pyramid next to it, built in the background.
struct RecordingsOpener;

impl RecordingOpener for RecordingsOpener {
    fn open(&self, path: &Path, _source: Option<&str>) -> StreamResult<OpenRecording> {
        let folder = path.parent().unwrap_or(path);
        if folder.join(dspk_io::live::SESSION_LOG_FILE).exists() && !SessionLog::ended_cleanly(folder) {
            return Err(StreamError::Protocol(format!("{}: the recording is still running: watch it through the live protocol", path.display())));
        }
        let source: Arc<dyn RecordingSource> = Arc::from(dspk_io::open(path).map_err(|e| StreamError::Protocol(e.to_string()))?);
        let (pyramid, _) = Pyramid::open_or_create(source.as_ref(), Some((path, ""))).map_err(|e| StreamError::Protocol(e.to_string()))?;
        let pyramid = Arc::new(pyramid);
        if !pyramid.is_complete() {
            PyramidBuilder::default().run(source.clone(), pyramid.clone(), 0, Arc::new(AtomicBool::new(false)), Arc::new(|_| {}));
        }
        Ok(OpenRecording::new(source, Some(pyramid)))
    }
}

/// Serves `device` until the endpoint closes or the process is asked to stop (SIGINT, SIGTERM),
/// which ends a running recording cleanly first. `pair`: make a pairing code for a director.
/// `ready` is called once listening.
pub async fn serve(device: Arc<LiveDevice>, pair: bool, ready: impl FnOnce(&Listening)) -> anyhow::Result<()> {
    let config = device.config().clone();
    let authority = Arc::new(Authority::open_or_create(&config.authority_dir())?);
    device.set_authority(authority.clone());
    let identity = ServerIdentity::SelfSigned { names: config.names.clone() };
    let tls = server_config_live(&authority, &identity)?;
    let server = Arc::new(Server::bind(config.listen, tls.config)?.with_authority(authority.clone()).with_pairing(config.names.clone(), false));
    let addr = server.local_addr()?;
    let pairing_code = if pair { Some(authority.new_pairing_code(Permission::Control)?.display()) } else { None };
    let fingerprint = dspk_stream::transport::auth::fingerprint(authority.ca_certificate());
    // Finished recordings, for the session protocol
    std::fs::create_dir_all(config.recordings_dir())?;
    let files = dspk_stream::Device {
        name: config.name.clone(),
        version: env!("CARGO_PKG_VERSION").into(),
        files: Some((DataRoot::new(&config.recordings_dir())?, Arc::new(RecordingsOpener) as Arc<dyn RecordingOpener>)),
        recording: None,
        jobs: None,
        runtime: String::new(),
        options: ServeOptions::default(),
        plugins: Vec::new(),
    };
    let files = Arc::new(files);
    let live: Arc<dyn Device> = device.clone();

    // Pages: the live protocol over a WebSocket (and the web app's files, if given)
    let mut web_addr = None;
    if let Some(web) = &config.web {
        use dspk_stream::transport::web::{WebServer, WebTls};
        let tls = if web.listen.ip().is_loopback() { WebTls::Plain } else { WebTls::Authority };
        let pages = WebServer::bind(web.listen, authority.clone(), files.clone(), config.names.clone(), web.root.clone(), tls).await?.with_live(live.clone());
        web_addr = Some(pages.local_addr()?);
        tokio::spawn(pages.run());
    }
    let listening = Listening { addr, pairing_code, authority_fingerprint: fingerprint, web_addr };
    ready(&listening);
    let accept = {
        let server = server.clone();
        async move {
            while let Some((connection, peer)) = server.accept().await {
                let remote = connection.remote_address();
                if is_live(&connection) {
                    let device = live.clone();
                    tokio::spawn(async move {
                        if let Err(e) = serve_device(connection, peer.clone(), device).await {
                            tracing::info!("{} ({remote}) left: {e}", peer.name);
                        }
                    });
                } else {
                    let files = files.clone();
                    tokio::spawn(async move {
                        if let Err(e) = serve_session(connection, peer.clone(), files).await {
                            tracing::info!("{} ({remote}, recordings) left: {e}", peer.name);
                        }
                    });
                }
            }
        }
    };
    tokio::select! {
        _ = accept => {}
        why = shutdown_signal() => {
            tracing::info!("{why}: stopping");
            let d = device.clone();
            tokio::task::spawn_blocking(move || d.shutdown(why)).await?;
            server.close();
        }
    }
    Ok(())
}

/// The first of SIGINT (Ctrl+C) and SIGTERM (`podman stop`, systemd).
async fn shutdown_signal() -> &'static str {
    #[cfg(unix)]
    {
        let Ok(mut term) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) else {
            let _ = tokio::signal::ctrl_c().await;
            return "interrupted";
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => "interrupted",
            _ = term.recv() => "terminated",
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
        "interrupted"
    }
}
