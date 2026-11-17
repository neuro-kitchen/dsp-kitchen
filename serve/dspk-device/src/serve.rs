//! The device on the network: QUIC + TLS, clients pair once with a code, then connect with their
//! certificate. No certificate, no session; a lost connection never stops the device.

use std::sync::Arc;

use dspk_stream::live::quic::serve_device;
use dspk_stream::live::Device;
use dspk_stream::transport::{server_config_live, Authority, Permission, Server, ServerIdentity};

use crate::device::LiveDevice;

/// What `serve` prints and returns once listening.
pub struct Listening {
    pub addr: std::net::SocketAddr,
    /// A pairing code for a director (control permission), valid for a few minutes.
    pub pairing_code: Option<String>,
    /// SHA-256 of the device's authority: what a paired client pins.
    pub authority_fingerprint: String,
}

/// Serves `device` until the endpoint closes. `pair`: make a pairing code for a director first.
/// `ready` is called once listening.
pub async fn serve(device: Arc<LiveDevice>, pair: bool, ready: impl FnOnce(&Listening)) -> anyhow::Result<()> {
    let config = device.config().clone();
    let authority = Arc::new(Authority::open_or_create(&config.authority_dir())?);
    let identity = ServerIdentity::SelfSigned { names: config.names.clone() };
    let tls = server_config_live(&authority, &identity)?;
    let server = Server::bind(config.listen, tls.config)?.with_authority(authority.clone()).with_pairing(config.names.clone(), false);
    let addr = server.local_addr()?;
    let pairing_code = if pair { Some(authority.new_pairing_code(Permission::Control)?.display()) } else { None };
    let listening = Listening { addr, pairing_code, authority_fingerprint: dspk_stream::transport::auth::fingerprint(authority.ca_certificate()) };
    ready(&listening);
    let device: Arc<dyn Device> = device;
    while let Some((connection, peer)) = server.accept().await {
        let device = device.clone();
        tokio::spawn(async move {
            let remote = connection.remote_address();
            if let Err(e) = serve_device(connection, peer.clone(), device).await {
                tracing::info!("{} ({remote}) left: {e}", peer.name);
            }
        });
    }
    Ok(())
}
