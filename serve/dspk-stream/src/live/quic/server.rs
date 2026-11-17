//! A device's side of one connection.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tokio::task::AbortHandle;

use super::super::{dispatch, wire, Caller, Device, Followup, Pushed, PROTOCOL_VERSION};
use crate::error::{StreamError, StreamResult};
use crate::protocol::{read_message, write_message};
use crate::transport::{Peer, Permission, DEFAULT_MAX_REQUEST_BYTES};

/// Subscriptions of one connection, to stop on `Unsubscribe`: by tap (`tap:<name>`) and by plot
/// (`plot:<key>`).
type Pumps = Arc<Mutex<HashMap<String, AbortHandle>>>;

/// Serves one client's connection until it closes: the welcome, the device's events, requests
/// (each on a stream of its own, answered off the network thread), and the subscriptions'
/// streams.
pub async fn serve_device(connection: quinn::Connection, peer: Peer, device: Arc<dyn Device>) -> StreamResult<()> {
    let caller = Caller { name: peer.name.clone(), may_control: peer.permission == Permission::Control };
    let (mut control, mut control_recv) = connection.accept_bi().await?;
    let hello: wire::DeviceHello = read_message(&mut control_recv, DEFAULT_MAX_REQUEST_BYTES).await?.ok_or_else(|| StreamError::protocol("no hello"))?;
    if hello.protocol_version != PROTOCOL_VERSION {
        let why = format!("this device speaks live protocol {PROTOCOL_VERSION}; the client ({}) speaks {}", hello.client_version, hello.protocol_version);
        connection.close(quinn::VarInt::from_u32(1), why.as_bytes());
        return Err(StreamError::protocol(why));
    }
    let permission = if caller.may_control { crate::protocol::wire::Permission::Control } else { crate::protocol::wire::Permission::Read };
    let welcome = wire::DeviceWelcome { protocol_version: PROTOCOL_VERSION, device: Some(device.info()), client_name: caller.name.clone(), permission: permission as i32 };
    write_message(&mut control, &welcome).await?;
    {
        let (device, caller) = (device.clone(), caller.clone());
        tokio::task::spawn_blocking(move || device.connected(&caller));
    }
    tracing::info!("{} connected from {}", caller.name, connection.remote_address());

    // The device's events, on the control stream
    let mut events = device.events();
    let event_pump = tokio::spawn(async move {
        loop {
            match events.recv().await {
                Ok(event) => {
                    if write_message(&mut control, &event).await.is_err() {
                        break;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => tracing::debug!("a slow client missed {n} device events"),
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });

    let pumps: Pumps = Arc::default();
    let result = loop {
        let (send, recv) = match connection.accept_bi().await {
            Ok(streams) => streams,
            Err(quinn::ConnectionError::ApplicationClosed(_) | quinn::ConnectionError::LocallyClosed) => break Ok(()),
            Err(e) => break Err(StreamError::from(e)),
        };
        let (device, caller, connection, pumps) = (device.clone(), caller.clone(), connection.clone(), pumps.clone());
        tokio::spawn(async move {
            if let Err(e) = answer(send, recv, device, caller, connection, pumps).await {
                tracing::debug!("a request failed: {e}");
            }
        });
    };
    event_pump.abort();
    for (_, pump) in pumps.lock().unwrap_or_else(|e| e.into_inner()).drain() {
        pump.abort();
    }
    // Kept open until here: dropping it earlier would stop the control stream
    drop(control_recv);
    tokio::task::spawn_blocking(move || device.disconnected(&caller));
    result
}

async fn answer(mut send: quinn::SendStream, mut recv: quinn::RecvStream, device: Arc<dyn Device>, caller: Caller, connection: quinn::Connection, pumps: Pumps) -> StreamResult<()> {
    let Some(request): Option<wire::LiveRequest> = read_message(&mut recv, DEFAULT_MAX_REQUEST_BYTES).await? else { return Ok(()) };
    let (response, followup) = {
        let (device, caller) = (device.clone(), caller.clone());
        tokio::task::spawn_blocking(move || dispatch(device.as_ref(), &caller, request)).await.map_err(|e| StreamError::protocol(e.to_string()))?
    };
    match followup {
        Followup::None => {}
        Followup::Unsubscribe { taps, plot_keys } => {
            let mut pumps = pumps.lock().unwrap_or_else(|e| e.into_inner());
            for key in taps.iter().map(|t| format!("tap:{t}")).chain(plot_keys.iter().map(|k| format!("plot:{k}"))) {
                if let Some(pump) = pumps.remove(&key) {
                    pump.abort();
                }
            }
        }
        // The tap's stream is opened (and its first message sent) before the answer, so a client
        // that waits for the answer then accepts the stream gets this one
        Followup::Subscribe(sub) => {
            let mut uni = connection.open_uni().await?;
            let key = if sub.plot_key != 0 { format!("plot:{}", sub.plot_key) } else { format!("tap:{}", sub.info.name) };
            write_message(&mut uni, &wire::TapOpened { tap: 0, info: Some(sub.info.clone()), plot_key: sub.plot_key }).await?;
            let mut frames = sub.frames;
            let pump = tokio::spawn(async move {
                while let Some(pushed) = frames.recv().await {
                    let sent = match &pushed {
                        Pushed::Frame(f) => write_message(&mut uni, f).await,
                        Pushed::View(v) => write_message(&mut uni, v).await,
                    };
                    if sent.is_err() {
                        break;
                    }
                }
                let _ = uni.finish();
            });
            if let Some(old) = pumps.lock().unwrap_or_else(|e| e.into_inner()).insert(key, pump.abort_handle()) {
                old.abort();
            }
        }
    }
    write_message(&mut send, &response).await?;
    send.finish().map_err(StreamError::protocol)?;
    Ok(())
}
