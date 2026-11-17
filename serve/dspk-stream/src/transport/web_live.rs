//! The live protocol over a page's WebSocket ([`super::web::LIVE_WS_PATH`]): what a device answers
//! over QUIC ([`crate::live::quic`]), for pages, which cannot open QUIC connections.
//!
//! The page says who it is as on `/ws` (a kept token, or pairing with a code; JSON text frames),
//! then everything is binary frames: `[kind: u8][id: u32 big-endian][protobuf]`.
//!
//! | kind | direction | id | message |
//! |---|---|---|---|
//! | 0 | page → device | the page's | [`wire::LiveRequest`] |
//! | 0 | device → page | the request's | [`wire::LiveResponse`] |
//! | 1 | device → page | 0 | [`wire::DeviceWelcome`] (first), then [`wire::DeviceEvent`]s |
//! | 2 | device → page | the subscribing request's | [`wire::TapOpened`] |
//! | 3 | device → page | the subscribing request's | [`wire::TapFrame`] |
//! | 4 | device → page | the subscribing request's | [`wire::LiveViewFrame`] |
//!
//! A subscription's frames carry the id of the request that made it, so a page tells its
//! subscriptions apart without the stream-per-subscription of QUIC.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use futures_util::{SinkExt, StreamExt};
use prost::Message as _;
use tokio::task::AbortHandle;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::WebSocketStream;

use super::web::{identify, web_error, State, HELLO_TIMEOUT};
use crate::error::{StreamError, StreamResult};
use crate::live::{dispatch, wire, Caller, Followup, Pushed, PROTOCOL_VERSION};
use crate::pair::Permission;

pub const KIND_RESPONSE: u8 = 0;
pub const KIND_EVENT: u8 = 1;
pub const KIND_TAP_OPENED: u8 = 2;
pub const KIND_TAP_FRAME: u8 = 3;
pub const KIND_VIEW_FRAME: u8 = 4;

/// One frame: its kind, id and message.
fn frame(kind: u8, id: u32, message: &impl prost::Message) -> Vec<u8> {
    let mut out = Vec::with_capacity(5 + message.encoded_len());
    out.push(kind);
    out.extend_from_slice(&id.to_be_bytes());
    message.encode(&mut out).expect("the buffer grows");
    out
}

/// One page on a device: who it is, the welcome, then its requests, the device's events and its
/// subscriptions' frames, all through one writer.
pub(super) async fn live_session<S>(mut socket: WebSocketStream<S>, state: Arc<State>) -> StreamResult<()>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let device = state.live.clone().ok_or_else(|| StreamError::protocol("this server is not a device"))?;
    let peer = match tokio::time::timeout(HELLO_TIMEOUT, identify(&mut socket, &state)).await {
        Ok(Ok(Some(peer))) => peer,
        Ok(Ok(None)) => {
            let _ = socket.close(None).await;
            return Err(StreamError::protocol("not let in"));
        }
        Ok(Err(e)) => return Err(e),
        Err(_) => return Err(StreamError::protocol("the page did not say who it is")),
    };
    let caller = Caller { name: peer.name.clone(), may_control: peer.permission == Permission::Control };
    let permission = if caller.may_control { crate::protocol::wire::Permission::Control } else { crate::protocol::wire::Permission::Read };
    let welcome = wire::DeviceWelcome { protocol_version: PROTOCOL_VERSION, device: Some(device.info()), client_name: caller.name.clone(), permission: permission as i32 };
    socket.send(Message::binary(frame(KIND_EVENT, 0, &welcome))).await.map_err(web_error)?;
    {
        let (device, caller) = (device.clone(), caller.clone());
        tokio::task::spawn_blocking(move || device.connected(&caller));
    }
    tracing::info!("web: {} reaches the device ({})", caller.name, peer.permission);

    let (mut sink, mut stream) = socket.split();
    let (out, mut ready) = tokio::sync::mpsc::channel::<Vec<u8>>(256);
    let writer = tokio::spawn(async move {
        while let Some(bytes) = ready.recv().await {
            if sink.send(Message::binary(bytes)).await.is_err() {
                break;
            }
        }
    });
    // The device's events
    let events = {
        let (mut rx, out) = (device.events(), out.clone());
        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(event) => {
                        if out.send(frame(KIND_EVENT, 0, &event)).await.is_err() {
                            break;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            }
        })
    };
    let pumps: Arc<Mutex<HashMap<String, AbortHandle>>> = Arc::default();
    let result = loop {
        let Some(message) = stream.next().await else { break Ok(()) };
        let bytes = match message.map_err(web_error) {
            Ok(Message::Binary(bytes)) => bytes,
            Ok(Message::Close(_)) => break Ok(()),
            Ok(_) => continue,
            Err(e) => break Err(e),
        };
        if bytes.len() < 5 || bytes[0] != KIND_RESPONSE {
            break Err(StreamError::protocol("a live request frame was expected"));
        }
        let id = u32::from_be_bytes(bytes[1..5].try_into().expect("four bytes"));
        let request = match wire::LiveRequest::decode(&bytes[5..]) {
            Ok(r) => r,
            Err(e) => break Err(StreamError::protocol(format!("not a live request: {e}"))),
        };
        let (device, caller, out, pumps) = (device.clone(), caller.clone(), out.clone(), pumps.clone());
        tokio::spawn(async move {
            let Ok((response, followup)) = tokio::task::spawn_blocking(move || dispatch(device.as_ref(), &caller, request)).await else { return };
            match followup {
                Followup::None => {}
                Followup::Unsubscribe { taps, plot_keys } => {
                    let mut pumps = pumps.lock().unwrap_or_else(|e| e.into_inner());
                    for key in taps.iter().map(|t| format!("tap:{t}")).chain(plot_keys.iter().map(|k| format!("plot:{k}"))) {
                        if let Some(p) = pumps.remove(&key) {
                            p.abort();
                        }
                    }
                }
                Followup::Subscribe(sub) => {
                    let key = if sub.plot_key != 0 { format!("plot:{}", sub.plot_key) } else { format!("tap:{}", sub.info.name) };
                    let _ = out.send(frame(KIND_TAP_OPENED, id, &wire::TapOpened { tap: id, info: Some(sub.info.clone()), plot_key: sub.plot_key })).await;
                    let (mut frames, o) = (sub.frames, out.clone());
                    let pump = tokio::spawn(async move {
                        while let Some(pushed) = frames.recv().await {
                            let bytes = match &pushed {
                                Pushed::Frame(f) => frame(KIND_TAP_FRAME, id, f),
                                Pushed::View(v) => frame(KIND_VIEW_FRAME, id, v),
                            };
                            if o.send(bytes).await.is_err() {
                                break;
                            }
                        }
                    });
                    if let Some(old) = pumps.lock().unwrap_or_else(|e| e.into_inner()).insert(key, pump.abort_handle()) {
                        old.abort();
                    }
                }
            }
            let _ = out.send(frame(KIND_RESPONSE, id, &response)).await;
        });
    };
    events.abort();
    for (_, p) in pumps.lock().unwrap_or_else(|e| e.into_inner()).drain() {
        p.abort();
    }
    drop(out);
    writer.abort();
    tokio::task::spawn_blocking(move || device.disconnected(&caller));
    result
}
