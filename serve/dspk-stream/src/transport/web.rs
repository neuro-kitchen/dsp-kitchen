//! The server for browsers: HTTPS on a TCP port, serving the web app's files and a WebSocket
//! (`/ws`) that carries the same requests and responses as a QUIC session.
//!
//! A browser cannot open QUIC connections of its own, nor present a certificate it made, so:
//!
//! - **Transport.** One WebSocket per page. Requests are binary frames, a 4-byte big-endian id
//!   chosen by the page followed by the protobuf [`wire::Request`]; each answer is the same id
//!   followed by the [`wire::Response`]. Requests are answered by the code that answers a QUIC
//!   session's request streams. The live signal and file downloads are not offered here yet.
//! - **Who it is.** The first frame is text (JSON, [`WebHello`]): a token the page kept from an
//!   earlier visit, or a pairing with a one-time code ([`super::pairing`]), which ends in a
//!   token instead of a certificate. A token is a client in the authority's registry like any
//!   other (its SHA-256 where a certificate's fingerprint is), so listing and revoking are the
//!   same. The server answers with [`WebWelcome`] (text), then takes requests.
//! - **Files.** Everything else is served from the web folder (`index.html`, the `.wasm` and
//!   `.js` of the app), read only, without following paths out of it.

use std::net::SocketAddr;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

use futures_util::{SinkExt, StreamExt};
use http_body_util::Full;
use hyper::body::{Bytes, Incoming};
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use prost::Message as _;
use serde::{Deserialize, Serialize};
use tokio_tungstenite::tungstenite::protocol::Role;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::WebSocketStream;

use super::auth::{Authority, Peer};
use super::pairing::{PairReply, PairToken};
use super::server::{answer, Current, Device};
use crate::error::{StreamError, StreamResult};
use crate::protocol::wire;

pub use crate::pair::{WebHello, WebWelcome, WS_PATH};
/// Largest frame a page may send (a request is small).
pub const MAX_WEB_FRAME_BYTES: usize = 1 << 20;
/// How long a page has to say who it is.
const HELLO_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// What the browser endpoint serves.
pub struct WebServer {
    listener: tokio::net::TcpListener,
    /// `None`: plain HTTP (see [`WebTls::Plain`]).
    tls: Option<tokio_rustls::TlsAcceptor>,
    state: Arc<State>,
}

/// How the browser endpoint is secured.
pub enum WebTls {
    /// Plain HTTP, for a loopback address only: a browser treats `http://127.0.0.1` and
    /// `http://localhost` as secure (no warning, threads allowed), nothing leaves the computer,
    /// and a proxy that holds a real certificate (`tailscale serve`, Caddy) can stand in front.
    Plain,
    /// A certificate the server's own authority issues for its names: encrypted, but a browser
    /// warns until that authority is trusted in it.
    Authority,
    /// A certificate chain and key from files (a public authority's: no warning).
    Files(super::tls::ServerIdentity),
}

struct State {
    authority: Arc<Authority>,
    device: Arc<Device>,
    /// The folder of the web app's files; `None`: only the WebSocket is served.
    web_root: Option<PathBuf>,
    names: Vec<String>,
}

impl WebServer {
    /// Listens on `addr` (TCP), secured as `tls` says (`names`: the server's, for its authority's
    /// certificate and for the tokens of the pages it pairs).
    pub async fn bind(addr: SocketAddr, authority: Arc<Authority>, device: Arc<Device>, names: Vec<String>, web_root: Option<PathBuf>, tls: WebTls) -> StreamResult<Self> {
        let identity = match &tls {
            WebTls::Plain if !addr.ip().is_loopback() => return Err(StreamError::Tls(format!("plain HTTP is only served on a loopback address, not on {addr}"))),
            WebTls::Plain => None,
            WebTls::Authority => Some(authority.server_certificate(&names)?),
            WebTls::Files(identity) => Some(super::tls::identity_chain(identity)?),
        };
        let tls = match identity {
            Some((chain, key)) => {
                let mut crypto = rustls::ServerConfig::builder().with_no_client_auth().with_single_cert(chain, key).map_err(|e| StreamError::Tls(e.to_string()))?;
                crypto.alpn_protocols = vec![b"http/1.1".to_vec()];
                Some(tokio_rustls::TlsAcceptor::from(Arc::new(crypto)))
            }
            None => None,
        };
        let listener = tokio::net::TcpListener::bind(addr).await?;
        Ok(Self { listener, tls, state: Arc::new(State { authority, device, web_root, names }) })
    }

    /// Whether pages are served over HTTPS (else plain HTTP on loopback).
    pub fn is_tls(&self) -> bool {
        self.tls.is_some()
    }

    pub fn local_addr(&self) -> StreamResult<SocketAddr> {
        Ok(self.listener.local_addr()?)
    }

    /// Serves pages until the task is dropped.
    pub async fn run(self) {
        loop {
            let (tcp, remote) = match self.listener.accept().await {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!("web: accept failed: {e}");
                    continue;
                }
            };
            let (tls, state) = (self.tls.clone(), self.state.clone());
            tokio::spawn(async move {
                let service = hyper::service::service_fn(move |request| handle(request, state.clone()));
                let http = hyper::server::conn::http1::Builder::new();
                let served = match tls {
                    Some(tls) => match tls.accept(tcp).await {
                        Ok(stream) => http.serve_connection(TokioIo::new(stream), service).with_upgrades().await,
                        Err(e) => {
                            tracing::debug!("web: TLS handshake with {remote} failed: {e}");
                            return;
                        }
                    },
                    None => http.serve_connection(TokioIo::new(tcp), service).with_upgrades().await,
                };
                if let Err(e) = served {
                    tracing::debug!("web: connection with {remote} ended: {e}");
                }
            });
        }
    }
}

type Reply = Response<Full<Bytes>>;

fn reply(status: StatusCode, content_type: &str, body: impl Into<Bytes>) -> Reply {
    Response::builder()
        .status(status)
        .header("content-type", content_type)
        // The page computes in threads and shared memory only when it is isolated like this
        .header("cross-origin-opener-policy", "same-origin")
        .header("cross-origin-embedder-policy", "require-corp")
        .header("x-content-type-options", "nosniff")
        .body(Full::new(body.into()))
        .expect("a response")
}

async fn handle(mut request: Request<Incoming>, state: Arc<State>) -> Result<Reply, std::convert::Infallible> {
    if request.uri().path() == WS_PATH {
        return Ok(upgrade(&mut request, state));
    }
    if request.method() != hyper::Method::GET {
        return Ok(reply(StatusCode::METHOD_NOT_ALLOWED, "text/plain", "GET only"));
    }
    let Some(root) = state.web_root.clone() else {
        return Ok(reply(StatusCode::NOT_FOUND, "text/plain", "this server has no web app (dspk serve --web <folder>)"));
    };
    let path = request.uri().path().to_string();
    Ok(tokio::task::spawn_blocking(move || file(&root, &path)).await.unwrap_or_else(|_| reply(StatusCode::INTERNAL_SERVER_ERROR, "text/plain", "failed")))
}

/// The file at `path` under `root`: only plain components, never out of the folder.
fn file(root: &Path, path: &str) -> Reply {
    let relative = path.trim_start_matches('/');
    let relative = if relative.is_empty() { "index.html" } else { relative };
    let plain = Path::new(relative).components().all(|c| matches!(c, Component::Normal(n) if !n.to_string_lossy().starts_with('.')));
    if !plain {
        return reply(StatusCode::NOT_FOUND, "text/plain", "not found");
    }
    let full = root.join(relative);
    // A link out of the folder is refused: the resolved path must still be under it
    let inside = match (full.canonicalize(), root.canonicalize()) {
        (Ok(f), Ok(r)) => f.starts_with(r) && f.is_file(),
        _ => false,
    };
    if !inside {
        return reply(StatusCode::NOT_FOUND, "text/plain", "not found");
    }
    let content_type = match full.extension().and_then(|e| e.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("wasm") => "application/wasm",
        Some("css") => "text/css; charset=utf-8",
        Some("json") => "application/json",
        Some("png") => "image/png",
        Some("svg") => "image/svg+xml",
        Some("ttf") => "font/ttf",
        _ => "application/octet-stream",
    };
    match std::fs::read(&full) {
        Ok(bytes) => reply(StatusCode::OK, content_type, bytes),
        Err(_) => reply(StatusCode::NOT_FOUND, "text/plain", "not found"),
    }
}

/// Answers the WebSocket handshake and hands the connection to [`session`].
fn upgrade(request: &mut Request<Incoming>, state: Arc<State>) -> Reply {
    let header = |name: &str| request.headers().get(name).and_then(|v| v.to_str().ok()).map(str::to_ascii_lowercase);
    let wants = header("upgrade").is_some_and(|v| v == "websocket");
    let Some(key) = request.headers().get("sec-websocket-key").cloned().filter(|_| wants) else {
        return reply(StatusCode::BAD_REQUEST, "text/plain", "a WebSocket is expected here");
    };
    let accept = tokio_tungstenite::tungstenite::handshake::derive_accept_key(key.as_bytes());
    let upgraded = hyper::upgrade::on(request);
    tokio::spawn(async move {
        match upgraded.await {
            Ok(io) => {
                let config = tokio_tungstenite::tungstenite::protocol::WebSocketConfig::default().max_message_size(Some(MAX_WEB_FRAME_BYTES)).max_frame_size(Some(MAX_WEB_FRAME_BYTES));
                let socket = WebSocketStream::from_raw_socket(TokioIo::new(io), Role::Server, Some(config)).await;
                if let Err(e) = session(socket, state).await {
                    tracing::debug!("web: session ended: {e}");
                }
            }
            Err(e) => tracing::debug!("web: upgrade failed: {e}"),
        }
    });
    Response::builder()
        .status(StatusCode::SWITCHING_PROTOCOLS)
        .header("upgrade", "websocket")
        .header("connection", "Upgrade")
        .header("sec-websocket-accept", accept)
        .body(Full::new(Bytes::new()))
        .expect("a response")
}

fn web_error(e: impl std::fmt::Display) -> StreamError {
    StreamError::Protocol(e.to_string())
}

async fn send_text<S>(socket: &mut WebSocketStream<S>, message: &impl Serialize) -> StreamResult<()>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let text = serde_json::to_string(message).map_err(web_error)?;
    socket.send(Message::text(text)).await.map_err(web_error)
}

async fn next_text<S, M: for<'de> Deserialize<'de>>(socket: &mut WebSocketStream<S>) -> StreamResult<M>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    loop {
        match socket.next().await {
            Some(Ok(Message::Text(text))) => return serde_json::from_str(text.as_str()).map_err(web_error),
            Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
            Some(Ok(_)) => return Err(StreamError::protocol("a text frame was expected")),
            Some(Err(e)) => return Err(web_error(e)),
            None => return Err(StreamError::protocol("the page left")),
        }
    }
}

/// Who the page is: by its token, or by pairing now (it then gets a token).
async fn identify<S>(socket: &mut WebSocketStream<S>, state: &State) -> StreamResult<Option<Peer>>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    match next_text::<_, WebHello>(socket).await? {
        WebHello::Token { token } => Ok(state.authority.authorize_token(&token)),
        WebHello::Pair { pair } => {
            let pairing = match state.authority.pairing_start(&pair) {
                Ok((pairing, reply)) => {
                    send_text(socket, &reply).await?;
                    pairing
                }
                Err(refusal) => {
                    send_text(socket, &refusal).await?;
                    return Ok(None);
                }
            };
            let proof: PairToken = next_text(socket).await?;
            let reply = state.authority.pairing_finish_token(pairing, &proof, &state.names);
            let peer = match &reply {
                PairReply::Token { token, .. } => state.authority.authorize_token(token),
                _ => None,
            };
            send_text(socket, &reply).await?;
            Ok(peer)
        }
    }
}

/// One page: who it is, then its requests, each answered with its id.
async fn session<S>(mut socket: WebSocketStream<S>, state: Arc<State>) -> StreamResult<()>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let peer = match tokio::time::timeout(HELLO_TIMEOUT, identify(&mut socket, &state)).await {
        Ok(Ok(Some(peer))) => peer,
        Ok(Ok(None)) => {
            let _ = socket.close(None).await;
            return Err(StreamError::protocol("not let in"));
        }
        Ok(Err(e)) => return Err(e),
        Err(_) => return Err(StreamError::protocol("the page did not say who it is")),
    };
    let device = state.device.clone();
    send_text(&mut socket, &WebWelcome { server_name: device.name.clone(), server_version: device.version.clone(), peer_name: peer.name.clone(), permission: peer.permission, has_files: device.files.is_some() }).await?;
    tracing::info!("web: {} connected ({})", peer.name, peer.permission);

    let current: Current = Arc::new(Mutex::new(device.recording.clone()));
    let peer = Arc::new(peer);
    let (mut sink, mut stream) = socket.split();
    // Answers are computed concurrently and written by one task, in the order they finish
    let (answers, mut ready) = tokio::sync::mpsc::channel::<Vec<u8>>(64);
    let writer = tokio::spawn(async move {
        while let Some(frame) = ready.recv().await {
            if sink.send(Message::binary(frame)).await.is_err() {
                break;
            }
        }
    });
    while let Some(frame) = stream.next().await {
        let bytes = match frame.map_err(web_error)? {
            Message::Binary(bytes) => bytes,
            Message::Close(_) => break,
            _ => continue,
        };
        if bytes.len() < 4 {
            break;
        }
        let id: [u8; 4] = bytes[..4].try_into().expect("four bytes");
        let request = wire::Request::decode(&bytes[4..]);
        let (device, current, peer, answers) = (device.clone(), current.clone(), peer.clone(), answers.clone());
        tokio::spawn(async move {
            let kind = match request {
                // Files are downloaded by the desktop clients; a page reads views
                Ok(wire::Request { kind: Some(wire::request::Kind::Fetch(_)) }) => wire::response::Kind::Error("downloads are not offered to browsers".into()),
                Ok(request) => answer(request, &device, &current, &peer).await.unwrap_or_else(|e| wire::response::Kind::Error(e.reason())),
                Err(e) => wire::response::Kind::Error(format!("not a request: {e}")),
            };
            let response = wire::Response { kind: Some(kind) };
            let mut frame = Vec::with_capacity(4 + response.encoded_len());
            frame.extend_from_slice(&id);
            response.encode(&mut frame).expect("the buffer grows");
            let _ = answers.send(frame).await;
        });
    }
    drop(answers);
    writer.abort();
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use dspk_core::MemoryRecording;
    use dspk_view::View;
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;

    use super::*;
    use crate::pair::Permission;
    use crate::protocol::convert::{envelope_from_frame, request_from_view};
    use crate::transport::pairing::ClientPairing;
    use crate::transport::server::{OpenRecording, ServeOptions};

    type Socket = WebSocketStream<tokio_rustls::client::TlsStream<tokio::net::TcpStream>>;

    /// A page's WebSocket to the server at `addr`, trusting the authority's CA.
    async fn connect(addr: SocketAddr, authority: &Authority) -> Socket {
        let mut roots = rustls::RootCertStore::empty();
        roots.add(rustls::pki_types::CertificateDer::from(authority.ca_certificate().to_vec())).unwrap();
        let config = rustls::ClientConfig::builder().with_root_certificates(roots).with_no_client_auth();
        let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
        let name = rustls::pki_types::ServerName::try_from("localhost").unwrap();
        let tls = tokio_rustls::TlsConnector::from(Arc::new(config)).connect(name, tcp).await.unwrap();
        tokio_tungstenite::client_async("wss://localhost/ws", tls).await.unwrap().0
    }

    async fn text<M: for<'de> Deserialize<'de>>(socket: &mut Socket) -> Option<M> {
        match socket.next().await? {
            Ok(Message::Text(t)) => serde_json::from_str(t.as_str()).ok(),
            _ => None,
        }
    }

    async fn say(socket: &mut Socket, message: &impl Serialize) {
        socket.send(Message::text(serde_json::to_string(message).unwrap())).await.unwrap();
    }

    /// A page pairs with a code and gets a token; with the token it comes back and reads a view,
    /// answered under the id it chose; a wrong token and a revoked one are not let in.
    #[tokio::test]
    async fn a_page_pairs_then_reads_views_with_its_token() {
        let dir = std::env::temp_dir().join(format!("dspk-stream-web-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let authority = Arc::new(Authority::open_or_create(&dir).unwrap());
        let (channels, samples) = (4usize, 2_000usize);
        let data: Vec<f32> = (0..channels * samples).map(|i| (i % 97) as f32).collect();
        let source = Arc::new(MemoryRecording::new("rec", data.clone(), channels, 1000.0).unwrap());
        let device = Arc::new(Device { name: "dspk".into(), version: "test".into(), files: None, recording: Some(Arc::new(OpenRecording::new(source, None))), jobs: None, runtime: String::new(), options: ServeOptions::default() });
        let names = vec!["localhost".to_string()];
        let server = WebServer::bind("127.0.0.1:0".parse().unwrap(), authority.clone(), device, names.clone(), None, WebTls::Authority).await.unwrap();
        let addr = server.local_addr().unwrap();
        tokio::spawn(server.run());

        // Pairing: the code is typed once, the page keeps a token
        let code = authority.new_pairing_code(Permission::Read).unwrap();
        let mut socket = connect(addr, &authority).await;
        let (client, hello) = ClientPairing::start(&code.display(), "browser").unwrap();
        say(&mut socket, &WebHello::Pair { pair: hello }).await;
        let (client, proof) = client.token(text(&mut socket).await.unwrap()).unwrap();
        say(&mut socket, &proof).await;
        let paired = client.finish(text(&mut socket).await.unwrap()).unwrap();
        let welcome: WebWelcome = text(&mut socket).await.unwrap();
        assert_eq!((welcome.peer_name.as_str(), welcome.permission, paired.names.clone()), ("browser", Permission::Read, names));
        drop(socket);

        // Later: the token, then a view under the page's own id
        let mut socket = connect(addr, &authority).await;
        say(&mut socket, &WebHello::Token { token: paired.token.clone() }).await;
        let _: WebWelcome = text(&mut socket).await.unwrap();
        let view = View { channels: vec![1, 3], start: 100, end: 110, width: 64 };
        let request = wire::Request { kind: Some(wire::request::Kind::View(request_from_view(&view).unwrap())) };
        let mut frame = 7u32.to_be_bytes().to_vec();
        request.encode(&mut frame).unwrap();
        socket.send(Message::binary(frame)).await.unwrap();
        let answer = loop {
            match socket.next().await.unwrap().unwrap() {
                Message::Binary(b) => break b,
                _ => continue,
            }
        };
        assert_eq!(&answer[..4], &7u32.to_be_bytes(), "the answer carries the request's id");
        let response = wire::Response::decode(&answer[4..]).unwrap();
        let Some(wire::response::Kind::Envelope(envelope)) = response.kind else { panic!("a view answers with an envelope") };
        let dspk_view::Envelope::Samples(got) = envelope_from_frame(envelope).unwrap() else { panic!("ten samples fit the width: raw samples") };
        let expected: Vec<f32> = [1usize, 3].iter().flat_map(|c| data[c * samples + 100..c * samples + 110].to_vec()).collect();
        assert_eq!(got, expected);
        drop(socket);

        // Not let in: a wrong token, then the revoked one
        let mut socket = connect(addr, &authority).await;
        say(&mut socket, &WebHello::Token { token: "nope".into() }).await;
        assert!(text::<WebWelcome>(&mut socket).await.is_none());
        authority.revoke("browser").unwrap();
        let mut socket = connect(addr, &authority).await;
        say(&mut socket, &WebHello::Token { token: paired.token }).await;
        assert!(text::<WebWelcome>(&mut socket).await.is_none(), "revoked");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn files_are_served_only_from_inside_the_web_folder() {
        let dir = std::env::temp_dir().join(format!("dspk-stream-webfiles-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("web/pkg")).unwrap();
        std::fs::write(dir.join("web/index.html"), "<html>").unwrap();
        std::fs::write(dir.join("web/pkg/app.wasm"), [0u8, 97, 115, 109]).unwrap();
        std::fs::write(dir.join("secret.txt"), "no").unwrap();
        let root = dir.join("web");
        let status = |p: &str| file(&root, p).status();
        assert_eq!(status("/"), StatusCode::OK, "the folder's index");
        assert_eq!(file(&root, "/pkg/app.wasm").headers()["content-type"], "application/wasm");
        for outside in ["/../secret.txt", "/pkg/../../secret.txt", "/.hidden", "/absent.js"] {
            assert_eq!(status(outside), StatusCode::NOT_FOUND, "{outside}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
