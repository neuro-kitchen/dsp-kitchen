//! A browser page's connection to the server that served it (`dspk serve --web`): one WebSocket
//! (`/ws`) carrying the same requests as a QUIC session, each a 4-byte id and a protobuf
//! `Request`, answered with the same id and a `Response`. It is a [`Link`], so a recording on the
//! server is a [`crate::RemoteSignal`] in a page as on a computer.
//!
//! - **Who the page is.** A token from an earlier visit, or a pairing code: both sides prove
//!   they know the code without sending it (`crate::pair`) and the server hands the page a token.
//!   Where the token is kept, and how the code is asked for, are the page's business.
//! - **Threads.** The socket belongs to the page's main thread, which never waits. A request made
//!   there is awaited ([`WebLink::request_async`]); one made on a worker (a view being drawn)
//!   waits for its answer ([`Link::request`]), which only a build with threads can do.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use futures_util::StreamExt as _;
use prost::Message as _;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast as _;

use crate::error::{StreamError, StreamResult};
use crate::pair::{ClientPairing, PairReply, WebHello, WebWelcome};
use crate::protocol::wire;
use crate::remote::{lock, Link};

type Answer = Result<wire::Response, String>;

/// The beginning of the error [`connect`] gives when nothing answered at `url` (a page served
/// by a plain file server has no socket: not an error for the page).
pub const UNREACHED: &str = "could not reach";
const CLOSED: &str = "the connection to the server closed";

/// How the page says who it is.
pub enum Hello {
    /// A token from an earlier pairing.
    Token(String),
    /// A pairing code, and the name the server will know this page by.
    Code { code: String, name: String },
}

/// Where a worker waits for its answer (a worker may wait; the page's main thread, which
/// delivers the answer, only takes the lock for an instant).
#[derive(Default)]
struct Slot {
    answer: Mutex<Option<Answer>>,
    arrived: std::sync::Condvar,
}

/// Who waits for an answer: a task (awaiting), or a worker (waiting).
enum Waiter {
    Task(futures_channel::oneshot::Sender<Answer>),
    Worker(Arc<Slot>),
}

impl Waiter {
    fn deliver(self, answer: Answer) {
        match self {
            Waiter::Task(tx) => {
                let _ = tx.send(answer);
            }
            Waiter::Worker(slot) => {
                *lock(&slot.answer) = Some(answer);
                slot.arrived.notify_all();
            }
        }
    }
}

/// An open connection, once the server let the page in.
pub struct WebLink {
    /// Frames for the main thread to send.
    out: futures_channel::mpsc::UnboundedSender<Vec<u8>>,
    /// Who waits for which answer.
    pending: Mutex<HashMap<u32, Waiter>>,
    next: AtomicU32,
    welcome: WebWelcome,
}

fn kind_of(answer: Answer) -> StreamResult<wire::response::Kind> {
    match answer.map_err(StreamError::protocol)?.kind {
        Some(wire::response::Kind::Error(e)) => Err(StreamError::Server(e)),
        Some(kind) => Ok(kind),
        None => Err(StreamError::protocol("the server sent an empty answer")),
    }
}

impl WebLink {
    /// Who the server says the page is.
    pub fn welcome(&self) -> &WebWelcome {
        &self.welcome
    }

    fn send(&self, kind: wire::request::Kind, waiter: Waiter) -> StreamResult<()> {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        lock(&self.pending).insert(id, waiter);
        let mut frame = id.to_be_bytes().to_vec();
        frame.extend(wire::Request { kind: Some(kind) }.encode_to_vec());
        self.out.unbounded_send(frame).map_err(|_| StreamError::protocol(CLOSED))
    }

    /// Asks the server without waiting (from the page's main thread, or any task).
    pub async fn request_async(&self, kind: wire::request::Kind) -> StreamResult<wire::response::Kind> {
        let (tx, rx) = futures_channel::oneshot::channel();
        self.send(kind, Waiter::Task(tx))?;
        kind_of(rx.await.map_err(|_| StreamError::protocol(CLOSED))?)
    }

    /// Everyone waiting learns the connection is gone.
    fn closed(&self) {
        let waiting: Vec<Waiter> = lock(&self.pending).drain().map(|(_, w)| w).collect();
        for w in waiting {
            w.deliver(Err(CLOSED.into()));
        }
        self.out.close_channel();
    }
}

impl Link for WebLink {
    /// Waits for the answer: from a worker of a build with threads only (a page's main thread
    /// may not wait, and a build without threads has no workers).
    fn request(&self, kind: wire::request::Kind) -> StreamResult<wire::response::Kind> {
        if cfg!(not(target_feature = "atomics")) {
            return Err(StreamError::protocol("a recording on the server is read from a worker: this build of the page has none"));
        }
        let slot = Arc::new(Slot::default());
        self.send(kind, Waiter::Worker(slot.clone()))?;
        let mut answer = slot.answer.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(answer) = answer.take() {
                return kind_of(answer);
            }
            answer = slot.arrived.wait(answer).unwrap_or_else(|e| e.into_inner());
        }
    }

    fn spawn(&self, name: &str, work: Box<dyn FnOnce() + Send>) -> StreamResult<()> {
        #[cfg(target_feature = "atomics")]
        return wasm_thread::Builder::new().name(name.into()).spawn(work).map(|_| ()).map_err(|e| StreamError::protocol(format!("no web worker: {e}")));
        #[cfg(not(target_feature = "atomics"))]
        {
            let _ = (name, work);
            Err(StreamError::protocol("this build of the page has no web workers"))
        }
    }
}

/// What the socket delivers before the page is let in.
enum Event {
    Open,
    Text(String),
    Closed,
}

/// Connects to `url` (`ws://…/ws` or `wss://…/ws`, on the page's main thread) and says who the
/// page is. Returns the connection, and the new token when the page paired.
pub async fn connect(url: &str, hello: Hello) -> StreamResult<(Arc<WebLink>, Option<String>)> {
    let js = |e: wasm_bindgen::JsValue| StreamError::protocol(format!("{e:?}"));
    let socket = web_sys::WebSocket::new(url).map_err(js)?;
    socket.set_binary_type(web_sys::BinaryType::Arraybuffer);
    let (events, mut incoming) = futures_channel::mpsc::unbounded::<Event>();
    // Answers go straight to whoever waits (set once the page is let in)
    let ready: Arc<OnceLock<Arc<WebLink>>> = Arc::default();

    let (tx, to) = (events.clone(), ready.clone());
    let on_message = Closure::<dyn FnMut(web_sys::MessageEvent)>::new(move |e: web_sys::MessageEvent| {
        let data = e.data();
        if let Some(text) = data.as_string() {
            let _ = tx.unbounded_send(Event::Text(text));
        } else if let Ok(buffer) = data.dyn_into::<js_sys::ArrayBuffer>() {
            let bytes = js_sys::Uint8Array::new(&buffer).to_vec();
            let (Some(link), true) = (to.get(), bytes.len() >= 4) else { return };
            let id = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let waiting = lock(&link.pending).remove(&id);
            if let Some(waiting) = waiting {
                waiting.deliver(wire::Response::decode(&bytes[4..]).map_err(|e| format!("the server's answer could not be read: {e}")));
            }
        }
    });
    socket.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
    on_message.forget();
    let tx = events.clone();
    let on_open = Closure::<dyn FnMut()>::new(move || {
        let _ = tx.unbounded_send(Event::Open);
    });
    socket.set_onopen(Some(on_open.as_ref().unchecked_ref()));
    on_open.forget();
    let (tx, to) = (events.clone(), ready.clone());
    let on_close = Closure::<dyn FnMut()>::new(move || {
        let _ = tx.unbounded_send(Event::Closed);
        if let Some(link) = to.get() {
            link.closed();
        }
    });
    socket.set_onclose(Some(on_close.as_ref().unchecked_ref()));
    on_close.forget();

    match incoming.next().await {
        Some(Event::Open) => {}
        _ => return Err(StreamError::protocol(format!("{UNREACHED} {url}"))),
    }
    let mut next_text = async || match incoming.next().await {
        Some(Event::Text(t)) => Ok(t),
        _ => Err(StreamError::protocol("the server closed the connection")),
    };
    let send = |message: String| socket.send_with_str(&message).map_err(js);
    let json = |e: serde_json::Error| StreamError::protocol(e.to_string());
    let mut token = None;
    match hello {
        Hello::Token(t) => send(serde_json::to_string(&WebHello::Token { token: t }).map_err(json)?)?,
        Hello::Code { code, name } => {
            let (pairing, hello) = ClientPairing::start(&code, &name)?;
            send(serde_json::to_string(&WebHello::Pair { pair: hello }).map_err(json)?)?;
            let reply: PairReply = serde_json::from_str(&next_text().await?).map_err(json)?;
            let (pairing, proof) = pairing.token(reply)?;
            send(serde_json::to_string(&proof).map_err(json)?)?;
            let reply: PairReply = serde_json::from_str(&next_text().await?).map_err(json)?;
            token = Some(pairing.finish(reply)?.token);
        }
    }
    let welcome: WebWelcome = serde_json::from_str(&next_text().await.map_err(|_| StreamError::protocol("the server did not let the page in"))?).map_err(json)?;

    let (out, mut frames) = futures_channel::mpsc::unbounded::<Vec<u8>>();
    let link = Arc::new(WebLink { out, pending: Mutex::default(), next: AtomicU32::new(1), welcome });
    let _ = ready.set(link.clone());
    // The one sender: frames from any thread leave from here. The page's memory is shared with
    // its workers, and a socket refuses bytes that live in shared memory, so each frame is
    // copied into a buffer of its own first.
    wasm_bindgen_futures::spawn_local(async move {
        while let Some(frame) = frames.next().await {
            let copy = js_sys::Uint8Array::new_with_length(frame.len() as u32);
            copy.copy_from(&frame);
            if socket.send_with_array_buffer(&copy.buffer()).is_err() {
                break;
            }
        }
    });
    Ok((link, token))
}
