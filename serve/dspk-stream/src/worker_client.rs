//! A page's connection to its engine worker (`dspk-engine-web`), for a page that runs alone
//! (GitHub Pages): the same protobuf requests as a server gets, sent to a dedicated web worker
//! with `postMessage` as `{id, bytes}` and answered as `{id, bytes}`. It is a [`Link`], so a
//! recording the worker opened is a [`crate::RemoteSignal`] like a server's.
//!
//! The worker script (`engine/engine-worker.js` next to the page) loads the engine's module and
//! answers with `Engine::answer`; [`WorkerLink::set_files`] hands it the files the user picked.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use futures_util::StreamExt as _;
use prost::Message as _;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast as _, JsValue};

use crate::error::{StreamError, StreamResult};
use crate::protocol::wire;
use crate::remote::{lock, Link};

type Answer = Result<wire::Response, String>;

/// Where a worker of the page (a view being drawn) waits for its answer.
#[derive(Default)]
struct Slot {
    answer: Mutex<Option<Answer>>,
    arrived: std::sync::Condvar,
}

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

/// What the page's main thread sends the engine worker.
enum Outgoing {
    Request(u32, Vec<u8>),
    Files(js_sys::Array),
}

// SAFETY: an `Outgoing::Files` array is made and sent on the page's main thread only
// (`set_files` is called there); requests carry plain bytes from any thread.
unsafe impl Send for Outgoing {}

/// The engine worker, as the page's threads see it.
pub struct WorkerLink {
    out: futures_channel::mpsc::UnboundedSender<Outgoing>,
    pending: Arc<Mutex<HashMap<u32, Waiter>>>,
    next: AtomicU32,
    /// Why the engine cannot process views ("" when it can).
    compute: String,
}

fn kind_of(answer: Answer) -> StreamResult<wire::response::Kind> {
    match answer.map_err(StreamError::protocol)?.kind {
        Some(wire::response::Kind::Error(e)) => Err(StreamError::Server(e)),
        Some(kind) => Ok(kind),
        None => Err(StreamError::protocol("the engine sent an empty answer")),
    }
}

impl WorkerLink {
    /// Why the engine cannot run ops here ("" when it can: the browser gave it WebGPU).
    pub fn compute_unavailable(&self) -> &str {
        &self.compute
    }

    /// Hands the engine the files the user picked (on the page's main thread).
    pub fn set_files(&self, files: js_sys::Array) -> StreamResult<()> {
        self.out.unbounded_send(Outgoing::Files(files)).map_err(|_| StreamError::protocol("the engine worker stopped"))
    }

    fn send(&self, kind: wire::request::Kind, waiter: Waiter) -> StreamResult<()> {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        lock(&self.pending).insert(id, waiter);
        let bytes = wire::Request { kind: Some(kind) }.encode_to_vec();
        self.out.unbounded_send(Outgoing::Request(id, bytes)).map_err(|_| StreamError::protocol("the engine worker stopped"))
    }

    /// Asks the engine without waiting (the page's main thread, or any task).
    pub async fn request_async(&self, kind: wire::request::Kind) -> StreamResult<wire::response::Kind> {
        let (tx, rx) = futures_channel::oneshot::channel();
        self.send(kind, Waiter::Task(tx))?;
        kind_of(rx.await.map_err(|_| StreamError::protocol("the engine worker stopped"))?)
    }
}

impl Link for WorkerLink {
    /// Waits for the answer: from a thread of a build with threads only (the page's main thread
    /// may not wait).
    fn request(&self, kind: wire::request::Kind) -> StreamResult<wire::response::Kind> {
        if cfg!(not(target_feature = "atomics")) {
            return Err(StreamError::protocol("a recording is read from a thread of the page: this build has none"));
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

/// Starts the engine worker from `script` (a module worker script URL, relative to the page) and
/// waits until it is ready: it has opened the browser's GPU for computing, or knows it cannot.
pub async fn start(script: &str) -> StreamResult<Arc<WorkerLink>> {
    let js = |e: JsValue| StreamError::protocol(format!("{e:?}"));
    let options = web_sys::WorkerOptions::new();
    options.set_type(web_sys::WorkerType::Module);
    let worker = web_sys::Worker::new_with_options(script, &options).map_err(js)?;
    let pending: Arc<Mutex<HashMap<u32, Waiter>>> = Arc::default();
    let (ready_tx, mut ready_rx) = futures_channel::mpsc::unbounded::<String>();

    let waiting = pending.clone();
    let on_message = Closure::<dyn FnMut(web_sys::MessageEvent)>::new(move |e: web_sys::MessageEvent| {
        let data = e.data();
        let get = |k: &str| js_sys::Reflect::get(&data, &k.into()).unwrap_or(JsValue::UNDEFINED);
        // {ready: "<why ops cannot run>" or ""}
        if let Some(why) = get("ready").as_string() {
            let _ = ready_tx.unbounded_send(why);
            return;
        }
        let (Some(id), Ok(buffer)) = (get("id").as_f64(), get("bytes").dyn_into::<js_sys::ArrayBuffer>()) else { return };
        let bytes = js_sys::Uint8Array::new(&buffer).to_vec();
        let waiter = lock(&waiting).remove(&(id as u32));
        if let Some(waiter) = waiter {
            waiter.deliver(wire::Response::decode(bytes.as_slice()).map_err(|e| format!("the engine's answer could not be read: {e}")));
        }
    });
    worker.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
    on_message.forget();
    let failed = pending.clone();
    let on_error = Closure::<dyn FnMut(web_sys::ErrorEvent)>::new(move |e: web_sys::ErrorEvent| {
        let waiting: Vec<Waiter> = lock(&failed).drain().map(|(_, w)| w).collect();
        for w in waiting {
            w.deliver(Err(format!("the engine worker failed: {}", e.message())));
        }
    });
    worker.set_onerror(Some(on_error.as_ref().unchecked_ref()));
    on_error.forget();

    let compute = ready_rx.next().await.ok_or_else(|| StreamError::protocol("the engine worker did not start"))?;
    let (out, mut outgoing) = futures_channel::mpsc::unbounded::<Outgoing>();
    // The one sender: messages from any thread leave from the page's main thread. The page's
    // memory may be shared with its threads, and shared bytes cannot be transferred, so each
    // request is copied into a buffer of its own, which is then transferred (not copied again)
    wasm_bindgen_futures::spawn_local(async move {
        while let Some(message) = outgoing.next().await {
            let sent = match message {
                Outgoing::Request(id, bytes) => {
                    let copy = js_sys::Uint8Array::new_with_length(bytes.len() as u32);
                    copy.copy_from(&bytes);
                    let message = js_sys::Object::new();
                    let _ = js_sys::Reflect::set(&message, &"id".into(), &JsValue::from(id));
                    let _ = js_sys::Reflect::set(&message, &"bytes".into(), &copy.buffer());
                    worker.post_message_with_transfer(&message, &js_sys::Array::of1(&copy.buffer()))
                }
                Outgoing::Files(files) => {
                    let message = js_sys::Object::new();
                    let _ = js_sys::Reflect::set(&message, &"files".into(), &files);
                    worker.post_message(&message)
                }
            };
            if sent.is_err() {
                break;
            }
        }
    });
    Ok(Arc::new(WorkerLink { out, pending, next: AtomicU32::new(1), compute }))
}
