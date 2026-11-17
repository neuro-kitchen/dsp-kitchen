//! Servers in a browser tab: one, named by a profile only. A page `dspk serve` served talks to
//! that server through its WebSocket (`dspk_stream::web_client`), keeping its token in the
//! browser's storage and asking for a pairing code when it has none. A page that runs alone
//! (GitHub Pages) starts the engine in a worker of its own (`dspk_stream::worker_client`) and
//! talks to it the same way: the files the user picks are its "server's" files. Downloads and
//! jobs are not offered to a page yet.

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use dspk_stream::pair::{WebWelcome, WS_PATH};
use dspk_stream::web_client::{self, Hello, WebLink};
use dspk_stream::worker_client::{self, WorkerLink};
use dspk_stream::{Link, RemoteSignal};

use crate::data::jobs::{Cancel, JobMessage};
use crate::data::remote::{parse_uri, remember, wire, ServerProfile, DEFAULT_SERVER_NAME};

/// The name of the page's server among the app's servers (`dspk://server/<path>`).
pub const PAGE_SERVER: &str = "server";
/// Where the page keeps its token.
const TOKEN_KEY: &str = "dspk-token";
const NOT_YET: &str = "not offered to a page yet (the desktop app does it)";

/// Where the engine worker's script is, next to the page.
const ENGINE_SCRIPT: &str = "./engine/engine-worker.js";

/// What the page works with: the server that served it, or its own engine worker.
enum PageLink {
    Server(Arc<WebLink>),
    Engine(Arc<WorkerLink>),
}

static LINK: OnceLock<PageLink> = OnceLock::new();

fn link() -> anyhow::Result<Arc<dyn Link>> {
    match LINK.get() {
        Some(PageLink::Server(l)) => Ok(l.clone()),
        Some(PageLink::Engine(l)) => Ok(l.clone()),
        None => Err(anyhow::anyhow!("the page is not connected to a server or its engine")),
    }
}

async fn request(kind: wire::request::Kind) -> anyhow::Result<wire::response::Kind> {
    Ok(match LINK.get() {
        Some(PageLink::Server(l)) => l.request_async(kind).await?,
        Some(PageLink::Engine(l)) => l.request_async(kind).await?,
        None => anyhow::bail!("the page is not connected to a server or its engine"),
    })
}

/// Where this page's processing happens, for the device line (a page computes nothing itself).
pub fn processing_label() -> Option<String> {
    Some(match LINK.get() {
        Some(PageLink::Server(l)) => {
            let w = l.welcome();
            let runtime = if w.runtime.is_empty() { String::new() } else { format!(" ({})", w.runtime) };
            format!("Processing on the server {}{runtime}", w.server_name)
        }
        Some(PageLink::Engine(l)) if l.compute_unavailable().is_empty() => "Processing on this browser's GPU (WebGPU)".into(),
        Some(PageLink::Engine(l)) => format!("Views only: {}", l.compute_unavailable()),
        None => "Not connected: no processing yet".into(),
    })
}

/// Whether the page runs alone, with its engine worker (no server).
pub fn runs_alone() -> bool {
    matches!(LINK.get(), Some(PageLink::Engine(_)))
}

/// Starts the engine worker for a page no server served. Returns why ops cannot run here ("" when
/// they can: the browser gave the engine WebGPU).
pub async fn start_engine() -> anyhow::Result<String> {
    if let Some(PageLink::Engine(l)) = LINK.get() {
        return Ok(l.compute_unavailable().to_string());
    }
    let engine = worker_client::start(ENGINE_SCRIPT).await?;
    let why = engine.compute_unavailable().to_string();
    let _ = LINK.set(PageLink::Engine(engine));
    Ok(why)
}

/// Asks the user for a folder (`folder`) or files, hands every file to the engine worker with
/// its path inside the folder, and returns the recordings among them (each `.bin` whose `.meta`
/// came too), by path, and, when there is none, what was picked instead. A dialog closed without
/// a choice fires no event: the page simply waits for the next one.
pub async fn pick_files(folder: bool) -> anyhow::Result<(Vec<String>, Option<String>)> {
    let Some(PageLink::Engine(engine)) = LINK.get() else { anyhow::bail!("picking files needs the page's engine") };
    let js = |e: wasm_bindgen::JsValue| anyhow::anyhow!("{e:?}");
    // Each picked file with its path "<folder picked>/<path inside>" (or its name alone)
    let picked: Vec<(String, web_sys::File)> = match folder.then(folder_picker).flatten() {
        // The loader's folder picker (File System Access API: "Let site view files?")
        Some(pick) => {
            let promise: js_sys::Promise = wasm_bindgen::JsCast::dyn_into(pick.call0(&wasm_bindgen::JsValue::NULL).map_err(js)?).map_err(|_| anyhow::anyhow!("the folder picker gave no promise"))?;
            let entries = wasm_bindgen_futures::JsFuture::from(promise).await.map_err(js)?;
            if entries.is_null() {
                return Ok((Vec::new(), None));
            }
            js_sys::Array::from(&entries)
                .iter()
                .filter_map(|e| {
                    let path = js_sys::Reflect::get(&e, &"path".into()).ok()?.as_string()?;
                    let file = wasm_bindgen::JsCast::dyn_into::<web_sys::File>(js_sys::Reflect::get(&e, &"file".into()).ok()?).ok()?;
                    Some((path, file))
                })
                .collect()
        }
        None => input_files(folder).await?,
    };
    let (files, mut paths) = (js_sys::Array::new(), Vec::new());
    for (relative, file) in picked {
        // "<folder>/<path>": the folder's own name is left out, unless the folder picked is a
        // Zarr recording itself (then it is the recording's name)
        let path = match relative.split_once('/') {
            Some((top, _)) if top.to_ascii_lowercase().ends_with(".zarr") => relative.clone(),
            Some((_, inside)) if !inside.is_empty() => inside.to_string(),
            _ => file.name(),
        };
        let entry = js_sys::Object::new();
        js_sys::Reflect::set(&entry, &"path".into(), &path.clone().into()).map_err(js)?;
        js_sys::Reflect::set(&entry, &"file".into(), &file).map_err(js)?;
        files.push(&entry);
        paths.push(path);
    }
    engine.set_files(files)?;
    let mut recordings: Vec<String> = paths.iter().filter(|p| p.to_ascii_lowercase().ends_with(".bin") && paths.contains(&p[..p.len() - 4].to_string().add_meta())).cloned().collect();
    // WAV files: their header says the layout
    recordings.extend(paths.iter().filter(|p| p.to_ascii_lowercase().ends_with(".wav")).cloned());
    // dspk Zarr recordings: a `.zarr` folder with a `/traces` array (an NWB folder is not one)
    for p in &paths {
        if let Some(folder) = p.strip_suffix("/traces/zarr.json").filter(|f| f.to_ascii_lowercase().ends_with(".zarr")) {
            recordings.push(folder.to_string());
        }
    }
    recordings.sort();
    recordings.dedup();
    let note = recordings.is_empty().then(|| not_a_recording(&paths));
    Ok((recordings, note))
}

/// Asks for a folder and reads every file in it into memory (awaited, one by one): what a
/// browser page does with a small folder it computes on (a `.sorting.zarr`). The folder's name,
/// and each file by its path inside it; `None` when the dialog was closed.
pub async fn pick_folder_bytes() -> anyhow::Result<Option<(String, Vec<(String, Vec<u8>)>)>> {
    let js = |e: wasm_bindgen::JsValue| anyhow::anyhow!("{e:?}");
    let picked: Vec<(String, web_sys::File)> = match folder_picker() {
        Some(pick) => {
            let promise: js_sys::Promise = wasm_bindgen::JsCast::dyn_into(pick.call0(&wasm_bindgen::JsValue::NULL).map_err(js)?).map_err(|_| anyhow::anyhow!("the folder picker gave no promise"))?;
            let entries = wasm_bindgen_futures::JsFuture::from(promise).await.map_err(js)?;
            if entries.is_null() {
                return Ok(None);
            }
            js_sys::Array::from(&entries)
                .iter()
                .filter_map(|e| {
                    let path = js_sys::Reflect::get(&e, &"path".into()).ok()?.as_string()?;
                    let file = wasm_bindgen::JsCast::dyn_into::<web_sys::File>(js_sys::Reflect::get(&e, &"file".into()).ok()?).ok()?;
                    Some((path, file))
                })
                .collect()
        }
        None => input_files(true).await?,
    };
    let Some(root) = picked.first().and_then(|(p, _)| p.split('/').next()).map(str::to_string) else { return Ok(None) };
    let mut files = Vec::with_capacity(picked.len());
    for (path, file) in picked {
        let inside = path.split_once('/').map_or(path.clone(), |(_, rest)| rest.to_string());
        let buffer = wasm_bindgen_futures::JsFuture::from(file.array_buffer()).await.map_err(js)?;
        files.push((inside, js_sys::Uint8Array::new(&buffer).to_vec()));
    }
    Ok(Some((root, files)))
}

/// Why none of `paths` is a recording here: the folders dspk writes next to a recording (its
/// derived store, epochs, a sorting), NWB, or nothing it reads.
fn not_a_recording(paths: &[String]) -> String {
    let folder = |suffix: &str| {
        paths.iter().find_map(|p| {
            let top = p.split('/').next().unwrap_or(p);
            top.to_ascii_lowercase().ends_with(suffix).then(|| top.to_string())
        })
    };
    if let Some(f) = folder(".dspk.zarr").or_else(|| folder(".epochs.zarr")) {
        return format!("{f} holds a recording's epochs and events, not the recording: open the folder that holds the recording");
    }
    if let Some(f) = folder(".sorting.zarr") {
        return format!("{f} is a sorting: sortings open in the desktop app or the container for now");
    }
    if let Some(f) = folder(".nwb.zarr").or_else(|| paths.iter().find(|p| p.to_ascii_lowercase().ends_with(".nwb")).cloned()) {
        return format!("{f} is NWB: NWB opens in the desktop app or the container for now");
    }
    if paths.iter().any(|p| p.to_ascii_lowercase().ends_with(".bin")) {
        return "A .bin needs its .meta (dspk's JSON layout) beside it".to_string();
    }
    format!("No recording among the {} files picked (a .zarr folder, a .wav, or a .bin with its .meta)", paths.len())
}

/// The loader's folder picker (`window.dspkPickFolder`), where the browser has the File System
/// Access API.
fn folder_picker() -> Option<js_sys::Function> {
    let window = web_sys::window()?;
    let pick: wasm_bindgen::JsValue = js_sys::Reflect::get(&window, &"dspkPickFolder".into()).ok()?;
    wasm_bindgen::JsCast::dyn_into::<js_sys::Function>(pick).ok()
}

/// Files picked with the page's file input (every browser): a folder (`webkitdirectory`, the
/// browser says "upload", though nothing leaves the computer) or files. Each with its path
/// "<folder>/<path inside>" for a folder, else its name. A dialog closed without a choice fires no
/// event: the page waits for the next one.
async fn input_files(folder: bool) -> anyhow::Result<Vec<(String, web_sys::File)>> {
    let document = web_sys::window().and_then(|w| w.document()).ok_or_else(|| anyhow::anyhow!("no document"))?;
    let js = |e: wasm_bindgen::JsValue| anyhow::anyhow!("{e:?}");
    let input: web_sys::HtmlInputElement = wasm_bindgen::JsCast::dyn_into(document.create_element("input").map_err(js)?).map_err(|_| anyhow::anyhow!("not an input"))?;
    input.set_type("file");
    input.set_multiple(true);
    if folder {
        input.set_attribute("webkitdirectory", "").map_err(js)?;
    } else {
        input.set_accept(".wav,.bin,.meta");
    }
    let (tx, rx) = futures_channel::oneshot::channel::<()>();
    let tx = std::cell::RefCell::new(Some(tx));
    let changed = wasm_bindgen::closure::Closure::<dyn FnMut()>::new(move || {
        if let Some(tx) = tx.borrow_mut().take() {
            let _ = tx.send(());
        }
    });
    input.set_onchange(Some(wasm_bindgen::JsCast::unchecked_ref(changed.as_ref())));
    input.click();
    let _ = rx.await;
    drop(changed);
    let Some(list) = input.files() else { return Ok(Vec::new()) };
    Ok((0..list.length())
        .filter_map(|i| list.get(i))
        .map(|file| {
            let relative = js_sys::Reflect::get(&file, &"webkitRelativePath".into()).ok().and_then(|v| v.as_string()).filter(|r| !r.is_empty()).unwrap_or_else(|| file.name());
            (relative, file)
        })
        .collect())
}

trait AddMeta {
    fn add_meta(self) -> String;
}

impl AddMeta for String {
    fn add_meta(mut self) -> String {
        self.push_str(".meta");
        self
    }
}

fn storage() -> Option<web_sys::Storage> {
    web_sys::window()?.local_storage().ok()?
}

/// What the loader found: `true` unless it asked and the host said it is not `dspk serve`.
fn served_by_dspk() -> bool {
    let flag = web_sys::window()
        .and_then(|w| js_sys::Reflect::get(&w, &"dspkWeb".into()).ok())
        .and_then(|web| js_sys::Reflect::get(&web, &"server".into()).ok())
        .and_then(|s| s.as_bool());
    flag != Some(false)
}

/// `ws://` or `wss://` to the page's own host; `None` when the page was not served over HTTP(S).
fn socket_url() -> Option<String> {
    let location = web_sys::window()?.location();
    let scheme = match location.protocol().ok()?.as_str() {
        "https:" => "wss",
        "http:" => "ws",
        _ => return None,
    };
    Some(format!("{scheme}://{}{WS_PATH}", location.host().ok().unwrap_or_default()))
}

/// Connects the page to the server that served it (once, on the main thread, after the window
/// is open): with the token kept from an earlier visit, else with a pairing code the user types.
/// A page no `dspk serve` served (a plain file server, GitHub Pages) has no server: `Ok(None)`.
pub async fn connect_page() -> anyhow::Result<Option<WebWelcome>> {
    match LINK.get() {
        Some(PageLink::Server(l)) => return Ok(Some(l.welcome().clone())),
        Some(PageLink::Engine(_)) => return Ok(None),
        None => {}
    }
    // The loader asked the host whether `dspk serve` sent the page (`window.dspkWeb.server`); a
    // plain file server or GitHub Pages did not: no socket, no pairing code
    if !served_by_dspk() {
        return Ok(None);
    }
    let Some(url) = socket_url() else { return Ok(None) };
    let kept = storage().and_then(|s| s.get_item(TOKEN_KEY).ok().flatten()).filter(|t| !t.is_empty());
    if let Some(token) = kept {
        match web_client::connect(&url, Hello::Token(token)).await {
            Ok((link, _)) => return Ok(Some(keep(link))),
            Err(_) => {
                if let Some(s) = storage() {
                    let _ = s.remove_item(TOKEN_KEY);
                }
            }
        }
    }
    // Nothing answers the socket: a plain file server, not an error
    let probe = web_client::connect(&url, Hello::Token(String::new())).await;
    if matches!(&probe, Err(dspk_stream::StreamError::Protocol(m)) if m.starts_with(web_client::UNREACHED)) {
        return Ok(None);
    }
    let prompt = "Pairing code (shown by `dspk serve --pair`, or `dspk clients pair`)";
    let code = web_sys::window().and_then(|w| w.prompt_with_message(prompt).ok().flatten()).unwrap_or_default();
    if code.trim().is_empty() {
        anyhow::bail!("no pairing code was given: reload the page to try again");
    }
    // A name of its own: the server knows each paired browser by name
    let name = format!("browser-{:06x}", (js_sys::Math::random() * 16_777_216.0) as u32);
    let (link, token) = web_client::connect(&url, Hello::Code { code, name }).await?;
    if let (Some(token), Some(s)) = (token, storage()) {
        let _ = s.set_item(TOKEN_KEY, &token);
    }
    Ok(Some(keep(link)))
}

fn keep(link: Arc<WebLink>) -> WebWelcome {
    let welcome = link.welcome().clone();
    let _ = LINK.set(PageLink::Server(link));
    welcome
}

/// The page's server (or its engine) as a profile, so the screens that list servers show it.
pub fn page_profile() -> ServerProfile {
    let address = if runs_alone() { "this browser" } else { "this page's server" };
    ServerProfile { name: PAGE_SERVER.into(), address: address.into(), server_name: DEFAULT_SERVER_NAME.into(), identity: PathBuf::new() }
}

/// Where the functions below run in a browser: on the page's own executor, when awaited.
pub struct PageRuntime;
/// A task of [`PageRuntime::spawn`]: awaiting it runs it (it never fails to join; the error type
/// matches a tokio task's, so callers are the same on both targets).
pub struct Task<T>(std::pin::Pin<Box<dyn std::future::Future<Output = T>>>);

impl PageRuntime {
    pub fn spawn<F: std::future::Future + 'static>(&self, future: F) -> Task<F::Output> {
        Task(Box::pin(future))
    }
}

impl<T> std::future::Future for Task<T> {
    type Output = Result<T, String>;
    fn poll(mut self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Self::Output> {
        self.0.as_mut().poll(cx).map(Ok)
    }
}

pub fn runtime() -> &'static PageRuntime {
    &PageRuntime
}

/// Opens the `dspk://` recording `path` on the page's server (from a worker: it waits). The page
/// has one connection, so one recording of the server is open at a time.
pub fn open_uri(path: &Path) -> anyhow::Result<Arc<RemoteSignal>> {
    let (_, remote) = parse_uri(path).ok_or_else(|| anyhow::anyhow!("{} is not a dspk:// path", path.display()))?;
    // The engine's own words for a page on its own (it is not a server)
    let signal = Arc::new(RemoteSignal::open(link()?, &remote).map_err(|e| if runs_alone() { anyhow::anyhow!("this browser's engine: {}", e.reason()) } else { anyhow::Error::from(e) })?);
    remember(&signal);
    Ok(signal)
}

pub async fn test(_profile: ServerProfile) -> anyhow::Result<wire::ServerInfo> {
    match LINK.get() {
        Some(PageLink::Server(l)) => {
            let w = l.welcome().clone();
            Ok(wire::ServerInfo { server_name: w.server_name, server_version: w.server_version, peer_name: w.peer_name, has_files: w.has_files, ..Default::default() })
        }
        Some(PageLink::Engine(_)) => Ok(wire::ServerInfo { server_name: "this browser".into(), has_files: true, ..Default::default() }),
        None => anyhow::bail!("the page is not connected to a server or its engine"),
    }
}

pub async fn list(_profile: ServerProfile, dir: String) -> anyhow::Result<wire::FileList> {
    match request(wire::request::Kind::ListFiles(wire::ListFiles { dir })).await? {
        wire::response::Kind::Files(list) => Ok(list),
        _ => anyhow::bail!("the server answered something else than a file list"),
    }
}

pub async fn fetch(_profile: ServerProfile, _path: String, _results: PathBuf) -> anyhow::Result<PathBuf> {
    anyhow::bail!(NOT_YET)
}

pub fn remote_sort(_profile: ServerProfile, _job: wire::Configure, _results: PathBuf, send: impl Fn(JobMessage) + Send + Sync + 'static) -> Cancel {
    send(JobMessage::Failed(NOT_YET.into()));
    Cancel::default()
}
