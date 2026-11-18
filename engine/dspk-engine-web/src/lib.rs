//! The dspk engine in a browser worker: what `dspk serve` is to a page served by a server, this
//! is to a page that runs alone (GitHub Pages). It runs in a dedicated web worker as a module of
//! its own, built without threads, so it can own the browser's WebGPU device (a device belongs to
//! the thread that made it) while the page draws on its own threads.
//!
//! - **Files.** The page hands over the `File`s the user picked; they are read here, in pieces,
//!   with the worker's synchronous reader (`FileReaderSync`: workers only), so no file is copied
//!   whole into memory.
//! - **Requests.** The page sends the same protobuf `Request`s it sends a server (views, list,
//!   open) and gets the same `Response`s: [`Engine::answer`]. The page's side is
//!   `dspk_stream::worker_client`.
//! - **Processing.** A view that carries ops is processed on WebGPU with the code of the app's
//!   preview (dspk-pipeline); the device's result is read back asynchronously
//!   (`ProcessedSignal::warm_view_async`), which is the only way a browser gives it.
#![cfg(target_family = "wasm")]

mod overview;

use std::sync::Arc;

use dspk_core::RecordingSource;
use dspk_io::{ByteSource, ByteSourceStore, RawParams, RawRecording, ZarrRecording};
use dspk_pipeline::processed::ProcessedChains;
use dspk_stream::engine::{answer_view, OpenRecording, ViewProcessor, WarmFuture};
use dspk_stream::protocol::{convert, wire};
use dspk_stream::{StreamError, StreamResult};
use dspk_view::{Pyramid, SignalBackend, StoredSignal, MEMORY_BASE, RAW_BLOCK_VALUES};
use prost::Message as _;
use wasm_bindgen::prelude::*;

/// A picked file, read at an offset with the worker's synchronous reader.
struct FileBytes(web_sys::File);

// SAFETY: this module is built without threads (no shared memory, no atomics): its one thread is
// the worker's, so the browser object is never touched from another thread.
unsafe impl Send for FileBytes {}
unsafe impl Sync for FileBytes {}

impl ByteSource for FileBytes {
    fn len(&self) -> u64 {
        self.0.size() as u64
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> std::io::Result<()> {
        let fail = |e: JsValue| std::io::Error::other(format!("{}: {e:?}", self.0.name()));
        let piece = self.0.slice_with_f64_and_f64(offset as f64, (offset + out.len() as u64) as f64).map_err(fail)?;
        let buffer = web_sys::FileReaderSync::new().map_err(fail)?.read_as_array_buffer(&piece).map_err(fail)?;
        let bytes = js_sys::Uint8Array::new(&buffer);
        if bytes.length() as usize != out.len() {
            return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof));
        }
        bytes.copy_to(out);
        Ok(())
    }
}

fn text_of(file: &web_sys::File) -> Result<String, String> {
    let reader = web_sys::FileReaderSync::new().map_err(|e| format!("{e:?}"))?;
    reader.read_as_text(file).map_err(|e| format!("{}: {e:?}", file.name()))
}

/// Runs a view's ops on the browser's WebGPU, with the chains kept so a view can be made ready
/// asynchronously before it is answered (the same code `dspk serve` uses).
struct Processor(ProcessedChains);

impl Default for Processor {
    fn default() -> Self {
        // Files the ops name (a remap's channel map) are not among a page's files yet
        Self(ProcessedChains::new(""))
    }
}

impl ViewProcessor for Processor {
    fn processed(&self, source: Arc<dyn RecordingSource>, pyramid: Option<Arc<Pyramid>>, channels: &[usize], ops: &[String]) -> StreamResult<Arc<dyn SignalBackend>> {
        let raw: Arc<dyn SignalBackend> = Arc::new(StoredSignal::new(source.clone(), pyramid));
        Ok(self.0.make(raw, source, channels, ops).map_err(StreamError::Server)?)
    }

    fn warm<'a>(&'a self, channels: &'a [usize], ops: &'a [String], view: &'a dspk_view::View) -> WarmFuture<'a> {
        Box::pin(async move { self.0.warm(channels, ops, view).await.map_err(|e| StreamError::Server(e.to_string())) })
    }
}

/// The engine of one page: the files it was given and the recording it opened.
struct Inner {
    /// The files picked, by their path inside the folder picked (or their name).
    /// By path: a picked folder can hold tens of thousands of files (an NWB-Zarr: ~20,000).
    files: std::collections::BTreeMap<String, web_sys::File>,
    current: Option<Arc<OpenRecording>>,
    processor: Arc<Processor>,
    /// Why ops cannot run here, when they cannot (no WebGPU for computing).
    no_compute: Option<String>,
    /// Bumped at each open: an overview still being built for an earlier recording stops.
    generation: u64,
    /// The open recording's overview while it builds (and after, for keeping it).
    overview: Option<overview::Overview>,
}

/// The engine, as the worker script holds it (`engine-worker.js`). Requests are answered one at
/// a time: the script queues them.
/// Runs the module's static constructors (the plugin registries of zarrs, CubeCL's IR): once,
/// called by the worker right after loading. A module that exports `_initialize` is a "reactor"
/// to the linker; without it the linker wraps every export to run them again on each call, which
/// cost seconds per request (29 s of 31 while opening a 19,754-file NWB-Zarr, 2026-11-18).
#[unsafe(no_mangle)]
pub extern "C" fn _initialize() {
    unsafe extern "C" {
        fn __wasm_call_ctors();
    }
    // SAFETY: called once, before anything else in the module runs
    unsafe { __wasm_call_ctors() }
}

#[wasm_bindgen]
pub struct Engine {
    inner: std::rc::Rc<std::cell::RefCell<Inner>>,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl Engine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Engine {
        console_error_panic_hook_set();
        let inner = Inner { files: std::collections::BTreeMap::new(), current: None, processor: Arc::default(), no_compute: Some("WebGPU not opened yet".into()), generation: 0, overview: None };
        let inner = std::rc::Rc::new(std::cell::RefCell::new(inner));
        ENGINE.with(|e| *e.borrow_mut() = Some(inner.clone()));
        Engine { inner }
    }

    /// Opens the browser's GPU for computing. Resolves to "" when ops can run, else why not
    /// (views of the recording still work).
    pub fn start(&self) -> js_sys::Promise {
        let inner = self.inner.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let why = dspk_core::compute::open_browser_gpu().await.err();
            inner.borrow_mut().no_compute = why.clone();
            Ok(JsValue::from_str(&why.unwrap_or_default()))
        })
    }

    /// Files the user picked, as `{path, file}` (the path inside the folder picked), added to
    /// earlier ones; a file of the same path replaces the earlier one.
    pub fn set_files(&self, files: js_sys::Array) {
        let mut inner = self.inner.borrow_mut();
        for entry in files.iter() {
            let path = js_sys::Reflect::get(&entry, &"path".into()).ok().and_then(|p| p.as_string());
            let file = js_sys::Reflect::get(&entry, &"file".into()).ok().and_then(|f| f.dyn_into::<web_sys::File>().ok());
            if let (Some(path), Some(file)) = (path, file) {
                inner.files.insert(path, file);
            }
        }
    }

    /// The overview's state as JSON (`null` before a recording is opened): its generation (one
    /// per open), pages built of all, whether it is complete and kept, its key, and whether the
    /// page's helpers can build it ([`overview`]).
    pub fn overview_state(&self) -> String {
        self.inner.borrow().overview.as_ref().map_or_else(|| "null".to_string(), |o| o.state().to_string())
    }

    /// What the helpers need (JSON), or "" when they cannot build this overview.
    pub fn overview_helper_meta(&self) -> String {
        self.inner.borrow().overview.as_ref().and_then(|o| o.helper_meta()).map(|m| m.to_string()).unwrap_or_default()
    }

    /// The helpers build the overview from now on (`true`), or this worker again (`false`).
    pub fn overview_use_helpers(&self, on: bool) {
        if let Some(o) = self.inner.borrow_mut().overview.as_mut() {
            o.helpers = on && o.plan.is_some();
            o.decided = true;
        }
    }

    /// The next unit for a helper, nearest the focus: `{generation, pages: [first, end],
    /// samples: [start, end], chunks: [{t0, c0, file}]}` (`file` null for a chunk not stored:
    /// all fill value), or null when every page is built or handed out.
    pub fn overview_next(&self) -> JsValue {
        let mut me = self.inner.borrow_mut();
        let Some(unit) = me.overview.as_mut().and_then(|o| o.next_unit()) else { return JsValue::NULL };
        let Some(o) = me.overview.as_ref() else { return JsValue::NULL };
        let chunks = js_sys::Array::new();
        for (t0, c0, path) in o.chunks_of(&unit.samples) {
            let c = js_sys::Object::new();
            let _ = js_sys::Reflect::set(&c, &"t0".into(), &(t0 as f64).into());
            let _ = js_sys::Reflect::set(&c, &"c0".into(), &(c0 as f64).into());
            let file = me.files.get(&path).map_or(JsValue::NULL, |f| f.clone().into());
            let _ = js_sys::Reflect::set(&c, &"file".into(), &file);
            chunks.push(&c);
        }
        let pair = |a: u64, b: u64| js_sys::Array::of2(&(a as f64).into(), &(b as f64).into());
        let out = js_sys::Object::new();
        let _ = js_sys::Reflect::set(&out, &"generation".into(), &(o.generation as f64).into());
        let _ = js_sys::Reflect::set(&out, &"pages".into(), &pair(unit.pages.start, unit.pages.end));
        let _ = js_sys::Reflect::set(&out, &"samples".into(), &pair(unit.samples.start, unit.samples.end));
        let _ = js_sys::Reflect::set(&out, &"chunks".into(), &chunks);
        out.into()
    }

    /// A helper's buckets of pages `first..end` of overview `generation` (see
    /// [`overview::Overview::put`]); resolves whether the overview is complete. A unit of an
    /// earlier recording is dropped.
    pub fn overview_put(&self, generation: u32, first: u32, end: u32, values: js_sys::Float32Array) -> Result<bool, JsValue> {
        let mut me = self.inner.borrow_mut();
        let Some(o) = me.overview.as_mut().filter(|o| o.generation == generation as u64) else { return Ok(false) };
        o.put(first as u64..end as u64, &values.to_vec()).map_err(|e| JsValue::from_str(&e))
    }

    /// A helper could not build pages `first..end`: they are free again.
    pub fn overview_failed(&self, generation: u32, first: u32, end: u32) {
        if let Some(o) = self.inner.borrow_mut().overview.as_mut().filter(|o| o.generation == generation as u64) {
            o.failed(first as u64..end as u64);
        }
    }

    /// The complete overview as bytes, to keep (`undefined` until complete).
    pub fn overview_bytes(&self) -> Option<js_sys::Uint8Array> {
        let me = self.inner.borrow();
        me.overview.as_ref().and_then(|o| o.pyramid.complete_bytes()).map(|b| js_sys::Uint8Array::from(b.as_slice()))
    }

    /// The overview kept from an earlier open of the same files: fills this one and marks it
    /// kept. Returns false (nothing changed) when it does not fit or building started.
    pub fn overview_restore(&self, bytes: js_sys::Uint8Array) -> bool {
        let mut me = self.inner.borrow_mut();
        let Some(o) = me.overview.as_mut() else { return false };
        let restored = o.pyramid.restore(&bytes.to_vec());
        o.kept |= restored;
        o.decided |= restored;
        restored
    }

    /// The worker script kept the complete overview.
    pub fn overview_mark_kept(&self) {
        if let Some(o) = self.inner.borrow_mut().overview.as_mut() {
            o.kept = true;
        }
    }

    /// Answers a protobuf `Request` with a protobuf `Response` (a `Uint8Array`; an error is a
    /// `Response` too).
    pub fn answer(&self, request: Vec<u8>) -> js_sys::Promise {
        let inner = self.inner.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let kind = match wire::Request::decode(request.as_slice()) {
                Ok(request) => Inner::answer_kind(&inner, request).await.unwrap_or_else(|e| wire::response::Kind::Error(e.reason())),
                Err(e) => wire::response::Kind::Error(format!("a request that could not be read: {e}")),
            };
            Ok(js_sys::Uint8Array::from(wire::Response { kind: Some(kind) }.encode_to_vec().as_slice()).into())
        })
    }
}

impl Inner {
    /// One request. The borrow of the engine ends before any await, so nothing is held across
    /// one (the script answers one request at a time anyway).
    async fn answer_kind(inner: &std::cell::RefCell<Inner>, request: wire::Request) -> StreamResult<wire::response::Kind> {
        use wire::request::Kind as Req;
        use wire::response::Kind as Resp;
        match request.kind.ok_or_else(|| StreamError::Server("empty request".into()))? {
            Req::ListFiles(list) => Ok(Resp::Files(inner.borrow().list(&list.dir))),
            Req::Open(open) => {
                let (recording, source, pyramid) = inner.borrow().open(&open.path)?;
                let recording = Arc::new(recording);
                let header = convert::header_from_info(recording.info());
                let generation = {
                    let mut me = inner.borrow_mut();
                    me.current = Some(recording);
                    me.generation += 1;
                    let generation = me.generation;
                    let (plan, key) = (me.helper_plan(&open.path, source.as_ref()), me.files_key(&open.path, source.as_ref()));
                    me.overview = Some(overview::Overview::new(generation, source, pyramid, plan, key));
                    generation
                };
                build_overview(inner_rc(inner), generation);
                Ok(Resp::Header(header))
            }
            Req::View(view) => {
                // The overview builds where the page looks next (a view of most of the recording is
                // the page asking how far it is: it moves nothing)
                if let Some(o) = inner.borrow_mut().overview.as_mut() {
                    let total = o.pyramid.samples();
                    if view.end > view.start && (view.end - view.start) * 2 < total {
                        o.focus = view.start + (view.end - view.start) / 2;
                    }
                }
                let (recording, no_compute) = {
                    let me = inner.borrow();
                    (me.current.clone().ok_or_else(|| StreamError::Server("no recording is open".into()))?, me.no_compute.clone())
                };
                if view.processing.as_ref().is_some_and(|p| !p.ops.is_empty()) {
                    if let Some(why) = no_compute {
                        return Err(StreamError::Server(format!("processing needs WebGPU: {why}")));
                    }
                    if let Some(why) = dspk_core::compute::browser_gpu_lost() {
                        return Err(StreamError::Server(format!("the browser took its GPU back ({why}): reload the page to process again; the recording itself still shows")));
                    }
                    dspk_stream::engine::warm_view(&recording, &view).await?;
                }
                Ok(Resp::Envelope(convert::frame_from_envelope(answer_view(&recording, &view))))
            }
            _ => Err(StreamError::Server("not offered by the page's engine (the desktop app or a server does it)".into())),
        }
    }

    /// The entries of `dir` among the files picked: its files, and its subfolders once each.
    fn list(&self, dir: &str) -> wire::FileList {
        let dir = dir.trim_matches('/');
        let prefix = if dir.is_empty() { String::new() } else { format!("{dir}/") };
        let mut entries: Vec<wire::FileEntry> = Vec::new();
        for (path, file) in &self.files {
            let Some(rest) = path.strip_prefix(&prefix) else { continue };
            match rest.split_once('/') {
                Some((sub, _)) => {
                    if !entries.iter().any(|e| e.directory && e.name == sub) {
                        // A Zarr recording is a folder (as `dspk serve` lists it): dspk's (a
                        // `/traces` array) or NWB's (an `/acquisition` group)
                        let zarr = sub.to_ascii_lowercase().ends_with(".zarr");
                        let recording = zarr && (self.file(&format!("{prefix}{sub}/traces/zarr.json")).is_some() || self.is_nwb(&format!("{prefix}{sub}")));
                        let kind = if recording { wire::FileKind::Recording } else if zarr { wire::FileKind::Other } else { wire::FileKind::Directory };
                        entries.push(wire::FileEntry { name: sub.to_string(), kind: kind as i32, size: 0, modified_unix: 0, directory: true });
                    }
                }
                None => {
                    // A `.bin` is a recording when its layout file came with it
                    let lower = rest.to_ascii_lowercase();
                    let recording = (lower.ends_with(".bin") && self.file(&format!("{}.meta", &path[..path.len() - 4])).is_some()) || dspk_io::generic::wav::is_wav_name(&lower);
                    let kind = if recording { wire::FileKind::Recording } else { wire::FileKind::Other };
                    entries.push(wire::FileEntry { name: rest.to_string(), kind: kind as i32, size: file.size() as u64, modified_unix: (file.last_modified() / 1000.0) as u64, directory: false });
                }
            }
        }
        entries.sort_by(|a, b| b.directory.cmp(&a.directory).then_with(|| a.name.cmp(&b.name)));
        wire::FileList { dir: dir.to_string(), entries }
    }

    /// How the page's helpers could build `name`'s overview themselves: an NWB-Zarr series stored
    /// as plain or gzip bytes ([`overview::plan_for`]); `None` otherwise.
    fn helper_plan(&self, name: &str, source: &dyn RecordingSource) -> Option<overview::Plan> {
        let info = source.info();
        let series = info.metadata.get("nwb_series")?;
        let prefix = format!("{}{}/data/", name.trim_end_matches('/'), series);
        let text = text_of(self.file(&format!("{prefix}zarr.json"))?).ok()?;
        let first = info.channels.first()?;
        // One scale for every channel (NWB has one per series)
        if info.channels.iter().any(|c| c.gain != first.gain || c.offset != first.offset) {
            return None;
        }
        overview::plan_for(&text, &prefix, first.gain as f64, first.offset as f64)
    }

    /// What identifies the files of recording `name` (a kept overview is used only for the same
    /// files): its shape and rate, and the count, total size and newest date of its files.
    fn files_key(&self, name: &str, source: &dyn RecordingSource) -> String {
        let prefix = name.trim_end_matches('/');
        let (mut count, mut bytes, mut newest) = (0u64, 0f64, 0f64);
        for (path, file) in self.files.range(prefix.to_string()..) {
            if !(path == prefix || path.starts_with(&format!("{prefix}/"))) {
                break;
            }
            count += 1;
            bytes += file.size();
            newest = newest.max(file.last_modified());
        }
        let info = source.info();
        format!("{prefix}|{}|{}|{}|{count}|{bytes}|{newest}|base {}", info.channel_count(), info.samples, info.sample_rate_hz(), MEMORY_BASE)
    }

    /// Whether the Zarr folder `name` is an NWB file (its root says `NWBFile`).
    fn is_nwb(&self, name: &str) -> bool {
        let root = format!("{}/zarr.json", name.trim_end_matches('/'));
        self.file(&root).and_then(|f| text_of(f).ok()).is_some_and(|t| t.contains("\"NWBFile\""))
    }

    /// The nodes under `/acquisition` of the Zarr folder `name`, from the picked files' paths.
    fn acquisition_of(&self, name: &str) -> Vec<String> {
        let prefix = format!("{}/acquisition/", name.trim_end_matches('/'));
        let mut out: Vec<String> = self.files.iter().filter_map(|(p, _)| p.strip_prefix(&prefix)?.split_once('/').map(|(node, _)| node.to_string())).collect();
        out.sort();
        out.dedup();
        out
    }

    fn file(&self, path: &str) -> Option<&web_sys::File> {
        self.files.get(path)
    }

    /// Opens a picked `.bin` with its `.meta` sidecar (dspk's JSON layout), and builds its
    /// overview here (the whole file is read once, in pieces).
    /// Opens `name` at once; its overview (the zoomed-out min / max levels) starts empty and is
    /// built in pieces between requests ([`build_overview`]), as `dspk serve` builds it in the
    /// background: views of parts not built yet say so and are asked again.
    fn open(&self, name: &str) -> StreamResult<(OpenRecording, Arc<dyn RecordingSource>, Arc<Pyramid>)> {
        let fail = |e: String| StreamError::Server(e);
        let lower = name.trim_end_matches('/').to_ascii_lowercase();
        let source = if lower.ends_with(".zarr") {
            self.open_zarr(name)?
        } else if dspk_io::generic::wav::is_wav_name(&lower) {
            self.open_wav(name)?
        } else {
            self.open_raw(name)?
        };
        let pyramid = Arc::new(Pyramid::in_memory(source.as_ref(), MEMORY_BASE).map_err(|e| fail(e.to_string()))?);
        let recording = OpenRecording::new(source.clone(), Some(pyramid.clone())).with_processor(self.processor.clone());
        Ok((recording, source, pyramid))
    }

    /// A Zarr recording: the picked files under the folder `name`, read by chunk through a
    /// read-only store over them (only the byte ranges zarrs asks for are read).
    fn open_zarr(&self, name: &str) -> StreamResult<Arc<dyn RecordingSource>> {
        let prefix = format!("{}/", name.trim_end_matches('/'));
        // dspk's layout (a `/traces` array), or NWB's (its largest continuous series in
        // `/acquisition`, read by the electrophysiology formats' reader)
        let nwb = self.file(&format!("{prefix}traces/zarr.json")).is_none();
        if nwb && !self.is_nwb(name) {
            return Err(StreamError::Server(format!("{name} is not a recording dspk reads (no /traces array, not NWB)")));
        }
        let files: Vec<(String, Arc<dyn ByteSource>)> =
            self.files.iter().filter_map(|(path, file)| path.strip_prefix(&prefix).map(|key| (key.to_string(), Arc::new(FileBytes(file.clone())) as Arc<dyn ByteSource>))).collect();
        if files.is_empty() {
            return Err(StreamError::Server(format!("{name} was not among the folders picked")));
        }
        let store: Arc<dyn zarrs::storage::ReadableStorageTraits + Send + Sync> = Arc::new(ByteSourceStore::new(files));
        if nwb {
            let recording = dspk_synapse::io::nwb::NwbZarrRecording::open_in(store, name.trim_end_matches('/'), &self.acquisition_of(name)).map_err(|e| StreamError::Server(format!("{name}: {e}")))?;
            return Ok(Arc::new(recording));
        }
        let recording = ZarrRecording::from_store(name.trim_end_matches('/'), store).map_err(|e| StreamError::Server(format!("{name}: {e}")))?;
        Ok(Arc::new(recording))
    }

    /// A `.bin` with its `.meta` (dspk's JSON layout).
    /// A WAV file: its header says the layout.
    fn open_wav(&self, name: &str) -> StreamResult<Arc<dyn RecordingSource>> {
        let fail = |e: String| StreamError::Server(e);
        let data = self.file(name).ok_or_else(|| fail(format!("{name} was not among the files picked")))?;
        Ok(Arc::new(dspk_io::wav_from_source(name, Arc::new(FileBytes(data.clone()))).map_err(|e| fail(e.to_string()))?))
    }

    fn open_raw(&self, name: &str) -> StreamResult<Arc<dyn RecordingSource>> {
        let fail = |e: String| StreamError::Server(e);
        let data = self.file(name).ok_or_else(|| fail(format!("{name} was not among the files picked")))?;
        let meta_name = format!("{}.meta", name.strip_suffix(".bin").unwrap_or(name));
        let meta = self.file(&meta_name).ok_or_else(|| fail(format!("{name} needs its layout file {meta_name} next to it")))?;
        let text = text_of(meta).map_err(fail)?;
        let params: RawParams = serde_json::from_str(&text).map_err(|e| {
            if text.lines().any(|l| l.starts_with("nSavedChans=")) {
                fail(format!("{meta_name} is a SpikeGLX meta file: SpikeGLX recordings open in the desktop app or the container for now"))
            } else {
                fail(format!("{meta_name} is not dspk's JSON layout ({e})"))
            }
        })?;
        Ok(Arc::new(RawRecording::from_source(name, Arc::new(FileBytes(data.clone())), &params).map_err(|e| fail(e.to_string()))?))
    }
}

/// The engine's state, shared with the tasks it starts (the overview being built).
type Shared = std::rc::Rc<std::cell::RefCell<Inner>>;

thread_local! {
    /// The engine of this worker (one per worker), for the tasks it starts.
    static ENGINE: std::cell::RefCell<Option<Shared>> = const { std::cell::RefCell::new(None) };
}

fn inner_rc(_inner: &std::cell::RefCell<Inner>) -> Option<Shared> {
    ENGINE.with(|e| e.borrow().clone())
}

/// Pages this worker builds between requests when it builds the overview itself: a page spans
/// parts of two chunks of a chunked source, so a few pages at a time decode each chunk about once.
const PAGES_HERE: u64 = 2;

/// Builds the overview of the recording opened as `generation`, nearest the focus first, a few
/// pages at a time, letting the worker answer the page's requests between them; while the page's
/// helpers build it ([`overview`]), only waits. Stops when it is complete or another recording
/// is opened.
fn build_overview(engine: Option<Shared>, generation: u64) {
    let Some(engine) = engine else { return };
    wasm_bindgen_futures::spawn_local(async move {
        loop {
            let job = {
                let me = engine.borrow();
                let Some(o) = me.overview.as_ref().filter(|o| o.generation == generation) else { return };
                if o.pyramid.is_complete() {
                    return;
                }
                let waiting = !o.decided && js_sys::Date::now() - o.opened_ms < overview::DECIDE_WITHIN_MS;
                (!o.helpers && !waiting).then(|| o.next_run(PAGES_HERE).map(|run| (o.source.clone(), o.pyramid.clone(), run))).flatten()
            };
            let Some((source, pyramid, run)) = job else {
                // The helpers build it, or every free page is with them: look again soon
                wait_ms(200).await;
                continue;
            };
            let ps = pyramid.page_samples();
            let (start, end) = (run.start * ps, (run.end * ps).min(pyramid.samples()));
            if let Err(e) = pyramid.fill(source.as_ref(), start, end, RAW_BLOCK_VALUES, || false) {
                web_sys_console_error(&format!("overview of {}: {e}", source.info().name));
                return;
            }
            wait_ms(0).await;
        }
    });
}

/// Waits `ms` (0: lets the worker take the page's next message before going on).
async fn wait_ms(ms: i32) {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        let global = js_sys::global();
        if let Ok(set_timeout) = js_sys::Reflect::get(&global, &"setTimeout".into()).and_then(|f| f.dyn_into::<js_sys::Function>()) {
            let _ = set_timeout.call2(&global, &resolve, &JsValue::from(ms));
        }
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

fn console_error_panic_hook_set() {
    std::panic::set_hook(Box::new(|info| web_sys_console_error(&info.to_string())));
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn web_sys_console_error(s: &str);
}
