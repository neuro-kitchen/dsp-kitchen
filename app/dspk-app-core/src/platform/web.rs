//! A browser tab (WebAssembly). Two builds: with threads (shared memory: the page must be served
//! cross-origin isolated, as `dspk serve --web` does) and without.

/// Whether work can run on threads of its own (web workers, in the threaded build).
pub const THREADS: bool = cfg!(target_feature = "atomics");

/// Runs `work` on a web worker called `name` (threaded build only).
pub fn spawn(name: &str, work: impl FnOnce() + Send + 'static) -> std::io::Result<()> {
    #[cfg(target_feature = "atomics")]
    return wasm_thread::Builder::new().name(name.into()).spawn(work).map(|_| ());
    #[cfg(not(target_feature = "atomics"))]
    {
        let _ = (name, work);
        Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "this build of the page has no web workers"))
    }
}

/// Where recordings come from, for the title bar: the server that served the page, or "This
/// browser" for a page that runs alone (the loader found no `dspk serve` behind it).
pub fn host_label() -> String {
    let window = web_sys::window();
    let alone = window
        .as_ref()
        .and_then(|w| js_sys::Reflect::get(w, &"dspkWeb".into()).ok())
        .and_then(|web| js_sys::Reflect::get(&web, &"server".into()).ok())
        .and_then(|s| s.as_bool())
        == Some(false);
    if alone {
        return "This browser".into();
    }
    window.and_then(|w| w.location().host().ok()).filter(|h| !h.is_empty()).unwrap_or_else(|| "This browser".into())
}

/// Keeps the last trace frames' timings in `window.dspkTimings` (`{t, render_ms, samples_per_px,
/// complete}`, `t` in the page's `performance.now()` ms when the frame reached the UI): what a
/// measuring script reads (`app/dspk-app-web/tools`). The newest 512 are kept.
pub fn record_frame_timing(render_ms: f64, samples_per_px: f64, complete: bool) {
    let Some(window) = web_sys::window() else { return };
    let key = wasm_bindgen::JsValue::from_str("dspkTimings");
    let list = js_sys::Reflect::get(&window, &key).ok().and_then(|v| wasm_bindgen::JsCast::dyn_into::<js_sys::Array>(v).ok()).unwrap_or_else(|| {
        let list = js_sys::Array::new();
        let _ = js_sys::Reflect::set(&window, &key, &list);
        list
    });
    let t = js_sys::Reflect::get(&window, &"performance".into())
        .ok()
        .and_then(|p| js_sys::Reflect::get(&p, &"now".into()).ok().and_then(|f| wasm_bindgen::JsCast::dyn_into::<js_sys::Function>(f).ok()).and_then(|f| f.call0(&p).ok()))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let entry = js_sys::Object::new();
    for (k, v) in [("t", t), ("render_ms", render_ms), ("samples_per_px", samples_per_px)] {
        let _ = js_sys::Reflect::set(&entry, &k.into(), &v.into());
    }
    let _ = js_sys::Reflect::set(&entry, &"complete".into(), &complete.into());
    list.push(&entry);
    if list.length() > 512 {
        list.shift();
    }
}

/// A page has no file dialog of the system's: its recordings are its server's.
pub const LOCAL_FILES: bool = false;


/// The page's GPU is asked for when work first needs it.
#[cfg(feature = "wgpu")]
pub fn wgpu_device() -> crate::data::device::DeviceInfo {
    crate::data::device::DeviceInfo { label: "GPU · WebGPU (the browser's)".into(), adapters: Vec::new() }
}
