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

/// Where recordings come from, for the title bar: the server that served the page.
pub fn host_label() -> String {
    web_sys::window().and_then(|w| w.location().host().ok()).filter(|h| !h.is_empty()).unwrap_or_else(|| "This browser".into())
}

/// A page has no file dialog of the system's: its recordings are its server's.
pub const LOCAL_FILES: bool = false;


/// The page's GPU is asked for when work first needs it.
#[cfg(feature = "wgpu")]
pub fn wgpu_device() -> crate::data::device::DeviceInfo {
    crate::data::device::DeviceInfo { label: "GPU · WebGPU (the browser's)".into(), adapters: Vec::new() }
}
