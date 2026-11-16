//! dspk-app in a browser tab: the entry point only. Everything the app does is `dspk-app-core`,
//! the same as the desktop's; what differs in a browser is `dspk_app_core::platform`.
//!
//! Built by `build.sh` (nightly: gpui's web platform needs it) into `dist/`: `index.html` (a
//! loader that checks the browser and picks a build), `mt/` (threaded) and `st/` (single thread).
//! Served by `dspk serve --web dist`, the page is a client of that server; served by anything
//! else it opens a made-up recording.
#![cfg(target_family = "wasm")]

use dspk_app_core::data::preferences::Session;
use dspk_app_core::data::recording::{Dataset, SourceSet};
use dspk_app_core::data::store::{Recording, Store};
use dspk_app_core::{app, di, navigation::actions, ui::assets};
use gpui_kit::component::TitleBar;
use gpui_kit::{App, AppContext as _, TitlebarOptions, WindowOptions};
use wasm_bindgen::prelude::*;

/// The font the page brings (a browser gives it none).
const FONT: &str = "Adwaita Sans";
/// The recording shown at start, made in memory: channels, sample rate (Hz), seconds.
const DEMO: (usize, f64, f64) = (32, 30_000.0, 10.0);

/// A made-up recording (µV-like values): a slow wave and a faster one per channel, noise, and a
/// spike now and then. Only so the page has something to show; it is not data.
fn demo() -> Result<Dataset, String> {
    let (channels, rate, seconds) = DEMO;
    let samples = (rate * seconds) as usize;
    let mut data = Vec::with_capacity(channels * samples);
    let mut state = 0x2545_f491_4f6c_dd1du64;
    for c in 0..channels {
        for i in 0..samples {
            // xorshift noise in [-1, 1)
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0;
            let t = i as f32 / rate as f32;
            let spike = if (i + c * 977) % 4201 < 30 { -180.0 * (1.0 - ((i + c * 977) % 4201) as f32 / 30.0) } else { 0.0 };
            data.push(40.0 * (t * 6.3 * 8.0 + c as f32).sin() + 12.0 * (t * 6.3 * 60.0).sin() + 18.0 * noise + spike);
        }
    }
    let recording = dspk_core::MemoryRecording::new("demo (made up)", data, channels, rate).map_err(|e| e.to_string())?;
    Dataset::local(std::sync::Arc::new(recording), None).map_err(|e| e.to_string())
}

unsafe extern "C" {
    /// Runs the static constructors (the `inventory` registrations of GPUI's actions and CubeCL's
    /// IR). Naming it here matters beyond calling it: when no code calls it, the linker wraps
    /// **every** export in a function that runs all constructors first, so each value that
    /// crosses to JavaScript (thousands per frame) paid for them (seen in a profile, 2026-11-15:
    /// about half the frame's time in WebAssembly).
    fn __wasm_call_ctors();
}

/// The value of `name` in the page's address (`?name=value`), decoded.
fn link_parameter(name: &str) -> Option<String> {
    let search = web_sys::window()?.location().search().ok()?;
    let value = search.trim_start_matches('?').split('&').find_map(|pair| pair.strip_prefix(name)?.strip_prefix('='))?;
    js_sys::decode_uri_component(value).ok().map(String::from).filter(|v| !v.is_empty())
}

/// What the page draws with, as the loader in `index.html` chose it after checking the browser
/// (`window.dspkWeb.drawing`: "webgpu" or "webgl"); WebGPU with a WebGL2 fallback when the page
/// was opened without the loader.
fn backend() -> gpui_kit::web::WebBackendPreference {
    use gpui_kit::web::WebBackendPreference as B;
    let chosen = web_sys::window()
        .and_then(|w| js_sys::Reflect::get(&w, &"dspkWeb".into()).ok())
        .and_then(|web| js_sys::Reflect::get(&web, &"drawing".into()).ok())
        .and_then(|d| d.as_string());
    match chosen.as_deref() {
        Some("webgpu") => B::WebGpu,
        Some("webgl") => B::WebGl,
        _ => B::Auto,
    }
}

#[wasm_bindgen(start)]
pub fn start() {
    static CONSTRUCTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !CONSTRUCTED.swap(true, std::sync::atomic::Ordering::SeqCst) {
        // SAFETY: the linker's own function; called once, before anything reads a registry
        unsafe { __wasm_call_ctors() };
    }
    gpui_kit::platform::web_init();
    // With threads in the build (shared memory, and a page served cross-origin isolated), GPUI's
    // background work and the app's render pool run on web workers; else everything runs here
    let threads = dspk_app_core::platform::THREADS;
    let platform = std::rc::Rc::new(gpui_kit::web::WebPlatform::new_with_backend(threads, backend()));
    let application = gpui_kit::Application::with_platform(platform);
    log::info!("dspk-app: threads in this build: {threads}, drawing: {:?}", backend());
    application.with_assets(assets::app_assets()).run(|cx: &mut App| {
        cx.text_system().add_fonts(vec![std::borrow::Cow::Borrowed(include_bytes!("AdwaitaSans-Regular.ttf").as_slice())]).expect("the font");
        gpui_kit::init(cx);
        gpui_kit::component::Theme::global_mut(cx).font_family = FONT.into();
        actions::bind_keys(cx);
        di::install(cx);
        let store = cx.new(|_| Store::new(Session::default(), None));
        let options = WindowOptions { titlebar: Some(TitlebarOptions { title: Some("dsp-kitchen".into()), ..TitleBar::title_bar_options() }), ..Default::default() };
        let s = store.clone();
        gpui_kit::open_window(options, cx, |window, cx| cx.new(|cx| app::DspApp::new(s, window, cx))).expect("the browser window");
        log::info!("dspk-app: window open");
        // Served by `dspk serve --web`: the page signs in to that server (a kept token, or a
        // pairing code) and File → Open from server lists its recordings. Served by anything
        // else (a plain file server): a made-up recording, so the page has something to show.
        let s = store.clone();
        cx.spawn(async move |cx| {
            match dspk_app_core::platform::remote::connect_page().await {
                Ok(Some(welcome)) => {
                    let _ = s.update(cx, |s, cx| {
                        s.set_servers(vec![dspk_app_core::platform::remote::page_profile()], cx);
                        s.set_status(format!("Connected to {} as {} ({}). File → Open from server… lists its recordings", welcome.server_name, welcome.peer_name, welcome.permission), cx);
                        // A link to a recording: `…/index.html?open=<path under the data root>`
                        if let Some(path) = link_parameter("open") {
                            s.open_remote(dspk_app_core::platform::remote::page_profile(), path, cx);
                        }
                    });
                }
                Ok(None) => {
                    let _ = s.update(cx, |s, cx| match demo() {
                        Ok(dataset) => s.install(Recording::new(SourceSet::single(dataset), None), cx),
                        Err(e) => s.set_status(format!("The demo recording could not be made: {e}"), cx),
                    });
                }
                Err(e) => {
                    let _ = s.update(cx, |s, cx| s.set_status(format!("Not connected to the server: {e}"), cx));
                }
            }
        })
        .detach();
    });
}
