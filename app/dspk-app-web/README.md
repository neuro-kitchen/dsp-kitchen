# dspk-app-web: the app in a browser tab

The entry point only. The app is `dspk-app-core`, the same code as the desktop's; what differs in
a browser (threads, locks on the page's thread, files, servers) is `dspk_app_core::platform`.

## Build

```bash
app/dspk-app-web/build.sh        # the app and the engine into app/dspk-app-web/dist/ (GitHub Pages)
app/dspk-app-web/build.sh mt     # the app only (what the container serves: its server computes)
```

Needs the nightly pinned in `build.sh` (with `rust-src` and `wasm32-unknown-unknown`; the script
installs it), `wasm-bindgen-cli` of the version in `Cargo.lock`, `wasm-opt` (binaryen) and
`brotli`. Nightly because gpui's web platform depends on `wasm_thread`, which uses an unstable
feature (`stdarch_wasm_atomic_wait`), and threads need `-Z build-std`.

`dist/` holds:

| | What | When the loader picks it |
|---|---|---|
| `index.html` | the loader: checks hardware drawing, WebGPU / WebGL2 and threads before downloading the app |
| `isolate-sw.js` | a service worker that makes the page cross-origin isolated where the host sends no headers (GitHub Pages) |
| `mt/` | the app, threaded: drawing on web workers |
| `engine/` | the engine (`dspk-engine-web`), `engine-worker.js`, `overview-helper.js` and `overview-fold.js` (module workers that build the overview on several cores with the browser's own gzip; the fold is checked against the engine's by `engine/dspk-engine-web/tests`): for a page no server served |

Two modules, because threads and computing cannot share one: a WebGPU device belongs to the
thread that made it, and CubeCL's wgpu runtime does not build with shared memory. So the app
(threaded) draws, and the engine (single thread, in a worker of its own) owns WebGPU. They talk
with the same requests as `dspk serve`: served by a server, the page talks to it instead and the
engine is not started.

## Serve

```bash
dspk serve --data <folder of recordings> --pair --web app/dspk-app-web/dist
```

Open `http://127.0.0.1:50052`, type the pairing code the server printed, then **File → Open
from server…**, or link to a recording: `…/?open=<path under the data folder>`. The page keeps a
token, so the code is asked once. The browser must draw with the GPU (`chrome://gpu`: "Hardware
accelerated"); the loader refuses to start otherwise.

## Measuring

`tools/` drives a headless Chrome over its DevTools port (`pip install websockets`), and
`tools/diag.html` shows what a browser offers the app (GPU, threads, frame pacing):

- `prof.py <url> <idle s> <busy s> <shot.png> [port]`: CPU profile of the page's main thread while
  it scrolls; frames per second, the functions by time, the console of the page and its workers.
- `trace.py <url> <idle s> <busy s> [port]`: Chrome's own trace, per thread.
- `pair.py <url> <code> <s> <shot.png> [port]`: answers the pairing prompt, then a screenshot.

Chrome for them, with the GPU: `google-chrome --headless=new --enable-gpu --ignore-gpu-blocklist
--use-gl=angle --use-angle=gl-egl --remote-debugging-port=9224 --user-data-dir=<dir> about:blank`.
Without those options headless Chrome composites in software and the numbers mean nothing.
Function names need the build before `wasm-opt`.
