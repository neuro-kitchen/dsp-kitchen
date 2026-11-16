# dspk-app-web: the app in a browser tab

The entry point only. The app is `dspk-app-core`, the same code as the desktop's; what differs in
a browser (threads, locks on the page's thread, files, servers) is `dspk_app_core::platform`.

## Build

```bash
app/dspk-app-web/build.sh        # both builds into app/dspk-app-web/dist/
app/dspk-app-web/build.sh mt     # the threaded build only (what the container serves)
```

Needs the nightly pinned in `build.sh` (with `rust-src` and `wasm32-unknown-unknown`; the script
installs it), `wasm-bindgen-cli` of the version in `Cargo.lock`, `wasm-opt` (binaryen) and
`brotli`. Nightly because gpui's web platform depends on `wasm_thread`, which uses an unstable
feature (`stdarch_wasm_atomic_wait`), and threads need `-Z build-std`.

`dist/` holds:

| | What | When the loader picks it |
|---|---|---|
| `index.html` | the loader: checks hardware drawing, WebGPU / WebGL2 and threads before downloading the app | always |
| `mt/` | threaded build: drawing on web workers; a client of `dspk serve`, which computes | the page is cross-origin isolated (`dspk serve --web`) |
| `st/` | single-thread build: ops on the browser's WebGPU | the page is not isolated (a plain file server) |

Threads and computing in the tab cannot be combined in one module: a WebGPU device belongs to the
thread that made it, and CubeCL's wgpu runtime does not build with shared memory. The next step
is an engine in a worker of its own (a second, single-thread module that owns WebGPU), talked to
with the same requests as `dspk serve`.

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
