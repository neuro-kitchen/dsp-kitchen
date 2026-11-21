// The page's loader (a file of its own: the page's Content-Security-Policy allows no inline
// script).
// ---- WebTransport for the app (as lucem's) ------------------------------------------------------
// Messages are a 4-byte big-endian length, then the bytes. `open(url, hash)` connects (a
// self-signed certificate is accepted by its SHA-256 only) and opens the control stream (who the
// page is, as JSON); `request(bytes)` opens a stream for one request and resolves with its answer.
// The request stream's close is not awaited: a server that answers at once ends the stream first,
// and the close then "fails" with STOP_SENDING although the answer is there.
const frame = (bytes) => {
  const out = new Uint8Array(4 + bytes.length);
  new DataView(out.buffer).setUint32(0, bytes.length);
  out.set(bytes, 4);
  return out;
};
const framed = (reader) => {
  let buf = new Uint8Array(0);
  const need = async (n) => {
    while (buf.length < n) {
      const { value, done } = await reader.read();
      if (done) throw new Error("the stream closed");
      const grown = new Uint8Array(buf.length + value.length);
      grown.set(buf);
      grown.set(value, buf.length);
      buf = grown;
    }
  };
  return {
    async next() {
      await need(4);
      const len = new DataView(buf.buffer, buf.byteOffset).getUint32(0);
      await need(4 + len);
      const out = buf.slice(4, 4 + len);
      buf = buf.slice(4 + len);
      return out;
    },
  };
};
if ("WebTransport" in window) {
  window.dspkTransport = {
    async open(url, hash) {
      const options = hash && hash.length ? { serverCertificateHashes: [{ algorithm: "sha-256", value: hash.slice() }] } : {};
      const wt = new WebTransport(url, options);
      await wt.ready;
      const control = await wt.createBidirectionalStream();
      const writer = control.writable.getWriter();
      const reader = framed(control.readable.getReader());
      const encoder = new TextEncoder(), decoder = new TextDecoder();
      return {
        control: {
          send: async (text) => writer.write(frame(encoder.encode(text))),
          next: async () => decoder.decode(await reader.next()),
        },
        async request(bytes) {
          const stream = await wt.createBidirectionalStream();
          const w = stream.writable.getWriter();
          await w.write(frame(bytes));
          w.close().catch(() => {});
          return framed(stream.readable.getReader()).next();
        },
        close() {
          wt.close();
        },
      };
    },
  };
}

// The loader: before the app (several MB) is downloaded, find what this browser offers and
// pick how the app runs. Nothing here guesses from the browser's name except the one case
// measured below.
//
// - Drawing: WebGPU where it is presented fast, else WebGL2; never in software (the page
//   would crawl whatever the app does): then the page says how to turn the GPU on.
// - Threads: the page must be cross-origin isolated (`dspk serve --web` serves it so; on
//   GitHub Pages a service worker of this page adds the headers, after one reload).
// - Computing: on a server (`dspk serve`), or in the engine worker the app starts when no
//   server served it (WebGPU; views only without it).
// - `?gpu=webgl` / `?gpu=webgpu` force the drawing backend (for measuring only).
// A project's place (or a `.zarr`) picked with the File System Access API (Chrome, Edge): the
// browser asks "Let site view files?", which is what happens (the files stay on the computer;
// the page reads them). Open file… uses the plain file chooser instead: no question there.
// Resolves to [{path: "<folder>/<path inside>", file}], or null when the user cancels.
// Browsers without it (Firefox, Safari) use the page's file input instead.
if (window.showDirectoryPicker) {
  window.dspkPickFolder = async () => {
    let root;
    try {
      root = await window.showDirectoryPicker({ mode: "read" });
    } catch {
      return null;
    }
    // Every file, but not hidden folders (.git, .venv…) nor deeper than recordings sit
    // (a Zarr's chunks are a few levels down), and at most MAX files: a large folder
    // (a home, a project) would take minutes to list
    // Names first, then the Files in parallel batches (a File is a handle: nothing is read until
    // a chunk is): an NWB-Zarr of ~20,000 files took minutes one getFile() at a time
    const handles = [];
    const MAX = 200000, DEPTH = 10;
    const walk = async (dir, prefix, depth) => {
      for await (const [name, handle] of dir.entries()) {
        if (handles.length >= MAX) return;
        if (handle.kind === "file") handles.push([`${prefix}/${name}`, handle]);
        else if (!name.startsWith(".") && depth < DEPTH) await walk(handle, `${prefix}/${name}`, depth + 1);
      }
    };
    await walk(root, root.name, 0);
    const out = [];
    for (let i = 0; i < handles.length; i += 512) {
      out.push(...(await Promise.all(handles.slice(i, i + 512).map(async ([path, h]) => ({ path, file: await h.getFile() })))));
    }
    return out;
  };
}
// Something dropped on the page (a recording's `.zarr`, a `.wav`, a `.bin` and its `.meta`): its
// files with their paths, handed to the app (`window.dspkDrop`, set by the app when it starts).
// A drop needs no permission prompt, even for a folder: the user handed the page these files.
// Folders are walked with the entries API (every browser), hidden folders skipped.
{
  const MAX = 200000, DEPTH = 10;
  const hint = document.createElement("div");
  hint.textContent = "Drop to open it in this browser (nothing is uploaded)";
  hint.style.cssText =
    "position:fixed;inset:12px;display:none;place-items:center;border:2px dashed var(--accent);border-radius:12px;" +
    "background:color-mix(in srgb, var(--bg) 85%, transparent);color:var(--fg);font:16px system-ui,sans-serif;z-index:10;pointer-events:none";
  document.body.append(hint);
  const files = (dragged) => [...(dragged?.types ?? [])].includes("Files");
  let depth = 0;
  window.addEventListener("dragenter", (e) => { if (files(e.dataTransfer)) { depth++; hint.style.display = "grid"; } }, true);
  window.addEventListener("dragleave", () => { if (--depth <= 0) { depth = 0; hint.style.display = "none"; } }, true);
  window.addEventListener("dragover", (e) => { if (files(e.dataTransfer)) e.preventDefault(); }, true);
  window.addEventListener("drop", (e) => {
    if (!files(e.dataTransfer)) return;
    e.preventDefault();
    depth = 0;
    hint.style.display = "none";
    // The entries must be taken now: the drop's list is gone once this handler returns
    const roots = [...e.dataTransfer.items].map((i) => i.webkitGetAsEntry?.()).filter(Boolean);
    const out = [];
    const fileOf = (entry) => new Promise((ok, fail) => entry.file(ok, fail));
    const children = (dir) => new Promise((ok, fail) => {
      const reader = dir.createReader(), all = [];
      const next = () => reader.readEntries((batch) => (batch.length ? (all.push(...batch), next()) : ok(all)), fail);
      next();
    });
    const walk = async (entry, prefix, level) => {
      if (out.length >= MAX) return;
      const path = prefix ? `${prefix}/${entry.name}` : entry.name;
      if (entry.isFile) out.push({ path, entry });
      else if (entry.isDirectory && !entry.name.startsWith(".") && level < DEPTH) {
        for (const child of await children(entry)) await walk(child, path, level + 1);
      }
    };
    (async () => {
      for (const root of roots) await walk(root, "", 0);
      const picked = [];
      for (let i = 0; i < out.length; i += 512) {
        picked.push(...(await Promise.all(out.slice(i, i + 512).map(async ({ path, entry }) => ({ path, file: await fileOf(entry) })))));
      }
      window.dspkDrop?.(picked);
    })().catch((err) => console.error("drop:", err));
  }, true);
}
const note = document.getElementById("note");
const say = (html) => (note.firstElementChild.innerHTML = html);
const saySub = (text) => {
  const sub = note.querySelector(".sub");
  if (sub) sub.textContent = text;
  else say(`<p>${text}</p>`);
};
const params = new URLSearchParams(location.search);

/** Hardware WebGL2: refused when the browser would draw in software. */
function webgl() {
  const gl = document.createElement("canvas").getContext("webgl2", { failIfMajorPerformanceCaveat: true });
  if (!gl) return null;
  const ext = gl.getExtension("WEBGL_debug_renderer_info");
  const renderer = String(ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
  gl.getExtension("WEBGL_lose_context")?.loseContext();
  return /swiftshader|llvmpipe|softpipe|software|basic render/i.test(renderer) ? null : renderer;
}

/** A hardware WebGPU adapter, or null. */
async function webgpu() {
  if (!navigator.gpu) return null;
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter || adapter.isFallbackAdapter || adapter.info?.isFallbackAdapter) return null;
    return `${adapter.info?.vendor ?? ""} ${adapter.info?.architecture ?? ""}`.trim() || "WebGPU";
  } catch {
    return null;
  }
}

// Threads need the page cross-origin isolated. `dspk serve` sends the headers; a host that
// cannot (GitHub Pages) gets them from a service worker of this page, after one reload
if (!window.crossOriginIsolated && "serviceWorker" in navigator && location.protocol !== "file:") {
  const tried = sessionStorage.getItem("dspk-isolate");
  if (!tried) {
    sessionStorage.setItem("dspk-isolate", "1");
    await navigator.serviceWorker.register("./isolate-sw.js");
    await navigator.serviceWorker.ready;
    location.reload();
    await new Promise(() => {});
  }
}

const [gl, gpu] = [webgl(), await webgpu()];
// Chrome on Linux presents a WebGPU canvas slowly (its WebGPU and compositor do not share
// textures there: chrome://gpu says "WebGPU interop: Disabled"; measured 2026-11-15), so it
// draws with WebGL2 even when it has WebGPU
const linuxChrome = /Linux/.test(navigator.userAgent) && !/Android/.test(navigator.userAgent) && /Chrome\//.test(navigator.userAgent);
let drawing = params.get("gpu") ?? (gpu && !linuxChrome ? "webgpu" : gl ? "webgl" : gpu ? "webgpu" : null);
const threads = window.crossOriginIsolated === true;
// Served by `dspk serve` (it answers this), or alone (a plain file server, GitHub Pages)
const server = await fetch("./dspk-serve.json", { cache: "no-store" })
  .then((r) => (r.ok ? r.json() : {}))
  .then((j) => j.server != null)
  .catch(() => false);

if (!threads) {
  say(`<h1>This browser cannot run dsp-kitchen here</h1>
    <p>The app draws on several threads, which a browser allows only to a page that is
    cross-origin isolated, and this one is not (a service worker could not make it so).</p>
    <p>Open it from <code>dspk serve --web</code>, or in a recent Chrome, Edge or Firefox.</p>`);
} else if (!drawing && !params.has("software")) {
  say(`<h1>This browser draws without the graphics card</h1>
    <p>dsp-kitchen draws every frame on the GPU; in software it would be too slow to use.</p>
    <ul>
      <li>Chrome / Edge: Settings → System → <em>Use graphics acceleration when available</em>, then restart; <code>chrome://gpu</code> should say “Hardware accelerated”.</li>
      <li>Firefox: Settings → Performance → <em>Use recommended performance settings</em>; <code>about:support</code> → Graphics.</li>
    </ul>
    <p><a href="?software">Start anyway (slow)</a></p>`);
} else {
  drawing ??= "webgl";
  // What the app reads at start (`dspk_app_web::backend`)
  window.dspkWeb = { drawing, server, renderer: drawing === "webgpu" ? gpu : gl };
  saySub(`Starting ${drawing === "webgpu" ? "WebGPU" : "WebGL2"} workspace…`);
  try {
    const app = await import("./mt/dspk_app_web.js");
    await app.default();
    note.remove();
  } catch (e) {
    // A Rust panic arrives here as "unreachable": the app's panic hook left what it was
  const why = window.dspkLastPanic ?? String(e);
  say(`<h1>dsp-kitchen did not start</h1><p><code>${why.replace(/</g, "&lt;")}</code></p>`);
  }
}
