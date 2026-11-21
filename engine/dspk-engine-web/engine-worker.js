// The engine worker: loads dspk-engine-web, opens the browser's GPU, then answers the page's
// requests ({id, bytes}: a protobuf Request) one at a time with {id, bytes} (a Response).
// {files: File[]} hands it the files the user picked. Built next to its module by build.sh.
//
// It also looks after the open recording's overview (src/overview.rs):
// - a copy kept from an earlier open of the same files (browser storage, the origin's private
//   file system) is restored at once;
// - else, when the recording's chunks are plain or gzip bytes, helpers (overview-helper.js, plain
//   workers) decode them with the browser's own gzip, on several cores, nearest the view first;
//   the engine builds what they cannot;
// - once complete, the overview is kept for next time.
import init, { Engine } from "./dspk_engine_web.js";

// The module's static constructors, once (`_initialize` in lib.rs says why)
const wasm = await init();
wasm._initialize();
const engine = new Engine();
let queue = Promise.resolve();
self.onmessage = (e) => {
  const message = e.data;
  if (message.files) {
    engine.set_files(message.files);
    return;
  }
  const arrived = performance.now();
  queue = queue.then(async () => {
    const out = await engine.answer(new Uint8Array(message.bytes));
    self.postMessage({ id: message.id, bytes: out.buffer }, [out.buffer]);
    // How long the page waited, queue included: what the overview's helpers adapt to
    answered(performance.now() - arrived);
  });
};

// ---- The overview -------------------------------------------------------------------------

// How many helpers: a ceiling from the device, then adjusted to how fast the page is answered.
// - Ceiling: the cores less two (the page's drawing and this worker), and no more than the memory
//   allows (MB_PER_HELPER each, from half the device's memory; `deviceMemory` is GB, Chrome only).
// - Adjusted like TCP's congestion window (additive increase, multiplicative decrease; Netflix's
//   concurrency limits): start at half the ceiling; while the page's requests wait longer than
//   SLOW_MS, halve the helpers (a helper stops after its unit); while they are answered within
//   QUICK_MS and work is left, add one, at most every GROW_EVERY_MS. A page being worked in gets its
//   cores back at once; an idle page builds with all of them.
const MB_PER_HELPER = 128;
const SLOW_MS = 150;
const QUICK_MS = 60;
const GROW_EVERY_MS = 2000;
const CEILING = (() => {
  const cores = navigator.hardwareConcurrency || 4;
  const memoryMb = (navigator.deviceMemory || 4) * 1024;
  return Math.max(1, Math.min(cores - 2, Math.floor(memoryMb / 2 / MB_PER_HELPER)));
})();
let target = Math.max(1, Math.floor(CEILING / 2));
let latency = 0; // moving average of the page's waits (ms)
let grown = 0;
function answered(ms) {
  latency = latency ? 0.7 * latency + 0.3 * ms : ms;
  if (!helpers.length) return;
  if (latency > SLOW_MS && target > 1) {
    target = Math.max(1, Math.floor(target / 2));
    latency = 0;
  }
}
function maybeGrow() {
  const now = performance.now();
  if (helpers.length && latency < QUICK_MS && target < CEILING && now - grown > GROW_EVERY_MS && helpers.every((h) => h.busy)) {
    target++;
    grown = now;
    addHelper();
  }
}
// A helper failing this many units in a row: the engine builds the rest itself
const MAX_FAILURES = 3;
let generation = -1;
let helpers = [];
let meta = null;
let failures = 0;
let keeping = false;
// When the open recording's overview started, and how it is built (for the console line)
let began = 0;
let how = "";

const store = async () => (await navigator.storage.getDirectory()).getDirectoryHandle("dspk-overviews", { create: true });
const fileName = async (key) => {
  const hash = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(key));
  return [...new Uint8Array(hash)].map((b) => b.toString(16).padStart(2, "0")).join("") + ".bin";
};
async function keptCopy(key) {
  try {
    const handle = await (await store()).getFileHandle(await fileName(key));
    return new Uint8Array(await (await handle.getFile()).arrayBuffer());
  } catch {
    return null;
  }
}
async function keep(key, bytes) {
  const handle = await (await store()).getFileHandle(await fileName(key), { create: true });
  const writable = await handle.createWritable();
  await writable.write(bytes);
  await writable.close();
}

function stopHelpers() {
  for (const h of helpers) h.worker.terminate();
  helpers = [];
}

// Gives helper `h` its next unit, leaves it idle when every page is built or handed out, or
// retires it when there are more helpers than the target
function feed(h) {
  if (helpers.length > target) {
    h.worker.terminate();
    helpers = helpers.filter((other) => other !== h);
    return;
  }
  const unit = engine.overview_next();
  h.busy = !!unit;
  if (unit) h.worker.postMessage({ meta, unit });
}

function startHelpers() {
  meta = JSON.parse(engine.overview_helper_meta());
  engine.overview_use_helpers(true);
  failures = 0;
  target = Math.max(1, Math.floor(CEILING / 2));
  latency = 0;
  for (let i = 0; i < target; i++) addHelper();
}

function addHelper() {
  const gen = generation;
  {
    const h = { worker: new Worker(new URL("./overview-helper.js", import.meta.url), { type: "module" }), busy: false };
    h.worker.onmessage = (e) => {
      const { generation: unitGen, pages, values, error } = e.data;
      if (gen !== generation || unitGen !== generation) return;
      if (error) {
        engine.overview_failed(unitGen, pages[0], pages[1]);
        console.warn(`dspk: an overview helper failed (${error})`);
        if (++failures >= MAX_FAILURES) {
          // The engine builds the rest (and what was handed out comes back free)
          stopHelpers();
          engine.overview_use_helpers(false);
          return;
        }
      } else {
        failures = 0;
        try {
          engine.overview_put(unitGen, pages[0], pages[1], new Float32Array(values));
        } catch (err) {
          console.warn("dspk: an overview unit did not fit", err);
          engine.overview_failed(unitGen, pages[0], pages[1]);
        }
      }
      feed(h);
      // Pages freed by a failure: wake the idle ones
      for (const other of helpers) if (!other.busy) feed(other);
      maybeGrow();
    };
    h.worker.onerror = (e) => {
      console.warn("dspk: an overview helper stopped", e.message);
      e.preventDefault();
    };
    helpers.push(h);
    feed(h);
  }
}

async function follow() {
  const state = JSON.parse(engine.overview_state());
  if (!state) return;
  if (state.generation !== generation) {
    generation = state.generation;
    stopHelpers();
    began = performance.now();
    const kept = await keptCopy(state.key);
    if (generation !== state.generation) return;
    if (kept && engine.overview_restore(kept)) {
      how = "restored from this browser";
      return;
    }
    if (state.helpers_possible && typeof DecompressionStream === "function" && !state.complete) {
      startHelpers();
      how = `helpers (up to ${CEILING} on this device)`;
    } else {
      engine.overview_use_helpers(false);
      how = "the engine";
    }
    return;
  }
  if (state.complete) {
    if (how) {
      console.info(`dspk: overview complete in ${((performance.now() - began) / 1000).toFixed(1)} s (${how}, ${state.pages} pages)`);
      how = "";
    }
    stopHelpers();
    if (!state.kept && !keeping) {
      keeping = true;
      try {
        const bytes = engine.overview_bytes();
        if (bytes) {
          await keep(state.key, bytes);
          engine.overview_mark_kept();
        }
      } catch (err) {
        // Kept or not, the overview works; it is only built again next time
        console.warn("dspk: the overview was not kept in this browser", err);
        engine.overview_mark_kept();
      } finally {
        keeping = false;
      }
    }
  }
}
setInterval(() => follow().catch((err) => console.warn("dspk: overview", err)), 250);

// "" when ops can run here, else why not
self.postMessage({ ready: await engine.start() });
