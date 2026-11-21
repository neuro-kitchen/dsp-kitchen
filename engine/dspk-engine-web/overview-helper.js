// An overview helper (a module worker, plain JS, no wasm): folds the units the engine worker
// hands it (`overview-fold.js`) and sends the level-0 buckets back (1/base of the data). The
// engine writes them into its pyramid (src/overview.rs). Several run at once (engine-worker.js
// sizes them to the device).
//
// {meta, unit} in. Out: {generation, pages, values} (see `fold`), or {generation, pages, error}.
import { fold } from "./overview-fold.js";

self.onmessage = async (e) => {
  const { meta, unit } = e.data;
  try {
    const values = await fold(meta, unit);
    self.postMessage({ generation: unit.generation, pages: unit.pages, values: values.buffer }, [values.buffer]);
  } catch (err) {
    self.postMessage({ generation: unit.generation, pages: unit.pages, error: String(err) });
  }
};
