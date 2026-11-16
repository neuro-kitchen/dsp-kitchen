// The engine worker: loads dspk-engine-web, opens the browser's GPU, then answers the page's
// requests ({id, bytes}: a protobuf Request) one at a time with {id, bytes} (a Response).
// {files: File[]} hands it the files the user picked. Built next to its module by build.sh.
import init, { Engine } from "./dspk_engine_web.js";

await init();
const engine = new Engine();
let queue = Promise.resolve();
self.onmessage = (e) => {
  const message = e.data;
  if (message.files) {
    engine.set_files(message.files);
    return;
  }
  queue = queue.then(async () => {
    const out = await engine.answer(new Uint8Array(message.bytes));
    self.postMessage({ id: message.id, bytes: out.buffer }, [out.buffer]);
  });
};
// "" when ops can run here, else why not
self.postMessage({ ready: await engine.start() });
