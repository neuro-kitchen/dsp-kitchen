// The page's overview fold (overview-fold.js) against the Rust fold's fixture
// (view/dspk-view/tests/overview_fold_fixture.rs writes and checks it): one fold, two languages.
// Run: node --test engine/dspk-engine-web/tests/
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { gzipSync } from "node:zlib";
import { fold } from "../overview-fold.js";

const fixture = JSON.parse(readFileSync(new URL("../../../view/dspk-view/tests/fixtures/overview-fold.json", import.meta.url)));

// The fixture's chunks as the files a page holds (plain, or gzip-compressed as NWB-Zarr stores them)
function unitWith(compression) {
  const chunks = fixture.unit.chunks.map((c) => {
    if (!c.values) return { t0: c.t0, c0: c.c0, file: null };
    const bytes = new Uint8Array(Int16Array.from(c.values).buffer);
    return { t0: c.t0, c0: c.c0, file: new Blob([compression === "gzip" ? gzipSync(bytes) : bytes]) };
  });
  return { ...fixture.unit, chunks };
}

for (const compression of ["none", "gzip"]) {
  test(`the JS fold equals the Rust fold (${compression})`, async () => {
    const got = await fold({ ...fixture.meta, compression }, unitWith(compression));
    assert.equal(got.length, fixture.expected.length);
    got.forEach((v, i) => assert.equal(v, Math.fround(fixture.expected[i]), `value ${i}`));
  });
}
