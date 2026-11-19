// The overview's fold in plain JS: a unit's chunk files decoded (the browser's own gzip,
// DecompressionStream) and folded into level-0 [min, max] buckets of `base` samples, as the
// engine's pyramid folds them (dspk-view `fold_row`, `Pyramid::build_page_from_buckets`).
// `overview-helper.js` runs it in a worker; `tests/overview-fold.test.mjs` checks it against the
// fixture the Rust fold writes (`view/dspk-view/tests/fixtures/overview-fold.json`), so the two
// stay one fold.
//
// meta: from Engine.overview_helper_meta (data type, compression, fill, gain, offset, channels,
// samples, base, page_samples, chunk_samples, chunk_channels). unit: from Engine.overview_next
// (pages, samples, chunks [{t0, c0, file}]). Returns a Float32Array: per channel, the unit's
// buckets in turn, min and max interleaved, [+Infinity, -Infinity] where no sample falls.
const TYPES = {
  float32: Float32Array, float64: Float64Array, int8: Int8Array, uint8: Uint8Array,
  int16: Int16Array, uint16: Uint16Array, int32: Int32Array, uint32: Uint32Array,
};

function decoded(file, compression) {
  if (!file) return Promise.resolve(null);
  if (compression === "gzip") return new Response(file.stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
  return file.arrayBuffer();
}

export async function fold(meta, unit) {
  const { channels: C, samples: total, base, page_samples: ps, chunk_samples: ct, chunk_channels: cc } = meta;
  const [p0, p1] = unit.pages;
  const [s0, s1] = unit.samples;
  // Buckets per channel: each page's in turn (pages are whole multiples of `base`, so a
  // sample's bucket is its offset from the unit's start / base; the last page may be shorter)
  let nb = 0;
  for (let p = p0; p < p1; p++) nb += Math.ceil((Math.min((p + 1) * ps, total) - p * ps) / base);
  const lo = new Float64Array(C * nb).fill(Infinity);
  const hi = new Float64Array(C * nb).fill(-Infinity);
  const Typed = TYPES[meta.data_type];
  if (!Typed) throw new Error(`samples of type ${meta.data_type}`);
  const buffers = await Promise.all(unit.chunks.map((c) => decoded(c.file, meta.compression)));
  unit.chunks.forEach((c, k) => {
    const width = Math.min(cc, C - c.c0);
    const first = Math.max(s0, c.t0);
    const last = Math.min(s1, c.t0 + ct, total);
    if (last <= first || width <= 0) return;
    const buffer = buffers[k];
    if (!buffer) {
      // A chunk not stored: every sample is the fill value
      const v = meta.fill;
      for (let t = first; t < last; t++) {
        const b = Math.floor((t - s0) / base);
        for (let j = 0; j < width; j++) {
          const i = (c.c0 + j) * nb + b;
          if (v < lo[i]) lo[i] = v;
          if (v > hi[i]) hi[i] = v;
        }
      }
      return;
    }
    // A chunk is stored whole ([chunk_samples, chunk_channels], C order), edge chunks too
    const a = new Typed(buffer);
    if (a.length < ct * cc) throw new Error(`a chunk of ${a.length} values (${ct * cc} expected)`);
    for (let t = first; t < last; t++) {
      const b = Math.floor((t - s0) / base);
      const row = (t - c.t0) * cc;
      for (let j = 0; j < width; j++) {
        // NaN fails both comparisons: skipped, as the engine's fold skips it
        const v = a[row + j];
        const i = (c.c0 + j) * nb + b;
        if (v < lo[i]) lo[i] = v;
        if (v > hi[i]) hi[i] = v;
      }
    }
  });
  // value = stored × gain + offset (as the reader scales); a negative gain swaps min and max
  const { gain: g, offset: o } = meta;
  const out = new Float32Array(2 * C * nb);
  for (let i = 0; i < C * nb; i++) {
    if (lo[i] > hi[i]) {
      out[2 * i] = Infinity;
      out[2 * i + 1] = -Infinity;
      continue;
    }
    const a = lo[i] * g + o;
    const b = hi[i] * g + o;
    out[2 * i] = g < 0 ? b : a;
    out[2 * i + 1] = g < 0 ? a : b;
  }
  return out;
}
