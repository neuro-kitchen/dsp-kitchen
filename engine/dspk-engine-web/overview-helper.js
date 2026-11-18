// An overview helper (plain JS, no wasm): reads a unit's chunk files, decodes them with the
// browser's own gzip (DecompressionStream, native code) and folds the samples into level-0
// [min, max] buckets of `base` samples, which it sends back (1/base of the data). The engine
// writes them into its pyramid (src/overview.rs). Several run at once, one per core.
//
// {meta, unit} in: meta from Engine.overview_helper_meta (data type, compression, fill, gain,
// offset, channels, samples, base, page_samples, chunk_samples, chunk_channels); unit from
// Engine.overview_next. Out: {generation, pages, values} (per channel, the unit's buckets in
// turn, min and max interleaved; [+Infinity, -Infinity] where no sample falls), or {error}.
const TYPES = {
  float32: Float32Array, float64: Float64Array, int8: Int8Array, uint8: Uint8Array,
  int16: Int16Array, uint16: Uint16Array, int32: Int32Array, uint32: Uint32Array,
};

self.onmessage = async (e) => {
  const { meta, unit } = e.data;
  try {
    const values = await fold(meta, unit);
    self.postMessage({ generation: unit.generation, pages: unit.pages, values: values.buffer }, [values.buffer]);
  } catch (err) {
    self.postMessage({ generation: unit.generation, pages: unit.pages, error: String(err) });
  }
};

function decoded(file, compression) {
  if (!file) return Promise.resolve(null);
  if (compression === "gzip") return new Response(file.stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
  return file.arrayBuffer();
}

async function fold(meta, unit) {
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
