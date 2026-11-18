//! The overview (min/max pyramid) of the recording a page opened, built nearest where the page
//! looks first, then outward.
//!
//! Two ways to build it:
//! - **Here**, a few pages at a time between requests ([`Overview::next_run`] +
//!   `Pyramid::fill`): any source, but in this one worker and with this module's decoders
//!   (gzip in wasm decodes ~3× slower than the browser's own).
//! - **By the page's helpers** (`overview-helper.js`, plain workers): for a Zarr array stored as
//!   plain or gzip-compressed bytes, the worker script asks for [`Unit`]s, the helpers read their
//!   chunk files, decode them with the browser's `DecompressionStream` and fold the samples into
//!   level-0 buckets, and the buckets come back ([`Overview::put`]). Only 1/`base` of the data
//!   crosses back, and decoding runs on several cores.
//!
//! Once complete, the pyramid can be kept by the worker script (browser storage) and restored at
//! the next open of the same files ([`Overview::key`]).

use std::collections::BTreeSet;
use std::sync::Arc;

use dspk_core::RecordingSource;
use dspk_view::Pyramid;
use serde_json::{json, Value};

/// Pages per unit a helper builds: units span several chunks, so the chunks at a unit's edges
/// (decoded by both neighbours) are a small share (about 5 % at 1-s pages and 1-s chunks).
pub const UNIT_PAGES: u64 = 16;

/// How the page's helpers read the samples of a Zarr array themselves.
pub struct Plan {
    /// Picked-file path of the array's folder, ending in `/`.
    pub prefix: String,
    /// The chunk key separator (`/` or `.`).
    pub separator: char,
    pub chunk_samples: u64,
    pub chunk_channels: u64,
    pub channel_chunks: u64,
    /// Sent to the helpers as it is (data type, codec, scale, layout).
    pub meta: Value,
}

/// Pages handed out as one piece of work (`first..end` pages, samples `start..stop`).
pub struct Unit {
    pub pages: std::ops::Range<u64>,
    pub samples: std::ops::Range<u64>,
}

pub struct Overview {
    pub generation: u64,
    pub source: Arc<dyn RecordingSource>,
    pub pyramid: Arc<Pyramid>,
    /// The sample the page looks at (the centre of its last view).
    pub focus: u64,
    /// Pages given to the helpers and not back yet.
    pub handed_out: BTreeSet<u64>,
    /// The helpers build it (the worker script took the plan); this worker only answers.
    pub helpers: bool,
    pub plan: Option<Plan>,
    /// Identifies the files the overview was built from (name, size, date of every file of the
    /// recording): the key of the kept copy.
    pub key: String,
    /// Whether the worker script kept the complete pyramid already (or restored it).
    pub kept: bool,
    /// The worker script chose how it is built (restored, helpers or this worker). Until then
    /// nothing is built here, so a kept copy can still be restored (a pyramid that started
    /// building is never overwritten); after [`DECIDE_WITHIN_MS`] this worker starts anyway.
    pub decided: bool,
    /// When the recording was opened (`Date.now()` ms).
    pub opened_ms: f64,
}

/// How long this worker waits for the worker script to choose before building itself.
pub const DECIDE_WITHIN_MS: f64 = 2000.0;

impl Overview {
    pub fn new(generation: u64, source: Arc<dyn RecordingSource>, pyramid: Arc<Pyramid>, plan: Option<Plan>, key: String) -> Self {
        Self { generation, source, pyramid, focus: 0, handed_out: BTreeSet::new(), helpers: false, plan, key, kept: false, decided: false, opened_ms: js_sys::Date::now() }
    }

    fn free(&self, page: u64) -> bool {
        !self.pyramid.page_built(page) && !self.handed_out.contains(&page)
    }

    /// Up to `max` consecutive pages not built nor handed out, starting at the free page nearest
    /// the focus (ahead of it first at equal distance), or `None` when every page is taken.
    pub fn next_run(&self, max: u64) -> Option<std::ops::Range<u64>> {
        let pages = self.pyramid.pages();
        if pages == 0 {
            return None;
        }
        let at = (self.focus / self.pyramid.page_samples().max(1)).min(pages - 1);
        let nearest = (0..pages).find_map(|d| [at.checked_add(d).filter(|&p| p < pages), at.checked_sub(d + 1)].into_iter().flatten().find(|&p| self.free(p)))?;
        // Back to the start of its unit while free: units stay aligned, so their edge chunks
        // (decoded by both neighbours) repeat little
        let unit_start = nearest / max * max;
        let mut first = nearest;
        while first > unit_start && self.free(first - 1) {
            first -= 1;
        }
        let mut end = first;
        while end < pages && end - first < max && self.free(end) {
            end += 1;
        }
        Some(first..end)
    }

    /// The next unit for a helper (handed out until [`Self::put`] or [`Self::failed`]).
    pub fn next_unit(&mut self) -> Option<Unit> {
        let pages = self.next_run(UNIT_PAGES)?;
        self.handed_out.extend(pages.clone());
        let ps = self.pyramid.page_samples();
        let samples = pages.start * ps..(pages.end * ps).min(self.pyramid.samples());
        Some(Unit { pages, samples })
    }

    /// The chunk keys (picked-file paths) holding samples `samples`: `(first sample of the chunk,
    /// first channel of the chunk, path)`.
    pub fn chunks_of(&self, samples: &std::ops::Range<u64>) -> Vec<(u64, u64, String)> {
        let Some(plan) = &self.plan else { return Vec::new() };
        let sep = plan.separator;
        let first = samples.start / plan.chunk_samples;
        let last = (samples.end.max(samples.start + 1) - 1) / plan.chunk_samples;
        (first..=last)
            .flat_map(|i| (0..plan.channel_chunks).map(move |j| (i, j)))
            .map(|(i, j)| (i * plan.chunk_samples, j * plan.chunk_channels, format!("{}c{sep}{i}{sep}{j}", plan.prefix)))
            .collect()
    }

    /// A helper's level-0 buckets of unit `pages`: per channel, every page's
    /// [`Pyramid::page_buckets`] in turn, `[min, max]` interleaved. Returns whether the overview
    /// is complete.
    pub fn put(&mut self, pages: std::ops::Range<u64>, values: &[f32]) -> Result<bool, String> {
        for p in pages.clone() {
            self.handed_out.remove(&p);
        }
        let channels = self.source.info().channel_count();
        let counts: Vec<usize> = pages.clone().map(|p| self.pyramid.page_buckets(p)).collect();
        let per_channel: usize = counts.iter().sum();
        if values.len() != 2 * channels * per_channel {
            return Err(format!("pages {pages:?}: {} values for {channels} channels × {per_channel} buckets", values.len()));
        }
        let pairs: &[[f32; 2]] = values.as_chunks::<2>().0;
        let mut offset = 0;
        let mut rows = Vec::new();
        for (p, &count) in pages.zip(&counts) {
            rows.clear();
            for ch in 0..channels {
                let at = ch * per_channel + offset;
                rows.extend_from_slice(&pairs[at..at + count]);
            }
            self.pyramid.build_page_from_buckets(p, &rows).map_err(|e| e.to_string())?;
            offset += count;
        }
        Ok(self.pyramid.is_complete())
    }

    /// A helper could not build unit `pages`: they are free again (built here or handed out anew).
    pub fn failed(&mut self, pages: std::ops::Range<u64>) {
        for p in pages {
            self.handed_out.remove(&p);
        }
    }

    /// What the worker script follows: the build, how far, and whether it may keep it.
    pub fn state(&self) -> Value {
        let pages = self.pyramid.pages();
        let built = (0..pages).filter(|&p| self.pyramid.page_built(p)).count();
        json!({
            "generation": self.generation,
            "pages": pages,
            "built": built,
            "complete": self.pyramid.is_complete(),
            "key": self.key,
            "kept": self.kept,
            "helpers_possible": self.plan.is_some(),
            "helpers": self.helpers,
        })
    }

    /// What the helpers need to read and fold the array's chunks.
    pub fn helper_meta(&self) -> Option<Value> {
        let plan = self.plan.as_ref()?;
        let mut meta = plan.meta.clone();
        let info = self.source.info();
        meta["channels"] = json!(info.channel_count());
        meta["samples"] = json!(info.samples);
        meta["base"] = json!(self.pyramid.base());
        meta["page_samples"] = json!(self.pyramid.page_samples());
        meta["chunk_samples"] = json!(plan.chunk_samples);
        meta["chunk_channels"] = json!(plan.chunk_channels);
        Some(meta)
    }
}

/// How the helpers could read the Zarr array described by `zarr_json` (its `zarr.json`, v3) at
/// picked-file folder `prefix`, scaled by `gain` and `offset`: a 2-D `[time, channel]` array,
/// regular chunks, the default key encoding, stored as little-endian bytes, plain or gzip.
/// `None` for anything else (the worker builds those itself).
pub fn plan_for(zarr_json: &str, prefix: &str, gain: f64, offset: f64) -> Option<Plan> {
    let meta: Value = serde_json::from_str(zarr_json).ok()?;
    if meta["zarr_format"].as_u64()? != 3 || meta["node_type"].as_str()? != "array" {
        return None;
    }
    let shape: Vec<u64> = meta["shape"].as_array()?.iter().filter_map(Value::as_u64).collect();
    if meta["chunk_grid"]["name"].as_str()? != "regular" {
        return None;
    }
    let chunk: Vec<u64> = meta["chunk_grid"]["configuration"]["chunk_shape"].as_array()?.iter().filter_map(Value::as_u64).collect();
    let (&[_, channels], &[chunk_samples, chunk_channels]) = (shape.as_slice(), chunk.as_slice()) else { return None };
    if chunk_samples == 0 || chunk_channels == 0 {
        return None;
    }
    if meta["chunk_key_encoding"]["name"].as_str()? != "default" {
        return None;
    }
    let separator = match meta["chunk_key_encoding"]["configuration"]["separator"].as_str().unwrap_or("/") {
        "/" => '/',
        "." => '.',
        _ => return None,
    };
    let codecs = meta["codecs"].as_array()?;
    let bytes_codec = codecs.first()?;
    if bytes_codec["name"].as_str()? != "bytes" || bytes_codec["configuration"]["endian"].as_str().unwrap_or("little") != "little" {
        return None;
    }
    let compression = match &codecs[1..] {
        [] => "none",
        [c] if c["name"].as_str() == Some("gzip") => "gzip",
        _ => return None,
    };
    let data_type = meta["data_type"].as_str()?;
    if !matches!(data_type, "float32" | "float64" | "int8" | "uint8" | "int16" | "uint16" | "int32" | "uint32") {
        return None;
    }
    let fill = meta["fill_value"].as_f64().unwrap_or(0.0);
    Some(Plan {
        prefix: prefix.to_string(),
        separator,
        chunk_samples,
        chunk_channels,
        channel_chunks: channels.div_ceil(chunk_channels),
        meta: json!({ "data_type": data_type, "compression": compression, "fill": fill, "gain": gain, "offset": offset }),
    })
}
