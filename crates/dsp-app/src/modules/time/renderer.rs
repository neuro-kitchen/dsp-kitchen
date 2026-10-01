//! Waveform rasterizer for the plot area.
//!
//! Draws only the plot itself (traces or heatmap, grid, spike ticks, scale bar); all text
//! (channel labels, time axis, readouts) is laid out by Slint around and over the image.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dsp_base::resampler::minmax::{finish, fold_block, Block, Columns, EMPTY};
use dsp_base::resampler::summary::BASE as SUMMARY_BASE;
use dsp_base::resampler::{min_max_decimate_into, MinMaxCache, MinMaxSummary};
use dsp_core::{MemoryOrder, RecordingSource};
use slint::{Rgba8Pixel, SharedPixelBuffer};

use crate::data::SpikeEventStore;
use crate::shared::canvas::{blend_color, Canvas};
use crate::shared::axis::nice_step;
use crate::shared::render_worker::{RenderContext, Rendered};

/// Palette for multi-channel visualization (vibrant, modern dark-theme colors)
pub const CHANNEL_COLORS: [Rgba8Pixel; 8] = [
    Rgba8Pixel { r: 56,  g: 189, b: 248, a: 255 }, // Cyan #38bdf8
    Rgba8Pixel { r: 52,  g: 211, b: 153, a: 255 }, // Emerald #34d399
    Rgba8Pixel { r: 168, g: 85,  b: 247, a: 255 }, // Purple #a855f7
    Rgba8Pixel { r: 251, g: 191, b: 36,  a: 255 }, // Amber #fbbf24
    Rgba8Pixel { r: 244, g: 63,  b: 94,  a: 255 }, // Rose #f43f5e
    Rgba8Pixel { r: 96,  g: 165, b: 250, a: 255 }, // Blue #60a5fa
    Rgba8Pixel { r: 249, g: 115, b: 22,  a: 255 }, // Orange #f97316
    Rgba8Pixel { r: 45,  g: 212, b: 191, a: 255 }, // Teal #2dd4bf
];

const BG_COLOR: Rgba8Pixel = Rgba8Pixel { r: 9,  g: 13, b: 19, a: 255 }; // Deep dark #090d13
const GRID_COLOR: Rgba8Pixel = Rgba8Pixel { r: 22, g: 27, b: 34, a: 255 }; // Grid line #161b22
const BASELINE_COLOR: Rgba8Pixel = Rgba8Pixel { r: 33, g: 38, b: 45, a: 255 }; // Baseline #21262d
const TEXT_COLOR: Rgba8Pixel = Rgba8Pixel { r: 139, g: 148, b: 158, a: 255 }; // Gray text #8b949e
const SPIKE_MARKER_COLOR: Rgba8Pixel = Rgba8Pixel { r: 250, g: 204, b: 21, a: 255 }; // Gold #facc15

/// Amplitude (µV) that maps to `LANE_FILL` of a lane's half-height at gain 1x, when a view is
/// not auto-scaled.
pub const NOMINAL_UV: f32 = 80.0;
/// Fraction of the lane half-height used by `NOMINAL_UV`.
const LANE_FILL: f32 = 0.84;
/// Most values (channels × samples) held by one raw read while streaming a window: bounds the
/// scratch memory, not the work (every sample of the window is still read once).
const BLOCK_VALUES: usize = 1 << 22;
/// How often a frame waiting on the min/max summary shows what is summarized so far.
const PREVIEW_EVERY: Duration = Duration::from_millis(100);

/// Time-module view kinds: how the plot area visualizes channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum TimeViewKind {
    /// One trace per lane for the visible channel page.
    #[default]
    Traces,
    /// All channels as rows, color = peak |amplitude| per screen column.
    Heatmap,
}

/// Everything needed to draw one frame of one view. Sendable to the worker thread.
#[derive(Clone)]
pub struct RenderRequest {
    pub source: Arc<dyn RecordingSource>,
    /// Complete min/max cache file of `source` for zoomed-out windows.
    pub lod: Option<Arc<MinMaxCache>>,
    /// Session min/max summary of `source`, filled as needed for zoomed-out windows (raw reads
    /// without either).
    pub summary: Option<Arc<MinMaxSummary>>,
    pub events: Arc<SpikeEventStore>,
    /// Physical pixel size of the plot area.
    pub width: u32,
    pub height: u32,
    /// Display scale factor (physical px per logical px); sets line and marker thickness.
    pub scale: f32,
    pub mode: TimeViewKind,
    /// Channels to draw, top to bottom (traces: the lanes on screen; heatmap: every row).
    pub channels: Vec<usize>,
    pub window_start_sec: f64,
    pub window_sec: f64,
    /// Session time of the source's first sample (sources can start at different times).
    pub start_time_sec: f64,
    pub amplitude_scale: f32,
    /// Fit the amplitude to the visible data (else [`NOMINAL_UV`] fills a lane).
    pub auto_scale: bool,
    /// Subtract each channel's offset over the window (center of its interquartile range).
    pub remove_dc: bool,
    /// Scale of the previous frame: kept while the data still fits it, so views do not jump.
    pub scale_hint: f32,
    /// Times of vertical grid lines (the time-axis ticks).
    pub grid_times: Vec<f64>,
    /// Draw the amplitude scale bar (traces).
    pub scale_bar: bool,
    /// Sorted spikes to mark in a cluster color: (color, [(time_sec, channel)]).
    pub highlights: Vec<(Rgba8Pixel, Vec<(f64, usize)>)>,
}

thread_local! {
    /// Scratch buffers of the worker thread, reused across frames.
    static RENDERER: std::cell::RefCell<WaveformRenderer> = std::cell::RefCell::new(WaveformRenderer::default());
}

/// Renders `req` with this thread's reusable renderer (call from the render worker). A window
/// the min/max summary does not cover yet is summarized first, with preview frames of the part
/// done; `None` when a newer frame cancelled it (the pages read are kept).
pub fn render_on_worker(req: &RenderRequest, ctx: &RenderContext) -> Option<Rendered> {
    RENDERER.with(|r| {
        let mut r = r.borrow_mut();
        let rendered = |r: &mut WaveformRenderer| {
            let (frame, scale) = r.render_scaled(req);
            Rendered { frame, scale: Some(scale) }
        };
        if let Some((summary, start, end)) = summary_fill(req) {
            let mut deadline = Instant::now() + PREVIEW_EVERY;
            loop {
                let stop = || ctx.cancel.load(Ordering::Relaxed) || Instant::now() >= deadline;
                match summary.fill(req.source.as_ref(), start, end, BLOCK_VALUES, stop) {
                    Ok(true) => break,
                    Ok(false) if ctx.cancel.load(Ordering::Relaxed) => return None,
                    Ok(false) => {
                        (ctx.preview)(rendered(&mut r));
                        deadline = Instant::now() + PREVIEW_EVERY;
                    }
                    Err(e) => {
                        tracing::warn!("min/max summary read failed: {e}");
                        break;
                    }
                }
            }
        }
        Some(rendered(&mut r))
    })
}

/// Visible sample range of `req`'s source and the pixel columns `x0..x1` it covers (sources can
/// start later or end earlier than the window); `None` when nothing is visible.
fn visible(req: &RenderRequest) -> Option<(usize, usize, usize, usize)> {
    let info = req.source.info();
    let samples = info.samples as usize;
    if samples == 0 || info.channel_count() == 0 || req.window_sec <= 0.0 {
        return None;
    }
    let sr = info.sample_rate_hz();
    let s0 = (req.window_start_sec - req.start_time_sec) * sr;
    let s1 = s0 + req.window_sec * sr;
    let (a, b) = (s0.max(0.0), s1.min(samples as f64));
    let w_px = req.width.max(1) as f64;
    let (x0, x1) = if b > a { (((a - s0) / (s1 - s0) * w_px).round() as usize, ((b - s0) / (s1 - s0) * w_px).round() as usize) } else { (0, 0) };
    let (start, end) = (a.round() as usize, (b.round() as usize).max(a.round() as usize));
    Some((start, end, x0, x1.min(req.width.max(1) as usize)))
}

/// The summary and sample range to fill before drawing `req`: zoomed out past the summary's
/// buckets, without a complete cache file, and not summarized yet.
fn summary_fill(req: &RenderRequest) -> Option<(&MinMaxSummary, u64, u64)> {
    let summary = req.summary.as_deref().filter(|_| req.lod.is_none())?;
    let (start, end, x0, x1) = visible(req)?;
    let (start, end) = (start as u64, end as u64);
    let zoomed_out = x1 > x0 && end - start >= SUMMARY_BASE * (x1 - x0) as u64;
    (zoomed_out && !summary.covers(start, end)).then_some((summary, start, end))
}

/// Pixels per unit for a lane of `lane_h` pixels, where `nominal` units fill `LANE_FILL` of
/// the half-lane at gain 1.
pub fn px_per_unit(lane_h: f32, amplitude_scale: f32, nominal: f32) -> f32 {
    lane_h * 0.5 * LANE_FILL * amplitude_scale / nominal.max(f32::MIN_POSITIVE)
}

/// Scale bar value (1-2-5 series) about 45% of a lane tall, for `k` pixels per unit.
pub fn scale_bar_value(lane_h: f32, k: f32) -> f32 {
    if k <= 0.0 || lane_h <= 0.0 {
        return 0.0;
    }
    nice_step((0.45 * lane_h / k) as f64) as f32
}

/// Auto-scale: the previous scale while the visible amplitude stays within 50–125 % of it,
/// else the new amplitude.
fn choose_scale(amplitude: f32, hint: f32) -> f32 {
    if !(amplitude > 0.0) || !amplitude.is_finite() {
        return if hint > 0.0 { hint } else { NOMINAL_UV };
    }
    if hint > 0.0 && amplitude >= 0.5 * hint && amplitude <= 1.25 * hint { hint } else { amplitude }
}

/// Rasterizer with reusable scratch buffers (no per-frame allocations besides the image).
#[derive(Default)]
pub struct WaveformRenderer {
    /// `[min, max]` per (row, pixel column), row-major.
    env: Vec<[f32; 2]>,
    /// Whether each requested row is a real channel.
    valid: Vec<bool>,
    /// Channels actually read, and the row each one fills.
    read_channels: Vec<usize>,
    read_rows: Vec<usize>,
    /// Raw samples of the read channels, channel-major.
    block: Vec<f32>,
    /// Cache envelope of the read channels, row-major.
    cached: Vec<[f32; 2]>,
    heat: Vec<f32>,
    /// Scratch for medians and percentiles.
    stats: Vec<f32>,
}

impl WaveformRenderer {
    #[cfg(test)]
    pub fn render(&mut self, req: &RenderRequest) -> SharedPixelBuffer<Rgba8Pixel> {
        self.render_scaled(req).0
    }

    /// Renders and returns the amplitude scale used (units filling a lane at gain 1). Zoomed out,
    /// columns come from the cache file or the summary (pages not summarized draw empty).
    pub fn render_scaled(&mut self, req: &RenderRequest) -> (SharedPixelBuffer<Rgba8Pixel>, f32) {
        let source = req.source.as_ref();
        let width = req.width.max(1);
        let height = req.height.max(1);
        let mut pixel_buffer = SharedPixelBuffer::<Rgba8Pixel>::new(width, height);
        let mut canvas = Canvas {
            pixels: pixel_buffer.make_mut_slice(),
            width: width as usize,
            height: height as usize,
        };
        canvas.pixels.fill(BG_COLOR);

        let Some((start, end, x0, x1)) = visible(req) else {
            return (pixel_buffer, req.scale_hint.max(0.0));
        };

        // Time grid aligned with the axis ticks
        let w = canvas.width as f64;
        for &t in &req.grid_times {
            let x = ((t - req.window_start_sec) / req.window_sec * w).round();
            if x >= 0.0 && x < w {
                canvas.vline(x as usize, 0, canvas.height, GRID_COLOR);
            }
        }

        self.envelope(source, req.lod.as_deref(), req.summary.as_deref(), &req.channels, start, end, x1.saturating_sub(x0));
        self.place_columns(req.channels.len(), x0, x1, canvas.width);
        let scale = self.adjust(req, canvas.width);
        let has_samples = end > start;
        match req.mode {
            TimeViewKind::Traces => self.draw_traces(&mut canvas, &req.events, req, has_samples, scale),
            TimeViewKind::Heatmap => self.draw_heatmap(&mut canvas, req, has_samples, scale),
        }

        (pixel_buffer, scale)
    }

    /// Spreads `cols`-wide envelope rows into `width`-wide rows starting at column `x0`; columns
    /// the source does not cover become NaN (drawn empty).
    fn place_columns(&mut self, rows: usize, x0: usize, x1: usize, width: usize) {
        let cols = x1.saturating_sub(x0);
        if cols == width {
            return;
        }
        let mut full = vec![[f32::NAN, f32::NAN]; rows * width];
        if cols > 0 {
            for r in 0..rows {
                full[r * width + x0..r * width + x1].copy_from_slice(&self.env[r * cols..(r + 1) * cols]);
            }
        }
        self.env = full;
    }

    /// Removes each row's offset (when asked: the midpoint of the 25th and 75th percentiles of
    /// its column midpoints — robust to spikes and unbiased for two-level signals) and returns
    /// the amplitude scale: the 99th percentile of |value| over the visible rows (auto-scale),
    /// else [`NOMINAL_UV`].
    fn adjust(&mut self, req: &RenderRequest, width: usize) -> f32 {
        let rows = req.channels.len();
        if req.remove_dc {
            for r in (0..rows).filter(|&r| self.valid[r]) {
                let row = &mut self.env[r * width..(r + 1) * width];
                self.stats.clear();
                self.stats.extend(row.iter().filter(|v| v[0].is_finite()).map(|v| 0.5 * (v[0] + v[1])));
                if self.stats.is_empty() {
                    continue;
                }
                let n = self.stats.len();
                let q1 = *self.stats.select_nth_unstable_by(n / 4, f32::total_cmp).1;
                let q3 = *self.stats.select_nth_unstable_by((3 * n / 4).min(n - 1), f32::total_cmp).1;
                let offset = 0.5 * (q1 + q3);
                for v in row.iter_mut() {
                    v[0] -= offset;
                    v[1] -= offset;
                }
            }
        }
        if !req.auto_scale {
            return NOMINAL_UV;
        }
        self.stats.clear();
        for r in (0..rows).filter(|&r| self.valid[r]) {
            self.stats.extend(self.env[r * width..(r + 1) * width].iter().filter(|v| v[0].is_finite()).map(|v| v[0].abs().max(v[1].abs())));
        }
        if self.stats.is_empty() {
            return choose_scale(0.0, req.scale_hint);
        }
        let at = ((self.stats.len() - 1) as f32 * 0.99) as usize;
        let (_, &mut p99, _) = self.stats.select_nth_unstable_by(at, f32::total_cmp);
        choose_scale(p99, req.scale_hint)
    }

    /// Fills `env` with the `[min, max]` of every requested row per pixel column over
    /// `start..end`. Zoomed out, the complete min/max cache file, else the session summary
    /// (filled beforehand by [`render_on_worker`]), supplies bucket-aligned columns; zoomed in, or
    /// without either, raw samples are read (see [`Self::stream_raw`]). Every sample of the
    /// window lands in a column, so no peak is dropped.
    #[allow(clippy::too_many_arguments)]
    fn envelope(
        &mut self,
        source: &dyn RecordingSource,
        lod: Option<&MinMaxCache>,
        summary: Option<&MinMaxSummary>,
        rows: &[usize],
        start: usize,
        end: usize,
        width: usize,
    ) {
        let total = source.info().channel_count();
        self.env.clear();
        self.env.resize(rows.len() * width, [0.0, 0.0]);
        self.valid.clear();
        self.valid.extend(rows.iter().map(|&c| c < total));
        self.read_channels.clear();
        self.read_rows.clear();
        for (r, &c) in rows.iter().enumerate().filter(|&(_, &c)| c < total) {
            self.read_channels.push(c);
            self.read_rows.push(r);
        }
        let nch = self.read_channels.len();
        if nch == 0 || end <= start || width == 0 {
            return;
        }

        if let Some(cache) = lod {
            self.cached.resize(nch * width, [0.0, 0.0]);
            match cache.envelope(&self.read_channels, start as u64, end as u64, width, &mut self.cached) {
                Ok(true) => {
                    for (k, &r) in self.read_rows.iter().enumerate() {
                        self.env[r * width..(r + 1) * width].copy_from_slice(&self.cached[k * width..(k + 1) * width]);
                    }
                    return;
                }
                Ok(false) => {}
                Err(e) => tracing::warn!("min/max cache read failed, reading raw samples: {e}"),
            }
        }

        if let Some(summary) = summary.filter(|_| (end - start) as u64 >= SUMMARY_BASE * width as u64) {
            self.cached.resize(nch * width, [0.0, 0.0]);
            summary.envelope(&self.read_channels, start as u64, end as u64, width, &mut self.cached);
            for (k, &r) in self.read_rows.iter().enumerate() {
                self.env[r * width..(r + 1) * width].copy_from_slice(&self.cached[k * width..(k + 1) * width]);
            }
            return;
        }

        self.stream_raw(source, start, end, width, BLOCK_VALUES);
    }

    /// Raw-sample envelope of `start..end` (the read channels into their rows of `env`). A window
    /// of at most `block_values` values is read at once; a longer one is streamed in blocks aligned
    /// to the source's storage chunks, so each chunk is read and decoded once per frame, and every
    /// block is folded into the columns it overlaps. Columns no read reached stay NaN (empty).
    fn stream_raw(&mut self, source: &dyn RecordingSource, start: usize, end: usize, width: usize, block_values: usize) {
        let (n, nch) = (end - start, self.read_channels.len());
        if n * nch <= block_values || n <= width {
            self.block.resize(n * nch, 0.0);
            if source.read(&self.read_channels, start as u64..end as u64, &mut self.block).is_err() {
                self.block.fill(0.0);
            }
            for (k, &r) in self.read_rows.iter().enumerate() {
                min_max_decimate_into(&self.block[k * n..(k + 1) * n], &mut self.env[r * width..(r + 1) * width]);
            }
            return;
        }

        // Here n > width, so every column holds at least one sample
        let columns = Columns::Even { start: start as u64, len: n as u64, width };
        self.cached.clear();
        self.cached.resize(nch * width, EMPTY);
        let chunk = source.chunk_samples().filter(|&c| c > 0).map_or(1, |c| c as usize);
        let step = ((block_values / nch) / chunk).max(1) * chunk;
        let mut b0 = start;
        while b0 < end {
            // Blocks end on chunk boundaries (the first one may start inside a chunk)
            let b1 = (b0 / chunk * chunk + step).min(end);
            let len = b1 - b0;
            self.block.resize(len * nch, 0.0);
            if source.read(&self.read_channels, b0 as u64..b1 as u64, &mut self.block).is_ok() {
                let block = Block { data: &self.block, order: MemoryOrder::ChannelMajor, channels: nch, samples: len, first: b0 as u64 };
                fold_block(&block, 0..len, columns, &mut self.cached);
            }
            b0 = b1;
        }
        finish(&mut self.cached);
        for (k, &r) in self.read_rows.iter().enumerate() {
            self.env[r * width..(r + 1) * width].copy_from_slice(&self.cached[k * width..(k + 1) * width]);
        }
    }

    fn draw_traces(&mut self, canvas: &mut Canvas, events: &SpikeEventStore, req: &RenderRequest, has_samples: bool, scale: f32) {
        let num_channels = req.channels.len();
        if num_channels == 0 {
            return;
        }

        let lane_h = canvas.height as f32 / num_channels as f32;
        let k = px_per_unit(lane_h, req.amplitude_scale, scale);
        let thickness = req.scale.max(1.0);
        let t_end = req.window_start_sec + req.window_sec;

        let w = canvas.width;
        for lane in 0..num_channels {
            let ch = req.channels[lane];
            if !self.valid[lane] {
                continue;
            }
            let color = CHANNEL_COLORS[ch % CHANNEL_COLORS.len()];
            let center = (lane as f32 + 0.5) * lane_h;

            // Dashed baseline
            let by = center.round() as usize;
            for x in (0..canvas.width).filter(|x| x % 4 != 0) {
                canvas.set(x, by, BASELINE_COLOR);
            }

            // Spikes of selected clusters: colored band + thick tick
            for (color, spikes) in &req.highlights {
                for &(t, sch) in spikes {
                    if sch != ch || t < req.window_start_sec || t > t_end {
                        continue;
                    }
                    let x = ((t - req.window_start_sec) / req.window_sec * canvas.width as f64) as f32;
                    let top = lane as f32 * lane_h;
                    canvas.rect(x - 2.0 * req.scale, top, x + 2.0 * req.scale, top + lane_h, *color, 0.16);
                    canvas.rect(x - thickness, top + 1.0, x + thickness, top + 1.0 + 10.0 * req.scale, *color, 1.0);
                }
            }

            // Spike ticks: faint full-lane line + solid tick at the lane top
            for &t in events.in_window(ch, req.window_start_sec, t_end) {
                let x = ((t - req.window_start_sec) / req.window_sec * canvas.width as f64) as usize;
                let top = (lane as f32 * lane_h) as usize;
                let bottom = ((lane + 1) as f32 * lane_h) as usize;
                canvas.vline_alpha(x, top, bottom, SPIKE_MARKER_COLOR, 0.18);
                let tick = (6.0 * req.scale) as usize;
                for dx in 0..(thickness as usize).max(1) {
                    canvas.vline(x + dx, top + 1, top + 1 + tick, SPIKE_MARKER_COLOR);
                }
            }

            if !has_samples {
                continue;
            }

            // Anti-aliased min/max envelope, joined to the previous column for continuity
            let (lo_clip, hi_clip) = (0.0, canvas.height as f32 - 1.0);
            let mut prev: Option<(f32, f32)> = None;
            for (x, &[mn, mx]) in self.env[lane * w..(lane + 1) * w].iter().enumerate() {
                if !mn.is_finite() {
                    // Outside the source's time range
                    prev = None;
                    continue;
                }
                let y_top = (center - mx * k).clamp(lo_clip, hi_clip);
                let y_bot = (center - mn * k).clamp(lo_clip, hi_clip);
                let (mut a, mut b) = (y_top, y_bot);
                if let Some((pa, pb)) = prev {
                    // Bridge the gap to the previous column so steep edges stay connected
                    a = a.min(pb);
                    b = b.max(pa);
                }
                prev = Some((y_top, y_bot));
                canvas.span_aa(x, a - thickness * 0.5, b + thickness * 0.5, color);
            }
        }

        // Amplitude scale bar in the bottom-right corner of the last lane
        let bar = if req.scale_bar { scale_bar_value(lane_h, k) } else { 0.0 };
        if bar > 0.0 {
            let bar_h = bar * k;
            let x = canvas.width.saturating_sub((12.0 * req.scale) as usize);
            let bottom = canvas.height as f32 - 6.0 * req.scale;
            let top = (bottom - bar_h).max(0.0);
            for dx in 0..(thickness as usize).max(1) {
                canvas.vline(x + dx, top as usize, bottom as usize, TEXT_COLOR);
            }
        }
    }

    fn draw_heatmap(&mut self, canvas: &mut Canvas, req: &RenderRequest, has_samples: bool, scale: f32) {
        let channels = req.channels.len();
        let (w, h) = (canvas.width, canvas.height);
        if !has_samples || channels == 0 {
            return;
        }

        // Peak |amplitude| per (row, column), normalized by the gain-scaled nominal range
        self.heat.resize(channels * w, 0.0);
        let norm = req.amplitude_scale / (scale * 1.5);
        for r in 0..channels {
            let row = &mut self.heat[r * w..(r + 1) * w];
            if !self.valid[r] {
                row.fill(0.0);
                continue;
            }
            for (v, &[mn, mx]) in row.iter_mut().zip(&self.env[r * w..(r + 1) * w]) {
                // Squared for contrast: background noise stays dark, spikes stand out
                let a = if mn.is_finite() { (mn.abs().max(mx.abs()) * norm).min(1.0) } else { 0.0 };
                *v = a * a;
            }
        }

        // Each pixel row shows the max over the channels it covers
        for y in 0..h {
            let c0 = y * channels / h;
            let c1 = ((y + 1) * channels / h).max(c0 + 1).min(channels);
            for x in 0..w {
                let mut v = 0.0f32;
                for ch in c0..c1 {
                    v = v.max(self.heat[ch * w + x]);
                }
                canvas.set(x, y, heat_color(v));
            }
        }
    }
}

/// Draws the timeline overview strip: one bar per column, height = normalized spike density.
pub fn render_overview(width: u32, height: u32, density: &[f32]) -> SharedPixelBuffer<Rgba8Pixel> {
    const STRIP_BG: Rgba8Pixel = Rgba8Pixel { r: 13, g: 17, b: 23, a: 255 }; // #0d1117
    const BAR: Rgba8Pixel = Rgba8Pixel { r: 121, g: 192, b: 255, a: 255 }; // #79c0ff
    let (w, h) = (width.max(1), height.max(1));
    let mut buffer = SharedPixelBuffer::<Rgba8Pixel>::new(w, h);
    let mut canvas = Canvas { pixels: buffer.make_mut_slice(), width: w as usize, height: h as usize };
    canvas.pixels.fill(STRIP_BG);

    let bins = density.len();
    if bins == 0 {
        return buffer;
    }
    for x in 0..canvas.width {
        let d = density[(x * bins / canvas.width).min(bins - 1)];
        if d <= 0.0 {
            continue;
        }
        let bar = d * canvas.height as f32;
        let top = canvas.height as f32 - bar;
        // Brighter where denser, anti-aliased top edge
        canvas.span_aa(x, top, canvas.height as f32, blend_color(STRIP_BG, BAR, 0.35 + 0.45 * d));
    }
    buffer
}

/// Dark-to-bright sequential colormap (navy → violet → orange → yellow).
fn heat_color(v: f32) -> Rgba8Pixel {
    const STOPS: [(f32, [f32; 3]); 5] = [
        (0.00, [9.0, 13.0, 19.0]),
        (0.25, [49.0, 36.0, 110.0]),
        (0.50, [150.0, 45.0, 120.0]),
        (0.75, [240.0, 110.0, 50.0]),
        (1.00, [252.0, 230.0, 90.0]),
    ];
    let v = v.clamp(0.0, 1.0);
    let i = STOPS.iter().position(|s| s.0 >= v).unwrap_or(4).max(1);
    let (t0, c0) = STOPS[i - 1];
    let (t1, c1) = STOPS[i];
    let f = (v - t0) / (t1 - t0);
    let mix = |j: usize| (c0[j] + (c1[j] - c0[j]) * f) as u8;
    Rgba8Pixel { r: mix(0), g: mix(1), b: mix(2), a: 255 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Dataset;

    /// Request for `mode` over `ds`: traces show the first 4 channels, heatmap every channel.
    fn request(ds: Dataset, mode: TimeViewKind) -> RenderRequest {
        let events = Arc::new(SpikeEventStore::detect(&ds));
        let channels = match mode {
            TimeViewKind::Traces => (0..4.min(ds.total_channels)).collect(),
            TimeViewKind::Heatmap => (0..ds.total_channels).collect(),
        };
        RenderRequest {
            lod: None,
            summary: None,
            source: Arc::new(ds),
            events,
            width: 800,
            height: 400,
            scale: 1.0,
            mode,
            channels,
            window_start_sec: 0.0,
            window_sec: 0.100,
            start_time_sec: 0.0,
            amplitude_scale: 1.0,
            auto_scale: false,
            remove_dc: false,
            scale_hint: 0.0,
            grid_times: vec![0.05],
            scale_bar: true,
            highlights: vec![(Rgba8Pixel { r: 255, g: 0, b: 0, a: 255 }, vec![(0.05, 0)])],
        }
    }

    #[test]
    fn test_renderer_output_dimensions_and_trace_pixels() {
        let mut r = WaveformRenderer::default();
        for mode in [TimeViewKind::Traces, TimeViewKind::Heatmap] {
            let req = request(Dataset::generate_synthetic(8, 30_000.0, 0.5), mode);
            let buf = r.render(&req);
            assert_eq!((buf.width(), buf.height()), (800, 400));
            // Something other than background was drawn
            assert!(buf.as_slice().iter().any(|p| *p != BG_COLOR));
        }
    }

    #[test]
    fn test_out_of_range_channels_are_skipped() {
        let mut req = request(Dataset::generate_synthetic(2, 10_000.0, 0.2), TimeViewKind::Traces);
        req.channels = vec![0, 99];
        let mut r = WaveformRenderer::default();
        assert_eq!(r.render(&req).width(), 800);
        req.mode = TimeViewKind::Heatmap;
        assert_eq!(r.render(&req).width(), 800);
    }

    /// One channel: a 1000 µV offset plus a ±100 µV square wave (100 Hz at 10 kHz).
    fn offset_square() -> Dataset {
        let data: Vec<f32> = (0..10_000).map(|s| 1000.0 + if (s / 50) % 2 == 0 { 100.0 } else { -100.0 }).collect();
        Dataset::from_samples("sq", data, 1, 10_000.0)
    }

    #[test]
    fn test_auto_scale_dc_and_hysteresis() {
        let mut r = WaveformRenderer::default();
        let base = RenderRequest { channels: vec![0], window_sec: 0.5, auto_scale: true, ..request(offset_square(), TimeViewKind::Traces) };
        // Without DC removal the offset dominates the scale
        let (_, with_dc) = r.render_scaled(&base);
        assert!((with_dc - 1100.0).abs() < 1.0, "{with_dc}");
        // Offset removed: only the ±100 swing remains
        let (_, no_dc) = r.render_scaled(&RenderRequest { remove_dc: true, ..base.clone() });
        assert!((no_dc - 100.0).abs() < 1.0, "{no_dc}");
        // A previous scale within 50–125 % of the data is kept (no jumping while scrolling)
        let (_, kept) = r.render_scaled(&RenderRequest { remove_dc: true, scale_hint: 110.0, ..base.clone() });
        assert_eq!(kept, 110.0);
        let (_, replaced) = r.render_scaled(&RenderRequest { remove_dc: true, scale_hint: 1000.0, ..base.clone() });
        assert!((replaced - 100.0).abs() < 1.0);
        // Not auto-scaled: the nominal range
        let (_, fixed) = r.render_scaled(&RenderRequest { auto_scale: false, ..base });
        assert_eq!(fixed, NOMINAL_UV);
    }

    #[test]
    fn test_source_starting_inside_the_window_draws_only_its_part() {
        let mut ds = Dataset::generate_synthetic(1, 1000.0, 1.0);
        ds.start_time_sec = 1.0;
        let req = RenderRequest {
            channels: vec![0],
            window_start_sec: 0.0,
            window_sec: 2.0,
            start_time_sec: 1.0,
            grid_times: Vec::new(),
            events: Arc::new(SpikeEventStore::default()),
            highlights: Vec::new(),
            ..request(ds, TimeViewKind::Traces)
        };
        let buf = WaveformRenderer::default().render(&req);
        let (w, h) = (buf.width() as usize, buf.height() as usize);
        let px = buf.as_slice();
        let drawn = |x0: usize, x1: usize| (0..h).any(|y| (x0..x1).any(|x| px[y * w + x] != BG_COLOR && px[y * w + x] != BASELINE_COLOR));
        assert!(!drawn(0, w / 2 - 2), "no data before the source starts");
        assert!(drawn(w / 2 + 2, w - 20), "data after it starts");
    }

    /// In-memory source that reports a storage chunk size and records every read range.
    struct Chunked {
        inner: Dataset,
        chunk: u64,
        reads: std::sync::Mutex<Vec<std::ops::Range<u64>>>,
    }

    impl RecordingSource for Chunked {
        fn info(&self) -> &dsp_core::RecordingInfo {
            self.inner.info()
        }
        fn chunk_samples(&self) -> Option<u64> {
            Some(self.chunk)
        }
        fn read(&self, channels: &[usize], samples: std::ops::Range<u64>, out: &mut [f32]) -> dsp_core::DspResult<()> {
            self.reads.lock().unwrap().push(samples.clone());
            self.inner.read(channels, samples, out)
        }
    }

    #[test]
    fn test_streamed_envelope_is_exact_and_reads_each_chunk_once() {
        let (nch, total, width) = (3usize, 10_000usize, 37usize);
        let data: Vec<f32> = (0..nch * total).map(|i| ((i * 7919) % 1013) as f32 - 500.0).collect();
        let src = Chunked { inner: Dataset::from_samples("chunked", data.clone(), nch, 1000.0), chunk: 300, reads: Default::default() };
        let (start, end) = (1_234usize, 9_876usize);

        let mut r = WaveformRenderer::default();
        let rows = [2usize, 0];
        r.env.resize(rows.len() * width, [0.0, 0.0]);
        r.read_channels = rows.to_vec();
        r.read_rows = vec![0, 1];
        // Room for 4 chunks of 2 channels per read
        r.stream_raw(&src, start, end, width, 2 * 4 * 300 + 17);

        let n = end - start;
        for (k, &ch) in rows.iter().enumerate() {
            for x in 0..width {
                let (c0, c1) = (start + x * n / width, start + (x + 1) * n / width);
                let col = &data[ch * total + c0..ch * total + c1];
                let exact = col.iter().fold([f32::INFINITY, f32::NEG_INFINITY], |[a, b], &v| [a.min(v), b.max(v)]);
                assert_eq!(r.env[k * width + x], exact, "row {k} column {x}");
            }
        }
        let reads = src.reads.lock().unwrap();
        assert!(reads.len() > 1, "the window is streamed");
        assert_eq!(reads.first().unwrap().start, start as u64);
        assert_eq!(reads.last().unwrap().end, end as u64);
        for w in reads.windows(2) {
            assert_eq!(w[0].end, w[1].start, "contiguous, no sample read twice");
            assert_eq!(w[0].end % 300, 0, "blocks end on chunk boundaries");
        }
    }

    #[test]
    fn test_heat_color_endpoints() {
        assert_eq!(heat_color(0.0), BG_COLOR);
        assert_eq!(heat_color(1.0), Rgba8Pixel { r: 252, g: 230, b: 90, a: 255 });
        assert_eq!(heat_color(2.0), heat_color(1.0));
    }

    /// Per-frame render cost on the local datasets. Run with:
    /// `cargo test -p dsp-app --release bench_render -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_render() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../playground/data");
        let cases = [
            ("mearec_32ch_10s.bin", TimeViewKind::Traces, 0.1),
            ("mearec_32ch_10s.bin", TimeViewKind::Traces, 1.0),
            ("mearec_32ch_10s.bin", TimeViewKind::Heatmap, 1.0),
            ("mock_signal_384ch.bin", TimeViewKind::Traces, 1.0),
            ("mock_signal_384ch.bin", TimeViewKind::Heatmap, 1.0),
        ];
        for (file, mode, window_sec) in cases {
            let Ok(ds) = Dataset::open(&root.join(file)) else {
                println!("skip {file} (not found)");
                continue;
            };
            let mut r = WaveformRenderer::default();
            let mut req = RenderRequest { width: 1136, height: 550, window_sec, ..request(ds, mode) };
            if mode == TimeViewKind::Traces {
                req.channels = (0..8).collect();
            }
            r.render(&req); // warm-up
            let n = 20;
            let t0 = std::time::Instant::now();
            for _ in 0..n {
                std::hint::black_box(r.render(&req));
            }
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
            println!("{file:24} {mode:?} {window_sec:>4}s  {ms:6.2} ms/frame");
        }
    }
}
