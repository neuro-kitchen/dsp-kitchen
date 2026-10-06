//! Waveform rasterizer for the plot area.
//!
//! Draws only the plot itself (traces or heatmap, grid, spike ticks, scale bar); all text
//! (channel labels, time axis, readouts), the playhead and the cursor are elements the view lays
//! out around and over the image.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use dsp_base::math::percentile;
use dsp_core::SignalUnit;
use dsp_view::{min_max_decimate_into, Envelope, SignalBackend, View};

use crate::engine::canvas::{Canvas, Frame};
use crate::engine::ticks::nice_step;
use crate::engine::palette::Palette;
use crate::engine::data::SpikeEventStore;

/// Voltage that fills `LANE_FILL` of a lane's half-height at gain 1x when a view is not
/// auto-scaled (80 µV: a large extracellular spike).
pub const NOMINAL_VOLTS: f64 = 80e-6;
/// The same for signals that are not voltages (no natural size: auto-scale is the useful mode).
pub const NOMINAL_OTHER: f32 = 1.0;
/// Fraction of the lane half-height used by the nominal amplitude.
const LANE_FILL: f32 = 0.84;

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
    /// The signal shown: asked for the envelope of the visible window (zoomed out from its
    /// pyramid, zoomed in from the samples).
    pub signal: Arc<dyn SignalBackend>,
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
    /// Amplitude (signal unit) filling a lane when not auto-scaled ([`nominal_amplitude`]).
    pub nominal: f32,
    /// Fit the amplitude to the visible data (else `nominal` fills a lane).
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
    pub highlights: Vec<(crate::engine::canvas::Pixel, Vec<(f64, usize)>)>,
    pub palette: Palette,
}

thread_local! {
    /// Scratch buffers of the worker thread, reused across frames.
    static RENDERER: std::cell::RefCell<WaveformRenderer> = std::cell::RefCell::new(WaveformRenderer::default());
}

/// Renders `req` with this thread's reusable renderer (call from a work thread): the frame and
/// the amplitude scale it was drawn with, and whether every column was available. Zoomed-out
/// windows draw from the signal's pyramid as far as it is built (the rest stays empty until its
/// builder reaches it: draw again on progress while incomplete); `None` when a newer frame for
/// the view cancelled this one.
pub fn render_on_worker(req: &RenderRequest, cancel: &AtomicBool) -> Option<(Frame, f32, bool)> {
    if cancel.load(Ordering::Relaxed) {
        return None;
    }
    Some(RENDERER.with(|r| {
        let mut r = r.borrow_mut();
        let (frame, scale) = r.render_scaled(req);
        (frame, scale, r.complete)
    }))
}

/// Visible sample range of `req`'s source and the pixel columns `x0..x1` it covers (sources can
/// start later or end earlier than the window); `None` when nothing is visible.
fn visible(req: &RenderRequest) -> Option<(usize, usize, usize, usize)> {
    let info = req.signal.info();
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

/// Nominal amplitude of a signal in `unit`: [`NOMINAL_VOLTS`] in that unit for voltages, else
/// [`NOMINAL_OTHER`].
pub fn nominal_amplitude(unit: &SignalUnit) -> f32 {
    let volts_per_unit = match unit {
        SignalUnit::Volt => 1.0,
        SignalUnit::Millivolt => 1e-3,
        SignalUnit::Microvolt => 1e-6,
        _ => return NOMINAL_OTHER,
    };
    (NOMINAL_VOLTS / volts_per_unit) as f32
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
fn choose_scale(amplitude: f32, hint: f32, nominal: f32) -> f32 {
    if amplitude.is_nan() || amplitude <= 0.0 || !amplitude.is_finite() {
        return if hint > 0.0 { hint } else { nominal };
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
    /// The signal's answer for the read channels (its buffer reused across frames).
    answer: Option<Envelope>,
    /// Whether the last frame had every column it needed (false while the pyramid builds).
    complete: bool,
    heat: Vec<f32>,
    /// Scratch for medians and percentiles.
    stats: Vec<f32>,
}

impl WaveformRenderer {
    #[cfg(test)]
    pub fn render(&mut self, req: &RenderRequest) -> Frame {
        self.render_scaled(req).0
    }

    /// Renders and returns the amplitude scale used (units filling a lane at gain 1). Zoomed out,
    /// columns come from the signal's pyramid (pages not built yet draw empty).
    pub fn render_scaled(&mut self, req: &RenderRequest) -> (Frame, f32) {
        let signal = req.signal.as_ref();
        let width = req.width.max(1);
        let height = req.height.max(1);
        let mut pixel_buffer = Frame::new(width, height);
        let mut canvas = pixel_buffer.canvas();
        canvas.pixels.fill(req.palette.background);

        let Some((start, end, x0, x1)) = visible(req) else {
            return (pixel_buffer, req.scale_hint.max(0.0));
        };

        // Time grid aligned with the axis ticks
        let w = canvas.width as f64;
        for &t in &req.grid_times {
            let x = ((t - req.window_start_sec) / req.window_sec * w).round();
            if x >= 0.0 && x < w {
                canvas.vline(x as usize, 0, canvas.height, req.palette.grid);
            }
        }

        self.envelope(signal, &req.channels, start, end, x1.saturating_sub(x0));
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
    /// else the request's nominal amplitude.
    fn adjust(&mut self, req: &RenderRequest, width: usize) -> f32 {
        let rows = req.channels.len();
        if req.remove_dc {
            for r in (0..rows).filter(|&r| self.valid[r]) {
                let row = &mut self.env[r * width..(r + 1) * width];
                self.stats.clear();
                self.stats.extend(row.iter().filter(|v| v[0].is_finite()).map(|v| 0.5 * (v[0] + v[1])));
                let (Some(q1), Some(q3)) = (percentile(&mut self.stats, 25.0), percentile(&mut self.stats, 75.0)) else { continue };
                let offset = 0.5 * (q1 + q3);
                for v in row.iter_mut() {
                    v[0] -= offset;
                    v[1] -= offset;
                }
            }
        }
        if !req.auto_scale {
            return req.nominal;
        }
        self.stats.clear();
        for r in (0..rows).filter(|&r| self.valid[r]) {
            self.stats.extend(self.env[r * width..(r + 1) * width].iter().filter(|v| v[0].is_finite()).map(|v| v[0].abs().max(v[1].abs())));
        }
        choose_scale(percentile(&mut self.stats, 99.0).unwrap_or(0.0), req.scale_hint, req.nominal)
    }

    /// Fills `env` with the `[min, max]` of every requested row per pixel column over
    /// `start..end`, as the signal answers the view (zoomed out from its pyramid, else from the
    /// samples, streamed in bounded blocks). A window of fewer samples than columns comes back as
    /// samples and is spread over the columns. Every sample of the window lands in a column, so no
    /// peak is dropped. Rows of channels the signal lacks, and every row when it cannot answer,
    /// stay NaN (drawn empty).
    fn envelope(&mut self, signal: &dyn SignalBackend, rows: &[usize], start: usize, end: usize, width: usize) {
        let total = signal.info().channel_count();
        self.complete = true;
        self.env.clear();
        self.env.resize(rows.len() * width, [f32::NAN, f32::NAN]);
        self.valid.clear();
        self.valid.extend(rows.iter().map(|&c| c < total));
        self.read_channels.clear();
        self.read_rows.clear();
        for (r, &c) in rows.iter().enumerate().filter(|&(_, &c)| c < total) {
            self.read_channels.push(c);
            self.read_rows.push(r);
        }
        if self.read_channels.is_empty() || end <= start || width == 0 {
            return;
        }
        let view = View { channels: self.read_channels.clone(), start: start as u64, end: end as u64, width };
        let answer = self.answer.get_or_insert_with(|| Envelope::Samples(Vec::new()));
        if let Err(e) = signal.view(&view, answer) {
            tracing::warn!("cannot draw samples {start}..{end}: {e}");
            return;
        }
        match answer {
            Envelope::Columns { values, complete } => {
                self.complete = *complete;
                for (k, &r) in self.read_rows.iter().enumerate() {
                    self.env[r * width..(r + 1) * width].copy_from_slice(&values[k * width..(k + 1) * width]);
                }
            }
            Envelope::Samples(samples) => {
                let n = end - start;
                for (k, &r) in self.read_rows.iter().enumerate() {
                    min_max_decimate_into(&samples[k * n..(k + 1) * n], &mut self.env[r * width..(r + 1) * width]);
                }
            }
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
            let color = req.palette.channel(ch);
            let center = (lane as f32 + 0.5) * lane_h;

            // Dashed baseline
            let by = center.round() as usize;
            for x in (0..canvas.width).filter(|x| x % 4 != 0) {
                canvas.set(x, by, req.palette.baseline);
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
                canvas.vline_alpha(x, top, bottom, req.palette.marker, 0.18);
                let tick = (6.0 * req.scale) as usize;
                for dx in 0..(thickness as usize).max(1) {
                    canvas.vline(x + dx, top + 1, top + 1 + tick, req.palette.marker);
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
                canvas.vline(x + dx, top as usize, bottom as usize, req.palette.text);
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
                canvas.set(x, y, req.palette.heat(v));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::data::Dataset;

    /// Request for `mode` over `ds`: traces show the first 4 channels, heatmap every channel.
    fn request(ds: Dataset, mode: TimeViewKind) -> RenderRequest {
        let events = Arc::new(SpikeEventStore::detect(ds.signal().as_ref()));
        let channels = match mode {
            TimeViewKind::Traces => (0..4.min(ds.total_channels)).collect(),
            TimeViewKind::Heatmap => (0..ds.total_channels).collect(),
        };
        RenderRequest {
            signal: ds.signal().clone(),
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
            nominal: nominal_amplitude(&SignalUnit::Microvolt),
            auto_scale: false,
            remove_dc: false,
            scale_hint: 0.0,
            grid_times: vec![0.05],
            scale_bar: true,
            highlights: vec![(crate::engine::canvas::Pixel::rgb(255, 0, 0), vec![(0.05, 0)])],
            palette: Palette::DARK,
        }
    }

    #[test]
    fn test_renderer_output_dimensions_and_trace_pixels() {
        let mut r = WaveformRenderer::default();
        for mode in [TimeViewKind::Traces, TimeViewKind::Heatmap] {
            let req = request(Dataset::generate_synthetic(8, 30_000.0, 0.5), mode);
            let buf = r.render(&req);
            assert_eq!((buf.width, buf.height), (800, 400));
            // Something other than background was drawn
            assert!(buf.pixels().iter().any(|p| *p != Palette::DARK.background));
        }
    }

    #[test]
    fn test_out_of_range_channels_are_skipped() {
        let mut req = request(Dataset::generate_synthetic(2, 10_000.0, 0.2), TimeViewKind::Traces);
        req.channels = vec![0, 99];
        let mut r = WaveformRenderer::default();
        assert_eq!(r.render(&req).width, 800);
        req.mode = TimeViewKind::Heatmap;
        assert_eq!(r.render(&req).width, 800);
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
        assert_eq!(fixed, nominal_amplitude(&SignalUnit::Microvolt));
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
        let (w, h) = (buf.width as usize, buf.height as usize);
        let px = buf.pixels();
        let drawn = |x0: usize, x1: usize| (0..h).any(|y| (x0..x1).any(|x| px[y * w + x] != Palette::DARK.background && px[y * w + x] != Palette::DARK.baseline));
        assert!(!drawn(0, w / 2 - 2), "no data before the source starts");
        assert!(drawn(w / 2 + 2, w - 20), "data after it starts");
    }

    /// In-memory source that reports a storage chunk size and records every read range.
    struct Chunked {
        inner: dsp_core::MemoryRecording,
        chunk: u64,
        reads: Arc<std::sync::Mutex<Vec<std::ops::Range<u64>>>>,
    }

    impl dsp_core::RecordingSource for Chunked {
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
    fn test_raw_windows_are_exact_and_read_each_chunk_once() {
        // Zoomed in below the pyramid's base: the envelope comes from the samples
        let (nch, total, width) = (3usize, 10_000usize, 37usize);
        let data: Vec<f32> = (0..nch * total).map(|i| ((i * 7919) % 1013) as f32 - 500.0).collect();
        let reads = Arc::new(std::sync::Mutex::new(Vec::new()));
        let inner = dsp_core::MemoryRecording::new("chunked", data.clone(), nch, 1000.0).unwrap();
        let src = Chunked { inner, chunk: 300, reads: reads.clone() };
        let ds = Dataset::local(Arc::new(src), None).unwrap();
        let (start, end) = (1_234usize, 9_876usize);
        assert!(((end - start) / width) < dsp_view::MEMORY_BASE as usize, "a raw window");

        let mut r = WaveformRenderer::default();
        let rows = [2usize, 0];
        r.envelope(ds.signal().as_ref(), &rows, start, end, width);

        let n = end - start;
        for (k, &ch) in rows.iter().enumerate() {
            for x in 0..width {
                let (c0, c1) = (start + x * n / width, start + (x + 1) * n / width);
                let col = &data[ch * total + c0..ch * total + c1];
                let exact = col.iter().fold([f32::INFINITY, f32::NEG_INFINITY], |[a, b], &v| [a.min(v), b.max(v)]);
                assert_eq!(r.env[k * width + x], exact, "row {k} column {x}");
            }
        }
        let reads = reads.lock().unwrap();
        assert_eq!(reads.first().unwrap().start, start as u64);
        assert_eq!(reads.last().unwrap().end, end as u64);
        for w in reads.windows(2) {
            assert_eq!(w[0].end, w[1].start, "contiguous, no sample read twice");
            assert_eq!(w[0].end % 300, 0, "blocks end on chunk boundaries");
        }
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
            let Ok(sources) = crate::engine::data::SourceSet::open(&root.join(file)) else {
                println!("skip {file} (not found)");
                continue;
            };
            let ds = Dataset::new(sources.default_dataset().signal().clone(), None);
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
