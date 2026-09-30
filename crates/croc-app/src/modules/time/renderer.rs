//! Waveform rasterizer for the plot area.
//!
//! Draws only the plot itself (traces or heatmap, grid, spike ticks, scale bar); all text
//! (channel labels, time axis, readouts) is laid out by Slint around and over the image.

use std::sync::Arc;

use dsp_stream::min_max_decimate_into;
use slint::{Rgba8Pixel, SharedPixelBuffer};

use crate::data::{SignalSource, SpikeEventStore};
use crate::shared::canvas::{blend_color, Canvas};
use crate::shared::render_worker::Frame;

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

/// Amplitude (µV) that maps to `LANE_FILL` of a lane's half-height at gain 1x.
pub const NOMINAL_UV: f32 = 80.0;
/// Fraction of the lane half-height used by `NOMINAL_UV`.
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
    pub source: Arc<dyn SignalSource>,
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
    pub amplitude_scale: f32,
    /// Times of vertical grid lines (the time-axis ticks).
    pub grid_times: Vec<f64>,
    /// Height of the amplitude scale bar in µV (0 = none).
    pub scale_bar_uv: f32,
    /// Sorted spikes to mark in a cluster color: (color, [(time_sec, channel)]).
    pub highlights: Vec<(Rgba8Pixel, Vec<(f64, usize)>)>,
}

thread_local! {
    /// Scratch buffers of the worker thread, reused across frames.
    static RENDERER: std::cell::RefCell<WaveformRenderer> = std::cell::RefCell::new(WaveformRenderer::default());
}

/// Renders `req` with this thread's reusable renderer (call from the render worker).
pub fn render_on_worker(req: &RenderRequest) -> Frame {
    RENDERER.with(|r| r.borrow_mut().render(req))
}

/// Pixels per µV for a lane of `lane_h` pixels at the given gain.
pub fn px_per_uv(lane_h: f32, amplitude_scale: f32) -> f32 {
    lane_h * 0.5 * LANE_FILL * amplitude_scale / NOMINAL_UV
}

/// Rasterizer with reusable scratch buffers (no per-frame allocations besides the image).
#[derive(Default)]
pub struct WaveformRenderer {
    buckets: Vec<[f32; 2]>,
    heat: Vec<f32>,
}

impl WaveformRenderer {
    pub fn render(&mut self, req: &RenderRequest) -> SharedPixelBuffer<Rgba8Pixel> {
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

        let samples = source.samples();
        if samples == 0 || source.channels() == 0 || req.window_sec <= 0.0 {
            return pixel_buffer;
        }

        // Visible sample range
        let sr = source.sample_rate();
        let start = ((req.window_start_sec * sr).round().max(0.0) as usize).min(samples);
        let end = (((req.window_start_sec + req.window_sec) * sr).round().max(0.0) as usize).min(samples);

        // Time grid aligned with the axis ticks
        let w = canvas.width as f64;
        for &t in &req.grid_times {
            let x = ((t - req.window_start_sec) / req.window_sec * w).round();
            if x >= 0.0 && x < w {
                canvas.vline(x as usize, 0, canvas.height, GRID_COLOR);
            }
        }

        self.buckets.resize(canvas.width, [0.0, 0.0]);
        match req.mode {
            TimeViewKind::Traces => self.draw_traces(&mut canvas, source, &req.events, req, start, end),
            TimeViewKind::Heatmap => self.draw_heatmap(&mut canvas, source, req, start, end),
        }

        pixel_buffer
    }

    fn draw_traces(
        &mut self,
        canvas: &mut Canvas,
        source: &dyn SignalSource,
        events: &SpikeEventStore,
        req: &RenderRequest,
        start: usize,
        end: usize,
    ) {
        let num_channels = req.channels.len();
        if num_channels == 0 {
            return;
        }

        let lane_h = canvas.height as f32 / num_channels as f32;
        let k = px_per_uv(lane_h, req.amplitude_scale);
        let thickness = req.scale.max(1.0);
        let t_end = req.window_start_sec + req.window_sec;

        for lane in 0..num_channels {
            let ch = req.channels[lane];
            if ch >= source.channels() {
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

            if end <= start {
                continue;
            }
            min_max_decimate_into(&source.channel(ch)[start..end], &mut self.buckets);

            // Anti-aliased min/max envelope, joined to the previous column for continuity
            let (lo_clip, hi_clip) = (0.0, canvas.height as f32 - 1.0);
            let mut prev: Option<(f32, f32)> = None;
            for (x, &[mn, mx]) in self.buckets.iter().enumerate() {
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
        if req.scale_bar_uv > 0.0 {
            let bar_h = req.scale_bar_uv * k;
            let x = canvas.width.saturating_sub((12.0 * req.scale) as usize);
            let bottom = canvas.height as f32 - 6.0 * req.scale;
            let top = (bottom - bar_h).max(0.0);
            for dx in 0..(thickness as usize).max(1) {
                canvas.vline(x + dx, top as usize, bottom as usize, TEXT_COLOR);
            }
        }
    }

    fn draw_heatmap(
        &mut self,
        canvas: &mut Canvas,
        source: &dyn SignalSource,
        req: &RenderRequest,
        start: usize,
        end: usize,
    ) {
        let rows = &req.channels;
        let channels = rows.len();
        let (w, h) = (canvas.width, canvas.height);
        if end <= start || channels == 0 {
            return;
        }

        // Peak |amplitude| per (row, column), normalized by the gain-scaled nominal range
        self.heat.resize(channels * w, 0.0);
        let norm = req.amplitude_scale / (NOMINAL_UV * 1.5);
        for (r, &ch) in rows.iter().enumerate() {
            let row = &mut self.heat[r * w..(r + 1) * w];
            if ch >= source.channels() {
                row.fill(0.0);
                continue;
            }
            min_max_decimate_into(&source.channel(ch)[start..end], &mut self.buckets);
            for (v, &[mn, mx]) in row.iter_mut().zip(&self.buckets) {
                // Squared for contrast: background noise stays dark, spikes stand out
                let a = (mn.abs().max(mx.abs()) * norm).min(1.0);
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
            source: Arc::new(ds),
            events,
            width: 800,
            height: 400,
            scale: 1.0,
            mode,
            channels,
            window_start_sec: 0.0,
            window_sec: 0.100,
            amplitude_scale: 1.0,
            grid_times: vec![0.05],
            scale_bar_uv: 50.0,
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

    #[test]
    fn test_heat_color_endpoints() {
        assert_eq!(heat_color(0.0), BG_COLOR);
        assert_eq!(heat_color(1.0), Rgba8Pixel { r: 252, g: 230, b: 90, a: 255 });
        assert_eq!(heat_color(2.0), heat_color(1.0));
    }

    /// Per-frame render cost on the local datasets. Run with:
    /// `cargo test -p croc-app --release bench_render -- --ignored --nocapture`
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
            let Ok(ds) = Dataset::load_from_file(&root.join(file), None, None) else {
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
