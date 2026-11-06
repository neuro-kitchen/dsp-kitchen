//! The Sorting workspace's plots, drawn on the CPU into frames (as the trace renderer does) on a
//! background thread: waveform overlays and the 2-D feature space.

use std::sync::Arc;

use dspk_app_core::ui::plot::canvas::{Canvas, Frame, Pixel};
use dspk_app_core::ui::theme::palette::Palette;
use gpui_kit::RenderImage;

use crate::sorting::Snippets;

/// Opacity of one spike's waveform (many overlap).
const WAVE_ALPHA: f32 = 0.12;
/// Opacity of a point of the feature space.
const POINT_ALPHA: f32 = 0.5;

/// A frame as an image GPUI shows.
pub fn to_image(frame: Frame) -> Option<Arc<RenderImage>> {
    let (w, h) = (frame.width, frame.height);
    let buffer = image::RgbaImage::from_raw(w, h, frame.into_bgra())?;
    Some(Arc::new(RenderImage::new(vec![image::Frame::new(buffer)])))
}

fn colour(unit: Option<u32>, palette: &Palette) -> Pixel {
    unit.map_or(palette.unsorted(), |u| palette.unit(u as usize))
}

/// A line through `points` (pixels), one vertical span per column between consecutive points.
fn polyline(canvas: &mut Canvas, points: &[(f32, f32)], c: Pixel, alpha: f32, width: f32) {
    for w in points.windows(2) {
        let ((x0, y0), (x1, y1)) = (w[0], w[1]);
        let (xa, xb) = (x0.round() as i64, x1.round() as i64);
        for x in xa.max(0)..=xb.max(xa) {
            if x as usize >= canvas.width {
                break;
            }
            let f = if xb > xa { (x - xa) as f32 / (xb - xa) as f32 } else { 0.0 };
            let f2 = if xb > xa { ((x + 1 - xa) as f32 / (xb - xa) as f32).min(1.0) } else { 1.0 };
            let (ya, yb) = (y0 + (y1 - y0) * f, y0 + (y1 - y0) * f2);
            let (top, bottom) = (ya.min(yb) - width * 0.5, ya.max(yb) + width * 0.5);
            canvas.span_aa_alpha(x as usize, top, bottom, c, alpha);
        }
    }
}

/// Every snippet in its unit's colour (unsorted grey), then each unit's mean on top.
pub fn waveforms(snippets: &Snippets, palette: &Palette, width: u32, height: u32, scale: f32) -> Frame {
    let mut frame = Frame::new(width, height);
    let mut canvas = frame.canvas();
    canvas.pixels.fill(palette.background);
    let n = (snippets.before + snippets.after) as usize;
    if snippets.rows.is_empty() || n < 2 {
        return frame;
    }
    // Amplitude range: 1st–99th percentile of every value, with a margin
    let mut all: Vec<f32> = snippets.rows.iter().flat_map(|(_, _, r)| r.iter().copied()).filter(|v| v.is_finite()).collect();
    all.sort_by(f32::total_cmp);
    let q = |p: f32| all[((all.len() - 1) as f32 * p) as usize];
    let (lo, hi) = (q(0.005), q(0.995));
    let span = (hi - lo).max(1e-6) * 1.1;
    let mid = 0.5 * (hi + lo);
    let (w, h) = (width as f32, height as f32);
    let to_px = |i: usize, v: f32| (i as f32 / (n - 1) as f32 * (w - 1.0), h * 0.5 - (v - mid) / span * h);
    // Spike time
    let x0 = to_px(snippets.before as usize, 0.0).0;
    canvas.vline(x0 as usize, 0, height as usize, palette.grid);
    let row_points = |r: &[f32]| r.iter().enumerate().map(|(i, &v)| to_px(i, v)).collect::<Vec<_>>();
    // Unsorted first (under), then units
    for (u, _, r) in snippets.rows.iter().filter(|(u, _, _)| u.is_none()).chain(snippets.rows.iter().filter(|(u, _, _)| u.is_some())) {
        polyline(&mut canvas, &row_points(r), colour(*u, palette), WAVE_ALPHA, scale);
    }
    let mut units: Vec<u32> = snippets.rows.iter().filter_map(|(u, _, _)| *u).collect();
    units.sort_unstable();
    units.dedup();
    for u in units {
        let rows: Vec<&Vec<f32>> = snippets.rows.iter().filter(|(x, _, _)| *x == Some(u)).map(|(_, _, r)| r).collect();
        let mean: Vec<f32> = (0..n).map(|i| rows.iter().map(|r| r[i]).sum::<f32>() / rows.len() as f32).collect();
        polyline(&mut canvas, &row_points(&mean), palette.text, 1.0, 3.0 * scale);
        polyline(&mut canvas, &row_points(&mean), colour(Some(u), palette), 1.0, 1.6 * scale);
    }
    frame
}

/// `points[k]` (PC1, PC2) of spike `units[k]`, coloured by unit, unsorted grey underneath.
pub fn scatter(points: &[[f32; 2]], units: &[Option<u32>], palette: &Palette, width: u32, height: u32, scale: f32) -> Frame {
    let mut frame = Frame::new(width, height);
    let mut canvas = frame.canvas();
    canvas.pixels.fill(palette.background);
    if points.is_empty() {
        return frame;
    }
    let range = |i: usize| {
        let mut v: Vec<f32> = points.iter().map(|p| p[i]).filter(|v| v.is_finite()).collect();
        v.sort_by(f32::total_cmp);
        let (a, b) = (v[(v.len() - 1) / 200], v[(v.len() - 1) - (v.len() - 1) / 200]);
        let pad = (b - a).max(1e-6) * 0.08;
        (a - pad, b + pad)
    };
    let ((x0, x1), (y0, y1)) = (range(0), range(1));
    let (w, h) = (width as f32, height as f32);
    // Axes through zero
    let zx = (-x0 / (x1 - x0) * w).clamp(0.0, w - 1.0);
    let zy = (h - (-y0 / (y1 - y0)) * h).clamp(0.0, h - 1.0);
    canvas.vline(zx as usize, 0, height as usize, palette.grid);
    canvas.rect(0.0, zy, w, zy + 1.0, palette.grid, 1.0);
    let r = 1.6 * scale;
    let order = (0..points.len()).filter(|&k| units[k].is_none()).chain((0..points.len()).filter(|&k| units[k].is_some()));
    for k in order {
        let [a, b] = points[k];
        let (x, y) = ((a - x0) / (x1 - x0) * w, h - (b - y0) / (y1 - y0) * h);
        canvas.rect(x - r, y - r, x + r, y + r, colour(units[k], palette), POINT_ALPHA);
    }
    frame
}

/// A histogram: `counts` as bars in `c`; `marker` draws a vertical line at that bin (e.g. zero
/// lag, or the refractory period).
pub fn bars(counts: &[u32], c: Pixel, marker: Option<f32>, palette: &Palette, width: u32, height: u32) -> Frame {
    let mut frame = Frame::new(width, height);
    let mut canvas = frame.canvas();
    canvas.pixels.fill(palette.background);
    let peak = counts.iter().copied().max().unwrap_or(0).max(1) as f32;
    let (w, h) = (width as f32, height as f32);
    let bw = w / counts.len().max(1) as f32;
    if let Some(m) = marker {
        canvas.vline((m * bw) as usize, 0, height as usize, palette.grid);
    }
    for (i, &n) in counts.iter().enumerate() {
        let top = h - (n as f32 / peak) * (h - 2.0);
        canvas.rect(i as f32 * bw + 0.5, top, (i + 1) as f32 * bw - 0.5, h, c, 0.9);
    }
    frame
}

/// Amplitude over time: `points[u]` = (time s, amplitude) of selected unit `units[u]`, over
/// `duration_sec`.
pub fn amplitude_time(points: &[(u32, Vec<(f32, f32)>)], duration_sec: f32, palette: &Palette, width: u32, height: u32, scale: f32) -> Frame {
    let mut frame = Frame::new(width, height);
    let mut canvas = frame.canvas();
    canvas.pixels.fill(palette.background);
    let all: Vec<f32> = points.iter().flat_map(|(_, p)| p.iter().map(|q| q.1)).filter(|v| v.is_finite()).collect();
    if all.is_empty() {
        return frame;
    }
    let (lo, hi) = all.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    let lo = lo.min(0.0);
    let span = (hi - lo).max(1e-6) * 1.08;
    let (w, h) = (width as f32, height as f32);
    let r = 1.2 * scale;
    for (u, pts) in points {
        let c = palette.unit(*u as usize);
        for &(t, a) in pts {
            let (x, y) = (t / duration_sec.max(1e-6) * w, h - (a - lo) / span * h);
            canvas.rect(x - r, y - r, x + r, y + r, c, POINT_ALPHA);
        }
    }
    frame
}

/// A unit's mean template on its channels (`rows`, labelled by the caller), one small panel per
/// channel in a grid, on one shared amplitude scale.
pub fn template_grid(rows: &[Vec<f32>], c: Pixel, palette: &Palette, width: u32, height: u32, scale: f32) -> Frame {
    let mut frame = Frame::new(width, height);
    let mut canvas = frame.canvas();
    canvas.pixels.fill(palette.background);
    let n = rows.len();
    if n == 0 {
        return frame;
    }
    let cols = (n as f32).sqrt().ceil() as usize;
    let lines = n.div_ceil(cols);
    let (cw, ch) = (width as f32 / cols as f32, height as f32 / lines as f32);
    let amp = rows.iter().flat_map(|r| r.iter()).fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    for (k, row) in rows.iter().enumerate() {
        let (x0, y0) = ((k % cols) as f32 * cw, (k / cols) as f32 * ch);
        let mid = y0 + ch * 0.5;
        canvas.rect(x0 + 4.0, mid, x0 + cw - 4.0, mid + 1.0, palette.grid, 1.0);
        let len = row.len().max(2);
        let pts: Vec<(f32, f32)> = row.iter().enumerate().map(|(i, &v)| (x0 + 4.0 + i as f32 / (len - 1) as f32 * (cw - 8.0), mid - v / amp * ch * 0.45)).collect();
        polyline(&mut canvas, &pts, c, 1.0, 1.4 * scale);
    }
    frame
}

/// A unit card (Plexon style): its waveform's mean ± one standard deviation over `rows` (top),
/// and its ISI histogram (bottom, the refractory bin marked).
pub fn unit_card(rows: &[&Vec<f32>], isi: &[u32], refractory_bin: f32, c: Pixel, palette: &Palette, width: u32, height: u32, scale: f32) -> Frame {
    let mut frame = Frame::new(width, height);
    let mut canvas = frame.canvas();
    canvas.pixels.fill(palette.background);
    let (w, h) = (width as f32, height as f32);
    let wave_h = h * 0.62;
    if let Some(first) = rows.first() {
        let n = first.len();
        if n >= 2 {
            let k = rows.len() as f32;
            let mean: Vec<f32> = (0..n).map(|i| rows.iter().map(|r| r[i]).sum::<f32>() / k).collect();
            let sd: Vec<f32> = (0..n).map(|i| (rows.iter().map(|r| (r[i] - mean[i]).powi(2)).sum::<f32>() / k).sqrt()).collect();
            let lo = mean.iter().zip(&sd).map(|(m, s)| m - s).fold(f32::INFINITY, f32::min);
            let hi = mean.iter().zip(&sd).map(|(m, s)| m + s).fold(f32::NEG_INFINITY, f32::max);
            let span = (hi - lo).max(1e-6) * 1.15;
            let mid = 0.5 * (hi + lo);
            let px_of = |i: usize, v: f32| (i as f32 / (n - 1) as f32 * (w - 1.0), wave_h * 0.5 - (v - mid) / span * wave_h);
            canvas.rect(0.0, wave_h * 0.5, w, wave_h * 0.5 + 1.0, palette.grid, 1.0);
            let band = |sign: f32| mean.iter().zip(&sd).enumerate().map(|(i, (m, s))| px_of(i, m + sign * s)).collect::<Vec<_>>();
            polyline(&mut canvas, &band(1.0), c, 0.6, 1.0 * scale);
            polyline(&mut canvas, &band(-1.0), c, 0.6, 1.0 * scale);
            let m: Vec<(f32, f32)> = mean.iter().enumerate().map(|(i, &v)| px_of(i, v)).collect();
            polyline(&mut canvas, &m, c, 1.0, 2.0 * scale);
        }
    }
    // ISI histogram under the waveform
    let (top, bottom) = (wave_h + 4.0, h - 1.0);
    canvas.rect(0.0, top - 2.0, w, top - 1.0, palette.grid, 1.0);
    let peak = isi.iter().copied().max().unwrap_or(0).max(1) as f32;
    let bw = w / isi.len().max(1) as f32;
    canvas.vline((refractory_bin * bw) as usize, top as usize, bottom as usize, palette.grid);
    for (i, &n) in isi.iter().enumerate() {
        let y = bottom - (n as f32 / peak) * (bottom - top);
        canvas.rect(i as f32 * bw + 0.3, y, (i + 1) as f32 * bw - 0.3, bottom, c, 0.9);
    }
    frame
}
