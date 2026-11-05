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
