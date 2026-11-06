//! Charts of the Curation views: data in its own units (ms, µV, counts, µm) as layers — lines,
//! points, histogram bars, markers — drawn for a visible range into an image on a background
//! thread; the axes (ticks, values, names with units) are text laid over it by the panel, so they
//! stay sharp and readable at any size. One chart, or a grid of charts sharing the x range
//! (correlograms of several units, a template's channels).

use dspk_app_core::ui::plot::canvas::{Frame, Pixel};
use dspk_app_core::ui::theme::palette::Palette;

use crate::plots::polyline;

#[derive(Debug, Clone)]
pub enum Layer {
    /// Polylines in data coordinates.
    Lines { color: Pixel, alpha: f32, width: f32, lines: Vec<Vec<(f32, f32)>> },
    Points { color: Pixel, alpha: f32, radius: f32, points: Vec<(f32, f32)> },
    /// A histogram: bin `i` spans `start + i·width … start + (i+1)·width`.
    Bars { color: Pixel, start: f32, width: f32, counts: Vec<f32> },
    /// A vertical marker (zero lag, the refractory period).
    Marker { x: f32 },
}

/// One chart: its layers, axis names (with units), and the full extent of its data.
#[derive(Debug, Clone)]
pub struct Chart {
    pub layers: Vec<Layer>,
    pub x_label: String,
    pub y_label: String,
    pub x: (f32, f32),
    pub y: (f32, f32),
    /// The y range follows what is visible (histograms: from 0 to the tallest visible bar).
    pub y_fit: bool,
    /// A title drawn in the cell (grid cells: the pair of units, the channel).
    pub title: String,
}

/// Charts in a grid (one cell: a single chart), sharing the x range.
#[derive(Debug, Clone)]
pub struct ChartGrid {
    pub cells: Vec<Chart>,
    pub cols: usize,
}

/// The visible range: x, and y (ignored by cells that fit y).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Range2 {
    pub x: (f32, f32),
    pub y: (f32, f32),
}

/// Space around a single chart's plot area for the axes (logical px).
pub const MARGIN_LEFT: f32 = 52.;
pub const MARGIN_BOTTOM: f32 = 30.;
pub const MARGIN_TOP: f32 = 8.;
pub const MARGIN_RIGHT: f32 = 10.;
/// Gap between the cells of a grid (logical px).
pub const CELL_GAP: f32 = 6.;

impl ChartGrid {
    pub fn one(chart: Chart) -> Self {
        Self { cells: vec![chart], cols: 1 }
    }

    pub fn rows(&self) -> usize {
        self.cells.len().div_ceil(self.cols.max(1)).max(1)
    }

    /// Everything the data covers.
    pub fn full(&self) -> Range2 {
        let mut r = Range2 { x: (f32::INFINITY, f32::NEG_INFINITY), y: (f32::INFINITY, f32::NEG_INFINITY) };
        for c in &self.cells {
            r.x = (r.x.0.min(c.x.0), r.x.1.max(c.x.1));
            r.y = (r.y.0.min(c.y.0), r.y.1.max(c.y.1));
        }
        if !(r.x.0 < r.x.1) {
            r.x = (0.0, 1.0);
        }
        if !(r.y.0 < r.y.1) {
            r.y = (0.0, 1.0);
        }
        r
    }

    /// The y range a cell shows in `view` (its own fit, or the view's).
    pub fn cell_y(&self, i: usize, view: &Range2) -> (f32, f32) {
        let c = &self.cells[i];
        if !c.y_fit {
            return view.y;
        }
        let mut top = 0.0f32;
        for l in &c.layers {
            if let Layer::Bars { start, width, counts, .. } = l {
                for (k, &n) in counts.iter().enumerate() {
                    let (a, b) = (start + k as f32 * width, start + (k + 1) as f32 * width);
                    if b > view.x.0 && a < view.x.1 {
                        top = top.max(n);
                    }
                }
            }
        }
        (0.0, (top * 1.1).max(1.0))
    }
}

/// Zooms `r` by `factor` (< 1: closer) around the fraction `at` of the visible range.
pub fn zoom(r: (f32, f32), factor: f32, at: f32) -> (f32, f32) {
    let span = (r.1 - r.0) * factor;
    let a = r.0 + (r.1 - r.0) * at - span * at;
    (a, a + span)
}

/// About `n` round tick values (1, 2, 5 × 10ᵏ steps) within `r`.
pub fn ticks(r: (f32, f32), n: usize) -> Vec<f32> {
    let span = (r.1 - r.0).abs();
    if !(span > 0.0) || !span.is_finite() {
        return Vec::new();
    }
    let raw = span / n.max(1) as f32;
    let mag = 10f32.powf(raw.log10().floor());
    let step = [1.0, 2.0, 5.0, 10.0].iter().map(|m| m * mag).find(|s| *s >= raw).unwrap_or(10.0 * mag);
    let first = (r.0.min(r.1) / step).ceil() * step;
    (0..=n * 3).map(|i| first + i as f32 * step).take_while(|v| *v <= r.0.max(r.1) + step * 1e-3).collect()
}

/// `0.5`, `12`, `1.2k`, `3e-4`: short tick labels.
pub fn tick_label(v: f32) -> String {
    let a = v.abs();
    if a == 0.0 {
        "0".into()
    } else if a >= 10_000.0 {
        format!("{:.0}k", v / 1000.0)
    } else if a >= 100.0 {
        format!("{v:.0}")
    } else if a >= 1.0 {
        let s = format!("{v:.2}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else if a >= 0.001 {
        let s = format!("{v:.4}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        format!("{v:.0e}")
    }
}

/// The grid's cells in an image of `w × h` px (the plot areas only: axes are the panel's text),
/// for `view`; `scale` px per logical px.
pub fn draw(grid: &ChartGrid, view: &Range2, w: u32, h: u32, palette: &Palette, scale: f32) -> Frame {
    let mut frame = Frame::new(w.max(1), h.max(1));
    let mut canvas = frame.canvas();
    canvas.pixels.fill(palette.background);
    let (cols, rows) = (grid.cols.max(1), grid.rows());
    let gap = CELL_GAP * scale;
    let (cw, ch) = ((w as f32 - gap * (cols - 1) as f32) / cols as f32, (h as f32 - gap * (rows - 1) as f32) / rows as f32);
    for (i, cell) in grid.cells.iter().enumerate() {
        let (ox, oy) = ((i % cols) as f32 * (cw + gap), (i / cols) as f32 * (ch + gap));
        let (x0, x1) = view.x;
        let (y0, y1) = grid.cell_y(i, view);
        let px = |x: f32| ox + (x - x0) / (x1 - x0).max(1e-12) * cw;
        let py = |y: f32| oy + ch - (y - y0) / (y1 - y0).max(1e-12) * ch;
        let inside = |x: f32, y: f32| x >= ox - 1.0 && x <= ox + cw + 1.0 && y >= oy - 1.0 && y <= oy + ch + 1.0;
        // Frame and grid lines at the ticks
        if grid.cells.len() > 1 {
            canvas.rect(ox, oy, ox + cw, oy + ch, palette.grid, 0.25);
        }
        for t in ticks(view.x, 6) {
            let x = px(t);
            if x >= ox && x < ox + cw {
                canvas.vline_alpha(x as usize, oy as usize, (oy + ch) as usize, palette.grid, 0.6);
            }
        }
        for t in ticks((y0, y1), 4) {
            let y = py(t);
            if y >= oy && y < oy + ch {
                canvas.rect(ox, y, ox + cw, y + 1.0, palette.grid, 0.6);
            }
        }
        for layer in &cell.layers {
            match layer {
                Layer::Bars { color, start, width, counts } => {
                    for (k, &n) in counts.iter().enumerate() {
                        let (a, b) = (px(start + k as f32 * width), px(start + (k + 1) as f32 * width));
                        let (a, b) = (a.max(ox), b.min(ox + cw));
                        if b <= a || n <= 0.0 {
                            continue;
                        }
                        let top = py(n).max(oy);
                        canvas.rect(a, top, (b - 0.4 * scale).max(a + 0.5), oy + ch, *color, 0.9);
                    }
                }
                Layer::Lines { color, alpha, width, lines } => {
                    for line in lines {
                        let pts: Vec<(f32, f32)> = line.iter().map(|&(x, y)| (px(x), py(y).clamp(oy, oy + ch))).filter(|&(x, _)| x >= ox - cw && x <= ox + 2.0 * cw).collect();
                        let clipped: Vec<(f32, f32)> = pts.into_iter().map(|(x, y)| (x.clamp(ox, ox + cw), y)).collect();
                        polyline(&mut canvas, &clipped, *color, *alpha, width * scale);
                    }
                }
                Layer::Points { color, alpha, radius, points } => {
                    let r = radius * scale;
                    for &(x, y) in points {
                        let (x, y) = (px(x), py(y));
                        if inside(x, y) {
                            canvas.rect(x - r, y - r, x + r, y + r, *color, *alpha);
                        }
                    }
                }
                Layer::Marker { x } => {
                    let x = px(*x);
                    if x >= ox && x < ox + cw {
                        canvas.vline(x as usize, oy as usize, (oy + ch) as usize, palette.text);
                    }
                }
            }
        }
    }
    frame
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ticks_zoom_labels() {
        assert_eq!(ticks((0.0, 10.0), 5), vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
        assert_eq!(ticks((-50.0, 50.0), 4), vec![-50.0, 0.0, 50.0]);
        assert_eq!(zoom((0.0, 10.0), 0.5, 0.5), (2.5, 7.5));
        assert_eq!(zoom((0.0, 10.0), 0.5, 0.0), (0.0, 5.0));
        assert_eq!(tick_label(1500.0), "1500");
        assert_eq!(tick_label(0.25), "0.25");
        assert_eq!(tick_label(2.0), "2");
    }

    #[test]
    fn test_histograms_fit_the_visible_bars() {
        let c = Chart { layers: vec![Layer::Bars { color: Palette::of(true).text, start: 0.0, width: 1.0, counts: vec![1.0, 10.0, 3.0] }], x_label: String::new(), y_label: String::new(), x: (0.0, 3.0), y: (0.0, 10.0), y_fit: true, title: String::new() };
        let g = ChartGrid::one(c);
        let (lo, hi) = g.cell_y(0, &Range2 { x: (2.0, 3.0), y: (0.0, 1.0) });
        assert!(lo == 0.0 && (hi - 3.3).abs() < 1e-4, "only the visible bar (3) counts: {hi}");
    }
}
