//! Reusable 2D plot kit (Step 5c): data ↔ pixel coordinate mapping, 1-2-5 axis ticks, wheel zoom,
//! drag pan, double-click reset, polygon lasso in data coordinates (`Ctrl+click` adds points,
//! `Ctrl+right-click` closes), Phy cluster selection palette, and GPU path / quad painters.

use gpui_kit::{fill, point, px, size, Bounds, Hsla, PathBuilder, Pixels, Point, Window};

use dsp_base::math::{point_in_polygon, ticks};
use crate::engine::canvas::Pixel;
use crate::views::trace::color;

/// Phy's cluster selection colour sequence (first selected = blue, second = red, third = green,
/// fourth = cyan, fifth = magenta, sixth = yellow, etc.).
pub const PHY_CLUSTER_COLORS: [Pixel; 12] = [
    Pixel::rgb(31, 119, 180),  // 0: Blue (best cluster)
    Pixel::rgb(255, 87, 87),   // 1: Red (top similar cluster)
    Pixel::rgb(44, 160, 44),   // 2: Green
    Pixel::rgb(23, 190, 207),  // 3: Cyan
    Pixel::rgb(227, 119, 194), // 4: Magenta
    Pixel::rgb(255, 187, 60),  // 5: Yellow / amber
    Pixel::rgb(255, 127, 14),  // 6: Orange
    Pixel::rgb(148, 103, 189), // 7: Purple
    Pixel::rgb(188, 189, 34),  // 8: Olive
    Pixel::rgb(140, 86, 75),   // 9: Brown
    Pixel::rgb(158, 218, 229), // 10: Light cyan
    Pixel::rgb(255, 152, 150), // 11: Salmon
];

pub fn cluster_color(selection_rank: usize) -> Pixel {
    PHY_CLUSTER_COLORS[selection_rank % PHY_CLUSTER_COLORS.len()]
}

pub fn cluster_hsla(selection_rank: usize) -> Hsla {
    color(cluster_color(selection_rank))
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlotTick {
    pub value: f64,
    /// Position along the axis in `[0.0, 1.0]` (for x: left to right; for y: top to bottom).
    pub frac: f32,
    pub label: String,
}

/// 2D data ↔ pixel coordinate mapping with pan, zoom around an anchor, and reset to initial bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlotTransform {
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
    default_x: (f64, f64),
    default_y: (f64, f64),
}

impl Default for PlotTransform {
    fn default() -> Self {
        Self::new(0.0, 1.0, -1.0, 1.0)
    }
}

impl PlotTransform {
    pub fn new(x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> Self {
        let (x0, x1) = if (x_max - x_min).abs() < 1e-9 { (x_min - 1.0, x_max + 1.0) } else { (x_min.min(x_max), x_min.max(x_max)) };
        let (y0, y1) = if (y_max - y_min).abs() < 1e-9 { (y_min - 1.0, y_max + 1.0) } else { (y_min.min(y_max), y_min.max(y_max)) };
        Self { x_min: x0, x_max: x1, y_min: y0, y_max: y1, default_x: (x0, x1), default_y: (y0, y1) }
    }

    pub fn set_defaults(&mut self, x_min: f64, x_max: f64, y_min: f64, y_max: f64) {
        let fresh = Self::new(x_min, x_max, y_min, y_max);
        if self.default_x != fresh.default_x || self.default_y != fresh.default_y {
            *self = fresh;
        }
    }

    pub fn reset(&mut self) {
        (self.x_min, self.x_max) = self.default_x;
        (self.y_min, self.y_max) = self.default_y;
    }

    pub fn x_span(&self) -> f64 {
        (self.x_max - self.x_min).max(1e-12)
    }

    pub fn y_span(&self) -> f64 {
        (self.y_max - self.y_min).max(1e-12)
    }

    /// Maps data `(x, y)` to normalized plot fractions `(fx, fy)` where `fy = 0` is top and `fy = 1`
    /// is bottom.
    pub fn data_to_frac(&self, x: f64, y: f64) -> (f32, f32) {
        let fx = ((x - self.x_min) / self.x_span()) as f32;
        let fy = (1.0 - (y - self.y_min) / self.y_span()) as f32;
        (fx, fy)
    }

    /// Maps normalized plot fractions `(fx, fy)` to data `(x, y)`.
    pub fn frac_to_data(&self, fx: f32, fy: f32) -> (f64, f64) {
        let x = self.x_min + (fx as f64) * self.x_span();
        let y = self.y_min + (1.0 - fy as f64) * self.y_span();
        (x, y)
    }

    pub fn data_to_px(&self, x: f64, y: f64, bounds: Bounds<Pixels>) -> Point<Pixels> {
        let (fx, fy) = self.data_to_frac(x, y);
        point(bounds.origin.x + bounds.size.width * fx, bounds.origin.y + bounds.size.height * fy)
    }

    pub fn px_to_data(&self, p: Point<Pixels>, bounds: Bounds<Pixels>) -> (f64, f64) {
        let w = bounds.size.width.as_f32().max(1.0);
        let h = bounds.size.height.as_f32().max(1.0);
        let fx = (p.x - bounds.origin.x).as_f32() / w;
        let fy = (p.y - bounds.origin.y).as_f32() / h;
        self.frac_to_data(fx, fy)
    }

    /// Pans by normalized plot fractions `(dfx, dfy)`.
    pub fn pan_frac(&mut self, dfx: f32, dfy: f32) {
        let dx = -(dfx as f64) * self.x_span();
        let dy = (dfy as f64) * self.y_span();
        self.x_min += dx;
        self.x_max += dx;
        self.y_min += dy;
        self.y_max += dy;
    }

    /// Zooms around normalized plot fractions `(fx, fy)` by `factor` (`< 1` zooms in, `> 1` zooms out).
    pub fn zoom_at(&mut self, factor_x: f64, factor_y: f64, fx: f32, fy: f32) {
        let (ax, ay) = self.frac_to_data(fx, fy);
        let nx = (self.x_span() * factor_x).clamp(1e-6, 1e9);
        let ny = (self.y_span() * factor_y).clamp(1e-6, 1e9);
        self.x_min = ax - (fx as f64) * nx;
        self.x_max = self.x_min + nx;
        self.y_min = ay - (1.0 - fy as f64) * ny;
        self.y_max = self.y_min + ny;
    }

    pub fn x_ticks(&self, target_count: usize, unit: &str) -> Vec<PlotTick> {
        ticks_in_range(self.x_min, self.x_max, target_count, unit, false)
    }

    pub fn y_ticks(&self, target_count: usize, unit: &str) -> Vec<PlotTick> {
        ticks_in_range(self.y_min, self.y_max, target_count, unit, true)
    }
}

fn ticks_in_range(min: f64, max: f64, target_count: usize, unit: &str, invert_frac: bool) -> Vec<PlotTick> {
    let span = (max - min).max(1e-12);
    let (values, decimals) = ticks(min, max, target_count);
    let suffix = if unit.is_empty() { String::new() } else { format!(" {unit}") };
    values
        .into_iter()
        .map(|v| {
            let raw = ((v - min) / span) as f32;
            PlotTick { value: v, frac: if invert_frac { 1.0 - raw } else { raw }, label: format!("{v:.decimals$}{suffix}") }
        })
        .collect()
}

/// Polygon lasso in data coordinates (Step 5c / Step 9):
/// - `Ctrl+click` adds a point `(x, y)`
/// - `Ctrl+right-click` closes the polygon (when at least 3 vertices exist)
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Lasso {
    pub vertices: Vec<[f32; 2]>,
    pub closed: bool,
}

impl Lasso {
    pub fn add_point(&mut self, x: f32, y: f32) {
        if self.closed {
            self.vertices.clear();
            self.closed = false;
        }
        self.vertices.push([x, y]);
    }

    pub fn close(&mut self) -> bool {
        if self.vertices.len() >= 3 {
            self.closed = true;
            true
        } else {
            false
        }
    }

    pub fn clear(&mut self) {
        self.vertices.clear();
        self.closed = false;
    }

    pub fn is_active(&self) -> bool {
        self.closed && self.vertices.len() >= 3
    }

    /// Whether `(x, y)` (data coordinates) is inside the polygon.
    pub fn contains(&self, x: f32, y: f32) -> bool {
        let vertices: Vec<(f64, f64)> = self.vertices.iter().map(|v| (v[0] as f64, v[1] as f64)).collect();
        point_in_polygon(x as f64, y as f64, &vertices)
    }
}

/// Paints a connected polyline on `window` using GPUI's `PathBuilder::stroke`.
pub fn paint_polyline(window: &mut Window, points: &[Point<Pixels>], stroke_width: f32, color: Hsla) {
    if points.len() < 2 {
        return;
    }
    let mut builder = PathBuilder::stroke(px(stroke_width));
    builder.move_to(points[0]);
    for &p in &points[1..] {
        builder.line_to(p);
    }
    if let Ok(path) = builder.build() {
        window.paint_path(path, color);
    }
}

/// Paints a polygon lasso overlay (vertices + connecting edges, closed when `lasso.closed`).
pub fn paint_lasso(window: &mut Window, lasso: &Lasso, transform: &PlotTransform, bounds: Bounds<Pixels>, accent: Hsla) {
    if lasso.vertices.is_empty() {
        return;
    }
    let mut pts: Vec<Point<Pixels>> = lasso
        .vertices
        .iter()
        .map(|&[x, y]| transform.data_to_px(x as f64, y as f64, bounds))
        .collect();
    for p in &pts {
        let dot = Bounds { origin: point(p.x - px(3.), p.y - px(3.)), size: size(px(6.), px(6.)) };
        window.paint_quad(fill(dot, accent));
    }
    if lasso.closed && pts.len() >= 3 {
        pts.push(pts[0]);
    }
    paint_polyline(window, &pts, 1.5, accent);
}

/// Paints subtle 1 px grid lines at the given x and y ticks.
pub fn paint_grid(window: &mut Window, bounds: Bounds<Pixels>, x_ticks: &[PlotTick], y_ticks: &[PlotTick], grid_color: Hsla) {
    for t in x_ticks {
        if (0.0..=1.0).contains(&t.frac) {
            let x = bounds.origin.x + bounds.size.width * t.frac;
            window.paint_quad(fill(Bounds { origin: point(x, bounds.origin.y), size: size(px(1.), bounds.size.height) }, grid_color));
        }
    }
    for t in y_ticks {
        if (0.0..=1.0).contains(&t.frac) {
            let y = bounds.origin.y + bounds.size.height * t.frac;
            window.paint_quad(fill(Bounds { origin: point(bounds.origin.x, y), size: size(bounds.size.width, px(1.)) }, grid_color));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lasso_point_in_polygon() {
        let mut lasso = Lasso::default();
        lasso.add_point(0.0, 0.0);
        lasso.add_point(10.0, 0.0);
        lasso.add_point(10.0, 10.0);
        lasso.add_point(0.0, 10.0);
        assert!(lasso.close());
        assert!(lasso.is_active());
        assert!(lasso.contains(5.0, 5.0));
        assert!(lasso.contains(1.0, 9.0));
        assert!(!lasso.contains(-1.0, 5.0));
        assert!(!lasso.contains(15.0, 5.0));
        assert!(!lasso.contains(5.0, 12.0));
    }

    #[test]
    fn test_plot_transform_zoom_pan_and_reset() {
        let mut tr = PlotTransform::new(0.0, 100.0, -50.0, 50.0);
        let (fx, fy) = tr.data_to_frac(50.0, 0.0);
        assert!((fx - 0.5).abs() < 1e-6 && (fy - 0.5).abs() < 1e-6);

        tr.zoom_at(0.5, 0.5, 0.5, 0.5);
        assert!((tr.x_min - 25.0).abs() < 1e-6 && (tr.x_max - 75.0).abs() < 1e-6);
        assert!((tr.y_min - -25.0).abs() < 1e-6 && (tr.y_max - 25.0).abs() < 1e-6);

        tr.reset();
        assert!((tr.x_min - 0.0).abs() < 1e-6 && (tr.x_max - 100.0).abs() < 1e-6);
    }
}
