//! Bounds-checked, anti-aliased pixel drawing shared by the plot rasterizers.

use slint::Rgba8Pixel;

pub fn blend_color(a: Rgba8Pixel, b: Rgba8Pixel, t: f32) -> Rgba8Pixel {
    let mix = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t) as u8;
    Rgba8Pixel { r: mix(a.r, b.r), g: mix(a.g, b.g), b: mix(a.b, b.b), a: 255 }
}

/// Bounds-checked pixel writer over a pixel buffer.
pub struct Canvas<'a> {
    pub pixels: &'a mut [Rgba8Pixel],
    pub width: usize,
    pub height: usize,
}

impl Canvas<'_> {
    pub fn set(&mut self, x: usize, y: usize, c: Rgba8Pixel) {
        if x < self.width && y < self.height {
            self.pixels[y * self.width + x] = c;
        }
    }

    pub fn blend(&mut self, x: usize, y: usize, c: Rgba8Pixel, alpha: f32) {
        if x < self.width && y < self.height && alpha > 0.0 {
            let p = &mut self.pixels[y * self.width + x];
            let a = alpha.min(1.0);
            let mix = |dst: u8, src: u8| (dst as f32 + (src as f32 - dst as f32) * a) as u8;
            *p = Rgba8Pixel { r: mix(p.r, c.r), g: mix(p.g, c.g), b: mix(p.b, c.b), a: 255 };
        }
    }

    pub fn vline(&mut self, x: usize, y0: usize, y1: usize, c: Rgba8Pixel) {
        for y in y0..y1.min(self.height) {
            self.set(x, y, c);
        }
    }

    pub fn vline_alpha(&mut self, x: usize, y0: usize, y1: usize, c: Rgba8Pixel, alpha: f32) {
        for y in y0..y1.min(self.height) {
            self.blend(x, y, c, alpha);
        }
    }

    /// Fills column `x` between fractional rows `top..bottom`, with partial coverage at the ends.
    pub fn span_aa(&mut self, x: usize, top: f32, bottom: f32, c: Rgba8Pixel) {
        let top = top.max(0.0);
        let bottom = bottom.min(self.height as f32);
        if bottom <= top {
            return;
        }
        let y_first = top.floor() as usize;
        let y_last = (bottom.ceil() as usize).min(self.height);
        for y in y_first..y_last {
            let coverage = (bottom.min(y as f32 + 1.0) - top.max(y as f32)).clamp(0.0, 1.0);
            self.blend(x, y, c, coverage);
        }
    }

    /// Fills an axis-aligned rectangle (clipped), blended with `alpha`.
    pub fn rect(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, c: Rgba8Pixel, alpha: f32) {
        let xa = x0.max(0.0).floor() as usize;
        let xb = (x1.min(self.width as f32).ceil().max(0.0) as usize).min(self.width);
        for x in xa..xb {
            let cover_x = (x1.min(x as f32 + 1.0) - x0.max(x as f32)).clamp(0.0, 1.0);
            if cover_x > 0.0 {
                self.span_aa_alpha(x, y0, y1, c, alpha * cover_x);
            }
        }
    }

    /// Like `span_aa`, with an overall opacity.
    pub fn span_aa_alpha(&mut self, x: usize, top: f32, bottom: f32, c: Rgba8Pixel, alpha: f32) {
        let top = top.max(0.0);
        let bottom = bottom.min(self.height as f32);
        if bottom <= top {
            return;
        }
        let y_first = top.floor() as usize;
        let y_last = (bottom.ceil() as usize).min(self.height);
        for y in y_first..y_last {
            let coverage = (bottom.min(y as f32 + 1.0) - top.max(y as f32)).clamp(0.0, 1.0);
            self.blend(x, y, c, coverage * alpha);
        }
    }

    /// Anti-aliased filled disc.
    pub fn dot(&mut self, cx: f32, cy: f32, r: f32, c: Rgba8Pixel, alpha: f32) {
        let (x0, x1) = ((cx - r - 1.0).max(0.0) as usize, ((cx + r + 1.0).max(0.0) as usize).min(self.width));
        let (y0, y1) = ((cy - r - 1.0).max(0.0) as usize, ((cy + r + 1.0).max(0.0) as usize).min(self.height));
        for y in y0..y1 {
            for x in x0..x1 {
                let d = ((x as f32 + 0.5 - cx).powi(2) + (y as f32 + 0.5 - cy).powi(2)).sqrt();
                let cover = (r + 0.5 - d).clamp(0.0, 1.0);
                if cover > 0.0 {
                    self.blend(x, y, c, cover * alpha);
                }
            }
        }
    }

    /// Polyline through points with increasing x, drawn as connected column spans.
    pub fn polyline(&mut self, pts: &[(f32, f32)], c: Rgba8Pixel, alpha: f32, thickness: f32) {
        let half = thickness.max(1.0) * 0.5;
        for seg in pts.windows(2) {
            let ((xa, ya), (xb, yb)) = (seg[0], seg[1]);
            let (x_start, x_end) = (xa.round() as i64, xb.round() as i64);
            for x in x_start..x_end.max(x_start + 1) {
                if x < 0 || x as usize >= self.width {
                    continue;
                }
                let t0 = ((x as f32 - xa) / (xb - xa).max(1e-6)).clamp(0.0, 1.0);
                let t1 = ((x as f32 + 1.0 - xa) / (xb - xa).max(1e-6)).clamp(0.0, 1.0);
                let (y0, y1) = (ya + (yb - ya) * t0, ya + (yb - ya) * t1);
                self.span_aa_alpha(x as usize, y0.min(y1) - half, y0.max(y1) + half, c, alpha);
            }
        }
    }

    pub fn hline(&mut self, y: usize, x0: usize, x1: usize, c: Rgba8Pixel) {
        for x in x0..x1.min(self.width) {
            self.set(x, y, c);
        }
    }
}
