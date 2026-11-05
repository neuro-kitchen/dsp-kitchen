//! Pixels, frames and bounds-checked, anti-aliased drawing shared by the plot rasterizers.
//!
//! A [`Frame`] is stored as BGRA bytes, the layout GPUI uploads as is (`RenderImage`), so a finished
//! frame reaches the screen without a conversion; [`Frame::to_rgba`] gives RGBA for PNG files.

use bytemuck::{Pod, Zeroable};

/// One 8-bit pixel, laid out in memory as B, G, R, A (see the module docs).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Pod, Zeroable)]
pub struct Pixel {
    pub b: u8,
    pub g: u8,
    pub r: u8,
    pub a: u8,
}

impl Pixel {
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { b, g, r, a: 255 }
    }
}

/// A rendered image: `width × height` pixels, row-major, BGRA bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    data: Vec<u8>,
}

impl Frame {
    /// A transparent black frame.
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height, data: vec![0; width as usize * height as usize * 4] }
    }

    pub fn pixels(&self) -> &[Pixel] {
        bytemuck::cast_slice(&self.data)
    }

    pub fn pixels_mut(&mut self) -> &mut [Pixel] {
        bytemuck::cast_slice_mut(&mut self.data)
    }

    /// A canvas drawing into this frame.
    pub fn canvas(&mut self) -> Canvas<'_> {
        let (width, height) = (self.width as usize, self.height as usize);
        Canvas { pixels: self.pixels_mut(), width, height }
    }

    /// The BGRA bytes (what GPUI uploads).
    pub fn into_bgra(self) -> Vec<u8> {
        self.data
    }

    /// RGBA bytes (PNG files).
    pub fn to_rgba(&self) -> Vec<u8> {
        self.pixels().iter().flat_map(|p| [p.r, p.g, p.b, p.a]).collect()
    }
}

/// Bounds-checked pixel writer over a pixel buffer.
pub struct Canvas<'a> {
    pub pixels: &'a mut [Pixel],
    pub width: usize,
    pub height: usize,
}

impl Canvas<'_> {
    pub fn set(&mut self, x: usize, y: usize, c: Pixel) {
        if x < self.width && y < self.height {
            self.pixels[y * self.width + x] = c;
        }
    }

    pub fn blend(&mut self, x: usize, y: usize, c: Pixel, alpha: f32) {
        if x < self.width && y < self.height && alpha > 0.0 {
            let p = &mut self.pixels[y * self.width + x];
            let a = alpha.min(1.0);
            let mix = |dst: u8, src: u8| (dst as f32 + (src as f32 - dst as f32) * a) as u8;
            *p = Pixel::rgb(mix(p.r, c.r), mix(p.g, c.g), mix(p.b, c.b));
        }
    }

    pub fn vline(&mut self, x: usize, y0: usize, y1: usize, c: Pixel) {
        for y in y0..y1.min(self.height) {
            self.set(x, y, c);
        }
    }

    pub fn vline_alpha(&mut self, x: usize, y0: usize, y1: usize, c: Pixel, alpha: f32) {
        for y in y0..y1.min(self.height) {
            self.blend(x, y, c, alpha);
        }
    }

    /// Fills column `x` between fractional rows `top..bottom`, with partial coverage at the ends.
    pub fn span_aa(&mut self, x: usize, top: f32, bottom: f32, c: Pixel) {
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
    pub fn rect(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, c: Pixel, alpha: f32) {
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
    pub fn span_aa_alpha(&mut self, x: usize, top: f32, bottom: f32, c: Pixel, alpha: f32) {
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
}
