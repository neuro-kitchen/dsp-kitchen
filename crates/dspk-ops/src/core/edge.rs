//! What a stencil reads past either end of a row.
//!
//! Defaults follow the scipy function each kernel mirrors (`sosfiltfilt`: odd, `lfilter`: zeros,
//! `ndimage.gaussian_filter1d`: reflect, `signal.medfilt`: zeros).

/// Extension of a row `x[0..n]` beyond its ends (shown for the left end; the right end mirrors it).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeMode {
    /// `0 0 0 | x0 x1 x2` (scipy `constant` with `cval = 0`).
    Zeros,
    /// Point reflection through the edge sample: `2x0−x3 2x0−x2 2x0−x1 | x0 x1 x2` (scipy
    /// `padtype="odd"`). Keeps the level and slope continuous, so filters start without a step.
    Odd,
    /// Mirror about the edge, repeating it: `x2 x1 x0 | x0 x1 x2` (scipy `reflect`).
    Reflect,
    /// Repeat the edge sample: `x0 x0 x0 | x0 x1 x2` (scipy `nearest`).
    Nearest,
}

impl EdgeMode {
    /// The comptime id kernels branch on.
    pub const fn id(self) -> u32 {
        match self {
            EdgeMode::Zeros => EDGE_ZEROS,
            EdgeMode::Odd => EDGE_ODD,
            EdgeMode::Reflect => EDGE_REFLECT,
            EdgeMode::Nearest => EDGE_NEAREST,
        }
    }
}

pub const EDGE_ZEROS: u32 = 0;
pub const EDGE_ODD: u32 = 1;
pub const EDGE_REFLECT: u32 = 2;
pub const EDGE_NEAREST: u32 = 3;

