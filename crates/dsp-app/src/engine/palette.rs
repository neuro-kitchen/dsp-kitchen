//! Plot colours. Renderers and plot overlays take a [`Palette`] instead of fixed colours, so plots
//! follow the app's light or dark theme (or stay dark when the user prefers dark plots).

use super::canvas::Pixel;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Palette {
    pub dark: bool,
    pub background: Pixel,
    pub grid: Pixel,
    pub baseline: Pixel,
    /// Scale bar, axis text.
    pub text: Pixel,
    /// Spike event ticks.
    pub marker: Pixel,
    /// Playhead.
    pub playhead: Pixel,
    /// One colour per channel, cycling.
    pub channels: [Pixel; 8],
    /// Sequential colour map stops for heatmaps, 0 → 1.
    pub heat: [(f32, Pixel); 5],
}

impl Palette {
    pub const DARK: Palette = Palette {
        dark: true,
        background: Pixel::rgb(9, 13, 19),
        grid: Pixel::rgb(22, 27, 34),
        baseline: Pixel::rgb(33, 38, 45),
        text: Pixel::rgb(139, 148, 158),
        marker: Pixel::rgb(250, 204, 21),
        playhead: Pixel::rgb(244, 63, 94),
        channels: [
            Pixel::rgb(56, 189, 248),
            Pixel::rgb(52, 211, 153),
            Pixel::rgb(168, 85, 247),
            Pixel::rgb(251, 191, 36),
            Pixel::rgb(244, 63, 94),
            Pixel::rgb(96, 165, 250),
            Pixel::rgb(249, 115, 22),
            Pixel::rgb(45, 212, 191),
        ],
        // navy → violet → orange → yellow
        heat: [
            (0.00, Pixel::rgb(9, 13, 19)),
            (0.25, Pixel::rgb(49, 36, 110)),
            (0.50, Pixel::rgb(150, 45, 120)),
            (0.75, Pixel::rgb(240, 110, 50)),
            (1.00, Pixel::rgb(252, 230, 90)),
        ],
    };

    pub const LIGHT: Palette = Palette {
        dark: false,
        background: Pixel::rgb(255, 255, 255),
        grid: Pixel::rgb(234, 238, 242),
        baseline: Pixel::rgb(208, 215, 222),
        text: Pixel::rgb(87, 96, 106),
        marker: Pixel::rgb(191, 135, 0),
        playhead: Pixel::rgb(207, 34, 46),
        channels: [
            Pixel::rgb(9, 105, 218),
            Pixel::rgb(26, 127, 55),
            Pixel::rgb(130, 80, 223),
            Pixel::rgb(154, 103, 0),
            Pixel::rgb(207, 34, 46),
            Pixel::rgb(5, 80, 174),
            Pixel::rgb(188, 76, 0),
            Pixel::rgb(27, 124, 131),
        ],
        // white → light blue → blue → dark blue
        heat: [
            (0.00, Pixel::rgb(255, 255, 255)),
            (0.25, Pixel::rgb(198, 219, 239)),
            (0.50, Pixel::rgb(107, 174, 214)),
            (0.75, Pixel::rgb(33, 113, 181)),
            (1.00, Pixel::rgb(8, 48, 107)),
        ],
    };

    pub fn of(dark: bool) -> Palette {
        if dark { Palette::DARK } else { Palette::LIGHT }
    }

    pub fn channel(&self, ch: usize) -> Pixel {
        self.channels[ch % self.channels.len()]
    }

    /// Colour of `v` (0..1, clamped) on the heat map.
    pub fn heat(&self, v: f32) -> Pixel {
        let v = v.clamp(0.0, 1.0);
        let stops = &self.heat;
        let i = stops.iter().position(|s| s.0 >= v).unwrap_or(stops.len() - 1).max(1);
        let ((t0, c0), (t1, c1)) = (stops[i - 1], stops[i]);
        let f = (v - t0) / (t1 - t0);
        let mix = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * f) as u8;
        Pixel::rgb(mix(c0.r, c1.r), mix(c0.g, c1.g), mix(c0.b, c1.b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heat_endpoints() {
        for p in [Palette::DARK, Palette::LIGHT] {
            assert_eq!(p.heat(0.0), p.heat[0].1);
            assert_eq!(p.heat(1.0), p.heat[4].1);
            assert_eq!(p.heat(2.0), p.heat(1.0));
        }
        assert_eq!(Palette::DARK.heat(0.0), Palette::DARK.background, "quiet reads as background");
    }
}
