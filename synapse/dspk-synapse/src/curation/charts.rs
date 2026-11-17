//! The Curation views' charts as data: layers (lines, points, histogram bars, markers) in the
//! data's own units (ms, µV, counts, µm), a chart's axis names and extent, a grid of charts sharing
//! the x range. Made where the sorting is ([`super::views`]: the app, or a server for a page) and
//! drawn by the app; colours are what they mean ([`Ink`]: a unit, the unsorted spikes, the grid),
//! turned into colours by whoever draws, in its theme.

use serde::{Deserialize, Serialize};

/// What a layer's colour means.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ink {
    /// Unit index `u`'s colour.
    Unit(usize),
    /// Spikes of no listed unit.
    Unsorted,
    /// The probe's channels, the plot's grid.
    Grid,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "layer", rename_all = "snake_case")]
pub enum Layer {
    /// Polylines in data coordinates.
    Lines { ink: Ink, alpha: f32, width: f32, lines: Vec<Vec<(f32, f32)>> },
    Points { ink: Ink, alpha: f32, radius: f32, points: Vec<(f32, f32)> },
    /// A histogram: bin `i` spans `start + i·width … start + (i+1)·width`.
    Bars { ink: Ink, start: f32, width: f32, counts: Vec<f32> },
    /// A vertical marker (zero lag, the refractory period).
    Marker { x: f32 },
}

/// One chart: its layers, axis names (with units), and the full extent of its data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChartGrid {
    pub cells: Vec<Chart>,
    pub cols: usize,
}

impl ChartGrid {
    pub fn one(chart: Chart) -> Self {
        Self { cells: vec![chart], cols: 1 }
    }

    pub fn rows(&self) -> usize {
        self.cells.len().div_ceil(self.cols.max(1)).max(1)
    }
}

impl Chart {
    /// A chart of `points` (x, y) in one layer per group, its range padded by 6 %; `None` when
    /// no point is finite.
    pub fn scatter(groups: Vec<(Ink, Vec<(f32, f32)>)>, radius: f32, x_label: String, y_label: String, title: String) -> Option<Self> {
        let all = groups.iter().flat_map(|(_, p)| p.iter());
        let ((x0, x1), (y0, y1)) = all.fold(((f32::INFINITY, f32::NEG_INFINITY), (f32::INFINITY, f32::NEG_INFINITY)), |((a, b), (c, d)), p| ((a.min(p.0), b.max(p.0)), (c.min(p.1), d.max(p.1))));
        if !(x0.is_finite() && y0.is_finite()) {
            return None;
        }
        let (px, py) = ((x1 - x0).max(1e-6) * 0.06, (y1 - y0).max(1e-6) * 0.06);
        let layers = groups.into_iter().map(|(ink, points)| Layer::Points { ink, alpha: 0.6, radius, points }).collect();
        Some(Self { layers, x_label, y_label, x: (x0 - px, x1 + px), y: (y0 - py, y1 + py), y_fit: false, title })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chart_travels_as_json_and_back() {
        let chart = Chart {
            layers: vec![
                Layer::Bars { ink: Ink::Unit(3), start: -1.0, width: 0.5, counts: vec![1.0, 2.0] },
                Layer::Points { ink: Ink::Unsorted, alpha: 0.5, radius: 1.0, points: vec![(0.0, 1.0)] },
                Layer::Marker { x: 0.0 },
            ],
            x_label: "lag (ms)".into(),
            y_label: "pairs".into(),
            x: (-1.0, 1.0),
            y: (0.0, 2.0),
            y_fit: true,
            title: String::new(),
        };
        let grid = ChartGrid::one(chart);
        let back: ChartGrid = serde_json::from_str(&serde_json::to_string(&grid).unwrap()).unwrap();
        assert_eq!(back, grid);
        assert_eq!(back.rows(), 1);
    }

    #[test]
    fn a_scatter_pads_its_range_and_needs_a_finite_point() {
        let c = Chart::scatter(vec![(Ink::Unit(0), vec![(0.0, 0.0), (10.0, 20.0)])], 1.0, "x".into(), "y".into(), String::new()).unwrap();
        assert!(c.x.0 < 0.0 && c.x.1 > 10.0 && c.y.1 > 20.0);
        assert!(Chart::scatter(vec![(Ink::Unit(0), vec![])], 1.0, String::new(), String::new(), String::new()).is_none());
    }
}
