//! Nominal geometries of common probes and electrode arrays.
//!
//! Sites are numbered in a fixed order (row-major for grids); real channel maps can differ by
//! configuration (e.g. the Neuropixels IMRO table) or vendor, so files that carry their own geometry
//! (SpikeGLX `.meta`, NWB electrodes) take precedence.

use super::{Position3D, SensorLayout, SensorSite};

/// Neuropixels 1.0 (384 channels, one shank): 20 µm rows, staggered columns at x = 43 / 11 µm on
/// even rows and 59 / 27 µm on odd rows (probeinterface `NP1010`).
pub fn neuropixels_1_0() -> SensorLayout {
    const X: [f32; 4] = [43.0, 11.0, 59.0, 27.0];
    let contacts = (0..384)
        .map(|i| SensorSite {
            channel_id: i,
            device_index: i,
            group_id: 0,
            shank_id: 0,
            position: Position3D::new(X[i % 4], (i / 2) as f32 * 20.0, 0.0),
            enabled: true,
        })
        .collect();
    SensorLayout::new("Neuropixels 1.0 (Standard 384)", contacts)
}

/// Shanks of a Neuropixels 2.0 four-shank probe.
const NP2_SHANKS: usize = 4;
/// Sites per shank in [`neuropixels_2_0`].
const NP2_SITES_PER_SHANK: usize = 96;
/// Neuropixels 2.0 spacing (µm): between shanks, between the two columns, between rows.
const NP2_SHANK_PITCH_UM: f32 = 250.0;
const NP2_COLUMN_PITCH_UM: f32 = 32.0;
const NP2_ROW_PITCH_UM: f32 = 15.0;

/// Neuropixels 2.0, four shanks of 96 sites (384 channels): two columns 32 µm apart, 15 µm rows,
/// shanks 250 µm apart, channels numbered shank by shank.
pub fn neuropixels_2_0() -> SensorLayout {
    let sites = (0..NP2_SHANKS * NP2_SITES_PER_SHANK)
        .map(|i| {
            let (shank_id, on_shank) = (i / NP2_SITES_PER_SHANK, i % NP2_SITES_PER_SHANK);
            let (col, row) = (on_shank % 2, on_shank / 2);
            let x = shank_id as f32 * NP2_SHANK_PITCH_UM + col as f32 * NP2_COLUMN_PITCH_UM;
            SensorSite {
                channel_id: i,
                device_index: i,
                group_id: shank_id,
                shank_id,
                position: Position3D::new(x, row as f32 * NP2_ROW_PITCH_UM, 0.0),
                enabled: true,
            }
        })
        .collect();
    SensorLayout::new("Neuropixels 2.0 (4-Shank 384)", sites)
}

/// Planar `rows × cols` grid with `pitch_um` spacing (HD-EMG, ECoG); channel `r · cols + c` sits at
/// `(c · pitch, r · pitch)`.
pub fn hdemg_grid(name: &str, rows: usize, cols: usize, pitch_um: f32) -> SensorLayout {
    let contacts = (0..rows * cols)
        .map(|ch| SensorSite::new(ch, Position3D::new((ch % cols) as f32 * pitch_um, (ch / cols) as f32 * pitch_um, 0.0), 0))
        .collect();
    SensorLayout::new(name, contacts)
}

/// 32-channel `4 × 8` HD-EMG grid.
pub fn hdemg_4x8(pitch_um: f32) -> SensorLayout {
    hdemg_grid("HDEMG-4x8", 4, 8, pitch_um)
}

/// 64-channel `8 × 8` HD-EMG grid.
pub fn hdemg_8x8(pitch_um: f32) -> SensorLayout {
    hdemg_grid("HDEMG-8x8", 8, 8, pitch_um)
}

/// Half the spacing between opposite wires of a tetrode (µm).
const TETRODE_HALF_SPACING_UM: f32 = 12.5;

/// Tetrode: four wires on a cross, `±12.5` µm along x then y.
pub fn tetrode() -> SensorLayout {
    let h = TETRODE_HALF_SPACING_UM;
    let positions = [(-h, 0.0), (h, 0.0), (0.0, -h), (0.0, h)];
    let sites = positions.iter().enumerate().map(|(c, &(x, y))| SensorSite::new(c, Position3D::new(x, y, 0.0), 0)).collect();
    SensorLayout::new("Tetrode (4-Channel)", sites)
}

/// Utah array side (electrodes per row / column) and pitch (µm).
const UTAH_SIDE: usize = 10;
const UTAH_PITCH_UM: f32 = 400.0;

/// Utah array: `10 × 10` at 400 µm without the four corners (96 channels), numbered row-major.
pub fn utah_array() -> SensorLayout {
    let last = UTAH_SIDE - 1;
    let sites = (0..UTAH_SIDE * UTAH_SIDE)
        .map(|i| (i / UTAH_SIDE, i % UTAH_SIDE))
        .filter(|&(row, col)| !((row == 0 || row == last) && (col == 0 || col == last)))
        .enumerate()
        .map(|(ch, (row, col))| SensorSite::new(ch, Position3D::new(col as f32 * UTAH_PITCH_UM, row as f32 * UTAH_PITCH_UM, 0.0), 0))
        .collect();
    SensorLayout::new("Utah Array (96-Channel)", sites)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_shapes() {
        assert_eq!(neuropixels_2_0().total_channels(), 384);
        assert_eq!(neuropixels_2_0().contacts[96].position, Position3D::new(250.0, 0.0, 0.0));
        let grid = hdemg_4x8(100.0);
        assert_eq!(grid.total_channels(), 32);
        assert_eq!(grid.contacts[7].position, Position3D::new(700.0, 0.0, 0.0));
        assert_eq!(grid.contacts[8].position, Position3D::new(0.0, 100.0, 0.0));
        assert_eq!(tetrode().total_channels(), 4);
        assert_eq!(utah_array().total_channels(), 96);
    }
}
