use dsp_core::layout::{Position3D, SensorLayout, SensorSite};

/// Builds a 2D planar High-Density EMG (`HD-EMG`) grid with `rows × cols` electrodes
/// separated by `pitch_um` micrometers.
pub fn hdemg_grid(name: &str, rows: usize, cols: usize, pitch_um: f32) -> SensorLayout {
    let mut contacts = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let channel_id = r * cols + c;
            let pos = Position3D::new((c as f32) * pitch_um, (r as f32) * pitch_um, 0.0);
            contacts.push(SensorSite::new(channel_id, pos, 0));
        }
    }
    SensorLayout::new(name, contacts)
}

/// Canonical 32-channel `4 × 8` planar Diaphragm HD-EMG grid.
pub fn hdemg_4x8(pitch_um: f32) -> SensorLayout {
    hdemg_grid("HDEMG-4x8", 4, 8, pitch_um)
}

/// Canonical 64-channel `8 × 8` planar HD-EMG grid.
pub fn hdemg_8x8(pitch_um: f32) -> SensorLayout {
    hdemg_grid("HDEMG-8x8", 8, 8, pitch_um)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdemg_4x8_layout() {
        let grid = hdemg_4x8(100.0);
        assert_eq!(grid.total_channels(), 32);
        assert_eq!(grid.contacts[0].position, Position3D::new(0.0, 0.0, 0.0));
        assert_eq!(grid.contacts[7].position, Position3D::new(700.0, 0.0, 0.0));
        assert_eq!(grid.contacts[8].position, Position3D::new(0.0, 100.0, 0.0));
    }
}
