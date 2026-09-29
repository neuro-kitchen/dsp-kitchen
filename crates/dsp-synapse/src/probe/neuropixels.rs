use dsp_core::layout::{SensorLayout, SensorSite, Position3D};

/// Generates a standard Neuropixels 1.0 probe layout (384 readout channels on 1 shank, staggered checkerboard).
pub fn neuropixels_1_0() -> SensorLayout {
    let mut sites = Vec::with_capacity(384);
    for i in 0..384 {
        let col = i % 2;
        let row = i / 2;
        let x = if col == 0 { 11.0 } else { 27.0 } + (if (row % 2) == 1 { 16.0 } else { 0.0 });
        let y = (row * 20) as f32;

        sites.push(SensorSite {
            channel_id: i,
            device_index: i,
            group_id: 0,
            shank_id: 0,
            position: Position3D::new(x, y, 0.0),
            enabled: true,
        });
    }

    SensorLayout::new("Neuropixels 1.0 (Standard 384)", sites)
}

/// Generates a Neuropixels 2.0 probe layout (4 shanks, 96 active recording sites per shank = 384 total).
pub fn neuropixels_2_0() -> SensorLayout {
    let mut sites = Vec::with_capacity(384);
    let shank_spacing_um = 250.0f32;

    for i in 0..384 {
        let shank_id = i / 96;
        let site_on_shank = i % 96;

        let col = site_on_shank % 2;
        let row = site_on_shank / 2;

        let x = (shank_id as f32 * shank_spacing_um) + (if col == 0 { 0.0 } else { 32.0 });
        let y = (row * 15) as f32;

        sites.push(SensorSite {
            channel_id: i,
            device_index: i,
            group_id: shank_id,
            shank_id,
            position: Position3D::new(x, y, 0.0),
            enabled: true,
        });
    }

    SensorLayout::new("Neuropixels 2.0 (4-Shank 384)", sites)
}
