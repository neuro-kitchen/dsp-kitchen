use dsp_core::layout::{SensorLayout, SensorSite, Position3D};

/// Generates a standard 10x10 Utah Array layout (100 electrodes, 400 um pitch, corners commonly unbonded = 96 channels).
pub fn utah_array() -> SensorLayout {
    let mut sites = Vec::with_capacity(96);
    let pitch_um = 400.0f32;
    let mut ch = 0;

    for row in 0..10 {
        for col in 0..10 {
            // Exclude standard 4 corners
            if (row == 0 || row == 9) && (col == 0 || col == 9) {
                continue;
            }

            sites.push(SensorSite {
                channel_id: ch,
                device_index: ch,
                group_id: 0,
                shank_id: 0,
                position: Position3D::new(col as f32 * pitch_um, row as f32 * pitch_um, 0.0),
                enabled: true,
            });
            ch += 1;
        }
    }

    SensorLayout::new("Utah Array (96-Channel)", sites)
}
