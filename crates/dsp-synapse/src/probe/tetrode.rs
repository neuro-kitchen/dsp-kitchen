use dsp_core::layout::{SensorLayout, SensorSite, Position3D};

/// Generates a standard 4-channel tetrode geometry.
pub fn tetrode() -> SensorLayout {
    let sites = vec![
        SensorSite { channel_id: 0, device_index: 0, group_id: 0, shank_id: 0, position: Position3D::new(-12.5, 0.0, 0.0), enabled: true },
        SensorSite { channel_id: 1, device_index: 1, group_id: 0, shank_id: 0, position: Position3D::new(12.5, 0.0, 0.0), enabled: true },
        SensorSite { channel_id: 2, device_index: 2, group_id: 0, shank_id: 0, position: Position3D::new(0.0, -12.5, 0.0), enabled: true },
        SensorSite { channel_id: 3, device_index: 3, group_id: 0, shank_id: 0, position: Position3D::new(0.0, 12.5, 0.0), enabled: true },
    ];
    SensorLayout::new("Tetrode (4-Channel)", sites)
}
