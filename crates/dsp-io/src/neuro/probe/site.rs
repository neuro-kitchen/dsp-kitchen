use serde::{Deserialize, Serialize};

/// Physical coordinates of a single sensor recording site (in micrometers or unit space).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Position3D {
    pub x_um: f32,
    pub y_um: f32,
    pub z_um: f32,
}

impl Position3D {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x_um: x, y_um: y, z_um: z }
    }

    pub fn distance_to(&self, other: &Self) -> f32 {
        let dx = self.x_um - other.x_um;
        let dy = self.y_um - other.y_um;
        let dz = self.z_um - other.z_um;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Metadata and physical specifications for an individual sensor site.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorSite {
    pub channel_id: usize,
    pub device_index: usize,
    pub group_id: usize,
    pub shank_id: usize,
    pub position: Position3D,
    pub enabled: bool,
}

impl SensorSite {
    pub fn new(channel_id: usize, position: Position3D, shank_id: usize) -> Self {
        Self {
            channel_id,
            device_index: channel_id,
            group_id: 0,
            shank_id,
            position,
            enabled: true,
        }
    }
}
