use serde::{Deserialize, Serialize};

/// 3D physical position of a sensor contact in micrometers (µm).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Position3D {
    pub x_um: f32,
    pub y_um: f32,
    pub z_um: f32,
}

impl Position3D {
    pub const fn new(x_um: f32, y_um: f32, z_um: f32) -> Self {
        Self { x_um, y_um, z_um }
    }

    pub fn distance_to(&self, other: &Self) -> f32 {
        let dx = self.x_um - other.x_um;
        let dy = self.y_um - other.y_um;
        let dz = self.z_um - other.z_um;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Description of a single sensor recording site.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorSite {
    pub channel_id: usize,
    pub position: Position3D,
    pub shank_id: usize,
    pub is_enabled: bool,
}

impl SensorSite {
    pub fn new(channel_id: usize, position: Position3D, shank_id: usize) -> Self {
        Self {
            channel_id,
            position,
            shank_id,
            is_enabled: true,
        }
    }
}
