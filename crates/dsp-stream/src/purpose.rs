use serde::{Deserialize, Serialize};

/// High-level purpose of an outgoing signal stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamPurpose {
    /// Full-fidelity lossless stream (e.g. 30 kHz continuous raw floats) for analytics or ML.
    Processing,

    /// Decimated Level-Of-Detail (LOD) stream for real-time UI/GPU monitors (e.g. 60 fps envelope).
    Visualization {
        target_points_per_channel: usize,
    },
}
