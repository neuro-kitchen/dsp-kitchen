//! Coordinate translation and boundary filtering for halo-windowed processing.

use dsp_core::HaloWindow;

/// Remaps a local sample index (relative to a window's read start) to a global recording timestamp,
/// returning `Some(global_sample)` if and only if the local index lies within the window's valid interior.
#[inline]
pub fn remap_event(window: &HaloWindow, local_sample: usize) -> Option<u64> {
    if window.is_interior_local(local_sample) {
        Some(window.to_global_sample(local_sample))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::ChunkSchedule;

    #[test]
    fn test_remap_event_boundaries() {
        let sched = ChunkSchedule::full_recording(100, 30, 5, 5);
        let w = &sched.windows()[1]; // Valid 30..60, read 25..65, local 5..35
        assert_eq!(remap_event(w, 4), None);
        assert_eq!(remap_event(w, 5), Some(30));
        assert_eq!(remap_event(w, 34), Some(59));
        assert_eq!(remap_event(w, 35), None);
    }
}
