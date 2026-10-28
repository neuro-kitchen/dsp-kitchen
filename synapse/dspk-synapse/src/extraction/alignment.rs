//! Sub-sample realignment of snippets: the trough position comes from a parabola through the three
//! samples around it (`dspk_ops::math::parabolic_vertex_offset`) and snippets are shifted by it
//! with the windowed-sinc fractional delay of `dspk_ops::resampler::fractional`.

/// Half-width of the windowed-sinc kernel used for snippet realignment (taps `−5..=5`).
pub const SINC_KERNEL_RADIUS: usize = 5;
