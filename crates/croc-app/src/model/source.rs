//! Data source abstraction consumed by views.
//!
//! Views read signals through this trait rather than a concrete `Dataset`, so the same
//! widget can later display a file, a processing-node output, or a live stream.

/// Read-only, thread-shareable multi-channel signal.
pub trait SignalSource: Send + Sync {
    fn channels(&self) -> usize;
    fn samples(&self) -> usize;
    fn sample_rate(&self) -> f64;
    /// Contiguous samples of one channel.
    fn channel(&self, ch: usize) -> &[f32];
}
