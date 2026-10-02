//! Myomatrix and EMUsort specialized spike-sorting engines for muscle electrophysiology.
//!
//! Provides:
//! - [`MyomatrixSortConfig`]: Presets for 8-channel threads, 32-channel, and 64-channel arrays.
//! - [`MyomatrixTemplateMatcher`] / [`MyomatrixDetector`]: 150-sample universal MUAP matched filtering.
//! - [`MyomatrixBasisEmbedder`]: 150-sample, 12-component spatiotemporal muscle basis projection.
//! - [`MyomatrixLatencyAligner`]: Cross-channel conduction velocity delay estimation and alignment.

pub mod basis;
pub mod config;
pub mod latency;
pub mod matcher;

pub use basis::{
    DEFAULT_MUAP_BASIS_COMPONENTS, DEFAULT_MUAP_BASIS_WINDOW_LEN, MYOMATRIX_BASIS_MODEL_ID,
    MyomatrixBasisEmbedder,
};
pub use config::{MyomatrixProbeKind, MyomatrixSortConfig};
pub use latency::MyomatrixLatencyAligner;
pub use matcher::{
    DEFAULT_MUAP_WINDOW_LEN, MYOMATRIX_TEMPLATES_MODEL_ID, MyomatrixDetector,
    MyomatrixTemplateMatcher,
};

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_synapse::{DeduplicatedSpike, FeatureEmbedder, SpikeDetector, extract_snippet_batch_multichannel, tetrode};

    #[test]
    fn test_canonical_myomatrix_detector_and_embedder() {
        let detector = MyomatrixDetector::from_canonical(5.0, 40, None).unwrap();
        assert_eq!(detector.num_templates(), 6);
        assert_eq!(detector.window_len(), 150);

        let embedder = MyomatrixBasisEmbedder::from_canonical(None).unwrap();
        assert_eq!(embedder.num_components(), 12);
        assert_eq!(embedder.window_len(), 150);

        // Verify basis rows are orthonormal
        let w_len = embedder.window_len();
        for i in 0..embedder.num_components() {
            let row_i = &embedder.basis()[i * w_len..(i + 1) * w_len];
            let norm_sq: f32 = row_i.iter().map(|&x| x * x).sum();
            assert!(
                (norm_sq - 1.0).abs() < 1e-3,
                "Basis row {i} norm_sq expected ~1.0, got {norm_sq}"
            );
            for j in 0..i {
                let row_j = &embedder.basis()[j * w_len..(j + 1) * w_len];
                let dot: f32 = row_i.iter().zip(row_j).map(|(&a, &b)| a * b).sum();
                assert!(
                    dot.abs() < 1e-3,
                    "Basis orthogonality {i} vs {j} expected ~0.0, got {dot}"
                );
            }
        }

        // Detect an injected 150-sample MUAP in synthetic trace
        let channels = 4usize;
        let samples = 1_500usize;
        let mut trace = vec![0.0f32; channels * samples];
        let center = 600usize;
        let tpl0 = &detector.templates()[..w_len];
        let start = center - detector.center_offset();
        for t in 0..w_len {
            trace[start + t] += tpl0[t] * 150.0;
        }

        let events = detector.detect(&trace, channels, samples, 24414.0).unwrap();
        assert!(!events.is_empty(), "expected MyomatrixDetector to detect MUAP");
        assert!(
            events.iter().any(|e| e.channel_id == 0 && (e.sample_index as isize - center as isize).abs() <= 3),
            "events: {events:?}"
        );

        // Test snippet batch extraction and 12-PC projection
        let layout = tetrode();
        let dedup = vec![DeduplicatedSpike {
            primary_channel: 0,
            sample_index: center as u64,
            peak_amplitude_uv: -150.0,
            participating_channels: vec![0, 1, 2, 3],
        }];
        let pre = detector.center_offset();
        let post = w_len - pre;
        let batch = extract_snippet_batch_multichannel(
            &trace,
            channels,
            samples,
            &dedup,
            &layout,
            4,
            pre,
            post,
            false,
        );
        assert_eq!(batch.shape(), [1, 4, 150]);

        let (feats, dim) = embedder.embed(&batch).unwrap();
        assert_eq!(dim, 4 * 12);
        assert_eq!(feats.len(), dim);

        let recon = embedder.reconstruct(&batch).unwrap();
        assert_eq!(recon.shape(), batch.shape());
    }
}
