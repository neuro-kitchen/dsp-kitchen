//! Kilosort4 pretrained temporal basis (`wPCA.npy`) and universal template matcher (`wTEMP.npy`).

pub mod basis;
pub mod matcher;

pub use basis::{KILOSORT4_BASIS_MODEL_ID, Kilosort4BasisEmbedder};
pub use matcher::{KILOSORT4_TEMPLATES_MODEL_ID, Kilosort4TemplateMatcher};

/// Backward-compatible type alias for [`Kilosort4TemplateMatcher`].
pub type Kilosort4Detector = Kilosort4TemplateMatcher;

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_synapse::{
        DeduplicatedSpike, FeatureEmbedder, SpikeDetector, extract_snippet_batch_multichannel,
        tetrode,
    };

    #[test]
    fn test_kilosort4_from_hub_loads_and_executes_real_weights() {
        let embedder = Kilosort4BasisEmbedder::from_hub(None).unwrap();
        assert_eq!(embedder.num_components(), 6);
        assert_eq!(embedder.window_len(), 61);

        // Verify rows of wPCA.npy are unit-norm orthonormal basis vectors
        let w_len = embedder.window_len();
        for c in 0..embedder.num_components() {
            let row = &embedder.basis()[c * w_len..(c + 1) * w_len];
            let norm_sq: f32 = row.iter().map(|v| v * v).sum();
            assert!(
                (norm_sq - 1.0).abs() < 1e-3,
                "wPCA row {c} norm_sq = {norm_sq}"
            );
        }

        let matcher = Kilosort4TemplateMatcher::from_hub(4.5, 30, None).unwrap();
        assert_eq!(matcher.num_templates(), 6);
        assert_eq!(matcher.window_len(), 61);

        // Build a 4-channel synthetic signal containing a scaled copy of template 0 at sample 400
        let channels = 4usize;
        let samples = 1_000usize;
        let mut trace = vec![0.0f32; channels * samples];
        for ch in 0..channels {
            for s in 0..samples {
                trace[ch * samples + s] = (((s * 17 + ch * 13) % 19) as f32 - 9.0) * 1.2;
            }
        }

        let tpl0 = &matcher.templates()[..w_len];
        let center = 400usize;
        let start = center - matcher.center_offset();
        for t in 0..w_len {
            trace[start + t] += tpl0[t] * 120.0;
        }

        let events = matcher.detect(&trace, channels, samples, 30_000.0).unwrap();
        assert!(!events.is_empty(), "expected Kilosort4TemplateMatcher to detect injected spike");
        assert!(
            events.iter().any(|e| e.channel_id == 0 && (e.sample_index as isize - center as isize).abs() <= 2),
            "detected events: {events:?}"
        );

        // Extract 61-sample snippets around sample 400 and project/reconstruct with Kilosort4BasisEmbedder
        let layout = tetrode();
        let dedup = vec![DeduplicatedSpike {
            primary_channel: 0,
            sample_index: center as u64,
            peak_amplitude_uv: -120.0,
            participating_channels: vec![0, 1, 2, 3],
        }];
        let pre = matcher.center_offset();
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
        assert_eq!(batch.shape(), [1, 4, w_len]);

        let (emb, dim) = embedder.embed(&batch).unwrap();
        assert_eq!(dim, 4 * embedder.num_components());
        assert_eq!(emb.len(), dim);

        let recon = embedder.reconstruct(&batch).unwrap();
        assert_eq!(recon.shape(), batch.shape());
    }
}
