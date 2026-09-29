//! `dsp-synapse-ml` (`synapseml`): Deep learning model zoo for electrophysiology
//! spike inference, denoising, latent embeddings, 3D dipole localization, and
//! automated single-unit curation.
//!
//! All models implement the polymorphic traits defined in [`dsp_synapse::traits`]
//! and operate on contiguous [`dsp_synapse::SnippetBatch`] tensors with zero-copy
//! memory hand-off.

pub mod backbones;
pub mod backend;
pub mod curation;
pub mod denoisers;
pub mod detectors;
pub mod embedders;
pub mod hub;
pub mod localizers;

pub use backbones::{
    BatchNorm1dLayer, Conv1dLayer, CrossElectrodeAttention, LayerNorm1D, LinearLayer, MlpBackbone,
    ResBlock1D, UNet1DBackbone,
};
pub use backend::{
    SnippetBatchMetadata, SynapseMlDevice, Tensor, Tensor1D, Tensor2D, Tensor3D,
    snippet_batch_into_tensor, snippet_batch_to_tensor, tensor_into_snippet_batch,
    tensor_to_snippet_batch,
};
pub use curation::{UnitCurationPrediction, UnitQualityClassifier, UnitQualityFeatures};
pub use denoisers::{
    CollisionSeparatorNet, SingleChannelDenoiser, SpatiotemporalUnetDenoiser,
};
pub use detectors::{EnsorArtifactRejector, SpikeClassProbabilities, SpikeDeeptector, YassNeuralDetector};
pub use embedders::{
    ContrastiveWaveformEmbedder, ConvAutoencoderEmbedder, DartsortVaeEmbedder, VaePosterior,
};
pub use hub::{ModelPresetConfig, ProbePreset, SafetensorEntryHeader, SafetensorsMap};
pub use localizers::{DipoleMlpLocalizer, DipoleSourceEstimate, MonopolarMlpLocalizer};

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_synapse::{
        DeduplicatedSpike, FeatureEmbedder, PeakLocalizer, SpikeDetector, UnitQualityLabel,
        WaveformDenoiser, extract_snippet_batch_multichannel, tetrode,
    };

    fn make_synthetic_tetrode_batch() -> (Vec<f32>, dsp_synapse::SnippetBatch, dsp_core::SensorLayout) {
        let layout = tetrode();
        let channels = 4;
        let samples = 2_000;
        let mut raw = vec![0.0f32; channels * samples];

        for ch in 0..channels {
            for s in 0..samples {
                raw[ch * samples + s] = (((s * 17 + ch * 31) % 21) as f32 - 10.0) * 1.5;
            }
        }

        // Inject two spikes at samples 500 and 1200
        for &center in &[500usize, 1200usize] {
            for ch in 0..channels {
                let atten = 1.0 / (1.0 + 0.3 * ch as f32);
                raw[ch * samples + center - 1] = -45.0 * atten;
                raw[ch * samples + center] = -110.0 * atten;
                raw[ch * samples + center + 1] = -50.0 * atten;
                raw[ch * samples + center + 5] = 35.0 * atten;
            }
        }

        let dedup_spikes = vec![
            DeduplicatedSpike {
                primary_channel: 0,
                sample_index: 500,
                peak_amplitude_uv: -110.0,
                participating_channels: vec![0, 1, 2, 3],
            },
            DeduplicatedSpike {
                primary_channel: 0,
                sample_index: 1200,
                peak_amplitude_uv: -110.0,
                participating_channels: vec![0, 1, 2, 3],
            },
        ];

        let batch = extract_snippet_batch_multichannel(
            &raw,
            channels,
            samples,
            &dedup_spikes,
            &layout,
            4,
            12,
            28,
            true,
        );

        (raw, batch, layout)
    }

    #[test]
    fn test_detectors_spikedeeptector_yass_and_ensor() {
        let (raw, batch, _layout) = make_synthetic_tetrode_batch();
        let dev = SynapseMlDevice::Cpu;

        let mut deeptector = SpikeDeeptector::new(1, 40, 42, dev);
        deeptector.spike_prob_threshold = 0.20;
        let detected = deeptector.detect(&raw, 4, 2_000, 30_000.0);
        assert!(!detected.is_empty());

        let mut yass = YassNeuralDetector::new(8, 42, dev);
        yass.probability_threshold = 0.25;
        let yass_events = yass.detect(&raw, 4, 2_000, 30_000.0);
        assert!(!yass_events.is_empty());

        let ensor = EnsorArtifactRejector::new(4, 42, dev);
        let scores = ensor.predict_artifact_scores(&batch);
        assert_eq!(scores.len(), 2);
        let filtered = ensor.filter_clean_snippets(&batch);
        assert!(filtered.num_spikes <= 2);
    }

    #[test]
    fn test_denoisers_single_channel_unet_and_collision() {
        let (_raw, batch, _layout) = make_synthetic_tetrode_batch();
        let dev = SynapseMlDevice::Cpu;

        let sc_denoiser = SingleChannelDenoiser::new(8, 101, dev);
        let d1 = sc_denoiser.denoise(&batch);
        assert_eq!(d1.shape(), batch.shape());

        let st_unet = SpatiotemporalUnetDenoiser::new(4, 40, 8, 202, dev);
        let d2 = st_unet.denoise(&batch);
        assert_eq!(d2.shape(), batch.shape());

        // Verify Safetensors weight save/load roundtrip on SpatiotemporalUnetDenoiser
        let mut st_map = SafetensorsMap::new();
        st_unet.save_weights(&mut st_map);
        let bytes = st_map.to_bytes().unwrap();
        let loaded_map = SafetensorsMap::from_bytes(&bytes).unwrap();
        let mut st_unet_reloaded = SpatiotemporalUnetDenoiser::new(4, 40, 8, 999, dev);
        st_unet_reloaded.load_weights(&loaded_map).unwrap();
        let d2_reloaded = st_unet_reloaded.denoise(&batch);
        assert_eq!(d2.data, d2_reloaded.data);

        let sep = CollisionSeparatorNet::new(4, 8, 303, dev);
        let (primary, secondary) = sep.separate_batch(&batch);
        assert_eq!(primary.shape(), batch.shape());
        assert_eq!(secondary.shape(), batch.shape());
    }

    #[test]
    fn test_embedders_autoencoder_vae_and_contrastive() {
        let (_raw, batch, _layout) = make_synthetic_tetrode_batch();
        let dev = SynapseMlDevice::Cpu;

        // 1. Conv Autoencoder
        let ae = ConvAutoencoderEmbedder::new(4, 40, 6, 42, dev);
        let (emb_ae, dim_ae) = ae.embed(&batch);
        assert_eq!(dim_ae, 6);
        assert_eq!(emb_ae.len(), 2 * 6);

        // 2. Dartsort Variational Autoencoder
        let vae = DartsortVaeEmbedder::new(4, 40, 6, 42, dev);
        let (emb_vae, dim_vae) = vae.embed(&batch);
        assert_eq!(dim_vae, 6);
        assert_eq!(emb_vae.len(), 2 * 6);
        let post = vae.encode_posterior(&snippet_batch_to_tensor(&batch, dev));
        let kls = post.kl_divergence_per_spike();
        assert_eq!(kls.len(), 2);
        assert!(kls[0] >= 0.0);

        // 3. Contrastive SimCLR Embedder (verify unit-norm vectors ||z||_2 == 1.0)
        let simclr = ContrastiveWaveformEmbedder::new(4, 8, 42, dev);
        let (emb_con, dim_con) = simclr.embed(&batch);
        assert_eq!(dim_con, 8);
        let norm0: f32 = emb_con[0..8].iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm0 - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_localizers_monopolar_and_dipole_mlp() {
        let (_raw, batch, layout) = make_synthetic_tetrode_batch();
        let dev = SynapseMlDevice::Cpu;

        let mono = MonopolarMlpLocalizer::new(4, 42, dev);
        let coords = mono.localize(&batch, &layout);
        assert_eq!(coords.len(), 2);
        assert!(coords[0][2] >= 1.0); // Positive z-distance

        let dipole = DipoleMlpLocalizer::new(4, 42, dev);
        let dipoles = dipole.localize_dipoles(&batch, &layout);
        assert_eq!(dipoles.len(), 2);
        let p = dipoles[0].dipole_moment;
        let p_norm = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        assert!((p_norm - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_automated_unit_quality_classifier() {
        let classifier = UnitQualityClassifier::new(42, SynapseMlDevice::Cpu);

        let clean_sua = UnitQualityFeatures {
            snr: 9.5,
            isi_violation_rate_pct: 0.0,
            firing_rate_hz: 12.0,
            amplitude_cutoff: 0.002,
            presence_ratio: 0.99,
            half_width_ms: 0.18,
            trough_to_peak_ms: 0.45,
            repolarization_slope: 85.0,
        };

        let noisy_artifact = UnitQualityFeatures {
            snr: 1.1,
            isi_violation_rate_pct: 0.2,
            firing_rate_hz: 0.5,
            amplitude_cutoff: 0.45,
            presence_ratio: 0.10,
            half_width_ms: 0.05,
            trough_to_peak_ms: 0.08,
            repolarization_slope: 10.0,
        };

        let preds = classifier.classify_units(&[clean_sua, noisy_artifact]);
        assert_eq!(preds.len(), 2);
        assert_eq!(preds[0].label, UnitQualityLabel::SingleUnit);
        assert_eq!(preds[1].label, UnitQualityLabel::Noise);
    }
}
