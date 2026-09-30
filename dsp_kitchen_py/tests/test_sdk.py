"""
Unit tests for `dsp_kitchen` Python SDK, native `dsp_kitchen_bindings`, and `synapse.ml` / `synapse.onnx`.
"""

import numpy as np
import dsp_kitchen as dk


def test_filters_and_pipeline():
    raw = np.sin(np.linspace(0, 20 * np.pi, 4 * 1000, dtype=np.float32)).reshape(4, 1000)
    bp = dk.filter.bandpass_filter(raw, low_hz=300.0, high_hz=6000.0, sample_rate=30000.0)
    assert bp.shape == (4, 1000)

    # Verify backward-compatible `dsp_kitchen.filters` alias
    notch = dk.filters.notch_filter(raw, freq_hz=60.0, sample_rate=30000.0)
    assert notch.shape == (4, 1000)

    pipe = dk.Pipeline(backend="cpu")
    pipe.add_stage(dk.BandpassFilter(300.0, 6000.0, 30000.0))
    pipe.add_stage(dk.CommonAverageReference())
    out = pipe.run(raw)
    assert out.shape == (4, 1000)


def test_synapse_ml_and_onnx():
    snippets = np.random.randn(3, 4, 40).astype(np.float32) * 25.0

    unet = dk.synapse.ml.SpatiotemporalUnetDenoiser(num_channels=4, num_samples=40, base_filters=8, seed=42)
    denoised = unet.denoise(snippets)
    assert denoised.shape == (3, 4, 40)

    vae = dk.synapse.ml.DartsortVaeEmbedder(num_channels=4, num_samples=40, latent_dim=6, seed=42)
    emb = vae.embed(snippets)
    assert emb.shape == (3, 6)

    simclr = dk.synapse.ml.ContrastiveWaveformEmbedder(num_channels=4, proj_dim=8, seed=42)
    emb_c = simclr.embed(snippets)
    assert emb_c.shape == (3, 8)
    norms = np.linalg.norm(emb_c, axis=1)
    np.testing.assert_allclose(norms, np.ones(3), atol=1e-4)

    curator = dk.synapse.ml.UnitQualityClassifier(seed=42)
    feats = np.array(
        [
            [9.5, 0.0, 12.0, 0.002, 0.99, 0.18, 0.45, 85.0],
            [1.1, 0.2, 0.5, 0.45, 0.10, 0.05, 0.08, 10.0],
        ],
        dtype=np.float32,
    )
    preds = curator.classify(feats)
    assert len(preds) == 2
    assert preds[0][0] == "SingleUnit"
    assert preds[1][0] == "Noise"
