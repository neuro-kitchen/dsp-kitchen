"""
Unit tests for `dsp_kitchen` Python SDK, native `dsp_kitchen_bindings`, and `synapse.ml` / `synapse.onnx`.
"""

import numpy as np
import dsp_kitchen as dk


def test_filters_and_pipeline():
    raw = np.sin(np.linspace(0, 20 * np.pi, 4 * 1000, dtype=np.float32)).reshape(4, 1000)
    bp = dk.filter.bandpass_filter(raw, low=300.0, high=6000.0, fs=30000.0)
    assert bp.shape == (4, 1000)

    notch = dk.filter.notch_filter(raw, freq=60.0, fs=30000.0)
    assert notch.shape == (4, 1000)

    pipe = dk.Pipeline()
    pipe.add(dk.BandpassFilter(300.0, 6000.0))
    pipe.add(dk.CommonAverageReference())
    out = pipe.run(raw, fs=30000.0)
    assert out.shape == (4, 1000)


def test_model_hub():
    hub = dk.ModelHub()
    entries = hub.list()
    assert len(entries) >= 7
    ks4 = hub.info("kilosort4/temporal-basis-v1")
    assert ks4["id"] == "kilosort4/temporal-basis-v1"
    assert ks4["family"] == "kilosort4"

    embedder = dk.Kilosort4BasisEmbedder.from_hub()
    assert embedder.num_components == 6
    assert embedder.window_len == 61
    basis = embedder.basis_matrix()
    assert basis.shape == (6, 61)

    recon_basis = embedder.reconstruct(basis)
    assert np.allclose(recon_basis, basis, atol=1e-4)

    snippets = np.stack([basis[:4], basis[1:5]], axis=0).astype(np.float32)
    embedded = embedder.embed(snippets)
    assert embedded.shape == (2, 24)
    denoised = embedder.denoise(snippets)
    assert denoised.shape == (2, 4, 61)
    assert np.allclose(denoised, snippets, atol=1e-4)

    detector = dk.Kilosort4Detector.from_hub(threshold_sigma=4.5, refractory_samples=30)
    assert detector.num_templates == 6
    assert detector.window_len == 61
    templates = detector.templates_matrix()
    assert templates.shape == (6, 61)


if __name__ == "__main__":
    test_filters_and_pipeline()
    test_model_hub()
    print("All dsp_kitchen_py SDK tests passed!")
