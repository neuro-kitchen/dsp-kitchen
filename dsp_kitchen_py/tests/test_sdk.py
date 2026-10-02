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


def test_whitening_ppca_and_clustering():
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((4, 500), dtype=np.float32)
    zca = dk.SpatialWhitening.fit_zca(raw, epsilon=1e-3)
    whitened = zca.run(raw)
    assert whitened.shape == (4, 500)

    probe = dk.hdemg_4x8_layout()
    assert probe.num_channels == 32
    lap = dk.SurfaceLaplacian.from_grid_2d(4, 8)
    grid_data = rng.standard_normal((32, 200), dtype=np.float32)
    lap_out = lap.run(grid_data)
    assert lap_out.shape == (32, 200)

    # Synthetic 2-cluster dataset for PPCA + GMM + IsoSplit
    c1 = rng.standard_normal((40, 6), dtype=np.float32) * 0.3 - 4.0
    c2 = rng.standard_normal((40, 6), dtype=np.float32) * 0.3 + 4.0
    feats = np.vstack([c1, c2]).astype(np.float32)

    ppca = dk.PPCA(n_components=2)
    ppca.fit(feats.T)
    proj = np.ascontiguousarray(ppca.transform(feats.T).T)
    assert proj.shape == (80, 2)

    gmm_res = dk.cluster_gmm(proj, min_clusters=1, max_clusters=4, covariance_type="diagonal")
    assert gmm_res["num_clusters"] == 2
    assert len(gmm_res["labels"]) == 80

    iso_res = dk.cluster_isosplit(proj, initial_k=6, dip_threshold=2.0, min_cluster_size=5)
    assert iso_res["num_clusters"] == 2
    assert len(iso_res["labels"]) == 80


def test_evoked_and_rate_metrics():
    # Continuous Gaussian firing rate
    spikes = [i * 300 for i in range(50)]
    fr = dk.compute_firing_rate(spikes, total_duration_sec=1.0, sample_rate_hz=10000.0, bin_dt_sec=0.01, sigma_ms=20.0)
    assert len(fr["rate_hz"]) == 100
    assert np.max(fr["rate_hz"]) > 20.0

    # STA and MEP quantification
    fs = 2000.0
    samples = 4000
    data = np.zeros((2, samples), dtype=np.float32)
    triggers = [500, 1500, 2500]
    for trig in triggers:
        for i in range(10, 40):
            t_ms = (i - 10) / 2.0
            wave = 400.0 * np.sin(t_ms * 0.4) * np.exp(-0.1 * t_ms)
            data[0, trig + i] = wave
            data[1, trig + i] = wave * 0.5

    sta = dk.compute_sta(data, triggers, sample_rate_hz=fs, pre_ms=10.0, post_ms=40.0)
    assert sta["mean"].shape[0] == 2
    assert sta["se"].shape == sta["mean"].shape

    mep = dk.quantify_mep(sta["mean"][0], list(sta["time_ms"]), baseline_window_ms=(-10.0, -1.0), response_window_ms=(2.0, 35.0))
    assert mep["peak_to_peak_uv"] > 200.0
    assert mep["onset_latency_ms"] is not None


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
    test_whitening_ppca_and_clustering()
    test_evoked_and_rate_metrics()
    test_model_hub()
    print("All dsp_kitchen_py SDK tests passed!")
