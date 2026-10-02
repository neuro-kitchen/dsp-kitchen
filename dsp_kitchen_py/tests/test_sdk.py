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


def test_sorting_output_storage_and_comparison():
    import tempfile
    import shutil
    from pathlib import Path

    # 1. Build SortingOutput from clusters
    times = [100, 150, 200, 250, 300, 350, 400, 450]
    labels = [0, 1, 0, 1, 0, 1, 0, 1]
    amps = [80.0, 60.0, 85.0, 65.0, 90.0, 70.0, 82.0, 62.0]
    locs = [[0.0, 10.0, 0.0] if l == 0 else [0.0, 20.0, 0.0] for l in labels]

    sorting = dk.SortingOutput.from_clusters(
        sorter_name="test_sorter",
        spike_samples=times,
        labels=labels,
        sample_rate_hz=30000.0,
        total_samples=1000,
        amplitudes=amps,
        locations=locs,
    )

    assert sorting.sorter_name == "test_sorter"
    assert sorting.sample_rate == 30000.0
    assert sorting.num_units == 2
    assert sorting.total_spikes == 8
    assert sorting.unit_ids() == [0, 1]

    # Verify spike trains and metrics
    u0_train = sorting.spike_train(0)
    assert len(u0_train) == 4
    assert list(u0_train) == [100, 200, 300, 400]

    u0_amps = sorting.spike_amplitudes(0)
    assert len(u0_amps) == 4

    metrics = sorting.unit_metrics(0)
    assert metrics["unit_id"] == 0
    assert metrics["num_spikes"] == 4
    assert metrics["quality_label"] in ["good", "mua", "noise"]

    summary = sorting.summary_table()
    assert len(summary) == 2

    # 2. Pairwise spike train matching and sorting comparison
    train_a = [100, 200, 300, 400]
    train_b = [100, 201, 300, 900]
    pw = dk.compare_spike_trains(train_a, train_b, sample_rate_hz=30000.0, delta_time_ms=0.4)
    assert pw["num_matches"] == 3
    assert pw["precision"] == 0.75
    assert pw["recall"] == 0.75

    comp = dk.compare_sortings(sorting, sorting, delta_time_ms=0.4, agreement_threshold=0.5)
    assert comp["agreement_matrix"].shape == (2, 2)
    assert np.allclose(np.diag(comp["agreement_matrix"]), [1.0, 1.0])
    assert comp["mean_agreement"] == 1.0
    assert len(comp["matches"]) == 2

    # 3. CBSS constructor
    cbss_units = [
        {
            "unit_id": 0,
            "spike_samples": [50, 150, 250],
            "pnr_db": 22.5,
            "cov_isi": 0.15,
            "ipt": np.zeros(300, dtype=np.float32),
        }
    ]
    cbss_sort = dk.SortingOutput.from_cbss(
        sorter_name="cbss_mu",
        cbss_units=cbss_units,
        sample_rate_hz=2048.0,
        total_samples=500,
    )
    assert cbss_sort.num_units == 1
    assert cbss_sort.total_spikes == 3

    # 4. Storage Round-trips
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # A. Phy format
        phy_dir = str(tmp / "phy_export")
        dk.save_sorting(sorting, phy_dir, format="phy")
        assert (tmp / "phy_export" / "spike_times.npy").exists()
        assert (tmp / "phy_export" / "spike_clusters.npy").exists()
        assert (tmp / "phy_export" / "params.py").exists()

        loaded_phy = dk.load_sorting(phy_dir)
        assert loaded_phy.num_units == 2
        assert loaded_phy.total_spikes == 8
        assert list(loaded_phy.spike_train(0)) == [100, 200, 300, 400]

        # B. Zarr SortingAnalyzer format
        zarr_dir = str(tmp / "test.sorting.zarr")
        dk.save_sorting(sorting, zarr_dir)
        assert (tmp / "test.sorting.zarr" / "zarr.json").exists()
        assert (tmp / "test.sorting.zarr" / "spikes" / "times.npy").exists()

        loaded_zarr = dk.load_sorting(zarr_dir)
        assert loaded_zarr.num_units == 2
        assert loaded_zarr.total_spikes == 8
        assert list(loaded_zarr.spike_train(1)) == [150, 250, 350, 450]

        # C. NWB /units format
        nwb_dir = str(tmp / "test.nwb.zarr")
        dk.save_nwb_units(sorting, nwb_dir)
        assert (tmp / "test.nwb.zarr" / "units" / "zarr.json").exists()

        loaded_nwb = dk.load_nwb_units(nwb_dir, sample_rate_hz=30000.0)
        assert loaded_nwb.num_units == 2
        assert loaded_nwb.total_spikes == 8
        assert list(loaded_nwb.spike_train(0)) == [100, 200, 300, 400]


if __name__ == "__main__":
    test_filters_and_pipeline()
    test_whitening_ppca_and_clustering()
    test_evoked_and_rate_metrics()
    test_model_hub()
    test_sorting_output_storage_and_comparison()
    print("All dsp_kitchen_py SDK tests passed!")

