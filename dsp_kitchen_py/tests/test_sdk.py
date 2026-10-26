"""The `dsp_kitchen` SDK against NumPy references (and scipy when installed), on the current
runtime."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

import dsp_kitchen as dk

FS = 30_000.0
RNG = np.random.default_rng(42)


def signal(channels=4, samples=6_000):
    return RNG.standard_normal((channels, samples)).astype(np.float32)


# --- runtime -----------------------------------------------------------------------------------


def test_runtime_selection():
    names = dk.runtime.available()
    assert names and dk.runtime.current() in names
    dk.runtime.set(names[-1])
    assert dk.runtime.current() == names[-1]
    dk.runtime.set(None)
    with pytest.raises(ValueError):
        dk.runtime.set("not-a-runtime")


# --- filters and pipelines ---------------------------------------------------------------------


def test_pointwise_and_spatial_stages_match_numpy():
    x = signal()
    np.testing.assert_allclose(dk.math.scale_samples(x, 0.5, 1.0), x * 0.5 + 1.0, rtol=1e-6)
    np.testing.assert_allclose(dk.spatial.common_average_reference(x), x - x.mean(axis=0), atol=1e-5)
    tk = dk.filter.non_linear.teager_kaiser_filter(x)
    np.testing.assert_allclose(tk[:, 1:-1], x[:, 1:-1] ** 2 - x[:, :-2] * x[:, 2:], atol=1e-4)


def test_median_matches_numpy_interior():
    x = signal(2, 1_000)
    out = dk.filter.non_linear.median_filter(x, 9)
    ref = np.median(np.lib.stride_tricks.sliding_window_view(x, 9, axis=1), axis=-1)
    np.testing.assert_allclose(out[:, 4:-4], ref, atol=1e-6)


def test_iir_filters_match_scipy():
    signal_mod = pytest.importorskip("scipy.signal")
    x = signal(3, 20_000)
    ours = dk.filter.iir.bandpass_filter(x, 300.0, 6_000.0, fs=FS)
    sos = signal_mod.butter(5, [300.0, 6_000.0], btype="bandpass", fs=FS, output="sos")
    np.testing.assert_allclose(ours, signal_mod.sosfiltfilt(sos, x), atol=1e-3)
    causal = dk.filter.iir.highpass_filter(x, 300.0, fs=FS, direction="forward")
    sos = signal_mod.butter(5, 300.0, btype="highpass", fs=FS, output="sos")
    np.testing.assert_allclose(causal, signal_mod.sosfilt(sos, x), atol=1e-3)


def test_pipeline_equals_its_stages_in_order():
    x = signal()
    pipe = dk.pipeline.Pipeline([dk.filter.iir.HighpassFilter(300.0), dk.spatial.CommonAverageReference()])
    staged = dk.spatial.common_average_reference(dk.filter.iir.highpass_filter(x, 300.0, fs=FS))
    np.testing.assert_allclose(pipe.run(x, fs=FS), staged, atol=1e-4)
    left, right = pipe.settling(fs=FS)
    assert left > 0 and right > 0


# --- linear algebra ----------------------------------------------------------------------------


def test_pca_variances_match_numpy():
    x = signal(5, 4_000) * np.array([[5.0], [3.0], [1.0], [0.5], [0.1]], dtype=np.float32)
    pca = dk.linalg.PCA(n_components=3).fit(x)
    eig = np.sort(np.linalg.eigvalsh(np.cov(x, bias=True)))[::-1][:3]
    np.testing.assert_allclose(pca.explained_variance, eig, rtol=1e-3)
    assert pca.transform(x).shape == (3, 4_000)


# --- spikes ------------------------------------------------------------------------------------


def spiky(channels=8, samples=30_000, every=1_500):
    x = signal(channels, samples)
    times = np.arange(every, samples - every, every)
    x[2, times] -= 20.0
    return x, times


def test_detection_finds_injected_spikes():
    x, times = spiky()
    sigmas = dk.synapse.estimate_noise(x)
    assert sigmas.shape == (8,) and np.all(np.abs(sigmas - 1.0) < 0.1)
    spikes = dk.synapse.detect_spikes(x, refractory_samples=30)
    found = {s.sample for s in spikes if s.channel == 2}
    assert set(times) <= found


def test_clustering_separates_two_blobs():
    a = RNG.standard_normal((60, 3)).astype(np.float32) * 0.3 - 3.0
    b = RNG.standard_normal((60, 3)).astype(np.float32) * 0.3 + 3.0
    feats = np.vstack([a, b])
    gmm = dk.synapse.cluster_gmm(feats, max_clusters=4)
    assert gmm["num_clusters"] == 2
    km = dk.synapse.kmeans(feats, 2)
    assert len(set(km["labels"][:60])) == 1 and len(set(km["labels"][60:])) == 1
    assert set(dk.synapse.hdbscan(feats, 10)) == {0, 1}


def test_metrics_follow_spikeinterface_conventions():
    spikes = list(range(0, 300_000, 3_000))
    isi = dk.synapse.compute_isi(spikes, fs=FS, duration_sec=10.0)
    assert isi["violation_count"] == 0 and isi["firing_rate_hz"] == pytest.approx(10.0)
    acg = dk.synapse.compute_autocorrelogram(spikes, fs=FS)
    assert acg["bin_ms"] == 1.0 and acg["window_ms"] == 50.0
    # 10 s of a 10 Hz unit: present in every 1 s bin; shorter than one 60 s bin (the default): no
    # ratio, as SpikeInterface
    assert dk.synapse.compute_presence_ratio(spikes, 300_000, fs=FS, bin_duration_sec=1.0) == pytest.approx(1.0)
    assert np.isnan(dk.synapse.compute_presence_ratio(spikes, 300_000, fs=FS))


def test_sorting_round_trips_through_every_format():
    samples = np.arange(0, 30_000, 300, dtype=np.uint64)
    labels = [i % 2 for i in range(len(samples))]
    sorting = dk.synapse.SortingOutput.from_clusters(
        "test", list(samples), labels, fs=FS, total_samples=30_000, primary_channels=[0] * len(samples)
    )
    assert np.isnan(sorting.spike_amplitudes(sorting.unit_ids()[0])).all()
    with tempfile.TemporaryDirectory() as tmp:
        for name, fmt in [("phy", "phy"), ("out.sorting.zarr", "sorting-zarr")]:
            path = Path(tmp) / name
            dk.synapse.save_sorting(sorting, str(path), fmt)
            back = dk.synapse.load_sorting(str(path))
            assert back.num_units == 2 and back.total_spikes == len(samples)
    with pytest.raises(ValueError):
        dk.synapse.SortingOutput.from_clusters("test", [0], [0], fs=FS, total_samples=10)


# --- sorters -----------------------------------------------------------------------------------


def test_sorter_provenance_and_defaults():
    ks = dk.synapse.ml.kilosort4
    assert "10.1038/s41592-024-02232-7" in ks.provenance().citation()
    config = ks.Config()
    assert (config.nt, config.th_universal, config.n_pcs) == (61, 10.0, 6)
    emu = dk.synapse.ml.emusort.Config()
    assert emu.th_single_ch == [6.0, 9.0, 12.0, 15.0] and not emu.do_car and not emu.do_notch
    assert (emu.n_pcs, emu.bandpass_high_hz) == (9, 5000.0)
    # EMUsort's config is flat: no nested Kilosort4 config can bring Kilosort4's defaults in
    assert not hasattr(emu, "kilosort4")
    with pytest.raises(TypeError):
        dk.synapse.ml.emusort.Config(kilosort4=dk.synapse.ml.kilosort4.Config())
    assert "10.7554/eLife.110417.1" in dk.synapse.ml.emusort.provenance().citation()
    with pytest.raises(TypeError):
        ks.Config(not_a_setting=1)


def test_channel_delays_are_recovered():
    emusort = dk.synapse.ml.emusort
    x = np.abs(signal(3, 20_000))
    x[1] = np.roll(x[0], 5)
    delays, reference = emusort.estimate_channel_delays([x], pad=25, max_lag=20)
    shifted = emusort.apply_channel_delays(x, delays)
    assert shifted.shape == x.shape and abs(delays[1] - delays[0]) == 5
    np.testing.assert_allclose(shifted[0], np.roll(x[0], -delays[0]))
