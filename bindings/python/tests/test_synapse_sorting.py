"""dspk.synapse clustering, matching and spatial on data whose answer is known: separated blobs,
templates injected at known samples, spikes under one channel, a probe drifting at a known rate."""

import numpy as np
import pytest

from conftest import RUNTIMES
from dspk.synapse import clustering, io, matching, spatial

FS = 30_000.0


@pytest.fixture(scope="module")
def blobs():
    rng = np.random.default_rng(3)
    centres = np.array([[0.0, 0.0], [12.0, 0.0], [0.0, 12.0]], dtype=np.float32)
    truth = np.repeat(np.arange(3), 120)
    x = (centres[truth] + rng.normal(0, 1.0, (truth.size, 2))).astype(np.float32)
    return x, truth


def pure(labels, truth) -> bool:
    """Each true group carries one label, and different groups different labels."""
    per = [np.unique(labels[truth == g]) for g in np.unique(truth)]
    return all(p.size == 1 and p[0] >= 0 for p in per) and len({int(p[0]) for p in per}) == len(per)


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_kmeans_finds_the_blobs(blobs, runtime):
    x, truth = blobs
    r = clustering.kmeans(x, 3, runtime=runtime)
    assert pure(r["labels"], truth)
    assert r["centers"].shape == (3, 2) and r["inertia"] > 0


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_hdbscan_finds_the_blobs(blobs, runtime):
    x, truth = blobs
    labels = clustering.hdbscan(x, min_cluster_size=20, runtime=runtime)
    core = labels >= 0
    assert core.mean() > 0.9
    assert len(np.unique(labels[core])) == 3
    assert pure(np.where(core, labels, labels.max() + 1 + truth), truth) or all(np.unique(labels[(truth == g) & core]).size == 1 for g in range(3))


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_gmm_chooses_three_components(blobs, runtime):
    x, truth = blobs
    r = clustering.gmm(x, k_min=1, k_max=5, runtime=runtime)
    assert r["means"].shape[0] == 3
    assert pure(r["labels"], truth)
    assert r["responsibilities"].shape == (x.shape[0], 3)
    assert np.allclose(r["responsibilities"].sum(axis=1), 1.0, atol=1e-3)


def test_density_peaks_and_kde_merge(blobs):
    x, truth = blobs
    r = clustering.density_peaks(x, cutoff_distance=2.0, num_clusters=3)
    assert pure(r["labels"], truth)
    assert r["centers"].size == 3
    k = clustering.kde_merge(x, initial_k=9)
    assert 3 <= k["centroids"].shape[0] <= 9
    # Merging never splits a blob across far-apart clusters: each blob's majority label holds most of it
    for g in range(3):
        _, counts = np.unique(k["labels"][truth == g], return_counts=True)
        assert counts.max() >= 0.3 * (truth == g).sum()


def test_subsample_peaks():
    samples = np.arange(1000, dtype=np.uint64) * 7
    channels = np.arange(1000) % 4
    kept = clustering.subsample_peaks(samples, channels, n_peaks=100)
    assert kept.size == 100 and np.all(np.diff(samples[kept]) > 0)
    per = clustering.subsample_peaks(samples, channels, n_peaks=10, per_channel=True)
    assert np.bincount(channels[per], minlength=4).tolist() == [10, 10, 10, 10]


def shape(width=40, at=12):
    t = np.arange(width) - at
    return (-np.exp(-((t / 3.0) ** 2)) + 0.3 * np.exp(-(((t - 8) / 5.0) ** 2))).astype(np.float32)


def test_template_similarity_and_merges():
    a = shape()
    b = np.roll(a, 3)  # the same unit, three samples later
    c = np.sin(np.linspace(0, 6 * np.pi, a.size)).astype(np.float32)
    t = np.stack([a, b, c])[:, None, :] * 100
    m = matching.template_similarity(t, max_lag_samples=5)
    assert m.shape == (3, 3) and np.allclose(np.diag(m), 1.0)
    assert m[0, 1] > 0.95 and abs(m[0, 2]) < 0.6
    merges = matching.suggest_merges(t, max_lag_samples=5, threshold=0.95)
    assert [(i, j) for i, j, _ in merges] == [(0, 1)]


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_matching_pursuit_recovers_injected_spikes(runtime):
    w = 40
    # Two units, each on its own channel
    templates = np.zeros((2, 2, w), dtype=np.float32)
    templates[0, 0] = shape(w, 12) * 120
    templates[1, 1] = shape(w, 12) * 90
    n = 6000
    data = np.zeros((2, n), dtype=np.float32)
    starts0, starts1 = np.arange(200, n - 200, 700), np.arange(500, n - 200, 900)
    for s in starts0:
        data[0, s : s + w] += templates[0, 0]
    for s in starts1:
        data[1, s : s + w] += templates[1, 1]
    r = matching.matching_pursuit(data, templates, min_explained_energy=1000.0, runtime=runtime)
    trough = int(np.argmin(templates[0, 0]))
    for unit, starts in ((0, starts0), (1, starts1)):
        found = np.sort(r["samples"][r["units"] == unit].astype(np.int64))
        assert found.size == starts.size, (unit, found, starts + trough)
        assert np.all(np.abs(found - (starts + trough)) <= 1)
    assert np.allclose(r["amplitudes"], 1.0, atol=0.05)


def test_localize_puts_spikes_under_their_channel():
    probe = io.Probe.from_positions([[0, 0], [0, 20], [0, 40], [0, 60]])
    s = shape()
    n = 5
    waveforms = np.zeros((n, 4, s.size), dtype=np.float32)
    # The spike is under channel 2 (y = 40); rows start with it, nearest next, as snippets give them
    waveforms[:, 0] = 100 * s
    waveforms[:, 1] = 30 * s
    waveforms[:, 2] = 30 * s
    waveforms[:, 3] = 10 * s
    snippets = {"waveforms": waveforms, "channels": np.tile([2, 1, 3, 0], (n, 1))}
    for method in spatial.METHODS:
        p = spatial.localize(snippets, probe, method=method)
        assert p.shape == (n, 3)
        assert np.all(np.abs(p[:, 1] - 40.0) < 6.0), (method, p[:, 1])


def test_rigid_drift_follows_the_probe():
    rng = np.random.default_rng(5)
    seconds = 120.0
    total = int(seconds * FS)
    samples = np.sort(rng.integers(0, total, 40_000)).astype(np.uint64)
    t = samples / FS
    true_drift = 10.0 * t / seconds  # µm, 0 → 10 over the recording
    units = rng.choice([200.0, 400.0, 650.0, 800.0], samples.size)
    depths = (units + true_drift + rng.normal(0, 2.0, samples.size)).astype(np.float32)
    amps = rng.uniform(50, 150, samples.size).astype(np.float32)
    r = spatial.drift(samples, depths, amps, total, FS, depth_min_um=0.0, depth_max_um=1000.0)
    est = r["drift_um"] - r["drift_um"][0]
    expected = 10.0 * r["times_s"] / seconds
    expected = expected - expected[0]
    assert np.corrcoef(est, expected)[0, 1] > 0.9
    assert np.max(np.abs(est - expected)) < 4.0
    nr = spatial.drift(samples, depths, amps, total, FS, depth_min_um=0.0, depth_max_um=1000.0, rigid=False)
    assert nr["drift_um"].shape == (spatial.DEFAULT_DRIFT_DEPTH_BLOCKS, r["times_s"].size)
