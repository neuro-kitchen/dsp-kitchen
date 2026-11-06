"""dspk.synapse: probes (presets, files, maps), sortings (round-trips), unit metrics against NumPy
definitions."""

import json

import numpy as np
import pytest

from dspk.synapse import io, metrics

FS = 30_000.0


def poisson_train(rate_hz, seconds, seed, refractory_s=0.0):
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(1 / rate_hz, int(rate_hz * seconds * 2)) + refractory_s
    t = np.cumsum(gaps)
    return np.round(t[t < seconds] * FS).astype(np.uint64)


# ---- probes -------------------------------------------------------------------------------------


def test_presets_and_geometry():
    np1 = io.Probe.preset("neuropixels1")
    assert len(np1) == 384
    np.testing.assert_array_equal(np1.positions[:4, :2], [[43, 0], [11, 0], [59, 20], [27, 20]])
    assert len(io.Probe.preset("hdemg8x8", pitch_um=4000.0)) == 64
    with pytest.raises(ValueError, match="unknown preset"):
        io.Probe.preset("nope")
    # Neighbours: itself first, then the nearest
    nb = np1.neighbours(3)
    assert nb.shape == (384, 3) and np.all(nb[:, 0] == np.arange(384))
    # JSON round-trip
    assert io.Probe.from_json(np1.to_json()).to_json() == np1.to_json()


def test_remap_select_and_depth_order():
    p = io.Probe.from_positions([[0, 60], [0, 0], [0, 40], [0, 20]], shanks=[0, 0, 0, 0])
    order = p.depth_order()
    sorted_probe = p.remap(order)
    np.testing.assert_array_equal(sorted_probe.positions[:, 1], [0, 20, 40, 60])
    np.testing.assert_array_equal(p.select([3, 1]).positions[:, 1], [20, 0])


def test_probe_files(tmp_path):
    f = tmp_path / "probe.json"
    f.write_text(json.dumps({"specification": "probeinterface", "probes": [{"contact_positions": [[0, 0], [0, 20], [0, 40]], "device_channel_indices": [2, -1, 0], "contact_ids": ["a", "b", "c"]}]}))
    probe, channel_map = io.read_probe_file(f)
    np.testing.assert_array_equal(probe.positions[:, 1], [0, 40])
    assert [(p["from"], p["to"], p.get("name")) for p in channel_map["pairs"]] == [(2, 0, "a"), (0, 1, "c")]
    mat = tmp_path / "chanMap.mat"
    mat.write_bytes(b"MATLAB")
    with pytest.raises(ValueError, match=".json"):
        io.read_probe_file(mat)


# ---- sortings -----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["out.sorting.zarr", "phy"])
def test_sortings_round_trip(tmp_path, name):
    trains = {0: poisson_train(5, 20, 1, 0.002), 3: poisson_train(12, 20, 2, 0.002)}
    samples = np.concatenate(list(trains.values()))
    units = np.concatenate([np.full(t.size, u) for u, t in trains.items()])
    s = io.Sorting.from_spikes(samples, units, FS, int(20 * FS), sorter_name="test", probe=io.Probe.preset("tetrode"))
    assert s.unit_ids == [0, 3] and len(s) == 2
    np.testing.assert_array_equal(s.spike_train(3), np.sort(trains[3]))
    rows = s.units()
    np.testing.assert_allclose(rows["firing_rate_hz"], [trains[0].size / 20, trains[3].size / 20])
    s.save(tmp_path / name, "phy" if name == "phy" else None)
    back = io.Sorting.load(tmp_path / name)
    assert back.unit_ids == s.unit_ids
    for u in s.unit_ids:
        np.testing.assert_array_equal(back.spike_train(u), s.spike_train(u))
    spikes = back.spikes()
    assert spikes["samples"].size == samples.size and np.all(np.diff(spikes["samples"].astype(np.int64)) >= 0)


# ---- metrics against their definitions ----------------------------------------------------------


def test_isi_violations_and_refractory_counts():
    t = poisson_train(20, 60, 3)  # no refractory period: some violations
    n, dur, thr = t.size, 60.0, metrics.DEFAULT_ISI_THRESHOLD_MS
    isis = np.diff(np.sort(t).astype(np.int64)) / FS
    v = int(np.sum(isis < thr / 1e3))
    r = metrics.isi_violations(t, FS, dur)
    assert r["violations"] == v
    # Hill et al.: violation rate over the violation time window, relative to the total rate
    np.testing.assert_allclose(r["ratio"], (v / (2 * n * thr / 1e3)) / (n / dur), rtol=1e-9)
    # All pairs (not only consecutive) within the refractory samples
    s = np.sort(t).astype(np.int64)
    pairs = sum(int(np.sum((s[i + 1 :] - s[i]) <= 45)) for i in range(s.size))
    assert metrics.refractory_violations(t, 45) == pairs


def test_correlograms_match_brute_force():
    a, b = poisson_train(30, 10, 4), poisson_train(25, 10, 5)
    bin_ms, window_ms = 1.0, 20.0
    acg = metrics.autocorrelogram(a, FS, bin_ms=bin_ms, window_ms=window_ms)
    half = int(round(window_ms / bin_ms))
    expect = np.zeros(2 * half + 1, dtype=np.int64)
    s = np.sort(a).astype(np.int64)
    for i in range(s.size):
        # Lags in ms (float32, as the kernel), binned by rounding half away from zero
        dt = ((s[i + 1 :] - s[i]) * 1e3 / FS).astype(np.float32)
        dt = dt[dt < np.float32((half + 0.5) * bin_ms)]
        k = np.minimum(np.floor(dt / np.float32(bin_ms) + np.float32(0.5)).astype(int), half)
        np.add.at(expect, half + k, 1)
        np.add.at(expect, half - k, 1)
    np.testing.assert_array_equal(acg["counts"], expect)
    np.testing.assert_allclose(acg["lags_ms"], np.arange(-half, half + 1) * bin_ms)
    ccg = metrics.crosscorrelogram(a, b, FS, bin_ms=bin_ms, window_ms=window_ms)
    assert ccg["counts"].sum() > 0 and ccg["counts"].size == 2 * half + 1


def test_presence_histogram_and_matching():
    t = poisson_train(10, 100, 6)
    t = t[(t < 40 * FS) | (t > 60 * FS)]  # silent from 40 to 60 s
    assert metrics.presence_ratio(t, int(100 * FS), FS, bin_s=10.0) == pytest.approx(0.8)
    h = metrics.isi_histogram(t, FS, bin_ms=5.0, max_ms=200.0)
    isis_ms = np.diff(np.sort(t).astype(np.int64)) * 1e3 / FS
    assert h["counts"].sum() == np.sum(isis_ms < 200.0)
    # A train against itself shifted by 2 samples, with 10 % of spikes removed
    a = poisson_train(20, 30, 7, 0.003)
    b = np.delete(a, np.arange(0, a.size, 10)) + 2
    r = metrics.compare_trains(a, b, tolerance_samples=5)
    assert r["matches"] == b.size and r["false_positives"] == 0 and r["false_negatives"] == a.size - b.size
    assert r["precision"] == pytest.approx(1.0) and r["recall"] == pytest.approx(b.size / a.size)


def test_sortings_compared_and_rates():
    trains = {0: poisson_train(5, 20, 8, 0.002), 1: poisson_train(8, 20, 9, 0.002)}
    flat = np.concatenate(list(trains.values()))
    units = np.concatenate([np.full(t.size, u) for u, t in trains.items()])
    a = io.Sorting.from_spikes(flat, units, FS, int(20 * FS))
    b = io.Sorting.from_spikes(flat + 3, units, FS, int(20 * FS))
    c = metrics.compare_sortings(a, b)
    assert c["agreement"].shape == (2, 2) and sorted(c["well_detected"]) == [0, 1]
    np.testing.assert_allclose(np.diag(c["agreement"]), 1.0)
    fr = metrics.firing_rate(trains[1], int(20 * FS), FS)
    assert fr["rate_hz"].mean() == pytest.approx(trains[1].size / 20.0, rel=0.1)
    p = metrics.psth(trains[1], (np.arange(1.0, 19.0) * FS).astype(np.uint64), FS, pre_ms=100.0, post_ms=200.0, bin_ms=20.0)
    assert p["trials"] == 18 and p["rate_hz"].mean() == pytest.approx(8.0, rel=0.5)
