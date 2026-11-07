"""dspk.synapse detection, snippets and features on a recording with spikes injected at known
times (two neighbouring channels see each spike)."""

import numpy as np
import pytest

import dspk
from dspk.synapse import detection, extraction, features, io

FS = 30_000.0
CHANNELS, SECONDS = 4, 3.0


@pytest.fixture(scope="module")
def recording():
    rng = np.random.default_rng(11)
    n = int(SECONDS * FS)
    x = rng.normal(0, 10, (CHANNELS, n)).astype(np.float32)
    t = np.arange(-15, 30)
    shape = -np.exp(-((t / 4.0) ** 2)) + 0.3 * np.exp(-(((t - 10) / 6.0) ** 2))
    shape = (shape / -shape.min()).astype(np.float32)
    times = np.arange(3_000, n - 3_000, 2_100)  # every 70 ms
    for s in times:
        x[1, s - 15 : s + 30] += 150 * shape
        x[2, s - 15 : s + 30] += 90 * shape
    probe = io.Probe.from_positions([[0, 0], [0, 20], [0, 40], [0, 60]])
    return x, times, probe, shape


def near(found, wanted, tol=3):
    return np.array([np.min(np.abs(found.astype(np.int64) - w)) <= tol for w in wanted])


def test_threshold_neo_and_matched_filter_find_the_spikes(recording):
    x, times, _, _ = recording
    for name, spikes in (("threshold", detection.threshold(x, FS)), ("neo", detection.neo(x, FS)), ("matched", detection.matched_filter(x, FS))):
        on1 = spikes["samples"][spikes["channels"] == 1]
        assert near(on1, times).mean() >= 0.95, name
        assert np.all(np.diff(spikes["samples"].astype(np.int64)) >= 0)
    # Noise only (channels 0, 3): few crossings at 5 σ
    t = detection.threshold(x, FS)
    assert np.isin(t["channels"], [0, 3]).sum() < 5
    # Positive polarity finds the rebound after each trough (~10 samples later), never the trough
    pos = detection.threshold(x, FS, polarity="positive")
    pos1 = pos["samples"][pos["channels"] == 1].astype(np.int64)
    assert pos1.size > 0 and near(times + 10, pos1, tol=4).mean() >= 0.95 and not near(times, pos1, tol=2).any()


def test_deduplicate_and_snippets(recording):
    x, times, probe, shape = recording
    spikes = detection.threshold(x, FS)
    d = detection.deduplicate(spikes, probe, FS, radius_um=50.0)
    # Each event seen on channels 1 and 2 becomes one spike, on channel 1 (the larger)
    signal = np.isin(d["channels"], [1, 2])
    assert np.all(d["channels"][signal] == 1) and near(d["samples"][signal], times).mean() >= 0.95
    assert any(set(p) == {1, 2} for p in d["participating"])
    sn = extraction.snippets(x, d, probe, FS, k=2, pre_ms=0.5, post_ms=1.0)
    assert sn["waveforms"].shape == (sn["kept"].size, 2, 15 + 30)
    assert np.all(sn["channels"][:, 0] == d["channels"][sn["kept"]])
    mean = sn["waveforms"][signal[sn["kept"]], 0].mean(axis=0)
    assert np.argmin(mean) == 15 and mean.min() == pytest.approx(-150, rel=0.1)


def test_morphology_wavelet_and_velocity(recording):
    _, _, _, shape = recording
    m = features.morphology(np.stack([100 * shape, 50 * shape]))
    np.testing.assert_allclose(m["trough_uv"], [-100, -50], rtol=1e-5)
    assert np.all(m["half_width_samples"] > 0)
    w = features.haar_wavelet(np.ones(64, dtype=np.float32), 3)
    assert w.size == 64 and np.allclose(w[8:], 0)
    # A wave crossing a 2 × 4 grid column by column, 0.5 ms per 5 mm column: 10 m/s
    rows, cols, pitch = 2, 4, 5_000.0
    t = np.arange(300) / FS
    tmpl = np.stack([np.exp(-(((t - 0.003 - c * 0.0005) / 0.0005) ** 2)) for r in range(rows) for c in range(cols)]).astype(np.float32)
    v = features.conduction_velocity(tmpl, rows, cols, pitch, FS)
    assert v["velocity_m_per_s"] == pytest.approx(10.0, rel=0.15)


def test_detect_recording_streams_a_file(tmp_path, recording):
    x, times, probe, _ = recording
    dspk.io.write_raw(x, tmp_path / "r.bin", FS, unit="µV")
    sorting, summary = detection.detect_recording(dspk.io.open(tmp_path / "r.bin"), probe, batch_s=0.5, calibration_s=1.0)
    assert summary["spikes"] > 0 and summary["sigmas_uv"].shape == (CHANNELS,)
    spikes = sorting.spikes()
    assert near(spikes["samples"], times).mean() >= 0.9
    # Every spike located between the two channels that see it (y 20 and 40 µm), nearer the larger
    y = spikes["locations"][:, 1]
    assert np.all(np.isfinite(spikes["locations"]))
    assert 20.0 <= np.median(y) <= 32.0, np.median(y)
    unlocated, _ = detection.detect_recording(dspk.io.open(tmp_path / "r.bin"), probe, batch_s=0.5, calibration_s=1.0, locate=False)
    assert np.all(np.isnan(unlocated.spikes()["locations"]))
