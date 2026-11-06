"""plating: the synthetic signal is what it says, figures render, records round-trip."""

import numpy as np

import plating
from plating import figures, results, signals


def test_synthetic_is_the_sum_of_its_components_and_reproducible():
    s = signals.synthetic(channels=4, seconds=0.5, seed=3)
    assert s.data.shape == (4, 15_000) and s.data.dtype == np.float32
    total = sum(c.astype(np.float64) for c in s.components.values())
    np.testing.assert_allclose(s.data, total, atol=1e-3)
    np.testing.assert_array_equal(s.data, signals.synthetic(channels=4, seconds=0.5, seed=3).data)
    # The hum and the common-mode artefact are the same on every channel
    for name in ("hum", "common"):
        assert np.ptp(s.components[name], axis=0).max() < 1e-4
    assert s.spike_samples.size > 0 and np.all(np.diff(s.spike_samples) >= 0)
    np.testing.assert_allclose(s.without("hum"), s.data - s.components["hum"], atol=1e-3)


def test_figures_render(tmp_path):
    s = signals.synthetic(channels=4, seconds=0.5)
    fig, _ = figures.before_after(s.data, s.without("hum"), s.sample_rate, title="Hum removed")
    assert plating.save(fig, tmp_path / "a.png", note="test").stat().st_size > 10_000
    fig, ax = plating.figure("Traces")
    figures.traces(ax, s.data, s.sample_rate, duration=0.1)
    plating.save(fig, tmp_path / "b.png")
    fig, ax = plating.figure("Raster")
    figures.raster(ax, s.spike_samples, s.spike_units, s.sample_rate)
    plating.save(fig, tmp_path / "c.png")


def test_records_round_trip_and_plot(tmp_path):
    x = np.ones((2, 10))
    cases = [results.compare("same", x, x, tolerance=1e-6, dspk_ms=1.0, reference_ms=20.0),
             results.compare("off", x * 1.1, x, tolerance=1e-6, dspk_ms=2.0, reference_ms=3.0)]
    assert cases[0].passed and not cases[1].passed
    path = results.write(tmp_path / "r.json", "test/bench", cases, reference={"tool": "numpy", "version": np.__version__})
    record = results.read(path)
    assert record["cases"][1]["pass"] is False and record["schema"] == results.SCHEMA_VERSION
    plating.save(figures.timings(path)[0], tmp_path / "t.png")
    plating.save(figures.agreement(record)[0], tmp_path / "g.png")
