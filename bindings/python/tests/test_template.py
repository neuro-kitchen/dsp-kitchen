"""`dspk.ops.filter.subtract_template`: planted waveforms are removed."""

import numpy as np
import pytest

import dspk
from conftest import RUNTIMES


def plant(channels=2, samples=3_000, events=(200, 900, 905, 2_500), lags=(0, 2, -1, 3), gains=(1.0, 1.4, 0.8, 1.1)):
    t = np.arange(25)
    template = np.stack([np.exp(-((t - 8) ** 2) / 8.0) * (c + 1) - 0.3 * np.exp(-((t - 14) ** 2) / 20.0) for c in range(channels)])
    rng = np.random.default_rng(6)
    background = (0.01 * rng.standard_normal((channels, samples))).astype(np.float32)
    x = background.copy()
    for ev, lag, g in zip(events, lags, gains):
        x[:, ev - 8 + lag : ev - 8 + lag + 25] += g * template
    return x.astype(np.float32), template.astype(np.float32), np.array(events, dtype=np.uint64), background


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_exact_subtraction_without_lag_or_scaling(runtime):
    x, template, events, background = plant(lags=(0, 0, 0, 0), gains=(1, 1, 1, 1), events=(200, 900, 1_500, 2_500))
    got = dspk.ops.filter.subtract_template(x, template, events, 8, max_lag=0, dynamic_scaling=False, runtime=runtime)
    np.testing.assert_allclose(got, background, atol=1e-5)


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_aligned_and_scaled_events_are_removed(runtime):
    x, template, events, background = plant()
    got = dspk.ops.filter.subtract_template(x, template, events, 8, runtime=runtime)
    residual = np.abs(got - background)
    planted = np.max(np.abs(x - background))
    isolated = max(residual[:, ev - 20 : ev + 30].max() for ev in (200, 2_500))
    overlap = residual[:, 880:940].max()
    assert isolated < 0.02 * planted  # isolated events: removed to ~0.5 % (lag 3, gain 1.1 included)
    # Overlapping events (900, 905) are aligned one after the other, so the first alignment is pulled
    # by its neighbour: ~10 % stays (the device equals the host reference here; a property of the method)
    assert overlap < 0.15 * planted


def test_template_arguments_are_checked():
    x, template, events, _ = plant()
    with pytest.raises(ValueError, match="channels"):
        dspk.ops.filter.subtract_template(x, template[:1], events, 8)
    with pytest.raises(ValueError, match="center_offset"):
        dspk.ops.filter.subtract_template(x, template, events, 99)
