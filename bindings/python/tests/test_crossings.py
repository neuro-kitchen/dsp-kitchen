"""`dspk.ops.crossings.detect` against a plain-Python reference."""

import numpy as np
import pytest

import dspk
from conftest import RUNTIMES


def reference(x, level, release, sign, min_duration, refractory):
    """One side (sign +1 rising, -1 falling) of every channel, as (channel, onset, offset, peak)."""
    out = []
    for c, row in enumerate(x.astype(np.float64)):
        s = sign * row
        on, ready, onset, peak, peak_at = False, 0, 0, -np.inf, 0
        for t in range(1, len(s)):
            if not on and s[t - 1] < level and s[t] >= level and t >= ready:
                on, onset, peak, peak_at = True, t, s[t], t
            elif on:
                if s[t - 1] >= release and s[t] < release:
                    on = False
                    if t - onset >= min_duration:
                        out.append((c, onset, t, peak_at))
                        ready = t + refractory
                elif s[t] > peak:
                    peak, peak_at = s[t], t
        if on:
            out.append((c, onset, -1, peak_at))
    return out


def bursts(seed=3, channels=3, samples=6_000):
    rng = np.random.default_rng(seed)
    t = np.arange(samples)
    x = rng.random((channels, samples)) + np.where(((t // 500)[None, :] + np.arange(channels)[:, None]) % 3 == 0, 6.0, 0.0)
    return x.astype(np.float32)


@pytest.mark.parametrize("runtime", RUNTIMES)
@pytest.mark.parametrize("direction", ["rising", "falling"])
def test_detect_matches_reference(runtime, direction):
    sign = 1.0 if direction == "rising" else -1.0
    x = sign * bursts()
    got = dspk.ops.crossings.detect(
        x, sign * 4.0, release=sign * 2.0, direction=direction, min_duration_samples=20, refractory_samples=100, runtime=runtime
    )
    expected = sorted(reference(x, 4.0, 2.0, sign, 20, 100), key=lambda e: (e[1], e[0]))
    rows = list(zip(got["channel"].tolist(), got["onset"].tolist(), got["offset"].tolist(), got["peak_sample"].tolist()))
    assert len(rows) > 5
    assert rows == expected
    np.testing.assert_array_equal(got["peak_value"], x[got["channel"], got["peak_sample"]])


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_both_directions(runtime):
    x = bursts(channels=1)
    x[0, 3_000:3_100] = -9.0
    got = dspk.ops.crossings.detect(x, 4.0, release=2.0, direction="both", runtime=runtime)
    assert (got["peak_value"] < 0).sum() == 1
    assert (got["peak_value"] > 0).sum() >= 3


def test_bad_options():
    x = bursts(channels=2)
    with pytest.raises(ValueError):
        dspk.ops.crossings.detect(x, 4.0, release=5.0)
    with pytest.raises(ValueError):
        dspk.ops.crossings.detect(x, 4.0, direction="up")
    with pytest.raises(ValueError):
        dspk.ops.crossings.detect(x, [1.0, 2.0, 3.0])
