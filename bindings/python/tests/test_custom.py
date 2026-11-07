"""dspk.custom: your own functions over a recording, window by window, equal to the same function
on the whole array (measures per window; transforms within their halo, written to Zarr)."""

import numpy as np
import pytest
from scipy.ndimage import uniform_filter1d

import dspk
from dspk import custom

FS = 1_000.0


@pytest.fixture(scope="module")
def rec(tmp_path_factory):
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, (3, 10_250)).astype(np.float32)  # 10.25 s: the last 1 s window is short
    x[1] *= 5.0
    path = tmp_path_factory.mktemp("custom") / "rec.zarr"
    dspk.io.write_zarr(x, path, FS, unit="µV")
    return dspk.io.open(path), x


def test_windows_cover_the_span_once_with_their_context(rec):
    r, x = rec
    seen = []
    for data, w in custom.windows(r, window_s=2.0, halo=7, start=100, stop=9_000, channels=[2, 0]):
        assert data.shape == (2, w.read_stop - w.read_start)
        np.testing.assert_array_equal(data, x[[2, 0], w.read_start : w.read_stop])
        assert w.read_start == max(0, w.start - 7) and w.read_stop == min(r.samples, w.stop + 7)
        seen.append((w.start, w.stop))
    assert seen[0][0] == 100 and seen[-1][1] == 9_000
    assert all(a[1] == b[0] for a, b in zip(seen, seen[1:])), "contiguous, no overlap"


def test_measure_gathers_one_row_per_window(rec):
    r, x = rec
    table = custom.measure(r, lambda d, w: {"rms": np.sqrt((d**2).mean(axis=1)), "n": d.shape[1]}, window_s=1.0)
    assert table["rms"].shape == (11, 3) and table["n"].tolist() == [1_000] * 10 + [250]
    np.testing.assert_allclose(table["rms"][3], np.sqrt((x[:, 3_000:4_000].astype(np.float64) ** 2).mean(axis=1)), rtol=1e-5)
    np.testing.assert_allclose(table["start_s"], np.arange(11.0))
    assert table["stop_s"][-1] == pytest.approx(10.25)
    # A value alone (data only) is "value"
    assert custom.measure(r, lambda d: d.shape[1], window_s=5.0)["value"].tolist() == [5_000, 5_000, 250]
    with pytest.raises(ValueError, match="same keys|returned"):
        custom.measure(r, lambda d, w: {"a": 1} if w.index == 0 else {"b": 1})


def test_transform_equals_the_function_on_the_whole_array(rec, tmp_path):
    r, x = rec
    smooth = custom.transform(lambda d: uniform_filter1d(d, 25, axis=1, mode="nearest"), halo=12, window_s=0.7)
    whole = uniform_filter1d(x, 25, axis=1, mode="nearest")
    np.testing.assert_allclose(smooth.apply(r), whole, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(smooth.apply(r, channels=[1], start=2_000, stop=5_000), whole[[1], 2_000:5_000], rtol=1e-5, atol=1e-6)
    # Too little context: differs at the window edges (the halo is what makes them equal)
    short = custom.transform(smooth.fn, halo=3, window_s=0.7)
    assert not np.allclose(short.apply(r), whole, rtol=1e-5, atol=1e-6)
    # Written window by window to Zarr: the same values, names and unit
    out = tmp_path / "smooth.zarr"
    smooth.write(r, out)
    back = dspk.io.open(out)
    np.testing.assert_allclose(back.read(), whole, rtol=1e-5, atol=1e-6)
    assert back.units == ["µV"] * 3 and back.sample_rate == FS


def test_decorator_window_argument_and_refusals(rec):
    r, x = rec

    @custom.transform(halo=0)
    def centred(d, w):
        assert isinstance(w, custom.Window)
        return d - d.mean(axis=0, keepdims=True)

    np.testing.assert_allclose(centred.apply(r), x - x.mean(axis=0, keepdims=True), rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match="keeps the shape"):
        custom.transform(lambda d: d[:1]).apply(r)
    with pytest.raises(ValueError, match="halo"):
        custom.transform(np.abs, halo=-1)
    with pytest.raises(ValueError, match="within"):
        custom.measure(r, np.mean, stop=r.samples + 1)
