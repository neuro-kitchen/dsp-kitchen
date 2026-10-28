"""`dspk.io`: NumPy → file → recording → NumPy round trips, sources, errors."""

import numpy as np
import pytest

import dspk

SAMPLE_RATE = 30_000.0


@pytest.fixture
def signal() -> np.ndarray:
    """4 channels × 3 000 samples, a different sine per channel (µV)."""
    t = np.arange(3_000, dtype=np.float32)
    return np.stack([100.0 * np.sin(0.01 * (c + 1) * t) for c in range(4)]).astype(np.float32)


def test_formats_are_the_generic_ones():
    assert dspk.io.formats() == ["zarr-traces", "raw"]


def test_raw_float32_round_trip(tmp_path, signal):
    path = tmp_path / "rec.bin"
    dspk.io.write_raw(signal, path, SAMPLE_RATE, unit="µV")
    rec = dspk.io.open(path)
    assert (rec.channel_count, rec.samples, rec.sample_rate, rec.format) == (4, 3_000, SAMPLE_RATE, "float32")
    assert rec.units == ["µV"] * 4
    assert rec.duration == pytest.approx(0.1)
    np.testing.assert_array_equal(rec.read(), signal)


def test_raw_int16_quantizes_by_gain(tmp_path, signal):
    path = tmp_path / "rec_i16.bin"
    gain = 0.25
    dspk.io.write_raw(signal, path, SAMPLE_RATE, format="int16", order="time_major", gain=gain)
    rec = dspk.io.open(path)
    assert rec.format == "int16"
    np.testing.assert_allclose(rec.read(), np.round(signal / gain) * gain, atol=1e-6)


def test_read_selects_channels_and_samples(tmp_path, signal):
    path = tmp_path / "rec.bin"
    dspk.io.write_raw(signal, path, SAMPLE_RATE)
    rec = dspk.io.open(path)
    out = rec.read(channels=[3, 1], start=100, stop=600)
    assert out.shape == (2, 500)
    np.testing.assert_array_equal(out, signal[[3, 1], 100:600])


def test_zarr_round_trip_and_sources(tmp_path, signal):
    path = tmp_path / "rec.zarr"
    dspk.io.write_zarr(signal, path, SAMPLE_RATE, unit="mV", chunk_samples=700)
    (source,) = dspk.io.sources(path)
    assert (source.channel_count, source.samples, source.sample_rate, source.unit) == (4, 3_000, SAMPLE_RATE, "mV")
    np.testing.assert_array_equal(dspk.io.open(path, source=source.id).read(), signal)


def test_float64_and_lists_are_converted(tmp_path, signal):
    path = tmp_path / "rec.bin"
    dspk.io.write_raw(signal.astype(np.float64).tolist(), path, SAMPLE_RATE)
    np.testing.assert_array_equal(dspk.io.open(path).read(), signal)


def test_errors(tmp_path, signal):
    with pytest.raises(OSError):
        dspk.io.open(tmp_path / "missing.bin")
    (tmp_path / "other.dat").write_bytes(b"\0" * 8)
    with pytest.raises(ValueError, match="cannot tell the format"):
        dspk.io.open(tmp_path / "other.dat")
    with pytest.raises(ValueError, match="2-D"):
        dspk.io.write_raw(signal[0], tmp_path / "x.bin", SAMPLE_RATE)
    path = tmp_path / "rec.bin"
    dspk.io.write_raw(signal, path, SAMPLE_RATE)
    with pytest.raises(IndexError):
        dspk.io.open(path).read(channels=[9])
