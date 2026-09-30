"""Soundness of the native bindings: zero-copy views keep their owner alive, heavy calls release
the GIL, reads use recording-relative defaults and raise precise errors (Task 05)."""

import gc
import json
import threading
import time

import numpy as np
import pytest

import dsp_kitchen_bindings as dk

FS = 30_000.0


def write_raw(tmp_path, data, dtype="float32", order="channel_major", gain_uv=1.0):
    path = tmp_path / "rec.bin"
    stored = data if order == "channel_major" else data.T
    np.ascontiguousarray(stored, dtype=dtype).tofile(path)
    sidecar = {"channels": data.shape[0], "sample_rate_hz": FS, "format": dtype, "order": order, "gain_uv": gain_uv}
    (tmp_path / "rec.meta").write_text(json.dumps(sidecar))
    return path


def test_numpy_view_outlives_recording(tmp_path):
    data = np.arange(4 * 50_000, dtype=np.float32).reshape(4, 50_000)
    rec = dk.MmapRecording(str(write_raw(tmp_path, data)))
    view = rec.to_numpy()
    mem = rec.memoryview()
    del rec
    gc.collect()
    # The view's base keeps the mapping alive: reading must not crash and must see the file.
    np.testing.assert_array_equal(view, data)
    assert mem.nbytes == data.nbytes
    assert not view.flags.writeable
    with pytest.raises(ValueError):
        view[0, 0] = 1.0


def test_views_and_reads_for_int16_time_major(tmp_path):
    data = (np.arange(3 * 1_000) % 2_000 - 1_000).astype(np.int16).reshape(3, 1_000)
    rec = dk.MmapRecording(str(write_raw(tmp_path, data, dtype="int16", order="time_major", gain_uv=0.5)))
    assert rec.dtype == "int16" and rec.shape == (3, 1_000)
    np.testing.assert_array_equal(rec.to_numpy(), data)
    np.testing.assert_allclose(rec.read(0, 1_000), data.astype(np.float32) * 0.5)


def test_explicit_layout_without_sidecar(tmp_path):
    data = np.random.default_rng(0).normal(size=(2, 3_000)).astype(np.float32)
    path = tmp_path / "plain.dat"
    data.tofile(path)
    rec = dk.MmapRecording(str(path), channels=2, sample_rate=FS)
    np.testing.assert_array_equal(rec.read(0, 3_000), data)
    with pytest.raises(ValueError):
        dk.MmapRecording(str(path), channels=2)  # sample_rate missing
    with pytest.raises(OSError):
        dk.MmapRecording(str(path))  # no sidecar


def test_read_defaults_to_one_second_and_validates(tmp_path):
    data = np.zeros((2, int(FS) * 3), dtype=np.float32)
    rec = dk.MmapRecording(str(write_raw(tmp_path, data)))
    assert rec.read().shape == (2, int(FS))
    assert rec.read(int(FS) * 3 - 10).shape == (2, 10)
    with pytest.raises(IndexError):
        rec.read(0, 10, channels=[0, 5])
    with pytest.raises(ValueError):
        rec.read(10, 5)


def test_pipeline_releases_the_gil():
    data = np.random.default_rng(1).normal(size=(64, 600_000)).astype(np.float32)
    ticks = 0
    done = threading.Event()

    def work():
        dk.bandpass_filter(data, 300.0, 6000.0, FS)
        done.set()

    worker = threading.Thread(target=work)
    start = time.perf_counter()
    worker.start()
    while not done.is_set():
        ticks += 1
        time.sleep(0.001)
    worker.join()
    elapsed = time.perf_counter() - start
    # With the GIL held for the whole call the main thread could not tick at all.
    assert elapsed > 0.05, "workload too small to observe"
    assert ticks > 10, f"main thread only ran {ticks} times in {elapsed:.2f} s"


def test_filter_output_is_a_fresh_float32_array():
    x = np.random.default_rng(2).normal(size=(3, 5_000))  # float64 input is converted
    y = dk.highpass_filter(x, 300.0, FS)
    assert y.dtype == np.float32 and y.shape == (3, 5_000) and y.flags.writeable
