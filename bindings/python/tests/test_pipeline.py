"""`dspk.ops.Pipeline`: chained stages on the device equal the same steps one by one; a stream in
uneven chunks equals the whole signal."""

import numpy as np
import pytest
from scipy import signal as sp

import dspk
from dspk.ops import Pipeline, stage

FS = 30_000.0
TOLERANCE = 2e-5  # of the signal peak, as the filter tests
from conftest import RUNTIMES


@pytest.fixture(scope="module")
def data() -> np.ndarray:
    rng = np.random.default_rng(11)
    t = np.arange(12_000) / FS
    common = np.sin(2 * np.pi * 50.0 * t)  # shared by every channel: CAR removes it
    x = np.stack([common + np.sin(2 * np.pi * (500.0 * (c + 1)) * t) + 0.2 * rng.standard_normal(t.size) for c in range(6)])
    return (100.0 * x).astype(np.float32)


def rel_err(got, expected) -> float:
    return float(np.max(np.abs(got - expected)) / np.max(np.abs(expected)))


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_chain_equals_steps(data, runtime):
    sos = sp.butter(5, 300.0, "highpass", fs=FS, output="sos")
    x = 0.5 * data.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    expected = sp.sosfilt(sos, x, axis=1)
    pipe = Pipeline([stage.scale(0.5), stage.car(), stage.iir("highpass", 300.0, mode="forward")], FS, runtime=runtime)
    assert (len(pipe), pipe.runtime) == (3, runtime)
    assert rel_err(pipe.process(data), expected) < TOLERANCE


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_stream_in_uneven_chunks_equals_whole_signal(data, runtime):
    sos = sp.butter(5, (300.0, 6_000.0), "bandpass", fs=FS, output="sos")
    expected = sp.sosfilt(sos, data.astype(np.float64), axis=1)
    pipe = Pipeline([stage.iir("bandpass", (300.0, 6_000.0), mode="forward")], FS, stream=True, runtime=runtime)
    edges = [0, 1_000, 1_001, 4_500, 9_000, data.shape[1]]
    got = np.concatenate([pipe.process(data[:, a:b]) for a, b in zip(edges, edges[1:])], axis=1)
    assert rel_err(got, expected) < TOLERANCE
    pipe.reset()
    assert rel_err(pipe.process(data), expected) < TOLERANCE  # a new stream after reset


def test_stream_refuses_zero_phase_and_channel_changes(data):
    with pytest.raises(ValueError, match="forward filters"):
        Pipeline([stage.iir("highpass", 300.0)], FS, stream=True)
    pipe = Pipeline([stage.car()], FS)
    pipe.process(data)
    with pytest.raises(ValueError, match="channels"):
        pipe.process(data[:3])
    pipe.reset()
    assert pipe.process(data[:3]).shape == (3, data.shape[1])


def test_settling_and_stage_checks():
    pipe = Pipeline([stage.iir("highpass", 300.0), stage.median(9)], FS)
    left, right = pipe.settling
    assert left > 4 and right > 4  # zero-phase filter needs both sides
    assert stage.MEDIAN_DEFAULT_EDGE == "zeros"
    with pytest.raises(ValueError, match="odd"):
        stage.median(4)
    with pytest.raises(ValueError, match="above max"):
        stage.clamp(1.0, -1.0)
    with pytest.raises(ValueError):
        Pipeline([stage.iir("lowpass", 20_000.0)], FS)  # above Nyquist
