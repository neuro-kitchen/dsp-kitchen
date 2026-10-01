"""
Unit tests for `dsp_kitchen` Python SDK, native `dsp_kitchen_bindings`, and `synapse.ml` / `synapse.onnx`.
"""

import numpy as np
import dsp_kitchen as dk


def test_filters_and_pipeline():
    raw = np.sin(np.linspace(0, 20 * np.pi, 4 * 1000, dtype=np.float32)).reshape(4, 1000)
    bp = dk.filter.bandpass_filter(raw, low=300.0, high=6000.0, fs=30000.0)
    assert bp.shape == (4, 1000)

    notch = dk.filter.notch_filter(raw, freq=60.0, fs=30000.0)
    assert notch.shape == (4, 1000)

    pipe = dk.Pipeline()
    pipe.add(dk.BandpassFilter(300.0, 6000.0))
    pipe.add(dk.CommonAverageReference())
    out = pipe.run(raw, fs=30000.0)
    assert out.shape == (4, 1000)


def test_model_hub():
    hub = dk.ModelHub()
    entries = hub.list()
    assert len(entries) >= 7
    ks4 = hub.info("kilosort4/temporal-basis-v1")
    assert ks4["id"] == "kilosort4/temporal-basis-v1"
    assert ks4["family"] == "kilosort4"

