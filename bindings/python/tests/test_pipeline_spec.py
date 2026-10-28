"""`dspk.pipeline`: specs as dicts and files, run end to end, against `dspk.ops.Pipeline`."""

import json

import numpy as np
import pytest

import dspk
from conftest import RUNTIMES
from dspk.ops import Pipeline, stage

FS = 30_000.0


@pytest.fixture
def recording(tmp_path):
    t = np.arange(40_000) / FS
    x = np.stack([np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 700 * (c + 1) * t) + 1.5 for c in range(4)])
    data = (100 * x).astype(np.float32)
    dspk.io.write_raw(data, tmp_path / "in.bin", FS, unit="µV")
    return data


def spec(out="out.bin", fmt="raw", **execution):
    return {
        "dspk_pipeline": 1,
        "nodes": {
            "in": {"op": "read", "path": "in.bin"},
            "c": {"op": "car"},
            "hp": {"op": "iir", "band": "highpass", "cutoff": [300.0], "mode": "forward"},
            "out": {"op": "write", "path": out, "format": fmt},
        },
        "edges": [["in", "c"], ["c", "hp"], ["hp", "out"]],
        "execution": {"chunk_samples": 7_001, **execution},
    }


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_run_equals_the_stream_pipeline(tmp_path, recording, runtime):
    report = dspk.pipeline.run(spec(runtime=runtime), base_dir=tmp_path)
    assert (report["mode"], report["runtime"], report["windows"]) == ("stream", runtime, 6)
    got = dspk.io.open(tmp_path / "out.bin").read()
    expected = Pipeline([stage.car(), stage.iir("highpass", 300.0, mode="forward")], FS, stream=True, runtime=runtime).process(recording)
    assert np.max(np.abs(got - expected)) / np.max(np.abs(expected)) < 1e-6
    assert dspk.io.open(tmp_path / "out.bin").units == ["µV"] * 4
    assert dspk.pipeline.load(report["provenance"]) == dspk.pipeline.canonical(spec(runtime=runtime))


def test_toml_file_and_default_runtime(tmp_path, recording):
    (tmp_path / "p.toml").write_text(
        """
        dspk_pipeline = 1
        edges = [["in", "c"], ["c", "out"]]

        [nodes.in]
        op = "read"
        path = "in.bin"

        [nodes.c]
        op = "car"   # common average reference

        [nodes.out]
        op = "write"
        path = "car.zarr"
        format = "zarr"
        """
    )
    report = dspk.pipeline.run(tmp_path / "p.toml")
    assert report["runtime"] == dspk.runtime.current()  # no execution.runtime: dspk.runtime decides
    got = dspk.io.open(tmp_path / "car.zarr").read()
    np.testing.assert_allclose(got, recording - recording.mean(axis=0, keepdims=True), atol=1e-3)


def test_canonical_fills_defaults_and_refuses_bad_specs():
    full = dspk.pipeline.canonical(spec())
    assert full["nodes"]["hp"]["order"] == 5 and full["execution"]["mode"] == "auto"
    bad = spec()
    bad["nodes"]["hp"]["cutof"] = [1.0]
    with pytest.raises(ValueError, match="unknown field"):
        dspk.pipeline.canonical(bad)
    bad = spec()
    bad["edges"].append(["in", "out"])
    with pytest.raises(ValueError, match="more than one output"):
        dspk.pipeline.canonical(bad)
    schema = dspk.pipeline.schema()
    assert "dspk_pipeline" in json.dumps(schema)
