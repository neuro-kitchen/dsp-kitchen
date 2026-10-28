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
    (report,) = dspk.pipeline.run(spec(runtime=runtime), base_dir=tmp_path)
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
    (report,) = dspk.pipeline.run(tmp_path / "p.toml")
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


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_subset_range_and_stored_upload(tmp_path, runtime):
    t = np.arange(30_000) / FS
    data = np.stack([200 * np.sin(2 * np.pi * 500 * (c + 1) * t) for c in range(5)]).astype(np.float32)
    dspk.io.write_raw(data, tmp_path / "in.bin", FS, format="int16", order="time_major", gain=0.05)
    s = {
        "dspk_pipeline": 1,
        "nodes": {
            "in": {"op": "read", "path": "in.bin", "channels": [3, 0], "start": 1_000, "stop": 21_000},
            "hp": {"op": "iir", "band": "highpass", "cutoff": [300.0], "mode": "forward"},
            "out": {"op": "write", "path": "out.bin"},
        },
        "edges": [["in", "hp"], ["hp", "out"]],
        "execution": {"chunk_samples": 6_000, "runtime": runtime},
    }
    (report,) = dspk.pipeline.run(s, base_dir=tmp_path)
    assert (report["channels"], report["samples"], report["stored_upload"]) == (2, 20_000, True)
    b = report["bytes"]
    assert b["uploaded"] < b["downloaded"]  # int16 up, float32 down
    assert b["written"] == 2 * 20_000 * 4
    stored = dspk.io.open(tmp_path / "in.bin").read(channels=[3, 0], start=1_000, stop=21_000)
    expected = Pipeline([stage.iir("highpass", 300.0, mode="forward")], FS, stream=True, runtime=runtime).process(stored)
    got = dspk.io.open(tmp_path / "out.bin").read()
    # The run scales int16 on the device, the reference on the host: float32 rounding (~1e-6),
    # carried through the filter, so the limit is 1e-5, below the 2e-5 of the filter tests
    assert np.max(np.abs(got - expected)) / np.max(np.abs(expected)) < 1e-5


def test_batch_and_merge(tmp_path):
    t = np.arange(10_000) / FS
    for name, f in [("a", 300.0), ("b", 700.0), ("c", 1_100.0)]:
        data = np.stack([np.sin(2 * np.pi * f * (c + 1) * t) for c in range(2)]).astype(np.float32)
        dspk.io.write_raw(data, tmp_path / f"{name}.bin", FS)
    batch = {
        "dspk_pipeline": 1,
        "nodes": {"in": {"op": "read", "path": "*.bin"}, "c": {"op": "car"}, "out": {"op": "write", "path": "out/{stem}.bin"}},
        "edges": [["in", "c"], ["c", "out"]],
    }
    reports = dspk.pipeline.run(batch, base_dir=tmp_path)
    assert [r["output"].name for r in reports] == ["a.bin", "b.bin", "c.bin"]
    merge = {
        "dspk_pipeline": 1,
        "nodes": {
            "in": {"op": "read", "inputs": [{"merge": [{"path": "a.bin"}, {"path": "b.bin", "channels": [1]}]}]},
            "out": {"op": "write", "path": "ab.bin"},
        },
        "edges": [["in", "out"]],
    }
    (report,) = dspk.pipeline.run(merge, base_dir=tmp_path)
    got = dspk.io.open(tmp_path / "ab.bin").read()
    expected = np.concatenate([dspk.io.open(tmp_path / "a.bin").read(), dspk.io.open(tmp_path / "b.bin").read(channels=[1])])
    np.testing.assert_array_equal(got, expected)
    batch["nodes"]["out"]["path"] = "same.bin"
    with pytest.raises(ValueError, match="stem"):
        dspk.pipeline.run(batch, base_dir=tmp_path)


def test_plan_progress_and_cancel(tmp_path):
    data = np.random.default_rng(3).standard_normal((3, 100_000)).astype(np.float32)
    dspk.io.write_raw(data, tmp_path / "x.bin", FS)
    s = {
        "dspk_pipeline": 1,
        "nodes": {
            "in": {"op": "read", "path": "x.bin"},
            "hp": {"op": "iir", "band": "highpass", "cutoff": [300.0], "mode": "forward"},
            "out": {"op": "write", "path": "y.bin"},
        },
        "edges": [["in", "hp"], ["hp", "out"]],
        "execution": {"chunk_samples": 30_000},
    }
    (p,) = dspk.pipeline.plan(s, base_dir=tmp_path)
    assert (p["mode"], p["windows"], p["channels"], p["output_samples"]) == ("stream", 4, 3, 100_000)
    assert p["bytes"]["read"] == 3 * 100_000 * 4
    assert not (tmp_path / "y.bin").exists(), "plan writes nothing"
    seen = []
    (report,) = dspk.pipeline.run(s, base_dir=tmp_path, progress=lambda *a: seen.append(a))
    assert seen[-1] == (0, 1, 4, 4) and len(seen) == 4
    assert report["windows"] == p["windows"] and report["bytes"]["read"] == p["bytes"]["read"]
    s["nodes"]["out"]["path"] = "z.bin"
    with pytest.raises(KeyboardInterrupt):
        dspk.pipeline.run(s, base_dir=tmp_path, progress=lambda i, n, done, total: done < 2)
    assert not (tmp_path / "z.bin").exists(), "a cancelled output is removed"

    def boom(*_):
        raise ZeroDivisionError("from the callback")

    with pytest.raises(ZeroDivisionError, match="from the callback"):
        dspk.pipeline.run(s, base_dir=tmp_path, progress=boom)
