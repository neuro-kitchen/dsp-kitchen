"""dspk.ops.layout: channels gathered on the device, channel maps (pairs) from dicts and files,
the reorder stage, and the remap node of pipeline specs."""

import numpy as np
import pytest

import dspk
from dspk.ops import layout, stage


def data(channels=5, samples=3000):
    rng = np.random.default_rng(7)
    return rng.standard_normal((channels, samples)).astype(np.float32)


def test_gather_channels_equals_numpy_indexing():
    x = data()
    order = [4, 0, 4, 2]
    np.testing.assert_array_equal(layout.gather_channels(x, order), x[order])
    # Integers cross as they are and are converted on the device
    xi = (x * 100).astype(np.int16)
    np.testing.assert_array_equal(layout.gather_channels(xi, [1, 3]), xi[[1, 3]].astype(np.float32))
    with pytest.raises(ValueError, match="does not exist"):
        layout.gather_channels(x, [5])


def test_maps_from_dicts_lists_and_files_agree(tmp_path):
    pairs = [{"from": 3, "to": 1, "name": "A1"}, {"from": 1, "to": 2}, {"from": 5, "to": 3, "name": "A3"}]
    one_based = {"base": 1, "pairs": pairs}
    order, names = layout.map_order(one_based, 5)
    assert order == [2, 0, 4] and names == ["A1", None, "A3"]
    assert layout.map_order(pairs and {"remap": one_based}) == (order, names)
    assert layout.map_order([{"from": 2, "to": 0}, {"from": 0, "to": 1}]) == ([2, 0], [None, None])
    # TOML and JSON files round-trip to the same map
    for suffix in (".toml", ".json"):
        path = layout.write_map(one_based, tmp_path / f"map{suffix}")
        assert layout.read_map(path) == layout.as_map(one_based)
    (tmp_path / "hand.toml").write_text('[remap]\nbase = 1\npairs = [{ from = 3, to = 1, name = "A1" }, { from = 1, to = 2 }, { from = 5, to = 3, name = "A3" }]\n')
    assert layout.map_order(tmp_path / "hand.toml") == (order, names)
    x = data()
    np.testing.assert_array_equal(layout.remap(x, tmp_path / "hand.toml"), x[order])
    # The rules: an output twice, an input twice, a gap, past the channels
    for bad in ([{"from": 0, "to": 0}, {"from": 1, "to": 0}], [{"from": 0, "to": 0}, {"from": 0, "to": 1}], [{"from": 0, "to": 1}]):
        with pytest.raises(ValueError, match="channel map"):
            layout.map_order(bad)
    with pytest.raises(ValueError, match="does not exist"):
        layout.map_order([{"from": 9, "to": 0}], 5)


def test_reorder_stage_in_a_pipeline():
    x = data()
    order = [3, 1, 4, 0, 2]
    p = dspk.ops.Pipeline([stage.scale(2.0), stage.reorder_channels(order)], 30_000.0)
    np.testing.assert_allclose(p.process(x), 2.0 * x[order], rtol=1e-6)
    p = dspk.ops.Pipeline([stage.remap([{"from": c, "to": i} for i, c in enumerate(order)])], 30_000.0)
    np.testing.assert_array_equal(p.process(x), x[order])
    with pytest.raises(ValueError, match="not an order"):
        stage.reorder_channels([0, 0, 1])


def test_remap_nodes_in_a_spec(tmp_path):
    x = data(4, 20_000)
    dspk.io.write_raw(x, tmp_path / "in.bin", 30_000.0)
    spec = {
        "dspk_pipeline": 1,
        "nodes": {
            "in": {"op": "read", "path": "in.bin"},
            "m": {"op": "remap", "pairs": [{"from": 2, "to": 0, "name": "A1"}, {"from": 0, "to": 1}]},
            "x2": {"op": "scale", "alpha": 2.0},
            "out": {"op": "write", "path": "out.zarr", "format": "zarr"},
        },
        "edges": [["in", "m"], ["m", "x2"], ["x2", "out"]],
    }
    (plan,) = dspk.pipeline.plan(spec, tmp_path)
    assert plan["channels"] == 2 and plan["output_channels"] == 2
    (report,) = dspk.pipeline.run(spec, tmp_path)
    assert report["channels"] == 2
    out = dspk.io.open(tmp_path / "out.zarr")
    np.testing.assert_allclose(out.read(), 2.0 * x[[2, 0]], rtol=1e-6)
    assert out.channel_names[0] == "A1"
