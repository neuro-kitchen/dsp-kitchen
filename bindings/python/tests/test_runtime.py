"""`dspk.runtime`: listing, choosing and resetting the compute runtime."""

import pytest

import dspk


def test_version_matches_the_crates():
    assert dspk.__version__ == "0.0.2"


def test_available_lists_compiled_runtimes_gpus_first():
    names = dspk.runtime.available()
    assert names, "at least one runtime is compiled in"
    assert set(names) <= {"wgpu", "cuda", "hip", "cpu"}
    if "cpu" in names:
        assert names[-1] == "cpu"


def test_set_and_reset():
    first = dspk.runtime.available()[0]
    for name in dspk.runtime.available():
        dspk.runtime.set(name)
        assert dspk.runtime.current() == name
    dspk.runtime.set(None)
    assert dspk.runtime.current() == first


def test_unknown_runtime_is_refused():
    with pytest.raises(ValueError, match="unknown compute runtime"):
        dspk.runtime.set("tpu")
