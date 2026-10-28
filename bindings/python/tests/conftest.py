"""Shared test settings.

``RUNTIMES``: the compiled-in runtimes, limited by ``DSPK_TEST_RUNTIMES`` (comma-separated names)
when it is set, as the Rust tests do (``ComputeTarget::for_tests``): CI machines have CUDA compiled
in but no GPU.
"""

import os

import dspk


def selected_runtimes() -> list[str]:
    available = dspk.runtime.available()
    wanted = [n.strip() for n in os.environ.get("DSPK_TEST_RUNTIMES", "").split(",") if n.strip()]
    if not wanted:
        return available
    unknown = set(wanted) - {"wgpu", "cuda", "hip", "cpu"}
    if unknown:
        raise ValueError(f"DSPK_TEST_RUNTIMES: unknown runtime(s) {sorted(unknown)}")
    return [r for r in available if r in wanted]


RUNTIMES = selected_runtimes()

# Tests that do not pass runtime= use the first selected runtime (not the first compiled in, which
# may have no device here: wgpu on a CI runner)
if RUNTIMES:
    dspk.runtime.set(RUNTIMES[0])
