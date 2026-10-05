# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""Generate scipy.signal reference fixtures for the dsp-base filter tests.

Run from the repository root:

    uv run crates/dsp-base/tests/scipy_reference.py

Writes crates/dsp-base/tests/fixtures/filters.json. The test signal is rebuilt in Rust from the
same formula (see `test_signal` in tests/common/mod.rs), so only outputs are stored.
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy import signal

FS = 30_000.0
N = 2_000
TOL = 1e-3
# Designs whose filtered outputs are stored (all designs are stored as coefficients).
SIGNAL_CASES = {("bandpass", 5), ("bandpass", 8), ("highpass", 1), ("highpass", 5), ("lowpass", 5), ("bandstop", 2)}


def test_signal(n: int, fs: float) -> np.ndarray:
    t = np.arange(n) / fs
    x = (
        -500.0
        + 40.0 * np.sin(2 * np.pi * 60.0 * t)
        + 25.0 * np.sin(2 * np.pi * 1_000.0 * t + 0.3)
        + 10.0 * np.sin(2 * np.pi * 4_500.0 * t + 1.1)
        + 5.0 * np.sin(2 * np.pi * 9_000.0 * t)
    )
    x[n // 2 :] += 80.0  # step
    x[n // 3 : n // 3 + 30] -= 150.0 * np.hanning(30)  # spike-like trough
    return x


def pole_radius(sec: np.ndarray) -> float:
    return float(np.max(np.abs(np.roots(sec[3:]))))


def rounded(v: np.ndarray) -> list[float]:
    return [float(f"{x:.10g}") for x in v]


def pole_bound(sos: np.ndarray) -> int:
    total = 0
    for sec in sos:
        r = pole_radius(sec)
        total += 2 if r < 1e-12 else max(2, math.ceil(math.log(TOL) / math.log(r)))
    return total


def settling(sos: np.ndarray) -> int:
    """First t with sum_{k>=t} |h[k]| <= TOL * sum |h[k]| (impulse response of the cascade)."""
    horizon = 2 * pole_bound(sos) + 64
    impulse = np.zeros(horizon)
    impulse[0] = 1.0
    h = np.abs(signal.sosfilt(sos, impulse))
    tail = np.cumsum(h[::-1])[::-1]
    above = np.nonzero(tail > TOL * h.sum())[0]
    return int(above[-1] + 1) if above.size else 0


def main() -> None:
    x = test_signal(N, FS)
    cases = []

    def add(name, spec, sos, run_signal=True):
        case = {"name": name, "spec": spec, "sos": sos.tolist(), "settling": settling(sos)}
        if run_signal:
            zi = signal.sosfilt_zi(sos) * x[0]
            case["forward"] = rounded(signal.sosfilt(sos, x, zi=zi)[0])
            case["forward_rest"] = rounded(signal.sosfilt(sos, x))
            pad = min(case["settling"], N - 1)
            case["padlen"] = pad
            case["forward_backward"] = rounded(signal.sosfiltfilt(sos, x, padtype="odd", padlen=pad))
        cases.append(case)

    bands = {
        "lowpass": 3_000.0,
        "highpass": 300.0,
        "bandpass": [300.0, 6_000.0],
        "bandstop": [55.0, 65.0],
    }
    for btype, wn in bands.items():
        for order in range(1, 9):
            sos = signal.butter(order, wn, btype=btype, fs=FS, output="sos")
            spec = {"kind": "butter", "order": order, "btype": btype, "wn": wn, "fs": FS}
            add(f"butter_{btype}_{order}", spec, sos, run_signal=(btype, order) in SIGNAL_CASES)

    # Low cutoffs (LFP / DC removal) put poles next to the unit circle.
    for order, wn in [(2, 0.5), (3, 1.0)]:
        sos = signal.butter(order, wn, btype="highpass", fs=FS, output="sos")
        spec = {"kind": "butter", "order": order, "btype": "highpass", "wn": wn, "fs": FS}
        add(f"butter_highpass_{order}_{wn}hz", spec, sos, run_signal=False)
    for order, wn in [(4, [0.5, 300.0])]:
        sos = signal.butter(order, wn, btype="bandpass", fs=FS, output="sos")
        spec = {"kind": "butter", "order": order, "btype": "bandpass", "wn": wn, "fs": FS}
        add("butter_bandpass_lfp", spec, sos, run_signal=False)

    for f0, q in [(60.0, 30.0), (50.0, 10.0)]:
        b, a = signal.iirnotch(f0, q, fs=FS)
        sos = np.concatenate([b, a])[None, :]
        spec = {"kind": "notch", "f0": f0, "q": q, "fs": FS}
        add(f"notch_{f0:g}_q{q:g}", spec, sos, run_signal=True)

    out = {"fs": FS, "n": N, "tol": TOL, "cases": cases}
    path = Path(__file__).parent / "fixtures" / "filters.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out))
    print(f"wrote {len(cases)} cases to {path}")


if __name__ == "__main__":
    main()
