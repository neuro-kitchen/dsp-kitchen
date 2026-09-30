"""Compare neuro-convert's TDT reader with TDT's own Python reader (`tdt.read_block`).

Usage:
    uv run --no-project --with tdt --with numpy tests/compare_tdt.py <block> [<block> ...]

For every store: stream rate, channels, sample count and first samples; epoc onsets, offsets
and values; scalar values; snip counts, times, channels, sort codes and first waveform values.
Exits non-zero on any mismatch.
"""

import json
import subprocess
import sys
import warnings

import numpy as np
import tdt

warnings.simplefilter("ignore")
BIN = "target/release/neuro-convert"


def close(a, b, tol=1e-6):
    a, b = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    return a.shape == b.shape and np.allclose(a, b, rtol=1e-5, atol=tol, equal_nan=True)


def compare(block: str) -> list[str]:
    ours = json.loads(subprocess.check_output([BIN, "inspect", "--json", block]))
    ref = tdt.read_block(block)
    problems = []

    def check(ok, what):
        if not ok:
            problems.append(what)

    for r in ours["recordings"]:
        key = tdt.fix_var_name(r["name"])
        s = getattr(ref.streams, key, None)
        if s is None:
            problems.append(f"stream {r['name']}: missing in tdt")
            continue
        data = np.atleast_2d(np.asarray(s.data))
        check(abs(s.fs - r["rate"]) < 1e-3, f"stream {r['name']}: rate {r['rate']} vs {s.fs}")
        check(data.shape[0] == r["channels"], f"stream {r['name']}: channels {r['channels']} vs {data.shape[0]}")
        # tdt pads a partial last packet with zeros; allow up to one packet of difference
        check(abs(data.shape[1] - r["samples"]) <= 4096, f"stream {r['name']}: samples {r['samples']} vs {data.shape[1]}")
        check(close(r["values"], data[0, : len(r["values"])]), f"stream {r['name']}: first values {r['values']} vs {data[0, :5]}")

    for e in ours["events"]:
        key = tdt.fix_var_name(e["name"])
        if e["channels"] > 1:
            s = getattr(ref.scalars, key, None)
            if s is None:
                problems.append(f"scalar {e['name']}: missing in tdt")
                continue
            vals = np.asarray(s.data)
            check(close(e["onsets"], np.asarray(s.ts)[: len(e["onsets"])], 1e-8), f"scalar {e['name']}: onsets")
            check(close(e["values"], vals[:, 0][: len(e["values"])]), f"scalar {e['name']}: values {e['values']} vs {vals[:3, 0]}")
            continue
        s = getattr(ref.epocs, key, None)
        if s is None:
            problems.append(f"epoc {e['name']}: missing in tdt")
            continue
        check(len(s.onset) == e["count"], f"epoc {e['name']}: count {e['count']} vs {len(s.onset)}")
        check(close(e["onsets"], s.onset[: len(e["onsets"])], 1e-8), f"epoc {e['name']}: onsets {e['onsets']} vs {s.onset[:3]}")
        check(close(e["values"], s.data[: len(e["values"])]), f"epoc {e['name']}: values {e['values']} vs {s.data[:3]}")
        if e["offsets"] is not None:
            finite = [o for o in s.offset[: len(e["offsets"])]]
            check(close(e["offsets"], finite, 1e-8), f"epoc {e['name']}: offsets {e['offsets']} vs {finite}")

    for sn in ours["snippets"]:
        s = getattr(ref.snips, tdt.fix_var_name(sn["name"]), None)
        if s is None:
            problems.append(f"snips {sn['name']}: missing in tdt")
            continue
        check(len(s.ts) == sn["count"], f"snips {sn['name']}: count {sn['count']} vs {len(s.ts)}")
        check(close(sn["timestamps"], np.ravel(s.ts)[:3], 1e-8), f"snips {sn['name']}: times")
        check(close(sn["channels"], np.ravel(s.chan)[:3]), f"snips {sn['name']}: channels")
        check(close(sn["sort_codes"], np.ravel(s.sortcode)[:3]), f"snips {sn['name']}: sort codes")
        check(close(sn["values"], np.asarray(s.data)[0, :3]), f"snips {sn['name']}: waveform {sn['values']} vs {np.asarray(s.data)[0, :3]}")

    stores = len(ours["recordings"]) + len(ours["events"]) + len(ours["snippets"])
    print(f"{block}: {stores} stores, {len(problems)} mismatches")
    for p in problems:
        print(f"  ✗ {p}")
    return problems


if __name__ == "__main__":
    failed = sum(len(compare(b)) for b in sys.argv[1:])
    sys.exit(1 if failed else 0)
