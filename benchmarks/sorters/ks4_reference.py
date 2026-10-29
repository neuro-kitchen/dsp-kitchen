"""Kilosort4's saved output (`data/kilosort4/saved_results/`, a Phy folder) as a reference, and the
per-channel comparison of a spike set against it. Used by the Kilosort4 validation scripts.

Spikes are compared by *where* (the channel nearest the spike's position, in Kilosort4's channel
space) and *when* (one-to-one matches within a tolerance), because unit ids differ between sorters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(os.environ.get("DSP_KITCHEN_DATA", Path(__file__).resolve().parents[2] / "data"))
BIN_PATH = DATA_DIR / "kilosort4" / "ZFM-02370_mini.imec0.ap.short.bin"
KS4_RESULTS = DATA_DIR / "kilosort4" / "saved_results"

#: Spike-matching tolerance (SpikeInterface default).
MATCH_MS = 0.4
#: Range of the time-difference histogram (± ms): wide enough to show an offset beyond `MATCH_MS`.
OFFSET_WINDOW_MS = 2.0
#: A spike also counts as matched on a channel within this distance.
NEIGHBOUR_UM = 40.0


def read_params(folder) -> dict:
    """`params.py` of a Phy folder as a dict (plain assignments, read without executing it)."""
    params = {}
    for line in (Path(folder) / "params.py").read_text().splitlines():
        name, sep, value = line.partition("=")
        if sep:
            params[name.strip()] = value.strip().strip("'\"")
    return params


@dataclass
class Reference:
    """Kilosort4's spikes, each placed on a channel: `placement="position"` (default) the channel
    nearest the spike's own position (`spike_positions.npy`, as ours are placed); `"template"` the
    peak channel of its template (what Phy shows: every spike of a unit on one channel)."""

    fs: float
    times: np.ndarray
    channel: np.ndarray
    channel_positions: np.ndarray
    ops: dict

    @classmethod
    def load(cls, folder=KS4_RESULTS, placement="position") -> "Reference":
        folder = Path(folder)
        times = np.load(folder / "spike_times.npy").astype(np.int64).ravel()
        order = np.argsort(times, kind="stable")
        ref = cls(
            fs=float(read_params(folder)["sample_rate"]),
            times=times[order],
            channel=np.empty(0, np.int64),
            channel_positions=np.load(folder / "channel_positions.npy").astype(np.float64),
            ops=np.load(folder / "ops.npy", allow_pickle=True).item(),
        )
        if placement == "position":
            ref.channel = ref.nearest_channel(np.load(folder / "spike_positions.npy")[:, :2])[order]
        elif placement == "template":
            templates = np.load(folder / "templates.npy")  # [templates, samples, channels]
            peak_channel = np.ptp(templates, axis=1).argmax(axis=1)
            ref.channel = peak_channel[np.load(folder / "spike_templates.npy").astype(np.int64).ravel()][order]
        else:
            raise ValueError(f"placement must be 'position' or 'template', got {placement!r}")
        return ref

    def nearest_channel(self, locations, chunk=20_000) -> np.ndarray:
        """Channel (Kilosort4's index) nearest each `[n, 2]` location in µm."""
        locations = np.asarray(locations, dtype=np.float64)
        out = np.empty(len(locations), dtype=np.int64)
        for i in range(0, len(locations), chunk):
            d = ((locations[i : i + chunk, None, :] - self.channel_positions[None]) ** 2).sum(axis=2)
            out[i : i + chunk] = d.argmin(axis=1)
        return out


def nearest_index(x, y):
    """Index in sorted `y` of the element nearest each of `x`."""
    if len(y) == 1:
        return np.zeros(len(x), np.int64)
    i = np.clip(np.searchsorted(y, x), 1, len(y) - 1)
    return np.where(np.abs(x - y[i - 1]) <= np.abs(x - y[i]), i - 1, i)


def match(a, b, tol) -> int:
    """One-to-one matches between sorted spike times `a` and `b`: mutual nearest neighbours within
    `tol` samples."""
    if len(a) == 0 or len(b) == 0:
        return 0
    ab, ba = nearest_index(a, b), nearest_index(b, a)
    return int(((ba[ab] == np.arange(len(a))) & (np.abs(a - b[ab]) <= tol)).sum())


def compare(ref: Reference, times, channel, *, shift=0, label="ours", table_rows=0) -> dict:
    """Per-channel comparison of spikes (`times` in samples, `channel` in Kilosort4's channel
    space) with the reference, `shift` samples added to `times`. Prints a report and returns
    its numbers."""
    times = np.asarray(times, dtype=np.int64) + shift
    channel = np.asarray(channel, dtype=np.int64)
    order = np.argsort(times, kind="stable")
    times, channel = times[order], channel[order]
    n_ch = len(ref.channel_positions)
    tol = max(1, round(MATCH_MS * 1e-3 * ref.fs))
    window = round(OFFSET_WINDOW_MS * 1e-3 * ref.fs)
    dist = np.sqrt(((ref.channel_positions[:, None] - ref.channel_positions[None]) ** 2).sum(axis=2))
    ours_by = [times[channel == c] for c in range(n_ch)]
    theirs_by = [ref.times[ref.channel == c] for c in range(n_ch)]

    same = near = 0
    diffs, rows = [], []
    for c in range(n_ch):
        a = theirs_by[c]
        b = np.sort(np.concatenate([ours_by[n] for n in np.flatnonzero(dist[c] <= NEIGHBOUR_UM)]))
        s, n = match(a, ours_by[c], tol), match(a, b, tol)
        same, near = same + s, near + n
        if len(a) and len(b):
            d = b[nearest_index(a, b)] - a
            diffs.append(d[np.abs(d) <= window])
        rows.append((c, len(a), len(ours_by[c]), s, n))
    diffs = np.concatenate(diffs) if diffs else np.empty(0, np.int64)

    ours_per = np.bincount(channel, minlength=n_ch)
    theirs_per = np.bincount(ref.channel, minlength=n_ch)
    out = {
        "spikes": len(times),
        "channels": int((ours_per > 0).sum()),
        "count_correlation": float(np.corrcoef(ours_per, theirs_per)[0, 1]),
        "recall_same_channel": same / len(ref.times),
        "recall_neighbours": near / len(ref.times),
        "precision_neighbours": near / max(len(times), 1),
        "offset_peak": None,
    }
    shift_note = f", shifted {shift:+d}" if shift else ""
    print(f"  [{label}{shift_note}] {len(times):,} spikes on {out['channels']} channels "
          f"(Kilosort4 {len(ref.times):,} on {int((theirs_per > 0).sum())}); count correlation {out['count_correlation']:.3f}")
    print(f"    Kilosort4 spikes matched ±{MATCH_MS} ms: same channel {out['recall_same_channel']:.3f}, "
          f"within {NEIGHBOUR_UM:g} µm {out['recall_neighbours']:.3f}; our spikes matched {out['precision_neighbours']:.3f}")
    if len(diffs):
        values, counts = np.unique(diffs, return_counts=True)
        out["offset_peak"] = int(values[counts.argmax()])
        print(f"    time difference ours − Kilosort4: peak {out['offset_peak']:+d} samples, "
              f"{(np.abs(diffs) <= tol).mean():.0%} of {len(diffs):,} within ±{tol}")
    if table_rows:
        print(f"    {'channel':>7} {'y µm':>6} {'KS4':>6} {'ours':>6} {'same':>6} {'near':>6}")
        for c, nt, no, s, n in sorted(rows, key=lambda r: -r[1])[:table_rows]:
            print(f"    {c:>7} {ref.channel_positions[c, 1]:>6.0f} {nt:>6} {no:>6} {s / max(nt, 1):>6.2f} {n / max(nt, 1):>6.2f}")
    return out


def best_cosine(ours, theirs) -> np.ndarray:
    """For each row of `theirs`, the best |cosine| with any row of `ours` (sign-free: templates
    and principal components are defined up to sign)."""
    a = np.asarray(ours, dtype=np.float64)
    b = np.asarray(theirs, dtype=np.float64)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.abs(b @ a.T).max(axis=1)


def unit_agreement(ref_times, ref_units, times, units, fs, match_ms=MATCH_MS):
    """For every reference unit, the unit of ours sharing the most spikes and their accuracy
    `matched / (n_ref + n_ours − matched)` (SpikeInterface's agreement score): a spike of the
    reference unit is matched by our nearest spike within ±`match_ms` (one-to-one per pair is not
    enforced; with ±0.4 ms windows double matches are rare). Returns `{ref unit: (our unit,
    accuracy)}`."""
    tol = max(1, round(match_ms * 1e-3 * fs))
    times = np.asarray(times, dtype=np.int64)
    units = np.asarray(units)
    order = np.argsort(times, kind="stable")
    times, units = times[order], units[order]
    ref_times = np.asarray(ref_times, dtype=np.int64)
    ref_units = np.asarray(ref_units)
    ours_count = dict(zip(*np.unique(units, return_counts=True)))
    j = nearest_index(ref_times, times)
    hit = np.abs(times[j] - ref_times) <= tol
    out = {}
    for u in np.unique(ref_units):
        sel = (ref_units == u) & hit
        n_ref = int((ref_units == u).sum())
        if not sel.any():
            out[u] = (None, 0.0)
            continue
        cand, counts = np.unique(units[j[sel]], return_counts=True)
        best = int(np.argmax(counts))
        matched = int(counts[best])
        out[u] = (cand[best], matched / (n_ref + ours_count[cand[best]] - matched))
    return out
