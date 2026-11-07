"""The signal strips behind the landing page's entry points: one channel each of the playground's
synthetic recording (spike band: slow waves, hum and the common artefact taken out), drawn as SVG
paths. Writes ``src/data/strips.json``; run again only to change them:

    uv run python docs/scripts/strips.py    (from the repository root)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from plating.signals import synthetic

#: One strip per entry point on the landing page.
STRIPS = 10
#: Seconds of signal in each strip, and where they start.
SECONDS = 0.25
START_S = 0.4
#: The SVG box of a strip (width × height, in user units) and its min/max columns.
WIDTH, HEIGHT = 240.0, 24.0
COLUMNS = 160

OUT = Path(__file__).resolve().parents[1] / "src" / "data" / "strips.json"


def strip_path(x: np.ndarray, scale: float) -> str:
    """``x`` as an SVG path: each column's minimum and maximum (what a trace view draws)."""
    cols = np.array_split(x, COLUMNS)
    points = []
    for i, c in enumerate(cols):
        px = (i + 0.5) * WIDTH / COLUMNS
        for v in (c.max(), c.min()):
            py = HEIGHT / 2 - float(v) * scale
            points.append(f"{px:.1f} {min(max(py, 0.5), HEIGHT - 0.5):.1f}")
    return "M" + " L".join(points)


def main() -> None:
    rec = synthetic(channels=STRIPS, seconds=START_S + SECONDS + 0.1, units=8)
    spikes = rec.without("lfp", "hum", "common")
    a, b = int(START_S * rec.sample_rate), int((START_S + SECONDS) * rec.sample_rate)
    window = spikes[:, a:b]
    # Each strip fills its height (a trace view's per-channel gain), so the spikes show.
    paths = [strip_path(ch, (HEIGHT / 2 - 1) / float(np.abs(ch).max())) for ch in window]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"width": WIDTH, "height": HEIGHT, "paths": paths}) + "\n")
    print(f"{OUT}: {len(paths)} strips, {OUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
