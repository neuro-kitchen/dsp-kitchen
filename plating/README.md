# plating

How dsp-kitchen's figures are presented (in a kitchen, *plating* is how a dish is presented).
The playground and the benchmarks draw every figure through this package, so they look like one
set and no styling lives in the scripts. It is committed so anyone can see how a figure was built.

```text
plating/
├── theme.py     palette (validated categorical order, light / dark), fonts, figure(), save()
├── figures.py   traces, overlay, spectrum, response, raster, waveforms; before_after; timings, agreement
├── signals.py   synthetic(): a recording whose content is known (LFP, hum, common mode, spikes, noise)
└── results.py   the benchmark JSON: Case, compare(), write(), read()
```

It is installed (editable) in the repository's environment: `uv sync` at the root.

## A figure, start to finish

```python
import plating
from plating import figures, signals

s = signals.synthetic()                         # 8 channels × 2 s at 30 kHz, every component known
clean = s.without("hum")                        # what an ideal hum filter would leave
fig, _ = figures.before_after(s.data, clean, s.sample_rate, title="Mains hum removed",
                              labels=("recorded", "without hum"))
plating.save(fig, "figures/hum.png", note="One line on what was done.")
```

`save` adds the footer (the note, then the machine and the dspk version / runtime) and writes a
PNG at 200 dpi.

## Rules the figures follow

- **Colour does a job.** Data before an operation, or a reference, is drawn in the muted ink; the
  result in series slot 1 (blue); several results take the slots in their fixed order (blue,
  orange, aqua, yellow, magenta, green, violet, red), never cycled. Pass / fail use the status
  colours, always with a word next to them.
- **Text stays in the ink colours**, never in a series colour; two or more series get a legend.
- **One y-axis per panel.** Two quantities of different scale go in two panels.
- **The frame:** a bold left-aligned title saying what the figure shows, a plain subtitle per
  panel, the footer saying how it was made.

## Benchmark records

```python
from plating import results
case = results.compare("bandpass", got, expected, tolerance=1e-5, dspk_ms=1.9, reference_ms=410.0)
results.write("results/filters.json", "signal-operations/filters", [case], reference={"tool": "scipy", "version": scipy.__version__})
fig, _ = figures.timings("results/filters.json")    # or figures.agreement(...)
```

The schema is in `results.py`; records are committed so runs can be compared across versions and
machines.
