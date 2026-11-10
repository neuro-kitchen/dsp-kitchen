---
title: Events and Analytics
description: "Find events with a threshold crossing, see them as ticks, turn them into epochs, and align signals and units to them."
sidebar:
  order: 3
---

An **event** is a stretch where a channel goes past a level and comes back: a muscle
contraction on an EMG envelope, a deflection of a raw trace, a trigger line going high. The app
finds events with the **Crossings** operation, draws them as ticks over the traces, and aligns
signals and units to them in **Analytics**.

## Find events

In **Pipeline**, build the chain and end it with **Events → Crossings**, then a **Write**. For EMG
contractions:

1. **Filter → IIR**, preset *EMG · band-pass 20–450 Hz*.
2. **Transform → Rectify**.
3. **Filter → IIR**, low-pass at about 6 Hz: the envelope.
4. **Events → Crossings**.
5. **Output → Write**, then double-click it.

The built-in recipe **EMG onsets** holds steps 1–4 (level 50, release 25, minimum 30 ms,
refractory 80 ms); drop it on the canvas and add a Write.

The Write finds the events and saves them as an **event set** of the recording. Its ticks appear on
the traces right away.

Crossings has these settings:

| Setting | What it does | Default |
|---|---|---|
| `level` | An event starts where the signal crosses it | — |
| `release` | The event ends where the signal crosses back past it. Between the baseline and the level, it keeps noise around the level from splitting one event into many | the level |
| `direction` | `rising` (up through the level), `falling` (down through it) or `both` (past +level upwards or −level downwards) | `rising` |
| `min_duration_ms` | Shorter events are dropped | 0 |
| `refractory_ms` | New onsets this soon after an event's end are ignored | 0 |

While an event is open, further crossings belong to it; a new event needs a fresh crossing. For an
EMG envelope, a release at about half the level and a refractory of 50–100 ms is a good start.

### Set the level on the traces

When a view's chain ends with Crossings, its traces show the level (and the release, fainter) as
lines in every lane. Drag a line to change the value; it's rounded to three digits and saved as one
change, so Ctrl+Z undoes it. The view shows the signal Crossings looks at, the envelope.

The ruler needs **Remove DC** off, since removing each lane's offset would move the lines away from
the values they stand for. A button on the plot turns it off.

## Event sets

Event sets are listed under **EVENTS** in Explore's Timeline tab, with their colour and number of
events:

- **Click** a set to move to its next event.
- The **eye** shows or hides its ticks.
- **Right-click** for the ticks' height, thickness and colour, showing offsets too (each event as
  a span), **Make epochs from events**, and **Remove from session**.

Making epochs from events turns every event with an offset into an epoch of type *response*, named
after the set. That's one change, so Ctrl+Z takes them all back.

## Where they're saved

What the app derives from a recording is kept in one Zarr store next to it,
`<recording>.dspk.zarr`, organized in groups as `.nwb.zarr` and `.sorting.zarr` are:

```text
rec.bin
rec.dspk.zarr/
  epochs/          start_s, end_s; ids, names, kinds and notes as attributes
  events/<name>/   channel, onset_sample, offset_sample (−1: still open at the end),
                   peak_sample, peak_value; sample rate and channel names as attributes
```

Event sets in the store are listed again when the recording opens. Sortings keep their own stores.
From Python:

```python
import zarr
g = zarr.open("rec.dspk.zarr/events/w1")
onsets = g["onset_sample"][:] / g.attrs["sample_rate_hz"]
```

A pipeline spec writes the same group when its `write` path ends in `.zarr` (or lies inside a
`.zarr` store); a `.parquet` or `.csv` path writes a table instead.

## Analytics: aligned to events

The **Analytics** workspace shows things aligned to an event set. Pick the set, the window (from
−50 … +150 ms to −1 … +3 s) and how to align:

- **Each channel to its own events**: e.g. each muscle to its own contractions.
- **Selected channels to every event**: the channels selected in Explore, to all events of the
  set.

**Signal** cards show each channel of the active view averaged around the events: the mean, and a
band of ± one standard deviation across events, with a line at the event. The view's processing
applies, so an envelope view averages envelopes.

When the active view has a linked sorting, **Units** cards show each unit's spikes around every
event (one row per event) over its peri-event time histogram (PETH, 10 ms bins, in spikes per
second). The 24 units with the most spikes near the events are shown.
