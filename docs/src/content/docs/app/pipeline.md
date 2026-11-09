---
title: Pipeline
description: "The session's processing graph on a canvas."
sidebar:
  order: 2
---

Pipeline shows the session as a graph: recordings feed operations, operations feed views and writes.
Every edit is a session command, so the canvas and Explore are always the same graph.

| Area | What it holds |
|---|---|
| Top | A menu per family of operations, and a floating toolbar on the canvas |
| Left | **Layers** (the graph as a tree) and **Recipes** |
| Centre | The canvas |
| Right | The inspector of what you select: values, presets, a recipe's uses |

## The canvas

Wheel pans; Ctrl+wheel zooms at the pointer. Drag to select or move, and drag from a handle to
connect two nodes. A minimap and controls sit in the corners.

## Operations

| Family | Operations |
|---|---|
| Sources | The open recording, or a recording from a file. A node picks its file's source from a dropdown and shows its metadata |
| Channels | Remap: reorder, drop and name channels |
| Filter | IIR, FIR, SOS, Notch, with presets such as "Spikes · band-pass 300–6000 Hz" and "EMG · band-pass 20–450 Hz" |
| Reference | CMR, CAR, Laplacian (with grid presets), Whiten |
| Smooth | Median, Gaussian |
| Transform | Scale, Subtract baseline, Clamp, Teager–Kaiser, Rectify, Square, Square root, Log₁₀ |
| Rate | Resample, as the last op before Write |
| Output | View (a stream shown in Explore), Write (a float32 Zarr file) |

Add one from a menu, or click it to add it after the selected node.

## Channel maps

A Remap node opens a channel table: reorder, drop, add back, reset, reverse, and import or export
map files in TOML or JSON. A remap can sit anywhere in a chain, and the preview follows it. In the
middle of a chain it has to keep every channel; a remap at the start or the end can drop channels.

## Branches

A node can feed several others. Branches share the work of their common first steps: one read, the
shared stages once, then each branch.

## Recipes

Select a group of nodes and **Make recipe** to reuse it. A recipe has typed ports and exposed
parameters. **Save to library** keeps it for other sessions, and **Ungroup** opens it back up.
Built-in recipes include an EMG envelope and an RMS envelope.

Recipes hold chains of operations for now, not branches.
