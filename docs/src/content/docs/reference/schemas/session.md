---
title: Session document
description: "The JSON file the app works from, field by field."
sidebar:
  order: 2
---

The app's [session](/dsp-kitchen/learn/concepts/sessions/) is one JSON document, version 1. Unknown
fields are refused, so a typo is an error, not silently ignored.

## The session

| Field | Type | Description |
|---|---|---|
| `dspk_session` | integer | Document version; must be `1` |
| `graph` | object | The processing graph (the Pipeline canvas) |
| `recipes` | object | The recipes the graph uses, by name |
| `epochs` | list | Marked spans of the recordings |
| `placements` | list | Where recordings sit on the session's time |
| `views` | list | Every view, in the order they are listed; a document with views and no graph gets a graph on load |
| `active_view` | string | Id of the view shown, and edited by default |

## A view

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique within the session (`v1`, `v2`, …) |
| `name` | string | Shown in tabs and lists |
| `recording` | object | What it reads, below |
| `ops` | list | Processing applied to every channel, in order |
| `display` | object | How it is drawn, below |
| `plugins` | object | What plugins keep for this view, by plugin id (for `synapse`, the linked `.sorting.zarr`) |

## `recording`

| Field | Type | Description |
|---|---|---|
| `path` | string | The file |
| `source` | string | Source id within the file; default: the file's default source |
| `channels` | list of integers | Channels read by this view, in this order; default: all |

## An op

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique within the view (`bandpass`, `cmr`, `op3`, …) |
| `enabled` | boolean | Off: kept in the chain but skipped. Default `true` |
| `node` | object | A [pipeline spec](/dsp-kitchen/reference/schemas/pipeline/) node |

## `display`

| Field | Type | Default | Description |
|---|---|---|---|
| `show` | `"processed"` or `"raw"` | `"processed"` | The chain's output, or the recording as stored |
| `gain` | number | `1.0` | Amplitude gain (1 = nominal) |
| `lanes` | integer | `8` | Traces on screen at once |
| `window` | `[start, end]` | | Visible time window, in seconds from the recording's start |
