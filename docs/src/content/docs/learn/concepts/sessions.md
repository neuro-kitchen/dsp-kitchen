---
title: Sessions
description: "The session document: everything the app remembers about your work."
sidebar:
  order: 5
---

Everything you do in the [app](/dsp-kitchen/app/) lives in one JSON document, the *session*: which
recordings are open, the processing graph, views, epochs and recipes. The app redraws from this
document and writes it back as you work.

Because it's a plain file, you can read it, keep it with your data, and edit it while the app is
open: the app watches the file and picks up the change.

On Linux the current session is `~/.config/dspk-app/current-session.json`.

| Part | What it holds |
|---|---|
| `graph` | The Pipeline canvas: sources, operations, views, writes |
| `views` | Each view: its recording, its operations, how it is drawn, what plugins linked to it |
| `recipes` | Reusable groups of operations the graph uses |
| `epochs` | Marked spans of the recordings |
| `placements` | Where recordings sit on the session's time |

The field-by-field description is in the
[session document reference](/dsp-kitchen/reference/schemas/session/).

A view's operations are the same nodes as a [pipeline spec](/dsp-kitchen/reference/schemas/pipeline/),
so what you tune in the app is what a run does.
