---
title: Settings and servers
description: "Servers, the runtime, the results folder and the theme."
sidebar:
  order: 3
---

**File → Settings…** (Ctrl+,). Every value shows its default.

| Setting | What it does |
|---|---|
| Servers | Add, edit, remove and test the servers the app works with |
| Results folder | Where results fetched from a server go |
| Compute runtime | Which runtime the app uses |
| Theme | Light or dark, or follow the system. **View** also has "Dark plots in the light theme" |

![Settings with the compute runtime menu open: automatic, wgpu, CUDA or CPU](/dsp-kitchen/img/app/10_Settings-select-runtime.png)

## Add a server

You need a [running server](/dsp-kitchen/deploy/run/) and an
[identity file](/dsp-kitchen/deploy/clients/) for this client.

1. In Settings, **Add server**: a name, its address (`192.168.1.20:50051`), the name on its certificate
   and your identity file.
2. **Test** connects and reports what it finds.

## Open from a server

**File → Open from server…** picks a server and walks its data root. Open a recording to view it in
Explore (read on the server), or fetch a file or folder, such as a sorting, into the results folder.

A remote recording behaves like a local one. A sort started on it runs on the server, and its
progress shows like the app's other jobs.
