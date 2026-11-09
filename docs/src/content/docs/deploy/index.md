---
title: Deploy
description: "Run dsp-kitchen on a GPU machine so a lab can open recordings and sort them from anywhere."
sidebar:
  order: 0
---

`dspk serve` turns a GPU machine into a server. It holds the recordings, answers views from their
min/max pyramids, and runs sorts on its GPU. Clients connect over QUIC with a certificate the
server issued them: the command line, the app and Python all work the same way.

Follow these in order the first time:

1. [Build the image](/dsp-kitchen/deploy/build/): one `Containerfile`, a target per kind of GPU.
2. [Run the server](/dsp-kitchen/deploy/run/): your folder of recordings at `/data`.
3. [Enroll clients](/dsp-kitchen/deploy/clients/): who may connect, and who may run jobs.
4. [Work from a client](/dsp-kitchen/deploy/remote/): list files, read, sort, fetch.

Then, as needed:

- [Pass the GPU in](/dsp-kitchen/deploy/gpu/), including SELinux on Fedora and RHEL.
- [Run it as a service](/dsp-kitchen/deploy/service/) that starts at boot.
- [Security](/dsp-kitchen/deploy/security/): what protects the server, and what doesn't yet.
- [Smoke test](/dsp-kitchen/deploy/smoke-test/): check an image end to end.
