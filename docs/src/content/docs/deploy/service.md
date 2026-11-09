---
title: Run as a service
description: "Start the server at boot with a Podman Quadlet."
sidebar:
  order: 6
---

`deploy/dspk-serve.container` is a Podman Quadlet: the server as a systemd user service, with the
hardening flags already set.

```sh
mkdir -p ~/.config/containers/systemd ~/dspk-data
cp deploy/dspk-serve.container ~/.config/containers/systemd/
systemctl --user daemon-reload
systemctl --user start dspk-serve
loginctl enable-linger "$USER"   # keeps it running after you log out
journalctl --user -u dspk-serve  # logs
```

Clients open recordings under `~/dspk-data`. To serve one recording to every client, add its path
to `Exec=` in the unit.

`%H` in the unit is this machine's host name, so the certificate carries it.
