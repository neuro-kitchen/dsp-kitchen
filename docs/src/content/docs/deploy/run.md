---
title: Run the server
description: "Serve a folder of recordings from a hardened container."
sidebar:
  order: 2
---

Try it first with a synthetic recording (384 channels at 30 kHz, one hour):

```sh
podman run --rm -p 50051:50051/udp --device nvidia.com/gpu=all \
  --security-opt=label=disable dspk-serve:nvidia
```

## Your own recordings

Put them in a folder, and mount it at `/data`:

```sh
mkdir -p ~/dspk-data && cp recording.ap.bin recording.ap.meta ~/dspk-data/
podman run --rm -p 50051:50051/udp --device nvidia.com/gpu=all \
  --security-opt=label=disable --userns=keep-id:uid=10001,gid=10001 \
  --read-only --cap-drop=all --security-opt=no-new-privileges --pids-limit=1024 \
  -v ~/dspk-data:/data:Z -v dspk-cache:/home/dspk/.cache dspk-serve:nvidia \
  serve --data /data --bind 0.0.0.0:50051 --runtime cuda --name $(hostname) --max-clients 8
```

| Part | Why |
|---|---|
| `:Z` | Lets SELinux allow the container to use the folder |
| `--userns=keep-id:uid=10001,gid=10001` | Maps you to the container's `dspk` user. Without it the server can't write to your folder ("Permission denied") |
| `--data /data` | Clients list and open the recordings under this folder; hidden files and links out of it are refused |
| `--name` | A name clients reach the server by, added to its certificate; repeat for several |
| `--read-only`, `--cap-drop`, … | Hardening, explained in [Security](/dsp-kitchen/deploy/security/) |

To stream one fixed recording to every client instead, name it: `serve /data/recording.ap.bin --loop`.

:::note
The pyramid is built the first time a recording is opened and written next to it, so the folder
must be writable. Later runs reuse it.
:::

## Open the firewall narrowly

QUIC runs over UDP. Open the port only to the network that needs it:

```sh
sudo firewall-cmd --permanent --add-rich-rule='rule family=ipv4 source address=192.168.1.0/24 port port=50051 protocol=udp accept'
sudo firewall-cmd --reload
```

Better still, reach the server through a VPN (WireGuard, Tailscale) and don't open the port to the
internet at all.

Next: [enroll clients](/dsp-kitchen/deploy/clients/).
