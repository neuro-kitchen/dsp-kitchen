---
title: Security
description: "What protects a dspk server, and what is still open."
sidebar:
  order: 7
---

## In place

| Threat | What stops it |
|---|---|
| Eavesdropping or tampering on the wire | QUIC is TLS 1.3, always on |
| Connecting to a fake server | Clients trust the server's authority and check its name |
| Anyone who reaches the port | Mutual TLS: only enrolled, unrevoked clients get in |
| A read-only client running jobs | Starting and stopping jobs needs a client enrolled with `--control` |
| Reading files outside the data root | Paths resolve under it; `..`, hidden files and links out of it are refused; identity and key files are never served |
| Requests that allocate without bound | Views are limited in size; client messages are length-checked first |
| Too many requests or connections | 8 requests at once per client, `--max-clients` (32), 30 s idle timeout |
| Spoofed source addresses | QUIC Retry before any handshake work |
| Job parameters as an attack path | At most 64 parameters, each one line of at most 256 characters |
| Not knowing who did what | Every control request, allowed or refused, goes to an audit log (JSON lines) |

## In the container

The image runs as non-root (uid 10001). The service file adds:

| Flag | What it does |
|---|---|
| `--read-only` | The image's files can't change; only `/data`, the cache volume and `/tmp` are writable |
| `--cap-drop=all` | No Linux capabilities: the server needs none |
| `--security-opt=no-new-privileges` | Nothing inside can gain privileges |
| `--pids-limit=1024` | Caps processes and threads |
| `--memory=<n>g` (optional) | Caps memory; sorts of long recordings need several GB |

## In the supply chain

- `cargo-deny` checks advisories, licences and sources on every pull request, weekly, and before
  publishing.
- Base images are pinned by digest, and the build uses `--locked`.
- Releases publish through trusted publishing, with no stored tokens, after a reviewer approves.

## Not done yet

- Container image scanning and an SBOM.
- Signed releases.

Report a vulnerability privately, as described in the repository's
[security policy](https://github.com/neuro-kitchen/dsp-kitchen/blob/main/SECURITY.md).
