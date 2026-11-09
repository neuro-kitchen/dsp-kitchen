---
title: Enroll clients
description: "Only clients the server has enrolled can connect."
sidebar:
  order: 3
---

The server keeps its own small certificate authority. Enrolling a client issues it a certificate,
and the server accepts only those, checking at every connection that it hasn't been revoked. Clients
trust the server through the same authority, so there is nothing else to hand out.

In the container the authority lives in `/data/.dspk-auth`. Elsewhere it defaults to
`<data dir>/serve`, such as `~/.local/share/dspk/serve` (`--auth-dir` or `DSPK_SERVE_AUTH_DIR` moves it).

## Enroll, list, revoke

The identity goes to standard output, so no copy stays in the container:

```sh
(umask 077; podman exec dspk-serve dspk clients enroll alice --out - > alice.dspk-identity.pem)
(umask 077; podman exec dspk-serve dspk clients enroll lab-pc --control --out - > lab-pc.dspk-identity.pem)
podman exec dspk-serve dspk clients list
podman exec dspk-serve dspk clients revoke alice      # refused from its next connection on
```

Give each client its file over a private channel: it holds the client's private key.

:::caution
Never put identity files under `/data`. The server never lists or sends `*.dspk-identity.pem`,
`*.key` or `*.key.pem` files anyway.
:::

## What to know

- A client can only read unless it was enrolled with `--control`. Control is what lets it run and stop jobs.
- Enrolling and revoking take effect without restarting the server.
- A file is never overwritten, and a name already enrolled must be revoked first.
- Certificates are valid for about two years; enroll again after.
- Back up the authority folder. Losing it means enrolling every client again, and anyone who copies
  it can enroll clients.

## Options that change who connects

- `--allow-anonymous` on `serve` also lets in clients without a certificate, read only. Anyone who
  reaches the port can then read the recording, so use it only on a network you trust.
- A certificate from your organisation's CA: pass `--certificate chain.pem` and `--private-key key.pem`
  to `serve`. Clients still need an enrollment, and also `--certificate` with your CA.

Next: [work from a client](/dsp-kitchen/deploy/remote/).
