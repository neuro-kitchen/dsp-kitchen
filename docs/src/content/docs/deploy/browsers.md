---
title: The endpoint for browsers
description: "dspk serve --web: what it serves, how a page signs in, and how to reach it without a browser warning."
sidebar:
  order: 5
---

`dspk serve --web <folder>` adds an endpoint a browser can use: the files in `<folder>` and a
WebSocket that carries the same requests as a QUIC session (views, files, opening a recording).

:::note
The web app itself is not published yet. This page describes the server's side, which is what a
page will connect to.
:::

```bash
dspk serve --data /data --pair --web <folder of the web app>
# Web app: http://127.0.0.1:50052 (browsers: a pairing code, then a token; this computer only; …)
```

- `--web-bind` sets the address (default `127.0.0.1:50052`, TCP).
- `<folder>` must hold an `index.html`.

## Signing in

A page cannot hold a client certificate, so it pairs:

1. The server shows a one-time code (`--pair` at start, or `dspk clients pair` while it runs):
   10 minutes, 3 tries.
2. The page sends the code and gets a token, which the browser keeps for that address.
3. The token is a client like any other: `dspk clients list` shows it, `dspk clients revoke`
   refuses it from its next connection on.

A page chooses the name it pairs under, and a name can be paired once: two browsers need two
names. A browser at another address (another host name, or `https` in place of `http`) is another
site to the browser and pairs again.

## Reaching it without a browser warning

A browser shows no warning in two cases: the address is this computer's own (loopback), or the
certificate comes from a public authority for a real host name. The server picks from the address
and the options:

| `--web-bind` | `--certificate` | Served as | In the browser |
|---|---|---|---|
| loopback (`127.0.0.1`, `::1`) | no | plain HTTP | no warning: browsers count loopback as secure |
| any | yes | HTTPS with that certificate | no warning, if the certificate is valid for the name |
| not loopback | no | HTTPS with the server's own certificate | a warning, until its authority is trusted |

Plain HTTP is never served on an address that is not loopback.

From there, by where the server runs:

| Where | How | Encrypted by |
|---|---|---|
| Your own computer | open `http://127.0.0.1:50052` | nothing leaves the computer |
| A cluster or any machine you reach by SSH | `ssh -L 50052:127.0.0.1:50052 <host>`, then open `http://127.0.0.1:50052` | SSH |
| A lab server with a host name | `--web-bind 0.0.0.0:50052 --certificate chain.pem --private-key key.pem` | TLS, your certificate |
| A lab server behind a reverse proxy (Caddy, nginx, `tailscale serve`) | the proxy holds the certificate and forwards to the loopback address | TLS, the proxy's certificate |

A proxy must pass WebSockets through and keep the two headers the server sends
(`Cross-Origin-Opener-Policy`, `Cross-Origin-Embedder-Policy`): a page needs them to use threads.

`--certificate` and `--private-key` apply to the QUIC endpoint too.

## Processing for a page

A page may have no compute of its own. It then sends a view's processing steps with each view
request, and the server runs them on its device with the same code the app's preview uses, and
answers the processed view. The server keeps the last few processed chains of each open recording.

## Not offered to browsers yet

The live signal, downloads and jobs.
