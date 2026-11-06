# Deploy: dspk as a server (Podman)

`dspk serve` streams a recording over QUIC to its clients: the exact description of the
recording, its stored samples at the recording's rate, and views answered from its min/max
pyramid. Clients include `dspk receive`, the app and Python. It listens on UDP port 50051, and
everything it sends is encrypted with TLS.

| File | What it is |
|---|---|
| `Containerfile` | The image: the `dspk` command line on Debian slim, run as a non-root user |
| `.containerignore` | Keeps build output, data and the Python environment out of the build context |
| `dspk-serve.container` | A Podman Quadlet: the server as a systemd service |

## Build

From the repository root (the build needs the whole workspace):

```bash
podman build -f deploy/Containerfile --ignorefile deploy/.containerignore -t dspk-serve .
```

The first build compiles everything and takes a while. Later builds reuse Podman's caches of
crates, build output and the LLVM bundle. For a smaller image without the GPU runtimes, add
`--build-arg FEATURES=cpu,hub`.

## Run

A synthetic recording (384 channels at 30 kHz, one hour):

```bash
podman run --rm -p 50051:50051/udp dspk-serve
```

Your own recording, from a folder mounted at `/data`:

```bash
mkdir -p ~/dspk-data && cp recording.ap.bin recording.ap.meta ~/dspk-data/
podman run --rm -p 50051:50051/udp --userns=keep-id:uid=10001,gid=10001 \
  -v ~/dspk-data:/data:Z dspk-serve \
  serve /data/recording.ap.bin --bind 0.0.0.0:50051 --loop \
  --certificate-out /data/server.der --name $(hostname)
```

- `:Z` lets SELinux allow the container to use the folder.
- `--userns=keep-id:uid=10001,gid=10001` maps you to the container's `dspk` user (uid 10001).
  Without it, rootless Podman maps that user to an unused uid on the host, and the server can't
  write to your folder ("Permission denied" writing `server.der`).
- The pyramid is built on the first run and written next to the recording, so the folder must
  be writable. Later runs reuse it.

## Connect

The server makes its own certificate and writes it to `/data/server.der`. Give that file to
clients so they can check they reach this server:

```bash
dspk receive <server-ip>:50051 --certificate ~/dspk-data/server.der --server-name <name>
```

`--server-name` must be one of the names in the certificate: `localhost`, or a name given with
`--name` (repeat it for several). For a certificate of your own, pass `--certificate chain.pem`
and `--private-key key.pem`, and mount them into the container.

## As a service (starts at boot)

```bash
mkdir -p ~/.config/containers/systemd ~/dspk-data
cp deploy/dspk-serve.container ~/.config/containers/systemd/
# edit the Exec= line: your recording's file name
systemctl --user daemon-reload
systemctl --user start dspk-serve
loginctl enable-linger "$USER"   # keeps it running after you log out
journalctl --user -u dspk-serve  # logs
```

`%H` in the unit is this machine's host name, so the certificate carries it.

## GPU

`dspk serve` streams and needs no GPU. The image can still run the other commands
(`dspk run`, …) on a GPU:

- **NVIDIA:** install the NVIDIA Container Toolkit, then add `--device nvidia.com/gpu=all`.
- **Vulkan (wgpu):** add `--device /dev/dri`. The image also needs a Vulkan driver
  (`mesa-vulkan-drivers`), which isn't installed by default.

Without either, every command runs on the CPU.

## Firewall

QUIC is UDP. If the server's firewall is on, open the port:

```bash
sudo firewall-cmd --add-port=50051/udp --permanent && sudo firewall-cmd --reload
```
