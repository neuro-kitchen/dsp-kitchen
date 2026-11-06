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

One `Containerfile`, three targets. They hold the same `dspk` and `dspk-cli-sorter` (CUDA, wgpu
and CPU compiled in) and differ only in the system libraries each GPU needs:

| Target | For | Adds | Run with |
|---|---|---|---|
| `cpu` (default) | no GPU | nothing (`DSPK_RUNTIME=cpu`) | — |
| `nvidia` | NVIDIA: CUDA and Vulkan | NVRTC 13.0 and the Vulkan loader; the driver comes from the host | `--device nvidia.com/gpu=all` |
| `mesa` | AMD, Intel: Vulkan (wgpu) | the Vulkan loader and Mesa's Vulkan drivers | `--device /dev/dri` |

From the repository root (the build needs the whole workspace):

```bash
podman build -f deploy/Containerfile --ignorefile deploy/.containerignore --target nvidia -t dspk-serve:nvidia .
```

- The first build compiles everything and takes a while. Building another target afterwards
  compiles nothing: they share the build stage and Podman's caches.
- `--build-arg CUDA_NVRTC=13-5` picks a newer NVRTC, at most the CUDA version of the host's
  driver (`nvidia-smi` shows it).
- `--build-arg FEATURES=cpu,hub` compiles the CPU runtime only.

## Run

A synthetic recording (384 channels at 30 kHz, one hour):

```bash
podman run --rm -p 50051:50051/udp --device nvidia.com/gpu=all --security-opt=label=disable dspk-serve:nvidia
```

Your own recording, from a folder mounted at `/data`:

```bash
mkdir -p ~/dspk-data && cp recording.ap.bin recording.ap.meta ~/dspk-data/
podman run --rm -p 50051:50051/udp --device nvidia.com/gpu=all --security-opt=label=disable --userns=keep-id:uid=10001,gid=10001 \
  --read-only --cap-drop=all --security-opt=no-new-privileges --pids-limit=1024 \
  -v ~/dspk-data:/data:Z -v dspk-cache:/home/dspk/.cache dspk-serve:nvidia \
  serve /data/recording.ap.bin --bind 0.0.0.0:50051 --loop --name $(hostname) --max-clients 8
```

- `:Z` lets SELinux allow the container to use the folder.
- `--userns=keep-id:uid=10001,gid=10001` maps you to the container's `dspk` user (uid 10001).
  Without it, rootless Podman maps that user to an unused uid on the host, and the server can't
  write to your folder ("Permission denied").
- The pyramid is built on the first run and written next to the recording, so the folder must
  be writable. Later runs reuse it.
- The second line hardens the container; see [Hardening](#hardening).

## Hardening

The image runs as a non-root user (uid 10001). The service file (`dspk-serve.container`) and the
command above add:

| Flag | What it does |
|---|---|
| `--read-only` | The image's files can't be changed; writable are only `/data`, the cache volume and `/tmp` |
| `--cap-drop=all` | No Linux capabilities: the server needs none |
| `--security-opt=no-new-privileges` | No program inside can gain privileges (setuid) |
| `--pids-limit=1024` | Caps processes and threads |
| `--memory=<n>g` (optional) | Caps memory; sorts of long recordings need several GB |

The server itself refuses requests that would make it allocate without bound, accepts at most
`--max-clients` clients (default 32) and 8 requests at once from each, closes idle connections
after 30 s, and makes a client prove its address before any work (QUIC Retry). Keep SELinux on
(see [GPU](#gpu)).

## Who can connect

Only clients the server has enrolled. The server keeps its own small certificate authority in
`/data/.dspk-auth` (mode 0700; its key 0600): enrolling a client issues it a certificate, and
the server accepts only those, checking at every connection that it is not revoked. Clients
trust the server through the same authority, so there is nothing else to hand out.

Enroll a client in the running container, then give it its file over a private channel (it
holds the client's private key):

```bash
# The identity goes to standard output, so no copy stays in the container or under /data
(umask 077; podman exec dspk-serve dspk clients enroll alice --out - > alice.dspk-identity.pem)
(umask 077; podman exec dspk-serve dspk clients enroll lab-pc --control --out - > lab-pc.dspk-identity.pem)
podman exec dspk-serve dspk clients list
podman exec dspk-serve dspk clients revoke alice      # refused from its next connection on
```

Never put identity files under `/data`: they hold the client's private key. The server never
lists or sends `*.dspk-identity.pem`, `*.key` or `*.key.pem` files anyway.

- Read only unless `--control`; control will allow configuring and running jobs.
- Revoking and enrolling take effect without restarting the server.
- A file is never overwritten, and a name already enrolled must be revoked first.
- Certificates are valid for about two years; enroll again after.

The client connects with its file:

```bash
dspk receive <server-ip>:50051 --identity alice.dspk-identity.pem --server-name <name> \
  --open mouse1/recording.ap.bin
```

`--server-name` must be one of the server's names: `localhost`, or a `--name` given to `serve`
(repeat it for several; the service file passes the host name).

- `--allow-anonymous` on `serve` also lets in clients without a certificate, read only. Anyone
  who reaches the port can then read the recording; use it only on a network you trust.
- A certificate of your own (from your organisation's CA): pass `--certificate chain.pem` and
  `--private-key key.pem` to `serve`. Clients still need an enrollment, and also
  `--certificate` with your CA.
- Back up `/data/.dspk-auth`. Losing it means enrolling every client again, and anyone who
  copies it can enroll clients.

## As a service (starts at boot)

```bash
mkdir -p ~/.config/containers/systemd ~/dspk-data
cp deploy/dspk-serve.container ~/.config/containers/systemd/
# clients open recordings under ~/dspk-data; to serve one to every client, add its path to Exec=
systemctl --user daemon-reload
systemctl --user start dspk-serve
loginctl enable-linger "$USER"   # keeps it running after you log out
journalctl --user -u dspk-serve  # logs
```

`%H` in the unit is this machine's host name, so the certificate carries it.

## GPU

Check what the container sees:

```bash
podman run --rm --device nvidia.com/gpu=all --security-opt=label=disable dspk-serve:nvidia doctor
```

Every runtime is listed as usable, or with the reason it isn't. The `Selected:` line is the one
commands use (`DSPK_RUNTIME`, else the first compiled in: wgpu); `doctor` warns when it has no
device and names one that has.

- **NVIDIA:** install the NVIDIA Container Toolkit and generate its CDI file once:
  `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`. Generate it again after a driver
  update. `--device nvidia.com/gpu=all` then passes the driver (CUDA and its Vulkan ICD) in.
- **AMD, Intel:** `--device /dev/dri` with the `mesa` target. Rootless containers also need
  `--group-add keep-groups` to keep your `render` and `video` groups.
- **No GPU:** the `cpu` target.

**SELinux (Fedora, RHEL).** SELinux gives every process and file a label. Podman starts each
container as `container_t` with a tag of its own, and the kernel lets it touch only files
labelled for that container (`:Z` on a volume relabels your folder for it). This holds even if
something inside the container breaks out of Podman's other limits. The GPU's device files
(`/dev/nvidia*`, `/dev/dri/*`) are not labelled for containers, so with the GPU passed in, CUDA
reports `CUDA_ERROR_NO_DEVICE` and Vulkan finds no adapter. Two ways around it:

| | `--security-opt=label=disable` (default here) | `setsebool -P container_use_devices on` |
|---|---|---|
| What it does | Turns the labels off for this one container | Keeps labelling on for every container and allows them to use device files in general: one rule is loosened for all containers instead of turning the labels off for one |
| Needs | Nothing (no sudo) | Root, once for the machine (`sudo setsebool -P container_use_devices on`) |
| Where | `podman run … --security-opt=label=disable`; `SecurityLabelDisable=true` in the Quadlet unit | Then drop `label=disable` from the commands and the unit |
| Matches | The developer container (`~/dev/devops`, `devop-cli`) | — |

Either way the other protections stay: non-root user, read-only root, no capabilities,
no-new-privileges, pids limit, and only enrolled clients.

`dspk serve` itself only streams; the GPU is for the pipelines and sorts the server runs.

### Tested (2026-11-06, RTX 2070, driver 615.71, rootless Podman 6.1, Fedora 45)

- `doctor` in the `nvidia` target: wgpu and CUDA usable.
- Kilosort4 on 30 s of a Neuropixels recording (384 channels) in the container, on CUDA:
  136 units, 63,014 spikes in 28.6 s.

```bash
podman run --rm --device nvidia.com/gpu=all --security-opt=label=disable --userns=keep-id:uid=10001,gid=10001 \
  -v ~/dspk-data:/data:Z dspk-serve:nvidia \
  sort --runtime cuda kilosort4 /data/recording.ap.bin --probe neuropixels1 \
  --output /data/recording.kilosort4.sorting.zarr --duration-sec 30
```

## Firewall

QUIC runs over UDP, and every connection is encrypted with TLS 1.3. The server lets in only
enrolled clients ([Who can connect](#who-can-connect)). Still open the port only to the network
that needs it (replace the address range with yours):

```bash
sudo firewall-cmd --permanent --add-rich-rule='rule family=ipv4 source address=192.168.1.0/24 port port=50051 protocol=udp accept'
sudo firewall-cmd --reload
```

Better still, reach the server through a VPN (WireGuard, Tailscale) and don't open the port to
the internet at all. Avoid `--add-port=50051/udp`: that opens it to every address.
