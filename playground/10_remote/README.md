# 10 · Remote: a GPU server, a laptop

`dspk serve` shares a folder of recordings and runs sorts on its GPU; clients connect over QUIC
with a certificate the server issues them (mutual TLS). This chapter is a walkthrough of
commands for now; a Python client is an open question (`.tasks/27`).

On the server (or in the container, `deploy/`):

```bash
dspk serve --data /srv/recordings --runtime cuda --bind 0.0.0.0:50051 --name gpu-box   # UDP 50051
dspk clients enroll laptop --control --out laptop.dspk-identity.pem                      # once per client
```

(`192.168.1.20` stands for the server's address; `--server-name` is a name its certificate carries, from `--name`.)

Give `laptop.dspk-identity.pem` to the client over a private channel (it holds a private key; it
never belongs under the data root, and the server never serves such files). On the laptop:

```bash
export DSPK_IDENTITY=laptop.dspk-identity.pem
dspk remote 192.168.1.20:50051 --server-name gpu-box info            # who it is, who you are, its runtime
dspk remote 192.168.1.20:50051 --server-name gpu-box files            # the recordings it shares
dspk remote 192.168.1.20:50051 --server-name gpu-box sort session1/rec.bin --sorter kilosort4 --wait
dspk remote 192.168.1.20:50051 --server-name gpu-box fetch session1/rec.kilosort4.sorting.zarr
```

The app does the same from Settings (servers) and File → Open from server: a remote recording
opens like a local one and sorts on the server.
