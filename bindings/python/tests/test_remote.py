"""dspk.remote against a real ``dspk serve`` on this machine (the CLI built in ``target/``; skipped
when it is not): enrol a client, serve a data root, then info, files, a recording read on the
server equal to the local file, a fetch, and the refusals."""

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

import dspk

ROOT = Path(__file__).resolve().parents[3]


def cli() -> Path | None:
    if os.environ.get("DSPK_CLI"):
        return Path(os.environ["DSPK_CLI"])
    for profile in ("validate", "release", "debug"):
        path = ROOT / "target" / profile / "dspk"
        if path.is_file():
            return path
    found = shutil.which("dspk")
    # The Python package's own small `dspk` command has no servers: only the full program counts
    if found and subprocess.run([found, "clients", "--help"], capture_output=True).returncode != 0:
        found = None
    return Path(found) if found else None


CLI = cli()
pytestmark = pytest.mark.skipif(CLI is None, reason="the dspk CLI is not built (cargo build -p dspk-cli)")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    base = tmp_path_factory.mktemp("remote")
    auth, data = base / "auth", base / "data"
    data.mkdir()
    env = {**os.environ, "DSPK_RUNTIME": "cpu"}
    identity = base / "tester.dspk-identity.pem"
    subprocess.run([CLI, "clients", "--auth-dir", auth, "enroll", "tester", "--control", "--out", identity], check=True, capture_output=True, env=env)
    x = np.random.default_rng(0).normal(0, 50, (4, 20_000)).astype(np.float32)
    dspk.io.write_zarr(x, data / "rec.zarr", 10_000.0)
    address = f"127.0.0.1:{free_port()}"
    proc = subprocess.Popen([CLI, "serve", "--data", data, "--bind", address, "--auth-dir", auth], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    try:
        session = None
        for _ in range(100):
            try:
                session = dspk.remote.Session(address, identity)
                break
            except RuntimeError:
                if proc.poll() is not None:
                    raise RuntimeError(f"dspk serve exited: {proc.stdout.read().decode(errors='replace')}") from None
                time.sleep(0.1)
        assert session is not None, "dspk serve did not answer"
        yield session, x, identity, address, base
        session.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_info_and_files(server):
    session, *_ = server
    info = session.info()
    assert info["peer_name"] == "tester" and info["permission"] == "control" and info["has_files"]
    names = {f["name"]: f for f in session.files()}
    assert "rec.zarr" in names and names["rec.zarr"]["directory"]


def test_a_recording_read_on_the_server_equals_the_file(server):
    session, x, *_ = server
    rec = session.open("rec.zarr")
    assert rec.read(start=1_000, stop=3_000).shape == (4, 2_000)
    np.testing.assert_allclose(rec.read(channels=[2, 0], start=5_000, stop=5_500), x[[2, 0], 5_000:5_500], rtol=1e-6)


def test_fetch_writes_a_new_path(server):
    session, x, _, _, base = server
    dest = base / "fetched.zarr"
    files, size = session.fetch("rec.zarr", dest)
    assert files > 0 and size > 0
    np.testing.assert_allclose(dspk.io.open(dest).read(), x)
    with pytest.raises(ValueError, match="exists"):
        session.fetch("rec.zarr", dest)


def test_status_and_refusals(server):
    session, _, identity, address, _ = server
    assert session.status()["state"] in ("stopped", "error")
    with pytest.raises(RuntimeError):
        session.open("../outside.zarr")
    with pytest.raises(ValueError, match="identity file"):
        dspk.remote.Session(address, identity.with_name("missing.pem"))
    with pytest.raises(ValueError, match="IP and port"):
        dspk.remote.Session("nowhere", identity)


def test_a_closed_session_says_so(server):
    _, _, identity, address, _ = server
    with dspk.remote.Session(address, identity) as s:
        assert s.info()["peer_name"] == "tester"
    with pytest.raises(RuntimeError, match="closed"):
        s.status()
