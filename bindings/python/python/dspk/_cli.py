"""The ``dspk`` command of the Python package: the desktop app and the sorter program.

The full command line (recordings, pipelines, servers) is the Rust program ``dspk``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import stat
import sys
import tempfile
import urllib.request
from pathlib import Path

import dspk

_COMMANDS = ("sorters", "app", "autotune")
_PROGRAM = "dspk-cli-sorter"
_MANIFEST = "dspk-cli-sorter-manifest.json"
_RELEASES = "https://github.com/neuro-kitchen/dsp-kitchen/releases/download"
_TARGETS = {
    ("Linux", "x86_64"): "x86_64-unknown-linux-gnu",
    ("Windows", "AMD64"): "x86_64-pc-windows-msvc",
    ("Darwin", "arm64"): "aarch64-apple-darwin",
}


def _data_dir() -> Path:
    named = os.environ.get("DSPK_DATA_DIR", "").strip()
    if named:
        return Path(named)
    system = platform.system()
    if system == "Windows":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "dspk"
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "dspk"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "dspk"


def _program_path(version: str) -> Path:
    return _data_dir() / "sorters" / version / (_PROGRAM + (".exe" if platform.system() == "Windows" else ""))


def _release_url(version: str) -> str:
    base = os.environ.get("DSPK_CLI_SORTER_RELEASE_URL", "").strip() or _RELEASES
    return f"{base.rstrip('/')}/v{version}"


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:  # noqa: S310 (https, or a URL the user set)
        return response.read()


def _install_sorters(version: str, force: bool) -> None:
    dest = _program_path(version)
    if dest.is_file() and not force:
        print(f"{_PROGRAM} {version} is installed: {dest}")
        return
    target = _TARGETS.get((platform.system(), platform.machine()))
    if target is None:
        sys.exit(f"no {_PROGRAM} build is released for {platform.system()} {platform.machine()}")
    base = _release_url(version)
    print(f"downloading {_PROGRAM} {version} from {base} ...")
    manifest = json.loads(_fetch(f"{base}/{_MANIFEST}"))
    if manifest.get("version") != version:
        sys.exit(f"{base}/{_MANIFEST} describes version {manifest.get('version')}, not {version}")
    asset = manifest["assets"].get(target)
    if asset is None:
        sys.exit(f"release v{version} has no {_PROGRAM} build for {target}")
    data = _fetch(f"{base}/{Path(asset['file']).name}")
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != asset["size"] or digest.lower() != asset["sha256"].lower():
        sys.exit(f"the download does not match the release manifest (SHA-256 {digest}, {len(data)} bytes)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as out:
        out.write(data)
    os.chmod(out.name, os.stat(out.name).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    os.replace(out.name, dest)
    print(f"installed {dest}")
    print(f"{_PROGRAM} is GPL-3.0 (it contains Kilosort4 and EMUsort); `{_PROGRAM} licenses` prints the texts")


def main(argv: list[str] | None = None) -> None:
    words = sys.argv[1:] if argv is None else argv
    if words and not words[0].startswith("-") and words[0] not in _COMMANDS:
        sys.exit(f"`dspk {words[0]}` is in the full command line: cargo install --path app/dspk-cli, from a clone of dsp-kitchen")
    parser = argparse.ArgumentParser(prog="dspk", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    sorters = commands.add_parser("sorters", help=f"the {_PROGRAM} program (GPL-3.0)").add_subparsers(dest="action", required=True)
    install = sorters.add_parser("install", help="download and verify it for this version and platform")
    install.add_argument("--force", action="store_true", help="download again even when it is installed")
    sorters.add_parser("status", help="where it is installed")
    tune = commands.add_parser("autotune", help="tune this machine's devices ahead of time (a minute per GPU, longer on the CPU)")
    tune.add_argument("--runtime", help="one runtime; by default every runtime that can open a device")
    app = commands.add_parser("app", help="open the desktop app")
    app.add_argument("path", nargs="?", help="a recording to open")
    args = parser.parse_args(argv)
    if args.command == "app":
        try:
            dspk.app.open(args.path)
        except RuntimeError as error:
            sys.exit(str(error))
    elif args.command == "autotune":
        for runtime, result in dspk.runtime.autotune(runtime=args.runtime).items():
            print(f"{runtime}: {result['sizes']:.0f} sizes in {result['seconds']:.0f} s")
    elif args.action == "install":
        _install_sorters(dspk.__version__, args.force)
    else:
        path = _program_path(dspk.__version__)
        print(f"{path} ({'installed' if path.is_file() else 'not installed: dspk sorters install'})")
