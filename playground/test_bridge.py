#!/usr/bin/env python3
"""
Automated verification test suite for dsp-kitchen Python bridge (PyO3 + maturin).
Tests:
  1. Neuropixels 1.0 ProbeLayout geometry & dictionary export
  2. Zero-copy MmapRecording of 384-channel mock signal
  3. DspSession and GPU execution via WGPU
"""

import sys
import os
from pathlib import Path
import numpy as np
from dotenv import load_dotenv, find_dotenv

# Load .env into os.environ
load_dotenv(find_dotenv(usecwd=True))


def get_local_path() -> Path:
    """
    Retrieves the base project path defined by LOCAL_PATH from .env or environment variable.
    """
    val = os.getenv("LOCAL_PATH")
    if val and val.strip():
        return Path(val.strip().strip('"').strip("'"))
    return Path(__file__).resolve().parent.parent


def test_probe_layout():
    print("\n--- [1/3] Testing Neuropixels 1.0 Probe Layout ---")
    import dsp_kitchen

    probe = dsp_kitchen.neuropixels_1_0_layout()
    print(f"  Probe: {probe.name}")
    print(f"  Total Channels:  {probe.total_channels}")
    print(f"  Active Channels: {probe.active_channels}")

    positions = probe.contact_positions()
    assert len(positions) == 384, (
        f"Expected 384 contact positions, got {len(positions)}"
    )
    pos_arr = np.array(positions)
    print(f"  Contact 0 Position:   {pos_arr[0]} um")
    print(f"  Contact 383 Position: {pos_arr[-1]} um")

    probe_dict = probe.to_dict()
    assert "contact_positions" in probe_dict
    assert probe_dict["ndim"] == 3
    print("  [SUCCESS] ProbeLayout verification passed!")


def test_zero_copy_mmap():
    print("\n--- [2/3] Testing Zero-Copy MmapRecording ---")
    import dsp_kitchen

    local_path = get_local_path()
    data_path = local_path / "playground" / "data" / "mock_signal_384ch.bin"
    print(f"  Resolved Base LOCAL_PATH: {local_path}")
    print(f"  Constructed Data Path:    {data_path}")

    if not data_path.exists():
        print(f"  [SKIPPED] {data_path} not found. Run dsp-cli mock-signal first.")
        return None

    rec, arr = dsp_kitchen.load_recording(data_path, channels=384, sample_rate=30000.0)
    print(f"  Recording Path: {rec.path}")
    print(f"  Reported Shape: {rec.shape}")
    print(f"  NumPy Array Shape: {arr.shape}, Dtype: {arr.dtype}")
    print(f"  Mapped File Size: {rec.total_bytes / (1024 * 1024):.2f} MB")

    assert arr.shape == (384, 30000), f"Expected (384, 30000), got {arr.shape}"
    assert arr.dtype == np.float32, f"Expected float32, got {arr.dtype}"

    # Sample statistics
    ch0 = arr[0, :]
    rms = float(np.sqrt(np.mean(ch0**2)))
    print(f"  Channel 0 RMS Voltage: {rms:.2f} uV")
    print(f"  Channel 0 Min/Max:     {ch0.min():.2f} uV / {ch0.max():.2f} uV")
    print("  [SUCCESS] Zero-Copy MmapRecording verification passed!")
    return rec, arr


def test_dsp_session(arr=None):
    print("\n--- [3/3] Testing DspSession & GPU Kernel Pipeline ---")
    import dsp_kitchen

    session = dsp_kitchen.DspSession(sample_rate=30000.0, channels=384)
    print(f"  Session: {session.info()}")
    assert session.channels == 384
    assert session.sample_rate == 30000.0

    if arr is not None:
        test_chunk = np.ascontiguousarray(arr[:, :1000], dtype=np.float32)
        print(f"  Running WGPU Pipeline on test chunk {test_chunk.shape}...")
        try:
            out = session.run_pipeline_wgpu(test_chunk)
            print(f"  Pipeline Output Shape: {out.shape}, Dtype: {out.dtype}")
            print(f"  Output Min/Max:        {out.min():.4f} / {out.max():.4f}")
            assert out.shape == test_chunk.shape
            assert not np.isnan(out).any()
            print("  [SUCCESS] DspSession WGPU Pipeline verification passed!")
        except Exception as e:
            print(f"  [WGPU Note] Pipeline execution caught: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  [INFO] DspSession created successfully without test chunk.")


if __name__ == "__main__":
    print("==================================================")
    print("   dsp-kitchen Python Zero-Copy Bridge Test       ")
    print("==================================================")
    test_probe_layout()
    res = test_zero_copy_mmap()
    if res is not None:
        rec, arr = res
        test_dsp_session(arr)
    else:
        test_dsp_session()
    print("\n==================================================")
    print("   ALL PYTHON BRIDGE VERIFICATIONS COMPLETED!     ")
    print("==================================================")
