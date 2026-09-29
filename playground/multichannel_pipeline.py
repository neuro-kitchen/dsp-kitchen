# %% [markdown]
# # dsp-kitchen: Multi-Channel Electrophysiology Pipeline & Scaling Evaluation
# Step-by-step interactive workflow:
# 1. Zero-copy binary loading via MmapRecording
# 2. Declarative In-VRAM GPU Pipeline: Scale -> Bandpass (300-6000 Hz) -> 60 Hz Notch -> Spatial CAR
# 3. 32-Channel vs. 384-Channel hardware throughput evaluation
# 4. Interactive comparison visualization (Original vs. Filtered)

# %% [1] Imports and Modular Components
import os
from pathlib import Path
import numpy as np
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import dsp_kitchen
from dsp_kitchen.pipeline import Pipeline
from dsp_kitchen.filter.iir import NotchFilter, BandpassFilter
from dsp_kitchen.spatial import CommonAverageReference
from dsp_kitchen.math import Scale

local_path = dsp_kitchen.get_local_path()
print(f"Workspace Base:      {local_path}")
print(f"dsp-kitchen Version: {dsp_kitchen.__version__}")

# %% [2] Load Electrophysiology Recording (Zero-Copy Mmap)
data_path = local_path / "playground" / "data" / "mock_signal_384ch.bin"
sample_rate = 30000.0  # 30 kHz Neuropixels AP band
total_channels = 384

rec, raw_data = dsp_kitchen.load_recording(
    data_path,
    channels=total_channels,
    sample_rate=sample_rate,
)

print(f"Recording Path:      {rec.path}")
print(f"Array Shape:         {raw_data.shape} (channels, samples)")
print(f"Mapped File Size:    {rec.total_bytes / (1024 * 1024):.2f} MB")
print(f"Channel 0 Raw RMS:   {float(np.sqrt(np.mean(raw_data[0]**2))):.2f} uV")

# %% [3] Define Composable In-VRAM GPU Pipeline
# Composable zero-host-PCIe-transfer pipeline using ping-pong VRAM buffers
pipeline = Pipeline([
    Scale(alpha=0.195, beta=0.0),                     # 1. ADC Bit-to-uV conversion
    BandpassFilter(low_hz=300.0, high_hz=6000.0),     # 2. 4th-order Butterworth AP bandpass
    NotchFilter(freq_hz=60.0, q=30.0),                # 3. Narrowband 60 Hz hum rejection
    CommonAverageReference(),                         # 4. Full-probe spatial referencing
])

print("Constructed In-VRAM Pipeline:")
for idx, stage in enumerate(pipeline.stages):
    print(f"  Stage {idx + 1}: {stage}")

# %% [4] Multi-Channel Processing: 384-Channel Probe
# Select a 50 ms test chunk (1,500 samples @ 30 kHz) across all 384 channels
num_samples = 1500
chunk_384 = np.ascontiguousarray(raw_data[:, :num_samples], dtype=np.float32)

filtered_384 = pipeline.run(chunk_384, fs=sample_rate)

print(f"\n[384-Channel Processing]")
print(f"  Input Matrix:      {chunk_384.shape}")
print(f"  Filtered Matrix:   {filtered_384.shape}")
print(f"  Raw Ch0 Min/Max:   {chunk_384[0].min():.2f} / {chunk_384[0].max():.2f} uV")
print(f"  Filt Ch0 Min/Max:  {filtered_384[0].min():.2f} / {filtered_384[0].max():.2f} uV")

# %% [5] Multi-Channel Processing: 32-Channel Sub-Array (Hardware Scaling Evaluation)
# Tests dynamically tuned SIMT workgroup launch geometry for small channel counts
chunk_32 = np.ascontiguousarray(raw_data[:32, :num_samples], dtype=np.float32)

pipeline_32 = Pipeline([
    Scale(alpha=0.195, beta=0.0),
    BandpassFilter(low_hz=300.0, high_hz=6000.0),
    NotchFilter(freq_hz=60.0, q=30.0),
    CommonAverageReference(),
])

filtered_32 = pipeline_32.run(chunk_32, fs=sample_rate)

print(f"\n[32-Channel Scaling Evaluation]")
print(f"  Input Matrix:      {chunk_32.shape}")
print(f"  Filtered Matrix:   {filtered_32.shape}")
print(f"  Raw Ch0 Min/Max:   {chunk_32[0].min():.2f} / {chunk_32[0].max():.2f} uV")
print(f"  Filt Ch0 Min/Max:  {filtered_32[0].min():.2f} / {filtered_32[0].max():.2f} uV")

# %% [6] Plot Original vs. Filtered Signal (384-Channel Probe vs. 32-Channel Sub-Array)
time_ms = np.arange(num_samples) / sample_rate * 1000.0
inspect_ch = 0

if HAS_MATPLOTLIB:
    plt.figure(figsize=(13, 8))

    # Subplot 1: Raw Electrophysiology Trace (Channel 0)
    plt.subplot(3, 1, 1)
    plt.plot(time_ms, chunk_384[inspect_ch], color="#1f77b4", linewidth=1.2, label=f"Raw Channel {inspect_ch}")
    plt.title(f"Channel {inspect_ch}: Raw Electrophysiology Recording (60 Hz Hum + AP Spikes)", fontsize=12, fontweight="bold")
    plt.ylabel("Raw (ADC / μV)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")

    # Subplot 2: 384-Channel Pipeline (Bandpassed + Notched + 384ch CAR)
    plt.subplot(3, 1, 2)
    plt.plot(time_ms, filtered_384[inspect_ch], color="#2ca02c", linewidth=1.2, label="384-ch Pipeline Output")
    plt.title(f"Channel {inspect_ch}: 384-Channel In-VRAM Pipeline (Bandpass 300-6000Hz + Notch 60Hz + CAR)", fontsize=12, fontweight="bold")
    plt.ylabel("Filtered (μV)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")

    # Subplot 3: 32-Channel Sub-Array Pipeline Output
    plt.subplot(3, 1, 3)
    plt.plot(time_ms, filtered_32[inspect_ch], color="#d62728", linewidth=1.2, label="32-ch Sub-Array Output")
    plt.title(f"Channel {inspect_ch}: 32-Channel Pipeline Output (Tuned Launch Geometry)", fontsize=12, fontweight="bold")
    plt.xlabel("Time (ms)", fontsize=11)
    plt.ylabel("Filtered (μV)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()
else:
    print("\n[Plotting skipped: matplotlib not installed in current environment]")

# %%
