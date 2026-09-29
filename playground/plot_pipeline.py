# %% [markdown]
# # dsp-kitchen: Step-by-Step Electrophysiology Processing & Visualization
# Real-time pipeline: Load zero-copy binary data -> Composable In-VRAM Pipeline -> Plot Comparison.

# %% [1] Imports and Modular Components
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import dsp_kitchen
from dsp_kitchen.pipeline import Pipeline
from dsp_kitchen.filter.iir import NotchFilter, BandpassFilter
from dsp_kitchen.spatial import CommonAverageReference
from dsp_kitchen.math import Scale

local_path = dsp_kitchen.get_local_path()
print(f"Base Workspace:      {local_path}")
print(f"dsp-kitchen Version: {dsp_kitchen.__version__}")

# %% [2] Load Data (Zero-Copy Mmap)
data_path = local_path / "playground" / "data" / "mock_signal_384ch.bin"
sample_rate = 30000.0  # 30 kHz Neuropixels acquisition
channels = 384

rec, raw_data = dsp_kitchen.load_recording(
    data_path,
    channels=channels,
    sample_rate=sample_rate,
)

print(f"Recording File:  {rec.path}")
print(f"Array Shape:     {raw_data.shape} (channels, samples)")
print(f"Data Type:       {raw_data.dtype}")
print(f"Mapped Size:     {rec.total_bytes / (1024 * 1024):.2f} MB")

# %% [3] Build and Execute In-VRAM GPU Pipeline
# Composable zero-copy pipeline: Scale -> Bandpass (300-6000Hz) -> 60 Hz Notch -> Spatial CAR
pipeline = Pipeline([
    Scale(alpha=0.195, beta=0.0),                     # ADC to uV conversion
    BandpassFilter(low_hz=300.0, high_hz=6000.0),     # Butterworth 4th-order AP band
    NotchFilter(freq_hz=60.0, q=30.0),                # 60 Hz power-line rejection
    CommonAverageReference(),                         # In-place spatial CAR
])

print("Pipeline configured:")
for idx, stage in enumerate(pipeline.stages):
    print(f"  Stage {idx + 1}: {stage}")

# Select a 50 ms slice (1,500 samples @ 30 kHz) across all 384 channels
num_samples_to_plot = 1500
test_chunk = np.ascontiguousarray(raw_data[:, :num_samples_to_plot], dtype=np.float32)

filtered_chunk = pipeline.run(test_chunk, fs=sample_rate)

print(f"\nExecution Results:")
print(f"  Input Shape:     {test_chunk.shape}")
print(f"  Processed Shape: {filtered_chunk.shape}")
print(f"  Raw Min / Max:   {test_chunk[0].min():.2f} / {test_chunk[0].max():.2f} uV")
print(f"  Filt Min / Max:  {filtered_chunk[0].min():.2f} / {filtered_chunk[0].max():.2f} uV")

# %% [4] Plot Original vs. Filtered Signal
ch = 0  # Channel to inspect
time_ms = np.arange(num_samples_to_plot) / sample_rate * 1000.0

plt.figure(figsize=(12, 7))

# Subplot 1: Raw Electrophysiology Signal (with line noise and spikes)
plt.subplot(2, 1, 1)
plt.plot(
    time_ms,
    test_chunk[ch],
    color="#1f77b4",
    linewidth=1.2,
    label=f"Raw Signal (Ch {ch})",
)
plt.title(
    f"Channel {ch}: Raw Electrophysiology Trace (384-ch, 30 kHz)",
    fontsize=13,
    fontweight="bold",
)
plt.ylabel("Voltage (μV)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper right")

# Subplot 2: Filtered Signal (Bandpass 300-6000 Hz + 60 Hz Notch + CAR)
plt.subplot(2, 1, 2)
plt.plot(
    time_ms,
    filtered_chunk[ch],
    color="#2ca02c",
    linewidth=1.2,
    label="Bandpass (300-6000Hz) + Notch (60Hz) + CAR",
)
plt.title(
    f"Channel {ch}: In-VRAM Filtered & Common Average Referenced",
    fontsize=13,
    fontweight="bold",
)
plt.xlabel("Time (ms)", fontsize=11)
plt.ylabel("Voltage (μV)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper right")

plt.tight_layout()
plt.show()

# %%
