# %% [markdown]
# # dsp-kitchen (`dsp-base`): Filtering Methods Guide
# Interactive notebook demonstrating every filtering method in `dsp_kitchen.filter`:
# 1. **IIR Second-Order Section (SOS) Filters**:
#    - `BandpassFilter` / `bandpass_filter`
#    - `HighpassFilter` / `highpass_filter`
#    - `LowpassFilter` / `lowpass_filter`
#    - `BandstopFilter`
#    - `NotchFilter` / `notch_filter`
#    - `ChebyshevFilter` (type I)
#    - Causal (`direction="forward"`) vs. Zero-Phase (`direction="forward-backward"`) & settling halos
# 2. **Non-Linear Filters**:
#    - `MedianFilter` / `median_filter` (impulse & stimulation artifact removal; width 9 by default)
#    - `TeagerKaiser` / `teager_kaiser_filter` (instantaneous action-potential energy operator)
# 3. **Template Subtraction**:
#    - `TemplateFilter` / `subtract_template` (1D & multi-channel dynamic lag/amplitude subtraction)

# %% [1] Imports & Synthetic Multi-Band Neural Signal
import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_PLT = True
except ImportError:
    HAS_PLT = False

import dsp_kitchen as dk
from dsp_kitchen.pipeline import Pipeline
from dsp_kitchen.filter.iir import (
    BandpassFilter,
    BandstopFilter,
    ChebyshevFilter,
    HighpassFilter,
    LowpassFilter,
    NotchFilter,
    bandpass_filter,
    highpass_filter,
    lowpass_filter,
    notch_filter,
)
from dsp_kitchen.filter.non_linear import (
    MedianFilter,
    TeagerKaiser,
    median_filter,
    teager_kaiser_filter,
)
from dsp_kitchen.filter.template import TemplateFilter, subtract_template

fs = 30_000.0  # 30 kHz sampling rate
duration_s = 0.100  # 100 ms window (3,000 samples)
n_samples = int(fs * duration_s)
n_channels = 8
t = np.arange(n_samples, dtype=np.float32) / fs
t_ms = t * 1000.0

rng = np.random.default_rng(42)

# A synthetic composite signal on 8 channels, built in µV:
#   - Slow LFP oscillation (8 Hz theta + 40 Hz gamma)
#   - 60 Hz power-line interference
#   - Sharp extracellular action potentials (~1 ms biphasic waveforms)
#   - Background thermal noise
raw = np.zeros((n_channels, n_samples), dtype=np.float32)
spike_times = [450, 1100, 1850, 2450]
spike_kernel_t = np.linspace(-0.001, 0.0015, 75, dtype=np.float32)
spike_wave = -140.0 * np.exp(-((spike_kernel_t / 0.00022) ** 2)) + 45.0 * np.exp(
    -(((spike_kernel_t - 0.00045) / 0.00035) ** 2)
)

for ch in range(n_channels):
    lfp = 80.0 * np.sin(2.0 * np.pi * 8.0 * t + 0.2 * ch) + 25.0 * np.sin(
        2.0 * np.pi * 40.0 * t
    )
    hum_60hz = 55.0 * np.sin(2.0 * np.pi * 60.0 * t)
    noise = rng.normal(0.0, 8.0, size=n_samples).astype(np.float32)
    raw[ch] = lfp + hum_60hz + noise
    for st in spike_times:
        scale = max(0.2, 1.0 - 0.15 * ch)
        raw[ch, st : st + len(spike_wave)] += scale * spike_wave

print(f"dsp-kitchen version: {dk.__version__}")
print(f"Composite signal shape: {raw.shape} (channels, samples) @ {fs:.0f} Hz")

# %% [2] Direct IIR Filtering Functions (Bandpass, Highpass, Lowpass, Notch)
# Extract the AP spike band (300 - 6000 Hz), LFP band (< 300 Hz), Highpass (> 300 Hz), and 60 Hz Notch
ap_band = bandpass_filter(raw, 300.0, 6000.0, fs=fs, order=5)
lfp_band = lowpass_filter(raw, 300.0, fs=fs, order=4)
hp_band = highpass_filter(raw, 300.0, fs=fs, order=4)
notched = notch_filter(raw, 60.0, 30.0, fs=fs)
# Chebyshev type I: steeper transition for the same order, with pass-band ripple
cheby_band = ChebyshevFilter(4, 0.5, "bandpass", 300.0, 6000.0)

print("[IIR Functional API Results on Channel 0]")
print(f"  Raw RMS:             {np.sqrt(np.mean(raw[0] ** 2)):6.2f} uV")
print(f"  60 Hz Notched RMS:   {np.sqrt(np.mean(notched[0] ** 2)):6.2f} uV")
print(f"  LFP (<300 Hz) RMS:   {np.sqrt(np.mean(lfp_band[0] ** 2)):6.2f} uV")
print(f"  AP (300-6000Hz) RMS: {np.sqrt(np.mean(ap_band[0] ** 2)):6.2f} uV")

# %% [3] Causal ("forward") vs. Zero-Phase ("forward-backward") & Settling Halos
# SpikeInterface-compatible direction control:
# - "forward-backward" (sosfiltfilt): zero phase distortion, ideal for spike waveform alignment
# - "forward" (sosfilt): causal single-pass, ideal for closed-loop low-latency BMI
pipe_zero_phase = Pipeline(
    [
        BandpassFilter(
            low_hz=300.0, high_hz=6000.0, order=5, direction="forward-backward"
        ),
        BandstopFilter(
            low_hz=58.0, high_hz=62.0, order=2, direction="forward-backward"
        ),
    ]
)
pipe_causal = Pipeline(
    [
        BandpassFilter(low_hz=300.0, high_hz=6000.0, order=5, direction="forward"),
    ]
)

out_zero_phase = pipe_zero_phase.run(raw, fs=fs)
out_causal = pipe_causal.run(raw, fs=fs)

halo_zp = pipe_zero_phase.settling(fs=fs)
halo_causal = pipe_causal.settling(fs=fs)
cheby_out = Pipeline([cheby_band]).run(raw, fs=fs)
print(f"Chebyshev I band-pass RMS (ch 0): {np.sqrt(np.mean(cheby_out[0] ** 2)):6.2f} uV")
print(f"Zero-phase pipeline settling halo (left, right): {halo_zp} samples")
print(f"Causal pipeline settling halo (left, right):     {halo_causal} samples")

# Measure trough timing at the first spike on Channel 0
win = slice(spike_times[0], spike_times[0] + 75)
trough_zp = int(np.argmin(out_zero_phase[0, win]))
trough_causal = int(np.argmin(out_causal[0, win]))
print(
    f"Spike trough index — Zero-phase: sample {trough_zp}, Causal: sample {trough_causal} (phase lag = {trough_causal - trough_zp} samples)"
)

# %% [4] Non-Linear Filters: Running Median & Teager-Kaiser Energy Operator (TKEO)
# Inject single-sample stimulation / ADC glitch spikes to test MedianFilter
glitchy = ap_band.copy()
glitch_indices = [300, 900, 1500, 2100]
for idx in glitch_indices:
    glitchy[:, idx] += 450.0  # Huge 1-sample impulse artifact

despiked = median_filter(glitchy, 9)
tkeo_energy = teager_kaiser_filter(ap_band)

print("\n[Non-Linear Filters]")
print(f"  Glitchy Peak Max:      {glitchy[0].max():7.2f} uV")
print(
    f"  After 9-point Median:  {despiked[0].max():7.2f} uV (impulse artifacts suppressed)"
)
print(
    f"  TKEO Energy Peak/Mean: {tkeo_energy[0].max() / (np.mean(np.abs(tkeo_energy[0])) + 1e-6):7.1f}x"
)

# %% [5] Template Subtraction Filter (1D and Multi-Channel)
# Build a multi-channel template [n_channels, 75] from the known spike waveform
template_2d = np.stack(
    [max(0.2, 1.0 - 0.15 * ch) * spike_wave for ch in range(n_channels)],
    axis=0,
).astype(np.float32)

# Subtract the template at the first 3 spike timestamps using dynamic lag & amplitude scaling
events_to_remove = [int(st + np.argmin(spike_wave)) for st in spike_times[:3]]
center_offset = int(np.argmin(spike_wave))

tmpl_filter = TemplateFilter(
    template=template_2d,
    center_offset=center_offset,
    max_lag=6,
    dynamic_scaling=True,
)
residual_2d = tmpl_filter.apply(ap_band, event_indices=events_to_remove)

# Also works via the direct function `subtract_template`
residual_1d = subtract_template(
    ap_band[0],
    spike_wave,
    event_indices=events_to_remove,
    center_offset=center_offset,
    max_lag=6,
    dynamic_scaling=True,
)

print("\n[Template Subtraction]")
print(f"  Filter: {tmpl_filter}")
print(
    f"  Pre-subtraction RMS around spike 0:  {np.sqrt(np.mean(ap_band[0, win] ** 2)):6.2f} uV"
)
print(
    f"  Post-subtraction RMS around spike 0: {np.sqrt(np.mean(residual_2d[0, win] ** 2)):6.2f} uV"
)

# %% [6] Plot Comparison Across Filtering Methods
if HAS_PLT:
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)

    axes[0].plot(
        t_ms, raw[0], color="#4c72b0", linewidth=1.0, label="Raw (LFP + 60Hz + Spikes)"
    )
    axes[0].plot(
        t_ms, lfp_band[0], color="#dd8452", linewidth=1.5, label="Lowpass LFP (<300 Hz)"
    )
    axes[0].set_title(
        "1. Raw Composite Signal vs. Extracted LFP Band", fontweight="bold"
    )
    axes[0].set_ylabel("μV")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        t_ms,
        out_zero_phase[0],
        color="#2ca02c",
        linewidth=1.1,
        label="Zero-Phase Bandpass (300-6000 Hz)",
    )
    axes[1].plot(
        t_ms,
        out_causal[0],
        color="#d62728",
        linewidth=0.9,
        alpha=0.75,
        label="Causal Bandpass (forward)",
    )
    axes[1].set_title(
        "2. IIR SOS Bandpass: Zero-Phase (forward-backward) vs. Causal (forward)",
        fontweight="bold",
    )
    axes[1].set_ylabel("μV")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        t_ms,
        tkeo_energy[0],
        color="#8172b3",
        linewidth=1.0,
        label="Teager-Kaiser Energy (TKEO)",
    )
    axes[2].set_title(
        "3. Non-Linear Teager-Kaiser Energy Operator (TKEO)", fontweight="bold"
    )
    axes[2].set_ylabel("Energy (μV²)")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(
        t_ms,
        ap_band[0],
        color="#999999",
        linewidth=1.0,
        alpha=0.7,
        label="Before Template Subtraction",
    )
    axes[3].plot(
        t_ms,
        residual_2d[0],
        color="#1f77b4",
        linewidth=1.2,
        label="After Template Subtraction (first 3 spikes removed)",
    )
    axes[3].set_title(
        "4. Dynamic-Scaling Multi-Channel Template Subtraction (4th spike left intact)",
        fontweight="bold",
    )
    axes[3].set_xlabel("Time (ms)")
    axes[3].set_ylabel("μV")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# %%
