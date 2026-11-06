# %% [markdown]
# # 02 · Filtering — 5. Non-linear: median and Teager–Kaiser energy
# Not every useful filter is linear. A running **median** ignores short events: wide enough, it
# removes the spikes and keeps the background. The **Teager–Kaiser** energy
# (`x[t]² − x[t−1]·x[t+1]`) does the opposite: it grows where the signal is both large and fast,
# so spikes stand out of the noise; spike detectors use it.

# %%
from pathlib import Path

import numpy as np

import dspk
import plating
from plating import signals

FIG = Path(__file__).parent / "figures"
s = signals.synthetic()
WIDTH = dspk.ops.filter.MAX_MEDIAN_WIDTH  # 31 samples, ~1 ms at 30 kHz
spikes_band = dspk.ops.filter.iir(s.data, s.sample_rate, "highpass", 300.0)
median = dspk.ops.filter.median(spikes_band, WIDTH)
energy = dspk.ops.filter.teager_kaiser(spikes_band)

ch = s.loudest_channel
on_ch = s.spike_samples[np.isin(s.spike_units, np.flatnonzero(s.unit_channels == ch))]
print(f"channel {ch}: {on_ch.size} spikes of units peaking there")

# %% Figure: the spike band, its running median, its energy (spike times marked)
p = plating.theme.palette()
win = slice(0, int(0.5 * s.sample_rate))
t = s.time[win]
marks = on_ch[on_ch < win.stop] / s.sample_rate
fig, axes = plating.figure("A median removes spikes; Teager–Kaiser energy finds them", rows=3, size=(11.0, 7.0), sharex=True)
axes[0].plot(t, spikes_band[ch, win], color=p.series[0], linewidth=0.8)
axes[0].set_title(f"high-pass 300 Hz, channel {ch} (ticks: the spikes put in)")
axes[1].plot(t, spikes_band[ch, win], color=p.muted, linewidth=0.8, label="high-pass")
axes[1].plot(t, median[ch, win], color=p.series[0], linewidth=1.0, label=f"running median, {WIDTH} samples")
axes[1].set_title("the median follows the background and skips the spikes")
axes[1].legend(loc="lower right", ncols=2)
axes[2].plot(t, energy[ch, win], color=p.series[2], linewidth=0.8)
axes[2].set_title("Teager–Kaiser energy: the spikes stand out")
for ax in axes:
    for m in marks:
        ax.axvline(m, ymin=0.94, ymax=1.0, color=p.ink, linewidth=1.2)
axes[0].set_ylabel("µV")
axes[1].set_ylabel("µV")
axes[2].set_ylabel("µV²")
axes[-1].set_xlabel("time (s)")
axes[-1].set_xlim(t[0], t[-1])
plating.save(fig, FIG / "05_median_and_energy.png", note=f"dspk.ops.filter.median (width {WIDTH}) and teager_kaiser, after a 300 Hz high-pass.")
