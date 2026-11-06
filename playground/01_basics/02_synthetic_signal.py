# %% [markdown]
# # 01 · Basics — 2. The synthetic recording
# Every chapter up to sorting uses a synthetic recording from `plating.signals`: its content is
# known, so each operation's effect can be seen *and checked*. It is the sum of five components,
# all kept separately:
#
# | component | what it is | which chapter removes it |
# |---|---|---|
# | `lfp` | slow oscillations (4 and 9 Hz) | low-pass keeps it, high-pass removes it (02) |
# | `hum` | 60 Hz mains and its third harmonic, the same on every channel | notch (02), referencing (03) |
# | `common` | a slow artefact shared by all channels | referencing (03) |
# | `spikes` | three units firing, each on a few neighbouring channels | what the filters keep |
# | `noise` | white noise, independent per channel | — |

# %%
from pathlib import Path

import plating
from plating import figures, signals

FIG = Path(__file__).parent / "figures"
s = signals.synthetic()  # 8 channels × 2 s at 30 kHz, seed 0: the same recording every time
print(f"{s.channels} channels × {s.samples:,} samples at {s.sample_rate:,.0f} Hz, {s.spike_samples.size} spikes from {len(s.templates)} units")

# %% Figure: the recording, then each component on the channel where the largest unit peaks
p = plating.theme.palette()
names = list(s.components)
fig, axes = plating.figure("What the synthetic recording is made of", rows=len(names) + 1, size=(11.0, 9.0), sharex=True,
                           height_ratios=[3] + [1] * len(names))
figures.traces(axes[0], s.data, s.sample_rate, duration=0.5)
axes[0].set_title("the recording (8 channels): the sum of the components below")
axes[0].set_xlabel("")
t = s.time[: int(0.5 * s.sample_rate)]
ch = s.loudest_channel
for ax, name, color in zip(axes[1:], names, p.series):
    ax.plot(t, s.components[name][ch, : t.size], color=color, linewidth=1.0)
    ax.set_title(f"{name} (channel {ch})", fontsize=9.5)
    ax.set_ylabel("µV")
    ax.set_xlabel("")
axes[-1].set_xlabel("time (s)")
plating.save(fig, FIG / "02_synthetic_signal.png", note="plating.signals.synthetic(): 8 channels × 2 s at 30 kHz, seed 0; every component in µV.")
