# %% [markdown]
# # 02 · Filtering — 4. Designing a FIR filter
# A FIR filter is a list of *taps*: each output is their weighted sum of the last inputs. More taps
# give a sharper edge between what passes and what is stopped, at the cost of time.
# `dspk.ops.filter.firwin` designs them (a windowed sinc, as scipy's `firwin`);
# `dspk.ops.filter.fir` runs them on the device (`centered=True`: no delay).

# %%
from pathlib import Path

import numpy as np

import dspk
import plating
from plating import figures, signals

FIG = Path(__file__).parent / "figures"
s = signals.synthetic()
CUTOFF = 1_000.0
designs = {n: dspk.ops.filter.firwin(n, CUTOFF, sample_rate=s.sample_rate) for n in (31, 101, 301)}

# Frequency response of each design (the Fourier transform of its taps)
nfft = 1 << 16
freqs = np.fft.rfftfreq(nfft, 1 / s.sample_rate)
responses = {n: np.abs(np.fft.rfft(taps, nfft)) for n, taps in designs.items()}
for n, r in responses.items():
    print(f"{n:4d} taps: gain at {2 * CUTOFF:.0f} Hz {20 * np.log10(r[np.searchsorted(freqs, 2 * CUTOFF)]):6.1f} dB")

# Run the 301-tap design centred (odd length: no delay)
out = dspk.ops.filter.fir(s.data, designs[301], centered=True)

# %% Figure: responses (left), the 301-tap filter applied (right)
p = plating.theme.palette()
fig, (left, right) = plating.figure(f"FIR low-pass at {CUTOFF:,.0f} Hz: more taps, a sharper edge", cols=2)
for (n, r), color in zip(responses.items(), p.series):
    keep = (freqs > 20) & (freqs < 10_000)
    figures.response(left, freqs[keep], r[keep], label=f"{n} taps", color=color)
left.axvline(CUTOFF, color=p.ink2, linewidth=0.8, linestyle=(0, (4, 3)))
left.set_ylim(-90, 5)
left.set_title("magnitude response")
left.legend(loc="lower left")
figures.spectrum(right, s.data, s.sample_rate, label="recorded", color=p.muted)
figures.spectrum(right, out, s.sample_rate, label="301 taps applied", color=p.series[2])
right.set_title("power spectrum, all channels")
right.legend(loc="upper right")
plating.save(fig, FIG / "04_fir_design.png", note="dspk.ops.filter.firwin (Hamming window) and dspk.ops.filter.fir(centered=True).")
