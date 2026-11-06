"""A synthetic recording whose content is known, so every chapter can show (and check) what an
operation did: slow oscillations, a mains hum, a common-mode artefact shared by all channels,
spikes from a few units on neighbouring channels, and noise. Each component is kept separately.

Amplitudes are in microvolts; the defaults resemble an extracellular recording at 30 kHz.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Defaults (visible in :func:`synthetic`'s signature too).
DEFAULT_CHANNELS = 8
DEFAULT_SECONDS = 2.0
DEFAULT_SAMPLE_RATE = 30_000.0
DEFAULT_SEED = 0


@dataclass
class Synthetic:
    """A synthetic recording: ``data`` ``(channels, samples)`` float32 µV, the sum of
    ``components`` (each the same shape), and the spikes' ground truth."""

    data: np.ndarray
    sample_rate: float
    components: dict[str, np.ndarray]
    #: Sample index of every spike, its unit, and the unit's peak channel.
    spike_samples: np.ndarray
    spike_units: np.ndarray
    unit_channels: np.ndarray
    #: Each unit's waveform (samples,), µV at its peak channel.
    templates: np.ndarray
    params: dict = field(default_factory=dict)

    @property
    def channels(self) -> int:
        return self.data.shape[0]

    @property
    def samples(self) -> int:
        return self.data.shape[1]

    @property
    def loudest_channel(self) -> int:
        """The channel where the largest unit peaks (where spikes show best)."""
        if not len(self.templates):
            return 0
        return int(self.unit_channels[int(np.argmax(np.abs(self.templates).max(axis=1)))])

    @property
    def time(self) -> np.ndarray:
        """Seconds of every sample."""
        return np.arange(self.samples) / self.sample_rate

    def without(self, *names: str) -> np.ndarray:
        """The recording minus some components (what an ideal operation would leave)."""
        out = self.data.astype(np.float64)
        for n in names:
            out -= self.components[n]
        return out.astype(np.float32)


def spike_waveform(sample_rate: float, *, width_ms: float = 1.2, amplitude_uv: float = -120.0) -> np.ndarray:
    """A biphasic extracellular spike (a sharp trough, then a slower positive bump)."""
    n = int(round(width_ms * 1e-3 * sample_rate)) | 1
    t = np.linspace(-0.4, 1.0, n) * width_ms
    trough = np.exp(-((t / (0.12 * width_ms)) ** 2))
    bump = 0.35 * np.exp(-(((t - 0.35 * width_ms) / (0.25 * width_ms)) ** 2))
    w = trough - bump
    return (amplitude_uv * w / np.max(np.abs(w))).astype(np.float32)


def synthetic(
    channels: int = DEFAULT_CHANNELS,
    seconds: float = DEFAULT_SECONDS,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    *,
    seed: int = DEFAULT_SEED,
    lfp_hz: tuple[float, ...] = (4.0, 9.0),
    lfp_uv: float = 80.0,
    hum_hz: float = 60.0,
    hum_uv: float = 40.0,
    common_uv: float = 60.0,
    noise_uv: float = 8.0,
    units: int = 3,
    rate_hz: float = 12.0,
    spike_uv: tuple[float, float] = (-90.0, -160.0),
    spread_channels: float = 1.2,
) -> Synthetic:
    """A recording of ``channels`` × ``seconds`` at ``sample_rate`` Hz (all amplitudes in µV):

    - ``lfp``: sines at ``lfp_hz``, phase-shifted across channels (``lfp_uv`` each);
    - ``hum``: mains at ``hum_hz`` and its third harmonic, identical on every channel;
    - ``common``: a slow common-mode artefact (movement-like steps), identical on every channel;
    - ``spikes``: ``units`` units firing at ``rate_hz`` (Poisson, 2 ms refractory), each peaking on
      one channel and fading over ``spread_channels`` channels, amplitudes in ``spike_uv``;
    - ``noise``: white Gaussian, ``noise_uv`` RMS, independent per channel.
    """
    rng = np.random.default_rng(seed)
    n = int(round(seconds * sample_rate))
    t = np.arange(n) / sample_rate
    ch = np.arange(channels)[:, None]

    lfp = sum(lfp_uv * np.sin(2 * np.pi * f * t + 0.4 * ch + k) for k, f in enumerate(lfp_hz))
    hum = np.broadcast_to(hum_uv * (np.sin(2 * np.pi * hum_hz * t) + 0.3 * np.sin(2 * np.pi * 3 * hum_hz * t)), (channels, n))
    # A random walk sampled every millisecond, smoothed over 100 ms: a slow, wandering artefact
    walk = np.cumsum(rng.normal(0, 1, max(2, int(seconds * 1000))))
    kernel = np.hanning(101)
    walk = np.convolve(np.pad(walk, 50, mode="edge"), kernel / kernel.sum(), mode="valid")
    common_1d = np.interp(t, np.linspace(0, seconds, walk.size), walk)
    common_1d = common_uv * (common_1d - common_1d.mean()) / (np.std(common_1d) or 1.0)
    common = np.broadcast_to(common_1d, (channels, n))
    noise = rng.normal(0, noise_uv, (channels, n))

    spikes = np.zeros((channels, n))
    unit_channels = rng.choice(channels, size=units, replace=units > channels)
    amplitudes = np.linspace(*spike_uv, units) if units > 1 else np.array([spike_uv[0]])
    templates = np.stack([spike_waveform(sample_rate, amplitude_uv=a) for a in amplitudes]) if units else np.zeros((0, 1))
    half = templates.shape[1] // 2
    refractory = int(0.002 * sample_rate)
    all_samples, all_units = [], []
    for u in range(units):
        gaps = rng.exponential(sample_rate / rate_hz, int(rate_hz * seconds * 2) + 4).astype(int) + refractory
        times = np.cumsum(gaps)
        times = times[(times > half) & (times < n - half - 1)]
        weights = np.exp(-(((np.arange(channels) - unit_channels[u]) / spread_channels) ** 2))
        for s in times:
            spikes[:, s - half : s - half + templates.shape[1]] += weights[:, None] * templates[u][None, :]
        all_samples.append(times)
        all_units.append(np.full(times.size, u))
    order = np.argsort(np.concatenate(all_samples)) if units else np.array([], dtype=int)
    spike_samples = np.concatenate(all_samples)[order] if units else np.array([], dtype=int)
    spike_units = np.concatenate(all_units)[order] if units else np.array([], dtype=int)

    components = {name: np.ascontiguousarray(c, dtype=np.float32) for name, c in
                  {"lfp": lfp, "hum": hum, "common": common, "spikes": spikes, "noise": noise}.items()}
    data = sum(c.astype(np.float64) for c in components.values()).astype(np.float32)
    params = dict(channels=channels, seconds=seconds, sample_rate=sample_rate, seed=seed, lfp_hz=lfp_hz, lfp_uv=lfp_uv,
                  hum_hz=hum_hz, hum_uv=hum_uv, common_uv=common_uv, noise_uv=noise_uv, units=units, rate_hz=rate_hz)
    return Synthetic(data, sample_rate, components, spike_samples, spike_units, unit_channels, templates, params)


__all__ = ["DEFAULT_CHANNELS", "DEFAULT_SAMPLE_RATE", "DEFAULT_SECONDS", "DEFAULT_SEED", "Synthetic", "spike_waveform", "synthetic"]
