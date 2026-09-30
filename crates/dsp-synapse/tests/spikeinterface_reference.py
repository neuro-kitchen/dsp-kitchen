# /// script
# requires-python = ">=3.11"
# dependencies = ["spikeinterface==0.105.0", "numpy", "scipy", "numba"]
# ///
"""Generate SpikeInterface quality-metric fixtures for dsp-synapse.

Run from the repository root:

    uv run crates/dsp-synapse/tests/spikeinterface_reference.py

Writes crates/dsp-synapse/tests/fixtures/quality_metrics.json with the inputs and SpikeInterface's
outputs of isi_violations, rp_contamination (Llobet), amplitude_cutoff and presence_ratio.
"""

import json
from pathlib import Path

import numpy as np
from spikeinterface.metrics.quality import misc_metrics as mm

FS = 30_000.0
TOTAL_SAMPLES = int(FS * 180)  # 3 minutes


def spike_train(rng, rate_hz, contamination, refractory_s=0.002):
    """Poisson train with a refractory period plus a fraction of unconstrained contaminating spikes."""
    n = rng.poisson(rate_hz * TOTAL_SAMPLES / FS)
    isis = rng.exponential(1 / rate_hz, n) + refractory_s
    t = np.cumsum(isis)
    t = t[t < TOTAL_SAMPLES / FS]
    extra = rng.uniform(0, TOTAL_SAMPLES / FS, int(len(t) * contamination))
    samples = np.unique(np.round(np.concatenate([t, extra]) * FS).astype(np.int64))
    return samples


def amplitudes(rng, n, mean, sd, cutoff):
    a = rng.normal(mean, sd, n)
    return a[a > cutoff]


def main():
    rng = np.random.default_rng(1234)
    units = []
    for i, (rate, contam, amp_mean, amp_sd, amp_cut) in enumerate(
        [(5.0, 0.0, 80, 15, 0), (12.0, 0.05, 60, 20, 45), (30.0, 0.2, 40, 10, 38), (1.0, 0.5, 100, 30, 0), (0.05, 0.0, 70, 5, 0)]
    ):
        st = spike_train(rng, rate, contam)
        amps = np.round(amplitudes(rng, max(len(st), 10), amp_mean, amp_sd, amp_cut), 4)
        t_s = st / FS
        ratio, rate_v, count = mm.isi_violations([t_s], TOTAL_SAMPLES / FS, isi_threshold_s=0.0015, min_isi_s=0.0)
        t_c = int(round(0.0 * FS * 1e-3))
        t_r = int(round(1.0 * FS * 1e-3))
        n_v = int(mm._compute_rp_violations_numba(st, t_c, t_r))
        rp = mm._compute_rp_contamination_one_unit(n_v, len(st), TOTAL_SAMPLES, t_c, t_r)
        cutoff = mm.amplitude_cutoff(amps, num_histogram_bins=100, histogram_smoothing_value=3, amplitudes_bins_min_ratio=5)
        bin_samples = int(20.0 * FS)
        edges = np.arange(0, TOTAL_SAMPLES // bin_samples * bin_samples + 1, bin_samples)
        pr = mm.presence_ratio(st, bin_edges=edges, bin_n_spikes_thres=0)
        nan = lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
        units.append(
            {
                "name": f"unit{i}",
                "spike_samples": st.tolist(),
                "amplitudes": amps.tolist(),
                "isi_violations_ratio": nan(float(ratio)),
                "isi_violations_rate": nan(float(rate_v)),
                "isi_violations_count": int(count),
                "rp_violations": n_v,
                "rp_contamination": nan(float(rp)),
                "amplitude_cutoff": nan(float(cutoff)),
                "presence_ratio": nan(float(pr)),
            }
        )
    out = {
        "spikeinterface": "0.105.0",
        "fs": FS,
        "total_samples": TOTAL_SAMPLES,
        "isi_threshold_ms": 1.5,
        "refractory_ms": 1.0,
        "censored_ms": 0.0,
        "presence_bin_s": 20.0,
        "units": units,
    }
    path = Path(__file__).parent / "fixtures" / "quality_metrics.json"
    path.write_text(json.dumps(out))
    print(f"wrote {len(units)} units to {path}")


if __name__ == "__main__":
    main()
