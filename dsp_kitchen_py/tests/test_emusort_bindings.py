"""
Unit tests for EMUsort / Myomatrix Python bindings:
- EmusortSortConfig (presets & properties)
- EmusortBasisEmbedder (12-PC temporal subspace & projections)
- EmusortDetector (150-sample template matched filtering & detection)
- EmusortLatencyAligner (cross-channel conduction latency estimation & alignment)
"""

import numpy as np
import pytest

from dsp_kitchen.synapse.ml import (
    EmusortSortConfig,
    EmusortBasisEmbedder,
    EmusortDetector,
    EmusortLatencyAligner,
    MyomatrixSortConfig,
    MyomatrixBasisEmbedder,
    MyomatrixDetector,
    MyomatrixLatencyAligner,
)


def test_emusort_sort_config():
    # Test canonical and alias equality
    assert EmusortSortConfig is MyomatrixSortConfig

    # Defaults
    cfg = EmusortSortConfig()
    assert cfg.template_samples == 150
    assert cfg.threshold_sigma == 6.5
    assert cfg.refractory_samples == 60
    assert cfg.dedup_window_samples == 36
    assert cfg.num_temporal_pcs == 12

    # 32ch grid preset
    p32 = EmusortSortConfig.preset_32ch_grid()
    assert p32.template_samples == 150
    assert p32.num_temporal_pcs == 12
    assert p32.spatial_radius_um == 6000.0

    # 64ch grid preset
    p64 = EmusortSortConfig.preset_64ch_grid()
    assert p64.spatial_radius_um == 6000.0
    assert p64.max_clusters == 16

    # 8ch thread preset
    p8 = EmusortSortConfig.preset_8ch_thread()
    assert p8.spatial_radius_um == 1500.0
    assert p8.max_clusters == 6


def test_emusort_basis_embedder():
    assert EmusortBasisEmbedder is MyomatrixBasisEmbedder
    embedder = EmusortBasisEmbedder.from_hub()
    assert embedder.num_components == 12
    assert embedder.window_len == 150

    basis = embedder.basis_matrix()
    assert basis.shape == (12, 150)

    # Check orthonormality: V * V^T ~ I
    gram = basis @ basis.T
    np.testing.assert_allclose(gram, np.eye(12), atol=1e-4)

    # Test single-channel waveform projection & reconstruction
    t = np.linspace(-3, 3, 150)
    synthetic_muap = (1.0 - t**2) * np.exp(-0.5 * t**2)
    synthetic_muap = np.ascontiguousarray(synthetic_muap, dtype=np.float32)

    coeffs = embedder.project(synthetic_muap)
    assert coeffs.shape == (12,)

    recon = embedder.reconstruct(coeffs)
    assert recon.shape == (150,)

    # Test multi-channel snippet embedding
    # Shape: [num_spikes, channels, 150]
    num_spikes = 5
    n_ch = 8
    snippets = np.zeros((num_spikes, n_ch, 150), dtype=np.float32)
    for i in range(num_spikes):
        snippets[i, 0, :] = synthetic_muap * (i + 1)

    features = embedder.embed(snippets)
    assert features.shape == (num_spikes, n_ch * 12)
    assert np.all(np.isfinite(features))


def test_emusort_detector():
    assert EmusortDetector is MyomatrixDetector
    fs = 24414.0625
    detector = EmusortDetector.from_hub(threshold_sigma=5.0, refractory_samples=60)
    assert detector.num_templates == 6
    assert detector.window_len == 150
    assert detector.center_offset == 70

    templates = detector.templates_matrix()
    assert templates.shape == (6, 150)

    # Synthetic continuous data with an injected MUAP
    num_channels = 32
    num_samples = 5000
    data = np.zeros((num_channels, num_samples), dtype=np.float32)

    # Inject template 0 into channel 4 at sample 1000
    t0 = templates[0]
    data[4, 1000 : 1000 + 150] = t0 * 100.0  # High SNR spike

    # Energy filter
    energy = detector.filter_energy(data)
    assert energy.shape == data.shape
    peak_sample = 1000 + detector.center_offset
    assert energy[4, peak_sample] > 10.0

    # Detection
    events = detector.detect(data, sample_rate_hz=fs)
    assert len(events) >= 1
    # Check that an event is detected near peak_sample on channel 4
    ch4_events = [e for e in events if e.channel_id == 4]
    assert len(ch4_events) >= 1
    assert abs(ch4_events[0].sample_index - peak_sample) <= 5


def test_emusort_latency_aligner():
    assert EmusortLatencyAligner is MyomatrixLatencyAligner
    aligner = EmusortLatencyAligner(max_lag_samples=20)
    n_channels = 4
    n_samples = 150
    snippet = np.zeros((n_channels, n_samples), dtype=np.float32)

    # Base waveform on ref_ch (ch 0) centered at 75
    t = np.arange(150)
    muap = np.exp(-0.5 * ((t - 75) / 10.0) ** 2)

    # Channel 0: lag = 0
    snippet[0] = muap
    # Channel 1: shifted by +5 samples (delayed propagation)
    snippet[1] = np.roll(muap, 5)
    # Channel 2: shifted by +10 samples
    snippet[2] = np.roll(muap, 10)
    # Channel 3: shifted by -4 samples
    snippet[3] = np.roll(muap, -4)

    lags = aligner.estimate_channel_lags(snippet, ref_ch=0)
    assert len(lags) == n_channels
    assert lags[0] == 0
    assert lags[1] == 5
    assert lags[2] == 10
    assert lags[3] == -4

    aligned = aligner.align_snippet(snippet, lags)
    assert aligned.shape == (n_channels, n_samples)

    # After alignment, all channel peaks should coincide with ch 0 peak (at sample 75)
    for ch in range(n_channels):
        peak_idx = int(np.argmax(aligned[ch]))
        assert peak_idx == 75, f"Channel {ch} peak was {peak_idx}, expected 75"
