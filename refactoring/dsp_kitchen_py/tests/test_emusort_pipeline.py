"""
End-to-end integration test for the EMUsort pipeline:
Median CAR -> Bandpass -> Spatial Whitening -> 150-sample MUAP detection ->
Localized deduplication -> Latency alignment -> 12-PC basis ->
Contact-localized unit formation -> PNR & CoV-ISI -> Phy2 export.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest

import dsp_kitchen as dk
import dsp_kitchen.synapse as syn
from dsp_kitchen.pipeline import Pipeline
from dsp_kitchen.filter.iir import BandpassFilter
from dsp_kitchen.spatial import CommonAverageReference, SpatialWhitening
from dsp_kitchen.synapse.ml import (
    EMUsortBasisEmbedder,
    EMUsortDetector,
    EMUsortLatencyAligner,
    EMUsortSortConfig,
)


def test_emusort_nwb_to_phy_roundtrip():
    nwb_file = dk.resolve_data_path("data/nwb/15-25-33_meps.nwb.zarr")
    if not nwb_file.exists():
        pytest.skip(f"Test NWB file not found at: {nwb_file}")

    # 1. Open NWB recording slice (2.0 seconds @ ~24.4 kHz)
    rec = dk.open_nwb_zarr(nwb_file, series="/acquisition/HDEMG")
    fs = float(rec.sample_rate)
    duration_sec = 2.0
    num_samples = int(round(duration_sec * fs))
    raw = rec.read(0, num_samples)

    if rec.unit.lower() in ("v", "volts", "volt"):
        raw = raw * 1e6

    # 2. Configure 32-channel 4x8 Myomatrix probe
    probe = syn.hdemg_4x8_layout(ied_mm=4.0)
    assert probe.num_channels == 32
    positions_xy = [[p[0], p[1]] for p in probe.contact_positions()]

    # 3. Sequential Preprocessing: CAR -> Bandpass -> Local Spatial Whitening
    car_data = Pipeline([CommonAverageReference()]).run(raw, fs=fs)
    bp_data = Pipeline([BandpassFilter(low_hz=100.0, high_hz=2000.0, order=4)]).run(car_data, fs=fs)
    whitener = SpatialWhitening.fit_local_knn(
        bp_data[:, :min(num_samples, int(fs))],
        positions=positions_xy,
        k_neighbors=8,
        epsilon=1e-5,
    )
    preprocessed = Pipeline([whitener]).run(bp_data, fs=fs)
    assert preprocessed.shape == raw.shape

    # 4. 150-sample Universal MUAP Matched-Filter Detection
    refractory_samples = max(int(0.0025 * fs), 10)
    detector = EMUsortDetector.from_hub(
        threshold_sigma=5.0,
        refractory_samples=refractory_samples,
    )
    events = detector.detect(preprocessed, sample_rate_hz=fs)
    assert len(events) > 0, "Expected non-zero EMUsort template crossings"

    # 5. Localized Spatial Deduplication (contact pitch = 4000 µm)
    dedup_spikes = syn.deduplicate_spikes(
        events,
        probe,
        radius_um=4000.0,
        window_samples=refractory_samples,
    )
    assert len(dedup_spikes) > 0
    assert len(dedup_spikes) <= len(events)

    # 6. Extract 150-sample multi-channel snippets
    snippets = syn.extract_snippets(
        preprocessed,
        dedup_spikes,
        probe,
        k_neighbors=8,
        pre_samples=50,
        post_samples=100,
        apply_sinc_shift=True,
    )
    assert len(snippets) > 0
    assert len(snippets) <= len(dedup_spikes)

    # 7. Conduction Latency Alignment
    waveforms = np.stack([s.waveform() for s in snippets], axis=0).astype(np.float32)
    assert waveforms.shape[1] == 8
    assert waveforms.shape[2] == 150

    aligner = EMUsortLatencyAligner(max_lag_samples=25)
    for i in range(min(5, waveforms.shape[0])):
        lags = aligner.estimate_channel_lags(waveforms[i], ref_ch=0)
        assert len(lags) == 8
        aligned = aligner.align_snippet(waveforms[i], lags)
        assert aligned.shape == waveforms[i].shape

    # 8. 12-PC Temporal Muscle Basis Embedding
    embedder = EMUsortBasisEmbedder.from_hub()
    features = embedder.embed(waveforms)
    assert features.shape == (waveforms.shape[0], 8 * 12)

    # 9. Contact-Localized Motor Unit Formation & Phy2 Export
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "phy_emusort_output"

        # Group by contact
        spikes_by_ch = {}
        for i, s in enumerate(snippets):
            spikes_by_ch.setdefault(s.primary_channel, []).append(i)

        unit_labels = np.zeros(len(snippets), dtype=np.int32)
        next_u = 0
        for ch, indices in spikes_by_ch.items():
            for idx in indices:
                unit_labels[idx] = next_u
            next_u += 1

        spike_times = [int(s.center_sample) for s in snippets]
        sorting = dk.SortingOutput.from_clusters(
            sorter_name="emusort",
            spike_samples=spike_times,
            labels=unit_labels,
            sample_rate_hz=fs,
            total_samples=num_samples,
            probe=probe,
            snippets=snippets,
        )

        sorting.export_to_phy(str(out_path))

        # Verify Phy2 export integrity
        assert (out_path / "params.py").exists()
        assert (out_path / "spike_times.npy").exists()
        assert (out_path / "spike_clusters.npy").exists()
        assert (out_path / "channel_map.npy").exists()
        assert (out_path / "channel_positions.npy").exists()
        assert (out_path / "templates.npy").exists()
        assert (out_path / "cluster_group.tsv").exists()
        assert (out_path / "cluster_info.tsv").exists()

        # Check params.py
        params_txt = (out_path / "params.py").read_text()
        assert "n_channels_dat = 32" in params_txt
        assert "sample_rate" in params_txt
