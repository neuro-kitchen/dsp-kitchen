"""
End-to-end integration test running EMUsort spike sorting on an NWB Zarr recording
and exporting full spiking data to Phy2 format.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest

import dsp_kitchen as dk
import dsp_kitchen.synapse as syn
from dsp_kitchen.pipeline import Pipeline
from dsp_kitchen.filter.iir import BandpassFilter
from dsp_kitchen.spatial import CommonAverageReference
from dsp_kitchen.synapse.ml import (
    EmusortBasisEmbedder,
    EmusortDetector,
    EmusortLatencyAligner,
    EmusortSortConfig,
)


def test_emusort_nwb_to_phy_roundtrip():
    nwb_file = dk.resolve_data_path("data/nwb/15-25-33_meps.nwb.zarr")
    if not nwb_file.exists():
        pytest.skip(f"Test NWB file not found at: {nwb_file}")

    # 1. Open NWB recording slice (1.0 second @ ~24.4 kHz)
    rec = dk.open_nwb_zarr(nwb_file, series="/acquisition/HDEMG")
    fs = float(rec.sample_rate)
    duration_sec = 1.0
    num_samples = int(round(duration_sec * fs))
    raw = rec.read(0, num_samples)

    if rec.unit.lower() in ("v", "volts", "volt"):
        raw = raw * 1e6

    # 2. Configure probe
    probe = syn.hdemg_4x8_layout(ied_mm=4.0)
    assert probe.num_channels == 32

    # 3. Filter with Pipeline
    pipe = Pipeline([
        BandpassFilter(low_hz=100.0, high_hz=2000.0, order=4),
        CommonAverageReference(),
    ])
    filtered = pipe.run(np.ascontiguousarray(raw, dtype=np.float32), fs=fs)
    assert filtered.shape == raw.shape

    # 4. EMUsort Matched-Filter Detection
    config = EmusortSortConfig.preset_32ch_grid()
    refractory_samples = max(int(0.0025 * fs), 10)
    detector = EmusortDetector.from_hub(
        threshold_sigma=5.0,
        refractory_samples=refractory_samples,
    )
    events = detector.detect(filtered, sample_rate_hz=fs)
    assert len(events) > 0, "Expected non-zero EMUsort template crossings"

    # 5. Spatial Deduplication & 150-sample Snippet Extraction
    dedup = syn.deduplicate_spikes(
        events,
        probe,
        radius_um=config.spatial_radius_um,
        window_samples=int(0.0015 * fs),
    )
    assert len(dedup) > 0

    k_neighbors = 8
    snippets = syn.extract_snippets(
        filtered,
        dedup,
        probe,
        k_neighbors=k_neighbors,
        pre_samples=70,
        post_samples=80,
        apply_sinc_shift=True,
    )
    assert len(snippets) > 0
    assert snippets[0].waveform().shape == (k_neighbors, 150)

    # 6. Conduction Latency Cross-Channel Alignment
    waveforms = np.stack([s.waveform() for s in snippets], axis=0).astype(np.float32)
    aligner = EmusortLatencyAligner(max_lag_samples=25)
    for i in range(waveforms.shape[0]):
        lags = aligner.estimate_channel_lags(waveforms[i], ref_ch=0)
        waveforms[i] = aligner.align_snippet(waveforms[i], lags)

    # 7. Project onto 12-PC Temporal Muscle Basis
    embedder = EmusortBasisEmbedder.from_hub()
    features = embedder.embed(waveforms)
    assert features.shape == (len(snippets), k_neighbors * 12)
    assert np.all(np.isfinite(features))

    # 8. Cluster into Motor Units via GMM
    cluster_res = dk.cluster_gmm(
        features,
        min_clusters=2,
        max_clusters=5,
        covariance_type="diagonal",
    )
    labels = cluster_res["labels"]
    assert len(labels) == len(snippets)

    # 9. Build SortingOutput Container
    spike_times = [int(s.center_sample) for s in snippets]
    sorting = dk.SortingOutput.from_clusters(
        sorter_name="emusort",
        spike_samples=spike_times,
        labels=labels,
        sample_rate_hz=fs,
        total_samples=num_samples,
        probe=probe,
        snippets=snippets,
    )
    assert sorting.num_units >= 2
    assert sorting.total_spikes == len(snippets)

    # 10. Export to Phy2 format in temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        phy_path = Path(tmp_dir)
        sorting.export_to_phy(str(phy_path))

        # Write raw trace buffer for Phy2 TraceView
        int16_dat = np.clip(filtered, -32768, 32767).astype(np.int16)
        int16_dat.T.tofile(phy_path / "recording.dat")

        # Additional Phy2 auxiliary files
        pc_feats = features.reshape(len(snippets), k_neighbors, 12).transpose(0, 2, 1).astype(np.float32)
        np.save(phy_path / "pc_features.npy", pc_feats)
        pc_ind = np.tile(np.arange(k_neighbors, dtype=np.int32), (sorting.num_units, 1))
        np.save(phy_path / "pc_feature_ind.npy", pc_ind)

        # Compute similarity matrix
        templates = np.load(phy_path / "templates.npy")
        flat = templates.reshape(sorting.num_units, -1)
        norms = np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-9)
        normed = flat / norms
        sim = (normed @ normed.T).astype(np.float32)
        np.save(phy_path / "similar_templates.npy", sim)

        # Verify expected files exist
        expected_files = [
            "params.py",
            "spike_times.npy",
            "spike_clusters.npy",
            "spike_templates.npy",
            "amplitudes.npy",
            "templates.npy",
            "templates_std.npy",
            "templates_se.npy",
            "channel_map.npy",
            "channel_positions.npy",
            "channel_shanks.npy",
            "cluster_group.tsv",
            "cluster_info.tsv",
            "pc_features.npy",
            "pc_feature_ind.npy",
            "similar_templates.npy",
            "recording.dat",
        ]
        for fname in expected_files:
            target = phy_path / fname
            assert target.exists(), f"Missing required Phy2 file: {fname}"
            assert target.stat().st_size > 0, f"File {fname} is empty"

        # Verify params.py contents can be parsed as Python
        params_content = (phy_path / "params.py").read_text()
        params_dict = {}
        exec(params_content, {}, params_dict)
        assert params_dict["n_channels_dat"] == 32
        assert abs(params_dict["sample_rate"] - fs) < 1.0
        assert params_dict["dtype"] == "int16"
        assert params_dict["dat_path"] == "recording.dat"

        # Verify cluster_info.tsv has header and all units
        tsv_lines = (phy_path / "cluster_info.tsv").read_text().strip().split("\n")
        assert len(tsv_lines) == sorting.num_units + 1

        # Verify reload from disk via dsp_kitchen
        reloaded = dk.load_sorting(str(phy_path))
        assert reloaded.num_units == sorting.num_units
        assert reloaded.total_spikes == sorting.total_spikes

        # Verify phylib TemplateModel can load the dataset seamlessly
        try:
            from phylib.io.model import load_model
            model = load_model(str(phy_path / "params.py"))
            assert model.n_spikes == sorting.total_spikes
            assert model.n_clusters == sorting.num_units
            assert model.n_channels == 32
        except ImportError:
            pass  # phylib optional in test environment
