"""
Test running Kilosort4 on an NWB Zarr recording and exporting spiking data to Phy2.
"""

import sys
from pathlib import Path
import tempfile
import numpy as np
import pytest

import dsp_kitchen as dk
import dsp_kitchen.synapse as syn
from dsp_kitchen.pipeline import Pipeline
from dsp_kitchen.filter.iir import BandpassFilter
from dsp_kitchen.spatial import CommonAverageReference
from dsp_kitchen.synapse.ml import Kilosort4BasisEmbedder, Kilosort4Detector


def test_kilosort4_nwb_to_phy_roundtrip():
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

    # 4. Kilosort4 Matched-Filter Detection
    detector = Kilosort4Detector.from_hub(threshold_sigma=5.0, refractory_samples=int(0.001 * fs))
    events = detector.detect(filtered, sample_rate_hz=fs)
    assert len(events) > 0, "Expected non-zero Kilosort4 template crossings"

    # 5. Spatial Deduplication & Snippet Extraction
    dedup = syn.deduplicate_spikes(events, probe, radius_um=6000.0, window_samples=int(0.0008 * fs))
    assert len(dedup) > 0

    k_neighbors = 8
    snippets = syn.extract_snippets(
        filtered,
        dedup,
        probe,
        k_neighbors=k_neighbors,
        pre_samples=20,
        post_samples=41,
        apply_sinc_shift=True,
    )
    assert len(snippets) > 0

    # 6. Kilosort4 wPCA Embedding
    embedder = Kilosort4BasisEmbedder.from_hub()
    waveforms = np.stack([s.waveform() for s in snippets], axis=0).astype(np.float32)
    features = embedder.embed(waveforms)
    assert features.shape == (len(snippets), k_neighbors * 6)

    # 7. Cluster into units
    cluster_res = dk.cluster_gmm(features, min_clusters=2, max_clusters=6, covariance_type="diagonal")
    labels = cluster_res["labels"]
    num_units = cluster_res["num_clusters"]
    assert num_units >= 2

    # 8. Build SortingOutput with snippets to compute multi-channel templates
    times = [int(s.center_sample) for s in snippets]
    sorting = dk.SortingOutput.from_clusters(
        "kilosort4",
        times,
        labels,
        fs,
        total_samples=raw.shape[1],
        probe=probe,
        snippets=snippets,
    )
    assert sorting.num_units == num_units
    assert sorting.total_spikes == len(snippets)

    # 9. Export to Phy2 folder
    with tempfile.TemporaryDirectory() as tmpdir:
        phy_dir = Path(tmpdir) / "phy_kilosort4"
        sorting.export_to_phy(str(phy_dir))

        # Also write recording.dat (int16 trace view)
        int16_dat = np.clip(filtered, -32768, 32767).astype(np.int16)
        int16_dat.T.tofile(phy_dir / "recording.dat")

        # Compute similarity matrix
        templates = np.load(phy_dir / "templates.npy")
        flat = templates.reshape(num_units, -1)
        norms = np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-9)
        normed = flat / norms
        sim = (normed @ normed.T).astype(np.float32)
        np.save(phy_dir / "similar_templates.npy", sim)

        # Verify all Phy2 required files exist and are non-empty
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
            "similar_templates.npy",
            "recording.dat",
        ]

        for fname in expected_files:
            file_path = phy_dir / fname
            assert file_path.exists(), f"Missing Phy2 file: {fname}"
            assert file_path.stat().st_size > 0, f"Empty Phy2 file: {fname}"

        # Verify params.py contents can be parsed as Python
        params_content = (phy_dir / "params.py").read_text()
        params_dict = {}
        exec(params_content, {}, params_dict)
        assert params_dict["n_channels_dat"] == 32
        assert abs(params_dict["sample_rate"] - fs) < 1.0
        assert params_dict["dtype"] == "int16"
        assert params_dict["dat_path"] == "recording.dat"

        # Verify cluster_info.tsv has header and all units
        tsv_lines = (phy_dir / "cluster_info.tsv").read_text().strip().split("\n")
        assert len(tsv_lines) == num_units + 1  # 1 header + N units

        # Verify reload from disk via dsp_kitchen
        reloaded = dk.load_sorting(str(phy_dir))
        assert reloaded.num_units == sorting.num_units
        assert reloaded.total_spikes == sorting.total_spikes
        assert list(reloaded.spike_train(0)) == list(sorting.spike_train(0))

        # Verify phylib TemplateModel can load the dataset directly if phylib is available
        try:
            from phylib.io.model import load_model
            model = load_model(str(phy_dir / "params.py"))
            assert model.n_spikes == sorting.total_spikes
            assert model.n_clusters == sorting.num_units
            assert model.n_channels == 32
        except ImportError:
            pass


if __name__ == "__main__":
    test_kilosort4_nwb_to_phy_roundtrip()
    print("test_kilosort4_nwb_to_phy_roundtrip passed!")
