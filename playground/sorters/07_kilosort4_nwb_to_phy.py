#!/usr/bin/env python3
"""
Run Kilosort4 (universal templates + wPCA basis) on an NWB Zarr recording and export to Phy2.

Pipeline:
  1. Open NWB Zarr recording (/acquisition/HDEMG or user-selected series).
  2. Pre-process through Pipeline (BandpassFilter 100-2000 Hz + CommonAverageReference).
  3. Detect MUAPs/spikes using Kilosort4 universal matched-filter templates (`wTEMP.npy`).
  4. Spatially deduplicate events across the multi-channel probe grid.
  5. Extract multi-channel sinc-realigned waveform snippets.
  6. Project snippets onto Kilosort4 temporal basis (`wPCA.npy`) for feature embedding.
  7. Cluster feature embeddings into motor units / neural clusters.
  8. Build a unified `SortingOutput` container with firing rates, SNR, and IBL quality metrics.
  9. Export all Phy2-compliant files (`params.py`, `.npy` arrays, `.tsv` tables, and `recording.dat`).

Usage:
  python playground/sorters/07_kilosort4_nwb_to_phy.py
  python playground/sorters/07_kilosort4_nwb_to_phy.py --duration-sec 30.0 --threshold-sigma 4.5
  phy template-gui data/sorters/phy_kilosort4_output/params.py
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import dsp_kitchen as dk
import dsp_kitchen.synapse as syn
from dsp_kitchen.pipeline import Pipeline
from dsp_kitchen.filter.iir import BandpassFilter, NotchFilter
from dsp_kitchen.spatial import CommonAverageReference
from dsp_kitchen.synapse.ml import Kilosort4BasisEmbedder, Kilosort4Detector


def run_kilosort4_nwb_to_phy(
    nwb_path: str = "data/nwb/15-25-33_meps.nwb.zarr",
    series: str = "/acquisition/HDEMG",
    output_dir: str = "data/sorters/phy_kilosort4_output",
    start_sec: float = 0.0,
    duration_sec: float = 5.0,
    threshold_sigma: float = 4.5,
    min_clusters: int = 3,
    max_clusters: int = 10,
    ied_mm: float = 4.0,
    export_raw_dat: bool = True,
) -> Path:
    resolved_nwb = dk.resolve_data_path(nwb_path)
    if not resolved_nwb.exists():
        raise FileNotFoundError(f"NWB Zarr file not found at: {resolved_nwb}")

    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  KILOSORT4 NWB SPIKE SORTING -> PHY2 EXPORTER")
    print("=" * 80)
    print(f"  Input NWB Store:    {resolved_nwb}")
    print(f"  Electrical Series:  {series}")
    print(f"  Output Phy2 Folder: {out_path}")
    print(f"  Window:             [{start_sec:.2f}s .. {start_sec + duration_sec:.2f}s] ({duration_sec:.2f} s)")
    print(f"  Detection Sigma:    {threshold_sigma:.1f} σ")
    print("-" * 80)

    # 1. Open NWB recording
    t0 = time.perf_counter()
    rec = dk.open_nwb_zarr(resolved_nwb, series=series)
    fs = float(rec.sample_rate)
    n_ch = rec.channels
    total_rec_samples = rec.samples
    start_sample = int(round(start_sec * fs))
    num_samples = int(round(duration_sec * fs))
    end_sample = min(start_sample + num_samples, total_rec_samples)
    actual_samples = end_sample - start_sample

    print(f"[1/8] Opened NWB stream: {n_ch} channels @ {fs:,.2f} Hz ({rec.duration_sec:.1f}s total)")
    print(f"      Reading slice [{start_sample:,} .. {end_sample:,}] ({actual_samples:,} samples)...")

    raw = rec.read(start_sample, end_sample)
    if rec.unit.lower() in ("v", "volts", "volt"):
        raw = raw * 1e6  # Convert to microvolts

    # 2. Configure Probe Layout
    if n_ch == 32:
        probe = syn.hdemg_4x8_layout(ied_mm=ied_mm)
        print(f"[2/8] Configured 4x8 HD-EMG grid layout (pitch = {ied_mm:.1f} mm)")
    elif n_ch == 64:
        probe = syn.hdemg_8x8_layout(ied_mm=ied_mm)
        print(f"[2/8] Configured 8x8 HD-EMG grid layout (pitch = {ied_mm:.1f} mm)")
    elif n_ch == 384:
        probe = syn.neuropixels_1_0_layout()
        print("[2/8] Configured Neuropixels 1.0 probe layout (384 ch)")
    else:
        # Linear or 2D fallback probe layout
        coords = [(0.0, float(i) * 50.0) for i in range(n_ch)]
        probe = syn.custom_layout(f"Probe-{n_ch}ch", coords)
        print(f"[2/8] Configured custom {n_ch}-channel probe layout")

    # 3. Filter Recording (Pipeline)
    print("[3/8] Running pre-processing pipeline (Bandpass 100-2000 Hz + CAR)...")
    pipe = Pipeline([
        BandpassFilter(low_hz=100.0, high_hz=2000.0, order=4, direction="forward-backward"),
        CommonAverageReference(),
    ])
    filtered = pipe.run(np.ascontiguousarray(raw, dtype=np.float32), fs=fs)

    # 4. Kilosort4 Matched-Filter Detection
    print(f"[4/8] Running Kilosort4 universal template matcher (`wTEMP.npy`, threshold = {threshold_sigma}σ)...")
    refractory_samples = max(int(0.001 * fs), 10)
    ks4_detector = Kilosort4Detector.from_hub(
        threshold_sigma=float(threshold_sigma),
        refractory_samples=refractory_samples,
    )
    raw_events = ks4_detector.detect(filtered, sample_rate_hz=fs)
    print(f"      Detected {len(raw_events):,} raw template crossings across {n_ch} channels")

    if not raw_events:
        print("WARNING: No events detected. Consider lowering threshold_sigma.")
        return out_path

    # 5. Spatial Deduplication & Snippet Extraction
    print("[5/8] Deduplicating events spatially & extracting sinc-aligned waveforms...")
    spatial_radius = max(float(ied_mm * 1500.0), 150.0)
    window_samples = max(int(0.0008 * fs), 10)
    dedup_events = syn.deduplicate_spikes(
        raw_events,
        probe,
        radius_um=spatial_radius,
        window_samples=window_samples,
    )
    print(f"      Retained {len(dedup_events):,} deduplicated spikes")

    k_neighbors = min(8, n_ch)
    snippets = syn.extract_snippets(
        filtered,
        dedup_events,
        probe,
        k_neighbors=k_neighbors,
        pre_samples=20,
        post_samples=41,
        apply_sinc_shift=True,
    )
    print(f"      Extracted {len(snippets):,} multi-channel snippets (K={k_neighbors}, T=61 samples)")

    # 6. Kilosort4 Basis Projection (wPCA)
    print("[6/8] Projecting snippets onto Kilosort4 temporal basis (`wPCA.npy`)...")
    ks4_embedder = Kilosort4BasisEmbedder.from_hub()
    waveforms = np.stack([s.waveform() for s in snippets], axis=0).astype(np.float32)
    features = ks4_embedder.embed(waveforms)
    print(f"      Feature embedding matrix: {features.shape} [spikes x (K*6)]")

    # 7. Cluster into Units
    print(f"[7/8] Clustering features via GMM (min_k={min_clusters}, max_k={max_clusters})...")
    cluster_res = dk.cluster_gmm(
        features,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        covariance_type="diagonal",
    )
    labels = cluster_res["labels"]
    num_units = cluster_res["num_clusters"]
    print(f"      Identified {num_units} distinct units (BIC = {cluster_res['bic']:.1f})")

    # 8. Build SortingOutput Container
    spike_times = [int(s.center_sample) + start_sample for s in snippets]
    sorting = dk.SortingOutput.from_clusters(
        sorter_name="kilosort4",
        spike_samples=spike_times,
        labels=labels,
        sample_rate_hz=fs,
        total_samples=total_rec_samples,
        probe=probe,
        snippets=snippets,
    )

    # 9. Export to Phy2
    print(f"[8/8] Exporting to Phy2 folder at: {out_path}...")
    sorting.export_to_phy(str(out_path))

    # Compute unit similarity matrix for Phy2 SimilarityView
    templates_npy_path = out_path / "templates.npy"
    if templates_npy_path.exists():
        templates = np.load(templates_npy_path)  # [n_units, n_samples, n_channels]
        n_u = templates.shape[0]
        flat_templates = templates.reshape(n_u, -1)
        norms = np.linalg.norm(flat_templates, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        normed = flat_templates / norms
        sim_matrix = (normed @ normed.T).astype(np.float32)
        np.save(out_path / "similar_templates.npy", sim_matrix)

    # Write PC features for Phy2 FeatureView (PC1 vs PC2)
    # features shape: [N, K * 6] -> reshape to [N, 6, K] for Kilosort format
    if features.shape[0] > 0:
        n_spk = features.shape[0]
        pc_feats = features.reshape(n_spk, k_neighbors, 6).transpose(0, 2, 1).astype(np.float32)
        np.save(out_path / "pc_features.npy", pc_feats)
        # pc_feature_ind: [n_units, k_neighbors] channel index map
        pc_feature_ind = np.tile(np.arange(k_neighbors, dtype=np.int32), (num_units, 1))
        np.save(out_path / "pc_feature_ind.npy", pc_feature_ind)

    # Write continuous recording.dat for Phy2 TraceView
    if export_raw_dat:
        dat_file = out_path / "recording.dat"
        print(f"      Writing raw trace buffer: {dat_file.name} ({actual_samples:,} samples x {n_ch} ch)...")
        int16_dat = np.clip(filtered, -32768, 32767).astype(np.int16)
        # Phy expects column-major / interleaved: [samples, channels]
        int16_dat.T.tofile(dat_file)

    elapsed = time.perf_counter() - t0
    speedup = duration_sec / elapsed

    print("\n" + "=" * 80)
    print("  SORTING & PHY2 EXPORT COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"  Units Resolved:     {sorting.num_units}")
    print(f"  Total Spikes:       {sorting.total_spikes:,}")
    print(f"  Elapsed Time:       {elapsed:.2f} s ({speedup:.1f}x real-time speedup)")
    print("\n  Unit Summary Metrics:")
    print("  " + "-" * 76)
    print(f"  {'Unit':>4s} | {'Ch':>3s} | {'Spikes':>7s} | {'Rate (Hz)':>9s} | {'SNR':>6s} | {'ISI Viol':>8s} | {'Quality':>7s}")
    print("  " + "-" * 76)
    for m in sorting.summary_table():
        print(
            f"  {m['unit_id']:>4d} | {m['primary_channel']:>3d} | {m['num_spikes']:>7,d} | "
            f"{m['firing_rate_hz']:>9.2f} | {m['snr']:>6.1f} | {m['isi_violation_ratio']:>8.4f} | {m['quality_label']:>7s}"
        )
    print("  " + "-" * 76)

    print("\n  Phy2 Exported Files:")
    for f in sorted(out_path.glob("*")):
        print(f"    - {f.name:<25s} ({f.stat().st_size:,} bytes)")

    print("\n  To open and inspect in Phy2:")
    print(f"    phy template-gui {out_path / 'params.py'}")
    print("=" * 80)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Run Kilosort4 spike sorting on an NWB Zarr recording and export to Phy2."
    )
    parser.add_argument(
        "--nwb-path",
        type=str,
        default="data/nwb/15-25-33_meps.nwb.zarr",
        help="Path to NWB Zarr recording directory.",
    )
    parser.add_argument(
        "--series",
        type=str,
        default="/acquisition/HDEMG",
        help="NWB series path to sort (e.g. /acquisition/HDEMG).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/sorters/phy_kilosort4_output",
        help="Target folder for Phy2 export.",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=0.0,
        help="Start timestamp in seconds.",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=5.0,
        help="Duration in seconds to sort.",
    )
    parser.add_argument(
        "--threshold-sigma",
        type=float,
        default=4.5,
        help="Matched-filter detection threshold multiplier.",
    )
    parser.add_argument(
        "--min-clusters",
        type=int,
        default=3,
        help="Minimum number of clusters for GMM.",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=8,
        help="Maximum number of clusters for GMM.",
    )
    parser.add_argument(
        "--ied-mm",
        type=float,
        default=4.0,
        help="Inter-electrode distance in millimeters for HD-EMG grid.",
    )
    parser.add_argument(
        "--no-raw-dat",
        action="store_true",
        help="Skip exporting recording.dat binary trace file.",
    )
    args = parser.parse_args()

    run_kilosort4_nwb_to_phy(
        nwb_path=args.nwb_path,
        series=args.series,
        output_dir=args.output_dir,
        start_sec=args.start_sec,
        duration_sec=args.duration_sec,
        threshold_sigma=args.threshold_sigma,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
        ied_mm=args.ied_mm,
        export_raw_dat=not args.no_raw_dat,
    )


if __name__ == "__main__":
    main()
