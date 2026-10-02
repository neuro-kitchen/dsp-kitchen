#!/usr/bin/env python3
"""
Run EMUsort (150-sample MUAP matched-filtering + conduction latency alignment + 12-PC temporal muscle basis)
on an NWB Zarr recording and export directly to Phy2.

Pipeline:
  1. Open NWB Zarr recording (/acquisition/HDEMG or user-selected series).
  2. Sequential Preprocessing: Median CAR -> Bandpass (100-2000 Hz) -> Local Spatial Whitening (ZCA).
  3. Detect MUAPs using EMUsort 150-sample universal matched-filter templates (`wTEMP_EMG`).
  4. Spatial deduplication enforcing motor unit biological refractory period (>= 2.5 ms) and array pitch.
  5. Multi-channel conduction velocity delay alignment (`EMUsortLatencyAligner`).
  6. Extract 150-sample multi-channel waveform snippets (pre=50, post=100) with sinc realignment.
  7. Project aligned snippets onto 12-PC temporal muscle basis (`EMUsortBasisEmbedder`).
  8. Contact-localized motor unit formation across the array grid (avoiding global GMM collapse).
  9. Calculate Pulse-to-Noise Ratio (PNR >= 20 dB) and CoV-ISI quality metrics.
  10. Export all Phy2-compliant files (`params.py`, `.npy` arrays, `.tsv` tables, and `recording.dat`).

Usage:
  python playground/sorters/08_emusort_nwb_to_phy.py
  python playground/sorters/08_emusort_nwb_to_phy.py --duration-sec 60.0 --threshold-sigma 6.5
  phy template-gui data/sorters/phy_emusort_output/params.py
"""

import argparse
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

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


def run_emusort_nwb_to_phy(
    nwb_path: str = "data/nwb/15-25-33_meps.nwb.zarr",
    series: str = "/acquisition/HDEMG",
    output_dir: str = "data/sorters/phy_emusort_output",
    start_sec: float = 0.0,
    duration_sec: float = 600.0,
    threshold_sigma: float = 6.5,
    refractory_ms: float = 2.5,
    ied_mm: float = 4.0,
    align_conduction_latency: bool = True,
    export_raw_dat: bool = True,
) -> Path:
    resolved_nwb = dk.resolve_data_path(nwb_path)
    if not resolved_nwb.exists():
        raise FileNotFoundError(f"NWB Zarr file not found at: {resolved_nwb}")

    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  EMUSORT / MYOMATRIX NWB SPIKE SORTING -> PHY2 EXPORTER")
    print("=" * 80)
    print(f"  Input NWB Store:          {resolved_nwb}")
    print(f"  Electrical Series:        {series}")
    print(f"  Output Phy2 Folder:       {out_path}")
    print(f"  Window:                   [{start_sec:.2f}s .. {start_sec + duration_sec:.2f}s] ({duration_sec:.2f} s / {duration_sec/60.0:.1f} min)")
    print(f"  Detection Sigma:          {threshold_sigma:.1f} σ")
    print(f"  Refractory Period:        {refractory_ms:.1f} ms")
    print(f"  Contact Pitch (IED):      {ied_mm:.1f} mm")
    print(f"  MUAP Snippet Window:      150 samples (~5.0 ms at 30 kHz)")
    print(f"  Temporal PCA Components:  12 PCs")
    print(f"  Conduction Align:         {align_conduction_latency}")
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
    print(f"      Reading slice [{start_sample:,} .. {end_sample:,}] ({actual_samples:,} samples, {actual_samples/fs:.1f}s)...")

    t_read_start = time.perf_counter()
    raw = rec.read(start_sample, end_sample)
    if rec.unit.lower() in ("v", "volts", "volt"):
        raw = raw * 1e6  # Convert to microvolts
    t_read = time.perf_counter() - t_read_start
    print(f"      Read complete in {t_read:.2f} s")

    # 2. Configure Probe Layout
    if n_ch == 32:
        probe = syn.hdemg_4x8_layout(ied_mm=ied_mm)
        print(f"[2/8] Configured 4x8 HD-EMG grid layout (pitch = {ied_mm:.1f} mm)")
    elif n_ch == 64:
        probe = syn.hdemg_8x8_layout(ied_mm=ied_mm)
        print(f"[2/8] Configured 8x8 HD-EMG grid layout (pitch = {ied_mm:.1f} mm)")
    else:
        coords = [(0.0, float(i) * float(ied_mm * 1000.0)) for i in range(n_ch)]
        probe = syn.custom_layout(f"Myomatrix-{n_ch}ch", coords)
        print(f"[2/8] Configured custom {n_ch}-channel probe layout")

    # Extract contact coordinates
    positions_xy = [[p[0], p[1]] for p in probe.contact_positions()]

    # 3. Sequential Preprocessing: CAR -> Bandpass -> Local Spatial Whitening
    print(f"[3/8] Running sequential preprocessing: CAR -> Bandpass (100-2000 Hz) -> Local Spatial Whitening...")
    t_prep_start = time.perf_counter()

    # Step A: Median-based Common Average Reference (CAR)
    car_pipe = Pipeline([CommonAverageReference()])
    car_data = car_pipe.run(raw, fs=fs)

    # Step B: Zero-phase bandpass filter
    bp_pipe = Pipeline([BandpassFilter(low_hz=100.0, high_hz=2000.0, order=4, direction="forward-backward")])
    bp_data = bp_pipe.run(car_data, fs=fs)

    # Step C: Local Spatial Whitening across neighboring contacts
    k_whiten = min(n_ch, 8)
    whiten_samples = min(actual_samples, int(2.0 * fs))
    whitener = SpatialWhitening.fit_local_knn(
        bp_data[:, :whiten_samples],
        positions=positions_xy,
        k_neighbors=k_whiten,
        epsilon=1e-5,
    )
    preprocessed = Pipeline([whitener]).run(bp_data, fs=fs)
    t_prep = time.perf_counter() - t_prep_start
    print(f"      Preprocessed {actual_samples:,} samples across {n_ch} channels in {t_prep:.2f} s ({actual_samples / fs / t_prep:.1f}x real-time)")

    # 4. Universal MUAP Matched-Filter Detection
    print(f"[4/8] Detecting MUAPs via EMUsort universal matched-filter templates ({threshold_sigma}σ, refractory={refractory_ms}ms)...")
    t_det_start = time.perf_counter()
    refractory_samples = max(int(refractory_ms * 1e-3 * fs), 10)
    emu_detector = EMUsortDetector.from_hub(
        threshold_sigma=float(threshold_sigma),
        refractory_samples=refractory_samples,
    )
    raw_events = emu_detector.detect(preprocessed, sample_rate_hz=fs)
    t_det = time.perf_counter() - t_det_start
    print(f"      Detected {len(raw_events):,} raw MUAP crossings in {t_det:.2f} s")

    if not raw_events:
        print("WARNING: No events detected. Consider lowering threshold_sigma.")
        return out_path

    # 5. Localized Spatial Deduplication (Contact Pitch Radius)
    spatial_radius_um = float(ied_mm * 1000.0)  # adjacent contact distance
    dedup_window_samples = max(int(refractory_ms * 1e-3 * fs * 0.6), 12)
    print(f"[5/8] Deduplicating events spatially (radius = {spatial_radius_um:.0f} µm, window = {dedup_window_samples} samples)...")
    t_dedup_start = time.perf_counter()
    dedup_events = syn.deduplicate_spikes(
        raw_events,
        probe,
        radius_um=spatial_radius_um,
        window_samples=dedup_window_samples,
    )
    t_dedup = time.perf_counter() - t_dedup_start
    print(f"      Retained {len(dedup_events):,} deduplicated MUAPs in {t_dedup:.2f} s")

    # 6. Extract 150-sample multi-channel snippets & Latency Align
    print(f"[6/8] Extracting 150-sample snippets (pre=50, post=100 samples) across K={min(8, n_ch)} channels...")
    t_snip_start = time.perf_counter()
    k_neighbors = min(8, n_ch)
    snippets = syn.extract_snippets(
        preprocessed,
        dedup_events,
        probe,
        k_neighbors=k_neighbors,
        pre_samples=50,
        post_samples=100,
        apply_sinc_shift=True,
    )
    t_snip = time.perf_counter() - t_snip_start
    print(f"      Extracted {len(snippets):,} multi-channel snippets in {t_snip:.2f} s")

    # Align propagation delays if requested
    waveforms = np.stack([s.waveform() for s in snippets], axis=0).astype(np.float32)  # [N, K, 150]
    if align_conduction_latency:
        t_align_start = time.perf_counter()
        aligner = EMUsortLatencyAligner(max_lag_samples=25)
        for i in range(waveforms.shape[0]):
            lags = aligner.estimate_channel_lags(waveforms[i], ref_ch=0)
            waveforms[i] = aligner.align_snippet(waveforms[i], lags)
        t_align = time.perf_counter() - t_align_start
        print(f"      Aligned conduction latency across {waveforms.shape[0]:,} snippets in {t_align:.2f} s")

    # 7. Contact-Localized Motor Unit Formation (Avoiding arbitrary global GMM)
    print("[7/8] Forming contact-localized motor units across probe contacts...")
    t_unit_start = time.perf_counter()
    spikes_by_ch = {}
    for i, s in enumerate(snippets):
        spikes_by_ch.setdefault(s.primary_channel, []).append((i, s))

    # Project waveforms onto 12-PC temporal muscle basis
    emu_embedder = EMUsortBasisEmbedder.from_hub()
    features = emu_embedder.embed(waveforms)  # [N, K * 12]

    units = []
    unit_labels = np.zeros(len(snippets), dtype=np.int32)
    next_unit_id = 0

    for ch in range(n_ch):
        ch_entries = spikes_by_ch.get(ch, [])
        if len(ch_entries) < 15:
            continue
        ch_indices = [idx for idx, _ in ch_entries]
        ch_feats = features[ch_indices]
        ch_amps = np.array([abs(snippets[idx].peak_amplitude_uv) for idx in ch_indices])

        # Check for multi-unit bimodality on this electrode
        med_amp = float(np.median(ch_amps))
        high_mask = ch_amps > med_amp * 1.5
        if np.sum(high_mask) >= 20 and np.sum(~high_mask) >= 20:
            # Unit A
            u_a = next_unit_id
            next_unit_id += 1
            for local_i, idx in enumerate(ch_indices):
                if not high_mask[local_i]:
                    unit_labels[idx] = u_a
            units.append({'unit_id': u_a, 'primary_ch': ch, 'indices': [idx for local_i, idx in enumerate(ch_indices) if not high_mask[local_i]]})

            # Unit B
            u_b = next_unit_id
            next_unit_id += 1
            for local_i, idx in enumerate(ch_indices):
                if high_mask[local_i]:
                    unit_labels[idx] = u_b
            units.append({'unit_id': u_b, 'primary_ch': ch, 'indices': [idx for local_i, idx in enumerate(ch_indices) if high_mask[local_i]]})
        else:
            u_id = next_unit_id
            next_unit_id += 1
            for idx in ch_indices:
                unit_labels[idx] = u_id
            units.append({'unit_id': u_id, 'primary_ch': ch, 'indices': ch_indices})

    num_units = len(units)
    t_unit = time.perf_counter() - t_unit_start
    print(f"      Resolved {num_units} motor units localized across active contacts in {t_unit:.2f} s")

    # 8. Compute Full Multi-Channel Templates [num_units, 150, n_ch]
    print(f"[8/8] Accumulating full multi-channel templates (150 samples x {n_ch} ch) and exporting to Phy2...")
    templates = np.zeros((num_units, 150, n_ch), dtype=np.float32)
    spike_times = [int(s.center_sample) + start_sample for s in snippets]

    for u_info in units:
        u_id = u_info['unit_id']
        u_indices = u_info['indices']
        sample_wfs = []
        for idx in u_indices[:150]:
            c_samp = int(snippets[idx].center_sample)
            if c_samp >= 50 and c_samp + 100 <= actual_samples:
                sample_wfs.append(preprocessed[:, c_samp - 50 : c_samp + 100])
        if sample_wfs:
            templates[u_id] = np.mean(sample_wfs, axis=0).T

    # Build SortingOutput container
    sorting = dk.SortingOutput.from_clusters(
        sorter_name="emusort",
        spike_samples=spike_times,
        labels=unit_labels,
        sample_rate_hz=fs,
        total_samples=total_rec_samples,
        probe=probe,
        snippets=snippets,
    )

    # Export to Phy2
    sorting.export_to_phy(str(out_path))

    # Overwrite templates with full 150-sample templates
    np.save(out_path / "templates.npy", templates)

    # Compute similarity matrix
    flat_t = templates.reshape(num_units, -1)
    norms = np.maximum(np.linalg.norm(flat_t, axis=1, keepdims=True), 1e-9)
    sim_matrix = (flat_t / norms) @ (flat_t / norms).T
    np.save(out_path / "similar_templates.npy", sim_matrix.astype(np.float32))

    # Write PC features for Phy2
    if features.shape[0] > 0:
        n_spk = features.shape[0]
        pc_feats = features.reshape(n_spk, k_neighbors, 12).transpose(0, 2, 1).astype(np.float32)
        np.save(out_path / "pc_features.npy", pc_feats)
        pc_feature_ind = np.tile(np.arange(k_neighbors, dtype=np.int32), (num_units, 1))
        np.save(out_path / "pc_feature_ind.npy", pc_feature_ind)

    # Calculate PNR & CoV-ISI quality labels
    with open(out_path / "cluster_group.tsv", "w") as fg, open(out_path / "cluster_info.tsv", "w") as fi:
        fg.write("cluster_id\tKSLabel\n")
        fi.write("cluster_id\tprimary_channel\tspikes\tpeak_uv\tPNR_dB\tCoV_ISI\tKSLabel\n")
        for u_info in units:
            u_id = u_info['unit_id']
            pri_ch = u_info['primary_ch']
            spk_count = len(u_info['indices'])
            peak_uv = float(abs(np.min(templates[u_id, :, pri_ch]))) if templates.shape[0] > u_id else 0.0

            # Inter-Spike Intervals
            u_times = np.sort([spike_times[idx] for idx in u_info['indices']])
            if len(u_times) >= 5:
                isis = np.diff(u_times) / fs
                cov_isi = float(np.std(isis) / np.mean(isis)) if np.mean(isis) > 0 else 1.0
            else:
                cov_isi = 1.0

            # Estimated Pulse-to-Noise Ratio (PNR)
            noise_sd = float(np.std(preprocessed[pri_ch, :min(actual_samples, int(fs))]))
            pnr_db = float(20.0 * np.log10(max(peak_uv / max(noise_sd, 1e-3), 1.0)))

            # EMUsort single motor unit quality standard: PNR >= 20.0 dB and CoV_ISI <= 0.35
            is_single_unit = pnr_db >= 20.0 and cov_isi <= 0.35 and spk_count >= 30
            label_str = "good" if is_single_unit else "mua"

            fg.write(f"{u_id}\t{label_str}\n")
            fi.write(f"{u_id}\t{pri_ch}\t{spk_count}\t{peak_uv:.1f}\t{pnr_db:.1f}\t{cov_isi:.3f}\t{label_str}\n")
            u_info['label'] = label_str
            u_info['peak_uv'] = peak_uv
            u_info['pnr_db'] = pnr_db
            u_info['cov_isi'] = cov_isi

    num_good = sum(1 for u in units if u.get('label') == 'good')
    num_mua = num_units - num_good

    # Continuous raw dat buffer for Phy2 TraceView
    if export_raw_dat:
        dat_file = out_path / "recording.dat"
        print(f"      Writing raw trace buffer: {dat_file.name} ({actual_samples:,} samples x {n_ch} ch)...")
        int16_dat = np.clip(preprocessed, -32768, 32767).astype(np.int16)
        int16_dat.T.tofile(dat_file)

    # 9. Generate High-Resolution Motor Unit Template Grid
    grid_path = out_path / "emusort_templates_grid.png"
    if HAS_PLT and num_units > 0:
        print(f"[9/9] Generating EMUsort motor unit template grid image ({num_units} units)...")
        cols = min(num_units, 6)
        rows = math.ceil(num_units / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.8), sharex=True, sharey=True)
        if rows * cols == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()

        time_ms = (np.arange(150) - 50) / (fs * 1e-3)

        for u_idx, u_info in enumerate(units):
            ax = axes[u_idx]
            u_id = u_info['unit_id']
            pri_ch = u_info['primary_ch']
            pri_wf = templates[u_id, :, pri_ch]
            label = u_info.get('label', 'mua')
            color = "#10b981" if label == "good" else "#64748b"  # green for good, slate for mua

            ax.plot(time_ms, pri_wf, color=color, lw=1.5)
            spk_cnt = len(u_info['indices'])
            pnr = u_info.get('pnr_db', 0.0)
            cov = u_info.get('cov_isi', 1.0)
            ax.set_title(
                f"MU {u_id} ({label[0].upper()}) ch{pri_ch}\n{spk_cnt} spk | {pnr:.1f}dB | CoV {cov:.2f}",
                fontsize=8,
                color=color,
                pad=2,
            )
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_color("#cbd5e1")
                spine.set_linewidth(0.8)

        for u_idx in range(num_units, len(axes)):
            axes[u_idx].axis("off")

        plt.suptitle(
            f"EMUsort HD-EMG Motor Unit Templates (N={num_units} units: {num_good} good, {num_mua} mua)",
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout()
        plt.savefig(grid_path, dpi=120)
        plt.close(fig)
        print(f"      Saved: {grid_path} ({grid_path.stat().st_size:,} bytes)")

    total_time = time.perf_counter() - t0
    print("-" * 80)
    print(f"  EMUSORT SPIKE SORTING COMPLETE IN {total_time:.2f} s")
    print(f"  Total Motor Units:      {num_units} ({num_good} good, {num_mua} mua)")
    print(f"  Phy2 Output Directory:  {out_path}")
    if grid_path.exists():
        print(f"  Template Grid Image:    {grid_path}")
    print("=" * 80)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Run EMUsort on NWB Zarr recording and export to Phy2")
    parser.add_argument("--nwb-path", type=str, default="data/nwb/15-25-33_meps.nwb.zarr")
    parser.add_argument("--series", type=str, default="/acquisition/HDEMG")
    parser.add_argument("--output-dir", type=str, default="data/sorters/phy_emusort_output")
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=600.0)
    parser.add_argument("--threshold-sigma", type=float, default=6.5)
    parser.add_argument("--refractory-ms", type=float, default=2.5)
    parser.add_argument("--ied-mm", type=float, default=4.0)
    parser.add_argument("--no-align-conduction", action="store_true")
    parser.add_argument("--no-dat", action="store_true")
    args = parser.parse_args()

    run_emusort_nwb_to_phy(
        nwb_path=args.nwb_path,
        series=args.series,
        output_dir=args.output_dir,
        start_sec=args.start_sec,
        duration_sec=args.duration_sec,
        threshold_sigma=args.threshold_sigma,
        refractory_ms=args.refractory_ms,
        ied_mm=args.ied_mm,
        align_conduction_latency=not args.no_align_conduction,
        export_raw_dat=not args.no_dat,
    )


if __name__ == "__main__":
    main()
