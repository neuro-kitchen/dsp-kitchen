#!/usr/bin/env python3
"""
Spike Sorting Pipeline Evaluation & Diagnostics
================================================
Benchmarks dsp-synapse classical electrophysiology algorithms against the
MEArec Neuropixels/Neuronexus 32-channel ground-truth dataset.

Steps Evaluated:
1. Robust Noise Standard Deviation Estimation (Quiroga MAD)
2. Multi-channel Threshold Crossing Detection (-5 * sigma_n)
3. Spatial Deduplication (spatial radius & temporal window)
4. Sub-sample Continuous Sinc Realignment (Lanczos/Sinc fractional shift)
5. Multi-channel K-NN Snippet Extraction (K=7 nearest neighbors)
6. Waveform PCA Feature Projection (PC1, PC2, PC3)
7. Inter-Spike Interval (ISI) & Refractory Violation Analysis (<1.5 ms)
8. Ground-Truth Accuracy Scorecard (Sensitivity, FDR, Timing Jitter)
"""

import os
import sys
import time
import json
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import dsp_kitchen as dsp
import dsp_kitchen.synapse as syn
import dsp_kitchen.linalg as linalg

# Output paths
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(FIG_DIR, exist_ok=True)

# Plot styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 200,
    "grid.alpha": 0.3,
})

def main():
    print("=" * 80)
    print("  dsp-synapse: Classical Electrophysiology Evaluation on MEArec Ground Truth")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 0. Load Dataset & Metadata
    # -------------------------------------------------------------------------
    bin_path = os.path.join(DATA_DIR, "mearec_32ch_10s.bin")
    meta_path = os.path.join(DATA_DIR, "mearec_32ch_10s.meta")
    gt_path = os.path.join(DATA_DIR, "mearec_32ch_ground_truth.json")
    h5_path = os.path.join(DATA_DIR, "mearec_test_10s.h5")

    with open(meta_path) as f:
        meta = json.load(f)
    with open(gt_path) as f:
        gt_data = json.load(f)
    with h5py.File(h5_path, "r") as f:
        pos_3d = f["channel_positions"][:]

    num_channels = meta["channels"]
    num_samples = meta["samples"]
    fs = meta["sample_rate_hz"]
    duration = meta["duration_seconds"]

    print(f"\n[Dataset Metadata]")
    print(f"  Channels:         {num_channels}")
    print(f"  Samples:          {num_samples:,} ({duration:.1f} s @ {fs:,.0f} Hz)")
    print(f"  Probe:            {meta['probe']}")
    print(f"  Ground-Truth:     {len(gt_data)} units")

    # Load binary raw recording [channels, samples]
    raw_data = np.fromfile(bin_path, dtype=np.float32).reshape((num_channels, num_samples))
    print(f"  Signal range:     [{raw_data.min():.1f}, {raw_data.max():.1f}] uV (std: {raw_data.std():.1f} uV)")

    # Build Probe Layout from 2D coordinates (y, z) in um
    probe_coords = [(float(p[1]), float(p[2])) for p in pos_3d]
    probe = syn.custom_layout("Neuronexus-32", probe_coords)
    print(f"  Probe layout:     {probe.total_channels} active recording sites")

    # Flatten ground-truth spikes: (sample_index, unit_id)
    gt_spikes = []
    for uid, udata in gt_data.items():
        for s in udata["sample_indices"]:
            gt_spikes.append((s, int(uid)))
    gt_spikes.sort(key=lambda x: x[0])
    total_gt = len(gt_spikes)
    print(f"  Total GT spikes:  {total_gt}")

    # -------------------------------------------------------------------------
    # 1. Step 1: Robust Noise Estimation (Quiroga MAD)
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    sigmas = np.array([syn.estimate_noise(raw_data[ch, :]) for ch in range(num_channels)])
    t_noise = (time.perf_counter() - t0) * 1000.0

    print(f"\n[Step 1: Noise Estimation (sigma_n = median(|x|) / 0.6745)]")
    print(f"  Mean sigma_n:     {np.mean(sigmas):.2f} uV (min: {np.min(sigmas):.2f}, max: {np.max(sigmas):.2f})")
    print(f"  Elapsed time:     {t_noise:.2f} ms ({num_samples * num_channels / (t_noise * 1e-3) / 1e6:.1f} MSamples/s)")

    # -------------------------------------------------------------------------
    # 2. Step 2: Multi-channel Threshold Crossing Detection (-5 * sigma_n)
    # -------------------------------------------------------------------------
    threshold_factor = 5.0
    refractory_samples = int(0.0008 * fs)  # 0.8 ms = 25.6 samples
    t0 = time.perf_counter()
    raw_spikes = syn.detect_spikes(
        raw_data,
        threshold_factor=threshold_factor,
        refractory_samples=refractory_samples,
    )
    t_detect = (time.perf_counter() - t0) * 1000.0

    print(f"\n[Step 2: Threshold Crossing Detection (-{threshold_factor} * sigma_n)]")
    print(f"  Crossings found:  {len(raw_spikes):,} across all channels")
    print(f"  Elapsed time:     {t_detect:.2f} ms ({num_samples * num_channels / (t_detect * 1e-3) / 1e6:.1f} MSamples/s)")

    # -------------------------------------------------------------------------
    # 3. Step 3: Spatial Deduplication
    # -------------------------------------------------------------------------
    dedup_radius_um = 60.0
    dedup_window_samples = int(0.0008 * fs)  # 25 samples
    t0 = time.perf_counter()
    dedup_spikes = syn.deduplicate_spikes(
        raw_spikes,
        probe,
        radius_um=dedup_radius_um,
        window_samples=dedup_window_samples,
    )
    t_dedup = (time.perf_counter() - t0) * 1000.0

    print(f"\n[Step 3: Spatial Deduplication (radius={dedup_radius_um} um, window={dedup_window_samples} samples)]")
    print(f"  Deduplicated:     {len(dedup_spikes):,} discrete action potentials")
    print(f"  Compression:      {len(raw_spikes) / len(dedup_spikes):.2f}x multi-channel collapse")
    print(f"  Elapsed time:     {t_dedup:.2f} ms")

    # -------------------------------------------------------------------------
    # 4. Step 4 & 5: Multi-channel Snippet Extraction & Sinc Realignment
    # -------------------------------------------------------------------------
    k_neighbors = 7
    pre_samples = 20
    post_samples = 40
    snippet_len = pre_samples + post_samples

    # Extract WITHOUT sinc shift (raw integer alignment)
    t0 = time.perf_counter()
    snippets_raw = syn.extract_snippets(
        raw_data,
        dedup_spikes,
        probe,
        k_neighbors=k_neighbors,
        pre_samples=pre_samples,
        post_samples=post_samples,
        apply_sinc_shift=False,
    )
    # Extract WITH sinc shift (sub-sample continuous realignment)
    snippets_sinc = syn.extract_snippets(
        raw_data,
        dedup_spikes,
        probe,
        k_neighbors=k_neighbors,
        pre_samples=pre_samples,
        post_samples=post_samples,
        apply_sinc_shift=True,
    )
    t_extract = (time.perf_counter() - t0) * 1000.0

    print(f"\n[Step 4 & 5: Multi-channel K-NN Extraction & Sinc Realignment]")
    print(f"  Extracted:        {len(snippets_sinc):,} waveforms ({k_neighbors} channels x {snippet_len} samples)")
    subsample_offsets = [s.subsample_offset for s in snippets_sinc]
    print(f"  Sub-sample offset: mean={np.mean(subsample_offsets):.3f}, std={np.std(subsample_offsets):.3f} samples")
    print(f"  Elapsed time:     {t_extract:.2f} ms ({len(snippets_sinc) / (t_extract * 1e-3):,.0f} waveforms/s)")

    # -------------------------------------------------------------------------
    # 5. Ground Truth Matching & Accuracy Evaluation
    # -------------------------------------------------------------------------
    tol_samples = int(0.0005 * fs)  # 0.5 ms tolerance = 16 samples
    gt_sample_arr = np.array([s[0] for s in gt_spikes])
    gt_unit_arr = np.array([s[1] for s in gt_spikes])
    det_sample_arr = np.array([s.sample_index for s in dedup_spikes])

    matched_gt_indices = {}
    matched_det_indices = {}
    timing_errors = []
    spike_unit_labels = np.full(len(dedup_spikes), -1, dtype=int)  # -1 = False positive / noise

    for i, ds in enumerate(det_sample_arr):
        diffs = np.abs(gt_sample_arr - ds)
        min_idx = np.argmin(diffs)
        if diffs[min_idx] <= tol_samples:
            if min_idx not in matched_gt_indices:
                matched_gt_indices[min_idx] = i
                matched_det_indices[i] = min_idx
                timing_errors.append(ds - gt_sample_arr[min_idx])
                spike_unit_labels[i] = gt_unit_arr[min_idx]

    tp_count = len(matched_gt_indices)
    sensitivity = (tp_count / total_gt) * 100.0
    fp_count = len(dedup_spikes) - len(matched_det_indices)
    fdr = (fp_count / len(dedup_spikes)) * 100.0
    timing_errors = np.array(timing_errors)
    mean_jitter_ms = np.mean(timing_errors) / (fs / 1000.0)
    std_jitter_ms = np.std(timing_errors) / (fs / 1000.0)

    print(f"\n[Ground-Truth Accuracy Benchmark (Tolerance: +-0.5 ms)]")
    print(f"  Total Ground Truth:  {total_gt}")
    print(f"  Detected Events:     {len(dedup_spikes)}")
    print(f"  True Positives (TP): {tp_count} / {total_gt} ({sensitivity:.1f}% sensitivity)")
    print(f"  False Positives (FP):{fp_count} ({fdr:.1f}% False Discovery Rate)")
    print(f"  Timing Jitter:       {mean_jitter_ms:+.3f} +- {std_jitter_ms:.3f} ms")

    # Unit-by-unit sensitivity breakdown
    unit_stats = {}
    for uid in range(len(gt_data)):
        u_gt = np.sum(gt_unit_arr == uid)
        u_tp = sum(1 for m_idx in matched_gt_indices.keys() if gt_unit_arr[m_idx] == uid)
        u_sens = (u_tp / u_gt) * 100.0 if u_gt > 0 else 0.0
        unit_stats[uid] = {"gt": u_gt, "tp": u_tp, "sens": u_sens}
        print(f"    Unit {uid:2d}: {u_tp:3d} / {u_gt:3d} TP ({u_sens:5.1f}% sensitivity)")

    # -------------------------------------------------------------------------
    # 6. Step 6: Waveform PCA Feature Space
    # -------------------------------------------------------------------------
    # Prepare primary channel waveform matrix [features, samples] = [snippet_len, N]
    waveform_matrix = np.array([s.waveform()[0, :] for s in snippets_sinc], dtype=np.float32).T
    pca = linalg.PCA(3)
    pca.fit(waveform_matrix)
    pcs = pca.transform(waveform_matrix)  # shape [3, N]
    var_exp = pca.explained_variance_ratio

    print(f"\n[Step 6: Principal Component Analysis (PCA)]")
    print(f"  Features shape:      {waveform_matrix.shape} ([features, spikes])")
    print(f"  Variance explained:  PC1: {var_exp[0]*100:.1f}%, PC2: {var_exp[1]*100:.1f}%, PC3: {var_exp[2]*100:.1f}% (Total: {np.sum(var_exp)*100:.1f}%)")

    # -------------------------------------------------------------------------
    # 7. Step 7: ISI & Refractory Violations
    # -------------------------------------------------------------------------
    print(f"\n[Step 7: Inter-Spike Intervals (ISI) & Refractory Violations (<1.5 ms)]")
    unit_isi_violations = {}
    for uid in range(len(gt_data)):
        unit_det_spikes = det_sample_arr[spike_unit_labels == uid]
        if len(unit_det_spikes) >= 2:
            isi_dict = syn.compute_isi(unit_det_spikes.tolist(), sample_rate_hz=fs, refractory_ms=1.5)
            unit_isi_violations[uid] = isi_dict
            print(f"    Unit {uid:2d}: {isi_dict['total_spikes']:3d} spikes, {isi_dict['violation_count']} violations ({isi_dict['violation_rate_pct']:.2f}%), firing rate: {isi_dict['firing_rate_hz']:.1f} Hz")

    # -------------------------------------------------------------------------
    # 8. Matplotlib Diagnostic Figures Generation
    # -------------------------------------------------------------------------
    print(f"\n[Generating Diagnostic Visualizations in {FIG_DIR}]")

    # -------------------------------------------------------------------------
    # Figure 1: Continuous Detection Traces Overlay
    # -------------------------------------------------------------------------
    fig1, axes1 = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    t_start_s = 0.50
    t_end_s = 0.65
    s_start = int(t_start_s * fs)
    s_end = int(t_end_s * fs)
    time_ms = np.linspace(0, (t_end_s - t_start_s) * 1000.0, s_end - s_start)

    # Pick 4 prominent channels showing clear action potentials
    selected_channels = [2, 5, 9, 24]
    for idx, ch in enumerate(selected_channels):
        ax = axes1[idx]
        trace = raw_data[ch, s_start:s_end]
        thresh = -threshold_factor * sigmas[ch]
        ax.plot(time_ms, trace, color="#1f77b4", lw=1.0, label="Filtered Signal" if idx == 0 else "")
        ax.axhline(thresh, color="#d62728", ls="--", lw=1.2, label=r"$-5\sigma_n$ Threshold" if idx == 0 else "")

        # Mark primary detected spikes on this channel
        primary_dets = [s for s in dedup_spikes if s.primary_channel == ch and s_start <= s.sample_index < s_end]
        for s in primary_dets:
            t_rel_ms = (s.sample_index - s_start) / fs * 1000.0
            val = raw_data[ch, s.sample_index]
            ax.scatter(t_rel_ms, val, color="#2ca02c", marker="o", s=70, zorder=5, label="Primary Channel AP" if idx == 0 and s == primary_dets[0] else "")

        # Mark participating co-detected spikes on this channel
        part_dets = [s for s in dedup_spikes if s.primary_channel != ch and ch in s.participating_channels and s_start <= s.sample_index < s_end]
        for s in part_dets:
            t_rel_ms = (s.sample_index - s_start) / fs * 1000.0
            val = raw_data[ch, s.sample_index]
            ax.scatter(t_rel_ms, val, color="#ff7f0e", marker="^", s=50, zorder=4, label="Participating AP" if idx == 0 and s == part_dets[0] else "")

        # Mark ground-truth spikes
        gt_in_win = [s[0] for s in gt_spikes if s_start <= s[0] < s_end]
        for gs in gt_in_win:
            t_rel_ms = (gs - s_start) / fs * 1000.0
            ax.axvline(t_rel_ms, color="#9467bd", ls=":", alpha=0.6, lw=1.5, label="Ground-Truth Spike" if idx == 0 and gs == gt_in_win[0] else "")

        ax.set_ylabel(f"Ch {ch}\n(uV)")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="upper right", framealpha=0.9, fontsize=8, ncol=4)

    axes1[-1].set_xlabel("Time (ms)")
    fig1.suptitle("1. Multi-Channel Spike Detection & Ground-Truth Overlay (150 ms Window)", fontweight="bold")
    fig1.tight_layout()
    fpath1 = os.path.join(FIG_DIR, "01_detection_overlay.png")
    fig1.savefig(fpath1)
    plt.close(fig1)
    print(f"  [+] Saved {os.path.basename(fpath1)}")

    # -------------------------------------------------------------------------
    # Figure 2: Sub-sample Sinc Realignment Before / After
    # -------------------------------------------------------------------------
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    time_waveform_us = (np.arange(snippet_len) - pre_samples) / fs * 1e6  # in microseconds

    # Pick spikes from a dominant unit
    target_unit = 0
    unit_spike_indices = np.where(spike_unit_labels == target_unit)[0][:40]

    # Panel A: Before sinc realignment (Integer sampled)
    for idx in unit_spike_indices:
        wf = snippets_raw[idx].waveform()[0, :]
        ax2a.plot(time_waveform_us, wf, color="#1f77b4", alpha=0.35, lw=1.2)
    mean_raw = np.mean([snippets_raw[i].waveform()[0, :] for i in unit_spike_indices], axis=0)
    ax2a.plot(time_waveform_us, mean_raw, color="#0b3c61", lw=2.5, label="Mean Template")
    ax2a.axvline(0, color="gray", ls="--", alpha=0.6)
    ax2a.set_title("Before Realignment (Integer Peak Jitter)", fontweight="bold")
    ax2a.set_xlabel(r"Time Relative to Peak ($\mu$s)")
    ax2a.set_ylabel("Amplitude (uV)")
    ax2a.legend(loc="lower right")
    ax2a.grid(True, alpha=0.3)

    # Panel B: After sinc realignment
    for idx in unit_spike_indices:
        wf = snippets_sinc[idx].waveform()[0, :]
        ax2b.plot(time_waveform_us, wf, color="#2ca02c", alpha=0.35, lw=1.2)
    mean_sinc = np.mean([snippets_sinc[i].waveform()[0, :] for i in unit_spike_indices], axis=0)
    ax2b.plot(time_waveform_us, mean_sinc, color="#0f5117", lw=2.5, label="Mean Template")
    ax2b.axvline(0, color="gray", ls="--", alpha=0.6)
    ax2b.set_title("After Sub-Sample Sinc Shift (Fractional Realignment)", fontweight="bold")
    ax2b.set_xlabel(r"Time Relative to Peak ($\mu$s)")
    ax2b.legend(loc="lower right")
    ax2b.grid(True, alpha=0.3)

    fig2.suptitle(f"2. Sub-Sample Fractional Sinc Realignment (Unit {target_unit}, N={len(unit_spike_indices)} Spikes)", fontweight="bold")
    fig2.tight_layout()
    fpath2 = os.path.join(FIG_DIR, "02_alignment_before_after.png")
    fig2.savefig(fpath2)
    plt.close(fig2)
    print(f"  [+] Saved {os.path.basename(fpath2)}")

    # -------------------------------------------------------------------------
    # Figure 3: Multi-channel Waveform Templates Across K Nearest Neighbors
    # -------------------------------------------------------------------------
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
    axes3 = axes3.flatten()
    demo_units = [0, 1, 2, 3]

    for p_idx, uid in enumerate(demo_units):
        ax = axes3[p_idx]
        u_indices = np.where(spike_unit_labels == uid)[0]
        if len(u_indices) == 0:
            continue
        u_snippets = [snippets_sinc[i] for i in u_indices]
        tmpl_res = syn.compute_template(u_snippets)
        if tmpl_res is None:
            continue
        mean_tmpl = tmpl_res["mean"]  # [k_neighbors, snippet_len]
        std_tmpl = tmpl_res["std"]

        # Plot K neighbor waveforms stacked vertically with spatial offsets
        k_ch = mean_tmpl.shape[0]
        v_offset = 60.0  # uV vertical separation
        for k in range(k_ch):
            y_base = -k * v_offset
            m = mean_tmpl[k, :] + y_base
            s = std_tmpl[k, :]
            ch_name = f"Ch {u_snippets[0].channel_ids[k]}" + (" (Primary)" if k == 0 else "")
            color = "#d62728" if k == 0 else "#1f77b4"
            ax.plot(time_waveform_us, m, color=color, lw=1.8, label=ch_name if k < 3 else "")
            ax.fill_between(time_waveform_us, m - s, m + s, color=color, alpha=0.15)
            ax.text(time_waveform_us[-1] + 15, y_base, f"Ch {u_snippets[0].channel_ids[k]}", va="center", fontsize=8)

        ax.set_title(f"Unit {uid} (N={len(u_indices)} Spikes)", fontweight="bold")
        ax.set_xlabel(r"Time ($\mu$s)")
        ax.set_ylabel(r"Electrode Potential ($\mu$V + spatial offset)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(time_waveform_us[0] - 20, time_waveform_us[-1] + 150)

    fig3.suptitle("3. Multi-Channel Waveform Footprints Across K-NN Electrodes (Mean +- 1 Std)", fontweight="bold")
    fig3.tight_layout()
    fpath3 = os.path.join(FIG_DIR, "03_multichannel_templates.png")
    fig3.savefig(fpath3)
    plt.close(fig3)
    print(f"  [+] Saved {os.path.basename(fpath3)}")

    # -------------------------------------------------------------------------
    # Figure 4: PCA Feature Space Colored by Ground-Truth Unit
    # -------------------------------------------------------------------------
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.get_cmap("tab10")

    # Plot noise/unmatched in light gray
    noise_mask = (spike_unit_labels == -1)
    ax4a.scatter(pcs[0, noise_mask], pcs[1, noise_mask], c="#d0d0d0", s=15, alpha=0.5, label="Noise / Unmatched")
    ax4b.scatter(pcs[0, noise_mask], pcs[2, noise_mask], c="#d0d0d0", s=15, alpha=0.5, label="Noise / Unmatched")

    # Plot each ground truth unit
    for uid in range(len(gt_data)):
        u_mask = (spike_unit_labels == uid)
        if np.sum(u_mask) > 0:
            color = cmap(uid % 10)
            ax4a.scatter(pcs[0, u_mask], pcs[1, u_mask], color=color, s=25, alpha=0.8, label=f"Unit {uid}")
            ax4b.scatter(pcs[0, u_mask], pcs[2, u_mask], color=color, s=25, alpha=0.8, label=f"Unit {uid}")

    ax4a.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% Variance)")
    ax4a.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% Variance)")
    ax4a.set_title("Waveform Feature Space: PC1 vs PC2", fontweight="bold")
    ax4a.grid(True, alpha=0.3)
    ax4a.legend(loc="upper right", ncol=2, fontsize=8, framealpha=0.9)

    ax4b.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% Variance)")
    ax4b.set_ylabel(f"PC3 ({var_exp[2]*100:.1f}% Variance)")
    ax4b.set_title("Waveform Feature Space: PC1 vs PC3", fontweight="bold")
    ax4b.grid(True, alpha=0.3)
    ax4b.legend(loc="upper right", ncol=2, fontsize=8, framealpha=0.9)

    fig4.suptitle("4. Waveform Feature Projection (PCA) Colored by Ground-Truth Neuron ID", fontweight="bold")
    fig4.tight_layout()
    fpath4 = os.path.join(FIG_DIR, "04_pca_feature_space.png")
    fig4.savefig(fpath4)
    plt.close(fig4)
    print(f"  [+] Saved {os.path.basename(fpath4)}")

    # -------------------------------------------------------------------------
    # Figure 5: Inter-Spike Interval (ISI) Distributions & Refractory Period
    # -------------------------------------------------------------------------
    fig5, axes5 = plt.subplots(2, 3, figsize=(14, 8))
    axes5 = axes5.flatten()
    top_units = [0, 1, 2, 4, 7, 8]

    for p_idx, uid in enumerate(top_units):
        ax = axes5[p_idx]
        u_spikes = det_sample_arr[spike_unit_labels == uid]
        if len(u_spikes) >= 2:
            intervals_ms = np.diff(u_spikes) / (fs / 1000.0)
            bins = np.linspace(0, 50, 51)
            ax.hist(intervals_ms, bins=bins, color="#1f77b4", edgecolor="black", alpha=0.7)
            ax.axvline(1.5, color="#d62728", ls="--", lw=2, label="1.5 ms Refractory")

            v_count = np.sum(intervals_ms < 1.5)
            v_rate = (v_count / len(intervals_ms)) * 100.0 if len(intervals_ms) > 0 else 0.0
            ax.set_title(f"Unit {uid}: {v_count} Violations ({v_rate:.1f}%)", fontweight="bold")
            ax.set_xlabel("Inter-Spike Interval (ms)")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)

    fig5.suptitle("5. Inter-Spike Interval (ISI) Distributions & Refractory Boundary (1.5 ms)", fontweight="bold")
    fig5.tight_layout()
    fpath5 = os.path.join(FIG_DIR, "05_isi_distributions.png")
    fig5.savefig(fpath5)
    plt.close(fig5)
    print(f"  [+] Saved {os.path.basename(fpath5)}")

    # -------------------------------------------------------------------------
    # Figure 6: Comprehensive Accuracy Scorecard Dashboard
    # -------------------------------------------------------------------------
    fig6 = plt.figure(figsize=(14, 9))
    gs = fig6.add_gridspec(2, 2, height_ratios=[1.2, 1.0])

    # Panel A: Unit-by-Unit Spike Counts (GT vs Detected)
    ax6a = fig6.add_subplot(gs[0, 0])
    u_ids = list(range(len(gt_data)))
    gt_counts = [unit_stats[u]["gt"] for u in u_ids]
    tp_counts = [unit_stats[u]["tp"] for u in u_ids]
    x_pos = np.arange(len(u_ids))
    width = 0.38

    ax6a.bar(x_pos - width/2, gt_counts, width, label="Ground Truth Spikes", color="#4a7bb0")
    ax6a.bar(x_pos + width/2, tp_counts, width, label="Detected (True Positives)", color="#55a868")
    ax6a.set_xticks(x_pos)
    ax6a.set_xticklabels([f"U{u}" for u in u_ids])
    ax6a.set_xlabel("Neuron Unit ID")
    ax6a.set_ylabel("Spike Count")
    ax6a.set_title("A. Spike Yield: Ground-Truth vs True Positives", fontweight="bold")
    ax6a.legend(loc="upper right", fontsize=9)
    ax6a.grid(True, alpha=0.3)

    # Panel B: Sensitivity (%) by Unit
    ax6b = fig6.add_subplot(gs[0, 1])
    sensitivities = [unit_stats[u]["sens"] for u in u_ids]
    bars = ax6b.bar(x_pos, sensitivities, color="#3470a3", edgecolor="black", alpha=0.85)
    ax6b.axhline(100.0, color="gray", ls=":", alpha=0.7)
    ax6b.axhline(sensitivity, color="#d62728", ls="--", lw=1.5, label=f"Overall Sensitivity: {sensitivity:.1f}%")
    ax6b.set_xticks(x_pos)
    ax6b.set_xticklabels([f"U{u}" for u in u_ids])
    ax6b.set_ylim(0, 115)
    ax6b.set_xlabel("Neuron Unit ID")
    ax6b.set_ylabel("Sensitivity / True Positive Rate (%)")
    ax6b.set_title("B. Unit-by-Unit Detection Sensitivity", fontweight="bold")
    ax6b.legend(loc="lower right", fontsize=9)
    ax6b.grid(True, alpha=0.3)

    for bar, val in zip(bars, sensitivities):
        ax6b.text(bar.get_x() + bar.get_width()/2, val + 2.0, f"{val:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Panel C: Sub-sample Timing Error Jitter
    ax6c = fig6.add_subplot(gs[1, 0])
    jitter_ms = timing_errors / (fs / 1000.0)
    ax6c.hist(jitter_ms, bins=35, color="#8172b3", edgecolor="black", alpha=0.8)
    ax6c.axvline(0, color="red", ls="--", lw=1.5)
    ax6c.set_xlabel("Timing Error: Detected - Ground Truth (ms)")
    ax6c.set_ylabel("Spike Count")
    ax6c.set_title(f"C. Peak Timing Error ({mean_jitter_ms:+.3f} +- {std_jitter_ms:.3f} ms)", fontweight="bold")
    ax6c.grid(True, alpha=0.3)

    # Panel D: Performance Summary Table
    ax6d = fig6.add_subplot(gs[1, 1])
    ax6d.axis("off")
    table_data = [
        ["Total Ground Truth Spikes", f"{total_gt:,}"],
        ["Total Detected & Deduped Events", f"{len(dedup_spikes):,}"],
        ["Matched True Positives (TP)", f"{tp_count:,}"],
        ["Overall Sensitivity (TPR)", f"{sensitivity:.1f}%"],
        ["False Discovery Rate (FDR)", f"{fdr:.1f}%"],
        ["Mean Timing Jitter", f"{mean_jitter_ms:+.3f} ms ({np.mean(timing_errors):.2f} samples)"],
        ["Refractory Violations (<1.5 ms)", f"{sum(v['violation_count'] for v in unit_isi_violations.values())} total across all units"],
        ["Processing Speed", f"{num_samples * num_channels / ((t_noise + t_detect + t_dedup + t_extract) * 1e-3) / 1e6:.1f} MSamples/sec"],
    ]
    tbl = ax6d.table(
        cellText=table_data,
        colLabels=["Metric / Benchmark", "Result"],
        loc="center",
        cellLoc="left",
        colWidths=[0.65, 0.35],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.4)
    # Style table headers
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2b5c8f")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f8f9fa" if row % 2 == 0 else "#ffffff")

    ax6d.set_title("D. Ground-Truth Validation Scorecard", fontweight="bold", pad=10)

    fig6.suptitle("6. Spike Sorting Pipeline Ground-Truth Benchmark & Accuracy Scorecard", fontweight="bold", fontsize=15)
    fig6.tight_layout()
    fpath6 = os.path.join(FIG_DIR, "06_accuracy_scorecard.png")
    fig6.savefig(fpath6)
    plt.close(fig6)
    print(f"  [+] Saved {os.path.basename(fpath6)}")

    print("\n" + "=" * 80)
    print(f"  SUCCESS: All 6 Diagnostic Figures Successfully Generated in: {FIG_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
