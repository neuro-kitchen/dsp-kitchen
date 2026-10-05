# %% [markdown]
# # dsp-kitchen (`dsp-synapse-ml`): Advanced Deep Learning & Burn-ONNX Spike Sorting
# Interactive notebook demonstrating all neural spike sorting stages in `dsp_kitchen.synapse.ml` and `dsp_kitchen.synapse.onnx`:
# 1. **Neural Waveform Denoising (`[N, K, T] -> [N, K, T]`)**:
#    - `SpatiotemporalUnetDenoiser`: Multi-channel 1D Spatio-Temporal UNet with `.safetensors` weight persistence
#    - `SingleChannelDenoiser`: Per-channel 1D Conv residual denoiser
# 2. **Deep Latent Feature Embedding (`[N, K, T] -> [N, D]`)**:
#    - `DartsortVaeEmbedder`: DARTsort-style Variational Autoencoder latent representation
#    - `ContrastiveWaveformEmbedder`: SimCLR / CEBRA-style unit-hypersphere representation ($\|\mathbf{z}\|_2 = 1$)
#    - Comparison against classical `PCA`
# 3. **Automated Allen/IBL Unit Quality Curation (`[N, 8] -> SingleUnit | MultiUnit | Noise`)**:
#    - `UnitQualityClassifier`
# 4. **External Sorter Bridge via Burn-ONNX (`OnnxModelRunner`)**:
#    - Running `.onnx` graphs with `"kilosort4"`, `"dartsort"`, and `"cebra"` profiles

# %% [1] Imports & Synthesize Multi-Unit 3D Waveform Snippets `[N, K, T]`
import tempfile
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_PLT = True
except ImportError:
    HAS_PLT = False

import dsp_kitchen as dk
import dsp_kitchen.synapse as syn
from dsp_kitchen.synapse.ml import (
    ContrastiveWaveformEmbedder,
    DartsortVaeEmbedder,
    SingleChannelDenoiser,
    SpatiotemporalUnetDenoiser,
    UnitQualityClassifier,
)
from dsp_kitchen.synapse.onnx import OnnxModelRunner
from dsp_kitchen.linalg import PCA

rng = np.random.default_rng(2026)

# Generate N = 90 multi-channel snippets across K = 4 neighboring channels, T = 40 samples (1.33 ms @ 30 kHz)
# belonging to 3 distinct neural units (fast-spiking interneuron, regular-spiking pyramidal, wide triphasic unit)
n_per_unit = 30
n_units = 3
num_spikes = n_per_unit * n_units
k_channels = 4
t_samples = 40
t_axis = np.linspace(-1.0, 1.5, t_samples, dtype=np.float32)

# Unit archetypes on primary channel + spatial decay across K=4 neighbors
unit_archetypes = [
    # Unit 0: Sharp fast-spiking interneuron
    -160.0 * np.exp(-((t_axis / 0.18) ** 2))
    + 75.0 * np.exp(-(((t_axis - 0.32) / 0.22) ** 2)),
    # Unit 1: Broad regular-spiking pyramidal cell
    -120.0 * np.exp(-((t_axis / 0.32) ** 2))
    + 45.0 * np.exp(-(((t_axis - 0.65) / 0.45) ** 2)),
    # Unit 2: Triphasic axonal / high-amplitude unit
    35.0 * np.exp(-(((t_axis + 0.30) / 0.18) ** 2))
    - 210.0 * np.exp(-((t_axis / 0.22) ** 2))
    + 60.0 * np.exp(-(((t_axis - 0.40) / 0.30) ** 2)),
]
spatial_profiles = [
    np.array([1.0, 0.65, 0.35, 0.15], dtype=np.float32),
    np.array([0.25, 1.0, 0.70, 0.30], dtype=np.float32),
    np.array([0.15, 0.40, 0.80, 1.00], dtype=np.float32),
]

clean_snippets = np.zeros((num_spikes, k_channels, t_samples), dtype=np.float32)
true_labels = np.zeros(num_spikes, dtype=np.int32)

for u in range(n_units):
    for i in range(n_per_unit):
        idx = u * n_per_unit + i
        amp_jitter = float(rng.uniform(0.85, 1.15))
        clean_snippets[idx] = amp_jitter * np.outer(
            spatial_profiles[u], unit_archetypes[u]
        )
        true_labels[idx] = u

# Add realistic background recording noise (sigma = 18 uV)
noisy_snippets = clean_snippets + rng.normal(
    0.0, 18.0, size=clean_snippets.shape
).astype(np.float32)

print(
    f"Prepared 3D Snippet Batch: shape={noisy_snippets.shape} [N, K, T], dtype={noisy_snippets.dtype}"
)

# %% [2] Stage 2: Neural Waveform Denoising (`SpatiotemporalUnetDenoiser` & `SingleChannelDenoiser`)
unet_denoiser = SpatiotemporalUnetDenoiser(
    num_channels=k_channels,
    num_samples=t_samples,
    base_filters=8,
    seed=42,
)
sc_denoiser = SingleChannelDenoiser(hidden_channels=16, seed=42)

denoised_unet = unet_denoiser.denoise(noisy_snippets)
denoised_sc = sc_denoiser.denoise(noisy_snippets)

# Demonstrate zero-copy `.safetensors` checkpoint export & reload
with tempfile.TemporaryDirectory() as tmpdir:
    weights_path = str(Path(tmpdir) / "spatiotemporal_unet.safetensors")
    unet_denoiser.save_safetensors(weights_path)
    unet_denoiser.load_safetensors(weights_path)
    reloaded_out = unet_denoiser.denoise(noisy_snippets[:2])
    assert np.allclose(denoised_unet[:2], reloaded_out, atol=1e-5)

print("[Stage 2: Neural Waveform Denoisers]")
print(f"  Noisy Input Shape:               {noisy_snippets.shape}")
print(f"  SpatiotemporalUnetDenoiser Out:  {denoised_unet.shape}")
print(f"  SingleChannelDenoiser Out:       {denoised_sc.shape}")
print(f"  Safetensors save/load verified:  OK")

# %% [3] Stage 3: Deep Latent Embeddings (`DartsortVaeEmbedder` & `ContrastiveWaveformEmbedder` vs. `PCA`)
latent_dim = 6

# 1. DARTsort-style Variational Autoencoder (VAE)
vae_embedder = DartsortVaeEmbedder(
    num_channels=k_channels,
    num_samples=t_samples,
    latent_dim=latent_dim,
    seed=42,
)
z_vae = vae_embedder.embed(noisy_snippets)  # [N, latent_dim]

# 2. Contrastive SimCLR / CEBRA-style Waveform Embedder (L2-normalized on unit hypersphere)
contrastive_embedder = ContrastiveWaveformEmbedder(
    num_channels=k_channels,
    proj_dim=latent_dim,
    seed=42,
)
z_contrastive = contrastive_embedder.embed(noisy_snippets)  # [N, latent_dim]
norms = np.linalg.norm(z_contrastive, axis=1)

# 3. Classical PCA baseline on flattened [K*T, N] waveforms
flat_for_pca = noisy_snippets.reshape(num_spikes, -1).T.astype(np.float32)
pca = PCA(n_components=latent_dim)
pca.fit(flat_for_pca)
z_pca = pca.transform(flat_for_pca, use_gpu=True).T  # [N, latent_dim]

print("\n[Stage 3: Latent Feature Embeddings]")
print(f"  DartsortVaeEmbedder shape:         {z_vae.shape}")
print(
    f"  ContrastiveWaveformEmbedder shape: {z_contrastive.shape} (mean L2 norm = {norms.mean():.4f} ± {norms.std():.4e})"
)
print(f"  Classical PCA shape:               {z_pca.shape}")

# %% [4] Stage 4: Automated Allen/IBL Unit Quality Curation (`UnitQualityClassifier`)
# Feature columns:
#   [snr, isi_viol_pct, firing_rate_hz, amp_cutoff, presence_ratio, half_width_ms, trough_to_peak_ms, repol_slope]
unit_features = np.array(
    [
        # Unit 0: High-SNR, zero ISI violations, high presence -> SingleUnit (SUA)
        [11.2, 0.02, 18.5, 0.004, 0.99, 0.16, 0.35, 92.0],
        # Unit 1: Good SNR regular-spiking pyramidal -> SingleUnit (SUA)
        [8.4, 0.08, 7.2, 0.012, 0.96, 0.24, 0.62, 65.0],
        # Candidate 2: Moderate SNR with refractory violations -> MultiUnit (MUA)
        [4.1, 1.45, 28.0, 0.085, 0.88, 0.22, 0.48, 40.0],
        # Candidate 3: Low SNR, high cutoff, fragmented presence -> Noise
        [1.2, 4.80, 0.6, 0.420, 0.15, 0.06, 0.09, 8.0],
    ],
    dtype=np.float32,
)

curator = UnitQualityClassifier(seed=42)
predictions = curator.classify(unit_features)

print("\n[Stage 4: Automated Unit Quality Curation (Allen/IBL Metrics)]")
print(
    f"  {'Unit':<12s} | {'Predicted Label':<12s} | {'P(SUA)':>8s} | {'P(MUA)':>8s} | {'P(Noise)':>8s}"
)
print("  " + "-" * 60)
for idx, (label, p_sua, p_mua, p_noise) in enumerate(predictions):
    print(
        f"  Candidate {idx:<2d} | {label:<14s} | {p_sua:8.3f} | {p_mua:8.3f} | {p_noise:8.3f}"
    )

# %% [5] Stage 5: External Sorter Profiles via Burn-ONNX (`OnnxModelRunner`)
# `OnnxModelRunner` loads any `.onnx` graph (`from_file(path)` or `from_bytes(buf)`) and runs it
# through `dsp-synapse-ml`'s pure-Rust Burn/ONNX-IR engine with built-in adapters for:
#   - sorter="kilosort4" (Kilosort4 spatio-temporal projection)
#   - sorter="dartsort"  (DARTsort peak-normalized denoiser & embedder)
#   - sorter="cebra"     (CEBRA contrastive latent projection)
onnx_files = sorted((dk.get_local_path() / "playground").rglob("*.onnx"))
if onnx_files:
    runner = OnnxModelRunner.from_file(str(onnx_files[0]))
    print(
        f"\n[Stage 5: Loaded ONNX Graph ({onnx_files[0].name}, nodes={runner.node_count})]"
    )
    ks4_emb = runner.embed_with_profile(
        noisy_snippets, sorter="kilosort4", embedding_dim=6
    )
    print(f"  Kilosort4 profile embedding shape: {ks4_emb.shape}")
else:
    print(
        "\n[Stage 5: Burn-ONNX External Sorter Runner Ready]\n"
        "  Usage with exported Kilosort4 / DARTsort / CEBRA `.onnx` models:\n"
        "    runner = OnnxModelRunner.from_file('model.onnx')\n"
        "    denoised = runner.denoise_dartsort(noisy_snippets)\n"
        "    emb_ks4  = runner.embed_with_profile(noisy_snippets, sorter='kilosort4', embedding_dim=8)\n"
        "    emb_cebra = runner.embed_with_profile(noisy_snippets, sorter='cebra', embedding_dim=8)"
    )

# %% [6] Visualize Denoised Waveforms, Latent Spaces & Unit Quality Probabilities
if HAS_PLT:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = ["#1f77b4", "#2ca02c", "#d62728"]

    # Panel 1: Clean vs. Noisy vs. UNet Denoised Waveform (Unit 0)
    t_ms = t_axis
    axes[0, 0].plot(
        t_ms,
        noisy_snippets[0, 0],
        color="#999999",
        alpha=0.7,
        linewidth=1.0,
        label="Noisy Input (Ch 0)",
    )
    axes[0, 0].plot(
        t_ms,
        clean_snippets[0, 0],
        color="#1f77b4",
        linestyle="--",
        linewidth=1.5,
        label="Ground-Truth Clean",
    )
    axes[0, 0].plot(
        t_ms,
        denoised_unet[0, 0],
        color="#2ca02c",
        linewidth=1.5,
        label="SpatiotemporalUnetDenoiser",
    )
    axes[0, 0].set_title(
        "1. 1D Spatio-Temporal UNet Waveform Denoising", fontweight="bold"
    )
    axes[0, 0].set_xlabel("Time from Trough (ms)")
    axes[0, 0].set_ylabel("Amplitude (μV)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: DARTsort Variational Autoencoder (VAE) Latent Space
    for u in range(n_units):
        mask = true_labels == u
        axes[0, 1].scatter(
            z_vae[mask, 0],
            z_vae[mask, 1],
            color=colors[u],
            s=28,
            alpha=0.85,
            label=f"Unit {u}",
        )
    axes[0, 1].set_title(
        "2. DartsortVaeEmbedder Latent Space (z₀ vs. z₁)", fontweight="bold"
    )
    axes[0, 1].set_xlabel("Latent Dim 0")
    axes[0, 1].set_ylabel("Latent Dim 1")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: Contrastive (SimCLR / CEBRA) Unit-Hypersphere Embedding
    for u in range(n_units):
        mask = true_labels == u
        axes[1, 0].scatter(
            z_contrastive[mask, 0],
            z_contrastive[mask, 1],
            color=colors[u],
            s=28,
            alpha=0.85,
            label=f"Unit {u}",
        )
    axes[1, 0].set_title(
        "3. ContrastiveWaveformEmbedder (Unit Hypersphere)", fontweight="bold"
    )
    axes[1, 0].set_xlabel("Projection Dim 0")
    axes[1, 0].set_ylabel("Projection Dim 1")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Panel 4: Automated Unit Quality Classifier Posterior Probabilities
    x = np.arange(len(predictions))
    width = 0.25
    p_sua = [p[1] for p in predictions]
    p_mua = [p[2] for p in predictions]
    p_noi = [p[3] for p in predictions]
    axes[1, 1].bar(x - width, p_sua, width, label="P(SingleUnit)", color="#2ca02c")
    axes[1, 1].bar(x, p_mua, width, label="P(MultiUnit)", color="#ff7f0e")
    axes[1, 1].bar(x + width, p_noi, width, label="P(Noise)", color="#d62728")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(
        [f"Cand {i}\n({predictions[i][0]})" for i in range(len(predictions))]
    )
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].set_title(
        "4. UnitQualityClassifier Posterior Probabilities", fontweight="bold"
    )
    axes[1, 1].set_ylabel("Probability")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

# %%
