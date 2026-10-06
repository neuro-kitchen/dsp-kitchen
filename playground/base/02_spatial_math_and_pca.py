# %% [markdown]
# # dsp-kitchen (`dsp-base`): Spatial Referencing, Math Stages & GPU PCA
# Interactive notebook demonstrating:
# 1. **Point-wise Math Stages (`dsp_kitchen.math`)**:
#    - `Scale` / `scale_samples` ($y = \alpha x + \beta$ ADC-to-$\mu\text{V}$ conversion)
#    - `SubtractBaseline` (DC offset removal)
#    - `Clamp` (saturation & rail clipping protection)
# 2. **Spatial Referencing (`dsp_kitchen.spatial`)**:
#    - `CommonAverageReference` / `common_average_reference` (multi-channel common-mode rejection)
# 3. **Dimensionality Reduction (`dsp_kitchen.linalg.PCA`)**:
#    - Fitting and projecting `PCA` on every compiled-in runtime (`dk.runtime.available()`, e.g. a GPU
#      through WebGPU and the CPU), which give the same result

# %% [1] Imports & Synthetic 32-Channel Array with Shared Common-Mode Artifact
import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_PLT = True
except ImportError:
    HAS_PLT = False

import dsp_kitchen as dk
from dsp_kitchen.pipeline import Pipeline
from dsp_kitchen.math import Clamp, Scale, SubtractBaseline, scale_samples
from dsp_kitchen.spatial import CommonAverageReference, common_average_reference
from dsp_kitchen.linalg import PCA

fs = 30_000.0
n_channels = 32
n_samples = 3_000  # 100 ms
t = np.arange(n_samples, dtype=np.float32) / fs
rng = np.random.default_rng(7)

STEP_UV = 0.195  # µV per ADC step of a typical extracellular headstage (e.g. Intan, Neuropixels 1.0 AP)
BASELINE_COUNTS = 500.0

# Simulate raw int16-scale ADC counts with a DC baseline offset, shared movement artifact,
# and localized neural activity on channels 4..8
adc_counts = rng.normal(0.0, 20.0, size=(n_channels, n_samples)).astype(np.float32)
adc_counts += BASELINE_COUNTS  # DC baseline in ADC counts

# Shared common-mode artifact across all 32 channels
common_artifact = 350.0 * np.sin(2.0 * np.pi * 25.0 * t) + 200.0 * np.exp(
    -((t - 0.050) ** 2) / (2.0 * 0.002**2)
)
adc_counts += common_artifact[None, :]

# Inject a localized spike burst on channel 5
spike_idx = 1200
adc_counts[5, spike_idx : spike_idx + 40] -= 900.0 * np.hanning(40).astype(np.float32)

# Inject a rail-saturation spike at sample 2200
adc_counts[5, 2200] = 25_000.0

print(
    f"Simulated ADC array: {adc_counts.shape}, range=[{adc_counts.min():.1f}, {adc_counts.max():.1f}] counts"
)

# %% [2] Math Stages: Scale, SubtractBaseline, and Clamp
# 1. Direct scaling helper: y = STEP_UV * x
scaled_uv = scale_samples(adc_counts, STEP_UV)

# 2. Composable math pipeline: Scale -> SubtractBaseline -> Clamp
math_pipe = Pipeline(
    [
        Scale(STEP_UV),  # ADC counts -> µV
        SubtractBaseline(BASELINE_COUNTS * STEP_UV),  # remove the DC offset (µV)
        Clamp(-500.0, 500.0),  # clip saturated samples to ±500 µV
    ]
)
cleaned_uv = math_pipe.run(adc_counts)  # no filter stage: no sample rate needed

print("[Math Pipeline Results]")
print(
    f"  After Scale mean:            {scaled_uv.mean():7.2f} uV (max={scaled_uv.max():.1f} uV)"
)
print(
    f"  After Baseline+Clamp mean:   {cleaned_uv.mean():7.2f} uV (max={cleaned_uv.max():.1f} uV)"
)

# %% [3] Spatial Referencing: Common Average Reference (CAR)
# Subtract the instantaneous across-channel mean at each sample to reject the shared movement artifact
car_referenced = common_average_reference(cleaned_uv)

# Measure common-mode rejection on a quiet channel (Ch 0) vs. active channel (Ch 5)
print("\n[Spatial Common Average Reference]")
print(
    f"  Ch 0 RMS before CAR: {np.sqrt(np.mean(cleaned_uv[0] ** 2)):6.2f} uV (dominated by common artifact)"
)
print(
    f"  Ch 0 RMS after CAR:  {np.sqrt(np.mean(car_referenced[0] ** 2)):6.2f} uV (background noise floor)"
)
print(
    f"  Ch 5 Peak after CAR: {car_referenced[5, spike_idx : spike_idx + 40].min():6.2f} uV (localized spike preserved)"
)

# %% [4] Principal Component Analysis (PCA) on every compiled-in runtime
# The same code runs on any GPU or the CPU; the runtime is chosen at run time.
projections = {}
for runtime in dk.runtime.available():
    pca = PCA(n_components=3).fit(car_referenced, runtime=runtime)
    projections[runtime] = pca.transform(car_referenced, runtime=runtime)
    evr = pca.explained_variance_ratio
    print(f"\n[PCA on {runtime}] explained variance ratio {np.round(evr, 4)}")

proj_gpu = next(iter(projections.values()))
if len(projections) > 1:
    a, b = list(projections.values())[:2]
    # Components are defined up to sign: compare magnitudes
    print(f"  Max |difference| between runtimes: {float(np.max(np.abs(np.abs(a) - np.abs(b)))):.3e}")

# %% [5] Visualize Math, CAR, and PCA Projections
if HAS_PLT:
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    t_ms = t * 1000.0

    axes[0].plot(
        t_ms,
        cleaned_uv[5],
        color="#d62728",
        alpha=0.75,
        label="Ch 5 After Scale + Baseline + Clamp",
    )
    axes[0].plot(
        t_ms, cleaned_uv[0], color="#999999", alpha=0.6, label="Ch 0 (Shared Artifact)"
    )
    axes[0].set_title(
        "1. After Math Pipeline (Scale → SubtractBaseline → Clamp)", fontweight="bold"
    )
    axes[0].set_ylabel("μV")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        t_ms,
        car_referenced[5],
        color="#2ca02c",
        linewidth=1.2,
        label="Ch 5 After Spatial CAR",
    )
    axes[1].plot(
        t_ms,
        car_referenced[0],
        color="#1f77b4",
        linewidth=0.9,
        alpha=0.7,
        label="Ch 0 After Spatial CAR",
    )
    axes[1].set_title(
        "2. After Spatial Common Average Reference (Common-Mode Artifact Removed)",
        fontweight="bold",
    )
    axes[1].set_ylabel("μV")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    for c in range(3):
        axes[2].plot(
            t_ms,
            proj_gpu[c],
            linewidth=1.0,
            label=f"PC{c + 1} ({evr[c] * 100:.1f}% var)",
        )
    axes[2].set_title(
        f"3. Principal Components (PC1–PC3) on {next(iter(projections))}", fontweight="bold"
    )
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_ylabel("Score")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# %%
