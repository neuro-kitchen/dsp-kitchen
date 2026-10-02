# EMUsort (Myomatrix): Parameter Optimization & Tuning Guide

This guide details all key parameters in the EMUsort / Myomatrix motor unit spike sorting engine and provides optimization heuristics based on muscle anatomy, array geometry, and force production levels.

---

## 1. Parameter Glossary

| Parameter | Type | Default | Units | Pipeline Stage | Description |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `probe_kind` | `enum` | `Grid32` | — | Ingestion | Array format (`Grid32`, `Thread8`, `Array64`). |
| `ied_mm` | `float` | `4.0` | mm | Geometry | Inter-Electrode Distance between neighboring contacts (1.0–8.0 mm). |
| `threshold_sigma` | `float` | `6.5` | $\sigma_{\text{MAD}}$ | Detection | MUAP detection threshold against the Quiroga MAD noise floor. |
| `refractory_ms` | `float` | `2.5` | ms | Deduplication | Minimum biological refractory period between discharges of the same motor unit. |
| `spatial_radius_um`| `float` | `4000.0`| $\mu\text{m}$ | Deduplication | Spatial distance within which simultaneous MUAP crossings are pruned. |
| `window_len` | `int` | `150` | samples | Extraction | Total snippet length (5.0 ms at 30 kHz; pre=50, post=100 samples). |
| `min_velocity` | `float` | `1.5` | m/s | Conduction | Minimum biological muscle fiber conduction velocity. |
| `max_velocity` | `float` | `6.0` | m/s | Conduction | Maximum biological muscle fiber conduction velocity. |
| `num_basis_pc` | `int` | `12` | components | Feature Embedding| Number of temporal principal components in `wPCA_EMG`. |
| `min_pnr_db` | `float` | `20.0` | dB | Curation | Minimum Pulse-to-Noise Ratio for a unit to be classified as a Single Motor Unit. |
| `max_cov_isi` | `float` | `0.35` | ratio | Curation | Maximum Coefficient of Variation of Inter-Spike Intervals ($\le 35\%$). |

---

## 2. Optimization Heuristics by Biological & Recording Assumptions

### A. Array Format & Muscle Geometry

#### 1. Surface & Epimysial Arrays (32-Channel 4×8 Grid, 4.0 mm Pitch)
* **Application**: Large superficial muscles (e.g., biceps brachii, tibialis anterior, gastrocnemius).
* **`ied_mm`**: Set to **`4.0` or `8.0` mm**.
* **`spatial_radius_um`**: Set to **$4000\text{--}8000\,\mu\text{m}$** ($1\text{--}1.5 \times \text{IED}$).
* **Why**: Motor unit territories in large muscles span 5–15 mm. A spatial deduplication radius below 4 mm creates duplicate units on adjacent rows for the same motor unit.

#### 2. Intramuscular Flexible Threads (8-Channel Fine-Wire Threads)
* **Application**: Deep, vocal, or small rodent muscles (e.g., laryngeal intrinsic muscles, rat forelimb).
* **`ied_mm`**: Set to **$0.25\text{--}0.5\,\text{mm}$** ($250\text{--}500\,\mu\text{m}$).
* **`spatial_radius_um`**: Lower to **$500\text{--}1000\,\mu\text{m}$**.
* **`threshold_sigma`**: Lower to **$5.0\text{--}5.5\,\sigma$** due to localized proximity to contracting fibers.

---

### B. Muscle Fiber Conduction Velocity (MFCV) Tuning

Action potentials propagate bidirectionally along muscle fibers away from the neuromuscular junction (innervation zone) toward the tendon insertions.

```mermaid
flowchart LR
    A["Innervation Zone\n(Tendon Center)"] -->|Propagates Right\nv ≈ 2–6 m/s| B["Contact Row 1\n(t = 0.0 ms)"]
    B -->|Δt = Δx / v| C["Contact Row 2\n(t = +0.8 ms)"]
    C -->|Δt = Δx / v| D["Contact Row 3\n(t = +1.6 ms)"]
```

#### Fiber Type Assumptions
* **Slow-Twitch (Type I) Postural Muscles (e.g., Soleus)**:
  - Muscle fibers have smaller diameters; conduction velocities are slow (**$2.0\text{--}3.5\,\text{m/s}$**).
  - Set `min_velocity = 1.5`, `max_velocity = 3.5`.
* **Fast-Twitch (Type II) Explosive Muscles (e.g., Gastrocnemius, Biceps)**:
  - Muscle fibers have larger diameters; conduction velocities are rapid (**$4.0\text{--}6.0\,\text{m/s}$**).
  - Set `min_velocity = 3.5`, `max_velocity = 6.5`.

---

### C. Contraction Level & Motor Unit Superposition

#### Low-Force Contractions ($< 20\%$ Maximum Voluntary Contraction, MVC)
* **Characteristics**: Sparse firing, low recruitment, distinct non-overlapping MUAPs.
* **Tuning**:
  - `threshold_sigma = 5.5`.
  - Standard GMM clustering (`min_clusters = 2`, `max_clusters = 6`).
  - High PNR yield ($> 25\,\text{dB}$).

#### High-Force Contractions ($> 50\%$ MVC) & Heavy Superposition
* **Characteristics**: High motor unit recruitment, frequent waveform collisions, destructive interference.
* **Tuning**:
  - Increase `threshold_sigma` to **`7.0` or `7.5`** to focus on the largest, well-isolated peaks.
  - Increase `refractory_ms` to **`3.0` or `3.5` ms** to avoid false triggers during polyphasic afterpotentials.
  - Employ Convolutive Blind Source Separation (`cBSS`) over standard GMM clustering to separate mixed source signals.

---

### D. Quality Curation Guidelines (PNR & CoV-ISI)

To guarantee publication-grade single motor unit isolation in accordance with standard HD-EMG decomposition criteria:

1. **Pulse-to-Noise Ratio ($\text{PNR}$)**:
   $$\text{PNR} = 10 \log_{10} \left( \frac{\mu_{\text{signal}}^2}{\sigma_{\text{noise}}^2} \right) \ge 20.0\,\text{dB}$$
   - Units with $\text{PNR} \ge 26\,\text{dB}$ represent near-perfect single motor unit isolation with $< 1\%$ false alarm rate.
   - Units with $20\,\text{dB} \le \text{PNR} < 26\,\text{dB}$ are acceptable for rate analyses.
   - Units with $\text{PNR} < 20\,\text{dB}$ must be relegated to Multi-Unit Activity (MUA).

2. **Coefficient of Variation of Inter-Spike Intervals ($\text{CoV}_{\text{ISI}}$)**:
   $$\text{CoV}_{\text{ISI}} = \frac{\sigma_{\text{ISI}}}{\mu_{\text{ISI}}} \le 0.35\quad (35\%)$$
   - Biological motor units during steady isometric contractions discharge with high rhythmic regularity ($\text{CoV}_{\text{ISI}} \approx 10\%\text{--}25\%$).
   - A unit with $\text{CoV}_{\text{ISI}} > 0.35$ indicates either missed discharges or double-counting of neighboring motor units.
