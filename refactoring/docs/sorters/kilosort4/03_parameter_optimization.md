# Kilosort4: Parameter Optimization & Tuning Guide

This guide details all critical algorithmic parameters in Kilosort4 and provides systematic optimization heuristics based on probe geometry, biological cell types, and noise characteristics.

---

## 1. Parameter Glossary

| Parameter | Type | Default | Units | Pipeline Stage | Description |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `n_chan_bin` | `int` | *Required* | channels | Ingestion | Total channels in binary file (including sync/auxiliary). |
| `fs` | `float` | `30000.0` | Hz | Ingestion | Sampling frequency of the recording hardware. |
| `do_CAR` | `bool` | `True` | — | Preprocessing | Apply median-based Common Average Reference before filtering. |
| `batch_size` | `int` | `60000` | samples | Streaming | Number of time samples per processing batch (2.0 s at 30 kHz). |
| `nblocks` | `int` | `1` | blocks | Drift Tracking | Number of non-overlapping depth blocks for non-rigid drift correction (0 = rigid). |
| `Th_universal` | `float` | `9.0` | $\sigma_{\text{MAD}}$ | Detection | Spike detection threshold for universal templates (`wTEMP.npy`). |
| `Th_learned` | `float` | `8.0` | $\sigma_{\text{MAD}}$ | Deconvolution | Detection threshold during learned template deconvolution. |
| `nt` | `int` | `61` | samples | Extraction | Length of extracted waveform snippets (must be an odd integer). |
| `dmin` | `float` | `None` | $\mu\text{m}$ | Templates | Minimum vertical distance between template seeds (defaults to vertical pitch). |
| `dminx` | `float` | `None` | $\mu\text{m}$ | Templates | Minimum horizontal distance between template seeds across shanks. |
| `nearest_templates`| `int` | `100` | count | Deconvolution | Number of nearby templates checked during deconvolution peeling. |
| `spatial_radius_um`| `float` | `35.0` | $\mu\text{m}$ | Deduplication | Spatial distance within which simultaneous crossings are pruned. |
| `refractory_ms` | `float` | `1.0` | ms | Deduplication | Minimum refractory interval between spikes assigned to the same channel/unit. |

---

## 2. Optimization Heuristics by Physical Assumptions

### A. Probe Geometry & Electrode Pitch

#### 1. High-Density Single-Shank Probes (Neuropixels 1.0)
* **Geometry**: Staggered vertical pitch of $20\,\mu\text{m}$, lateral width of $32\,\mu\text{m}$.
* **`spatial_radius_um`**: Set to **$30\text{--}40\,\mu\text{m}$**.
  > [!WARNING]
  > Setting `spatial_radius_um = 150\,\mu\text{m}` on Neuropixels suppresses over 30 channels along the shank, destroying simultaneously active neighboring neurons and reducing unit yield by up to 85%.
* **`dmin`**: $20.0\,\mu\text{m}$.
* **`dminx`**: Leave default (`None` or $32.0\,\mu\text{m}$).

#### 2. Multi-Shank Probes (Neuropixels 2.0 4-Shank, Silicon Polytrodes)
* **Geometry**: Multiple shanks separated by $250\,\mu\text{m}$.
* **`dminx`**: Must be set equal to the shank spacing ($250\,\mu\text{m}$) or larger.
* **Why**: Without setting `dminx`, Kilosort will attempt to interpolate templates across empty tissue between shanks, corrupting template footprints and causing out-of-memory errors.

#### 3. Low-Density Probes & Tetrodes
* **Geometry**: Contacts spaced $>50\,\mu\text{m}$ apart with fewer total channels ($4\text{--}32$).
* **`spatial_radius_um`**: Increase to **$60\text{--}80\,\mu\text{m}$** to encompass the full tetrode footprint.
* **`nearest_templates`**: Reduce from 100 to **16 or 32** to reflect the smaller channel count.
* **`whiten_k_neighbors`**: Reduce from 32 to `min(channels, 8)`.

---

### B. Signal-to-Noise Ratio (SNR) & Noise Floor

#### High Electrical Noise / In Vivo Awake Recordings
* **Symptom**: Large numbers of false-positive noise crossings; unisolated MUA clusters.
* **Action**:
  - Increase `Th_universal` from `9.0` to **`10.0` or `11.0`**.
  - Increase `Th_learned` from `8.0` to **`9.0`**.
  - Verify that `do_CAR = True` is active to eliminate ground reference ripple.

#### Low-Amplitude Deep Brain Structures (Thalamus, Granule Cells)
* **Symptom**: Missed small units; firing rates below biological ground truth.
* **Action**:
  - Lower `Th_universal` to **`6.5` or `7.0`**.
  - Lower `Th_learned` to **`6.0`**.
  - Ensure that ZCA whitening is fit using an adequate regularization factor (`epsilon = 1e-4`) to prevent amplifying low-variance channel noise.

---

### C. Biological Cell-Type Constraints

#### Pyramidal Neurons vs. Fast-Spiking Interneurons
* **Fast-Spiking Interneurons**: Refractory periods can be as short as **$0.8\text{--}1.0\,\text{ms}$** with narrow waveforms.
  - Set `refractory_ms = 0.8` (24 samples at 30 kHz).
  - Use `nt = 61` (2.0 ms window) to capture repolarization.
* **Bursting Neurons (Hippocampal CA1/CA3 Complex Spikes)**:
  - Spike amplitudes decrease systematically within a 10 ms burst.
  - Set `Th_learned` low enough ($6.0\text{--}7.0\,\sigma$) so that trailing attenuated burst spikes are not missed during deconvolution.

---

### D. Probe Drift & Non-Stationary Recordings

#### Acute Recordings with Tissue Relaxation
* **Setting**: `nblocks = 1` or `nblocks = 2`.
* **Heuristic**:
  - If recording is $< 10$ minutes, rigid drift correction (`nblocks = 1`) is sufficient.
  - If recording is $> 30$ minutes with substantial brain movement or postural shifts in awake animals, use non-rigid drift with `nblocks = 3` to `5` to track non-uniform compression along the probe shank.
