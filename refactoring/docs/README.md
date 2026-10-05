# Parked documentation

`sorters/` is the former `docs/sorters/` (commit `1a83454`, 2026-10-02), moved out on 2026-10-05
when the mdBook became the documentation. Replaced by `docs/book/src/sorters/`.

Why it was replaced (see `refactoring/dsp-synapse-ml/REVIEW.md`, DOC1–DOC2):
- EMUsort pages describe a design EMUsort does not have (150-sample `wTEMP_EMG`, 12-PC
  `wPCA_EMG`, cBSS clustering, CAR on) and cite only the Myomatrix array paper, not EMUsort's.
- Kilosort4 pages present `wTEMP.npy` / `wPCA.npy` as pretrained weights (Kilosort4 learns them per
  recording by default; the predefined set is one `wTEMP.npz`).
- Code examples use APIs that do not exist (e.g. `dsp_synapse::probe::neuropixels_1_0`,
  `CommonAverageReference::new`) and Python bindings that were not verified.
