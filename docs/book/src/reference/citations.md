# Citations

If you use a sorter or model, cite its authors. Every entry below is also available in code
(`Attributed::provenance()`, `Provenance::citation()`). DOIs were resolved through doi.org.

## Sorters

| Sorter | Paper | Code |
|---|---|---|
| Kilosort4 | Pachitariu, Sridhar, Pennington, Stringer. *Spike sorting with Kilosort4.* Nature Methods (2024). [10.1038/s41592-024-02232-7](https://doi.org/10.1038/s41592-024-02232-7) | [MouseLand/Kilosort](https://github.com/MouseLand/Kilosort), GPL-3.0 |
| EMUsort | O'Connell, Michaels, Wang, Mamidipaka, Venkatesh, Aresh, Pachitariu, Pruszynski, Sober, Pandarinath. *High performance sorting of motor unit action potentials with EMUsort.* openRxiv (2026), CC-BY 4.0. [10.64898/2026.01.06.697952](https://doi.org/10.64898/2026.01.06.697952) | [snel-repo/EMUsort](https://github.com/snel-repo/EMUsort), [snel-repo/Kilosort4](https://github.com/snel-repo/Kilosort4), GPL-3.0 |

## Artifacts

| Artifact | Source | SHA-256 |
|---|---|---|
| Kilosort4 `wTEMP.npz` (`wPCA`, `wTEMP`, 6 × 61) | [osf.io/download/6807fb5958b763aae139aa60](https://osf.io/download/6807fb5958b763aae139aa60/) | `cae1c96f8f4150be0a39627515750b70c4bc3548177cf487ae3c013f1ca6abd8` |

## Hardware

- Chung et al. *Myomatrix arrays for high-definition muscle recording.* eLife (2023).
  [10.7554/eLife.88551](https://doi.org/10.7554/eLife.88551) — the arrays EMUsort was built for.

## Algorithms the sorters rely on

- Campello, Moulavi, Sander. *Density-Based Clustering Based on Hierarchical Density Estimates.*
  PAKDD, LNCS (2013). [10.1007/978-3-642-37456-2_14](https://doi.org/10.1007/978-3-642-37456-2_14)
  — HDBSCAN.
- McInnes, Healy, Astels. *hdbscan: Hierarchical density based clustering.* JOSS (2017).
  [10.21105/joss.00205](https://doi.org/10.21105/joss.00205) — the reference implementation
  whose defaults (via scikit-learn) `dsp_synapse::sorting::hdbscan` follows.
- k-means with k-means++ seeding follows scikit-learn's `KMeans` defaults.
