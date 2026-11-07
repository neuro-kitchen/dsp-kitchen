# 08 · Detection

From a recording to units, on the synthetic recording (its spikes are known, so every step is
scored, not just looked at).

| Script | What it shows |
|---|---|
| `01_threshold_and_scoring` | threshold, NEO and matched-filter detection, merging across neighbouring channels; recall and precision against the truth |
| `02_features_and_clusters` | snippets on every channel → principal components → k-means / HDBSCAN; how far the clusters agree with the true units |
| `03_templates_and_matching` | templates (cluster means), their similarity, and matching pursuit finding every spike again |

The functions: `dspk.synapse.detection`, `extraction`, `clustering`, `matching`, and
`dspk.ops.linalg.PCA`. A sorter does the same steps on a whole recording, on the device; chapter
09 does them on real data.
