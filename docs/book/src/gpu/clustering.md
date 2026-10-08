# Case study: Kilosort4 clustering and matching

Kilosort4's later stages, clustering the detected spikes, learning templates from the clusters,
matching those templates back against the data and clustering again, look like host work: graphs,
trees, sequential pursuit. This chapter shows how each became GPU work, what stayed on the host and
why, and the one measurement that changed the design. All of it is written from the paper's Methods
(see [the pipeline](../sorters/kilosort4/pipeline.md)); timings are on the RTX 2070 of
[Benchmarks](../sorters/benchmarks.md).

## The data that crosses the bus

The rule from [Moving data to the device](data-movement.md) still decides the shape: data crosses
once, in its compact form.

| Data | Crosses |
|---|---|
| Spike features (`[spikes, 10 channels, 6 PCs]`) | down once during detection; they stay on the host, as Kilosort4 keeps them, because an hour of recording does not fit in VRAM |
| A section's spikes | up once, compact; the wider section embedding (`[channels in section · 6, spikes]`) is built **on the device** |
| k-NN, seeding, assignment, edge counts, regression Gram, projections | device only |
| Back per section | labels (one `u32` per spike), the ≤ 200 × 200 edge counts, per split check one `d × d` Gram and a 400-bin histogram |

The merging tree (≤ 200 leaves) and the 400-bin histogram stay on the host: a few thousand
operations each, less than one launch's overhead.

## k-nearest neighbours are a matrix product

The paper's graph links every spike to its 10 nearest neighbours among a subset of the spikes. With
`‖x − y‖² = ‖x‖² + ‖y‖² − 2·xᵀy`, all the distances of a block of spikes to the subset are one
[`matmul`](matmul.md) (`X · Yᵀ`) plus two norm vectors; a small kernel then keeps the 10 smallest per
row. The points are stored feature-major (`[d, n]`), so `X` is a **transposed view** of the stored
buffer and `Yᵀ` is the subset as stored: no copy. Rows go in chunks so the `rows × m` scratch stays
bounded (`MAX_CHUNK_DOTS`).

An existing HDBSCAN kernel finds neighbours too, but keeps all `d` features of its point in
registers; with `d` = channels × 6 ≈ 60–200 here they would spill. The matrix product has no such
limit and reuses each loaded feature across a whole tile.

## The measurement that changed the design: serial chains inside a thread

k-means++ picks 200 seeds; each new seed is the best of `2 + ln k` candidates drawn by squared
distance. The first version kept the existing device k-means++, which asks the host for each draw:
~9 round trips per seed. Profiling the first full run showed clustering at **165 s**, 163 of them in
seeding, so the obvious fix was to remove the round trips: uniforms made on the host and uploaded
once, every draw, score and pick on the device, five queued launches per seed and no sync.

It changed nothing: **158 s**. The round trips were not the cost. The kernels that score the
candidates summed blocks of `TRIAL_BLOCK = 1024` points per thread, so for a section of 1 416
points, seven candidates ran on **fourteen threads**, each looping a hundred thousand multiply-adds.
A GPU thread is slow; a GPU is fast only with many of them. Measured on one section (200 seeds,
1 416 points, 102 features):

| Points per thread | Seeding |
|---|---|
| 1024 | 3 431 ms |
| 256 | 871 ms |
| 64 | 241 ms |
| **16** | **76 ms** |

Linear in the block size: the time was the length of the serial chain inside each thread. With 16,
the launches' own overhead dominates. Clustering of the whole recording went from 158 s to 5 s.

Two lessons. Measure before fixing: the first fix was reasonable and wrong. And "few syncs" is not
"fast": a kernel with a long loop and few threads is as serial as the host.

## Assignment in parallel: why the graph is bipartite

Modularity optimizers such as Louvain move one node at a time, because each move changes the
others' gains. The paper's way around it: make the graph **bipartite**. Left nodes are all spikes,
right nodes a copy of the subset, and every edge joins a left and a right node. Given the right
labels, each left node's best cluster depends only on its own neighbours, so all left nodes are
assigned at once, then all right nodes, alternately, one kernel each:

- right step: every edge adds 1 to `counts[r, label(t)]` (a global `Atomic<u32>`), then one thread
  per right node takes the cluster maximizing `n_rc − γ·k_r·K_c / 2m`;
- left step: one thread per spike counts its 10 neighbours' labels in registers and takes the best;
  a counter of moved spikes is the only value read back per round.

It converges in a handful of rounds on these data (a 200-round cap is never reached).

## Merges and splits

The merging tree needs only the edge counts between clusters, counted on the device (one more atomic
kernel) and read back once: at most 200 × 200 values. Building it is host work.

Each split decision fits the paper's weighted regression axis between two halves. Its normal
equations are again matrix products on the device: a kernel writes `√w·x`, `√w·y` and `√w` for the
node's spikes, then `matmul` gives `Σ w x xᵀ`, `Σ w y x` and `Σ w x`. Only the `d × d` system comes
back to be solved in `f64` (Cholesky with a small ridge); the projections are binned into a 400-bin
histogram with atomics, and only the histogram returns.

A detail the paper leaves out: its axis has no intercept. Spike features are not centred, and a half
sitting near the origin then projects onto the histogram's trough. The device test with two
separated blobs caught it (a negative bimodality score); the fix adds a bias term, which only needs
`Σ w x` more.

## All pairs at all lags, without waveforms

Two later steps need the correlation of every pair of templates at every time lag: merging
near-duplicate templates, and updating scores during matching pursuit. Done on waveforms that is
`templates² × lags × channels × samples`: 2·10¹¹ operations for 279 templates.

In PC space it collapses. A template is `U[ch, p]` (channels × 6 PCs), its waveform `Σ_p U[ch,p] ·
wPCA[p, t]`. The lagged correlation of two templates is

```text
Σ_ch Σ_t w_i[ch,t] · w_j[ch,t+lag] = Σ_{p,q} (Σ_ch U_i[ch,p] U_j[ch,q]) · (Σ_t wPCA[p,t] wPCA[q,t+lag])
                                  =  Σ_{p,q}      UtU[(i,p),(j,q)]     ·         wtw[p,q,lag]
```

`UtU` for all pairs is **one `matmul`** (`X · Xᵀ`, X = `[templates · 6, channels]`), and `wtw` is a
6 × 6 × 121 table of the PCs' own lagged products. One kernel combines them: 36 multiply-adds per
pair and lag. The same factorization gives matching's scores: every template's projection at every
sample is one `matmul` of the templates in PC form with the data's PC projections.

## Matching pursuit without atomics or sorting

Each round of matching pursuit finds spikes that are local maxima over ±`nt` samples, then subtracts
them from the data and the scores. Subtracting in parallel is a write conflict when two spikes'
ranges (±`nt − 1`) overlap, and spikes only `nt + 1` apart do overlap. Floating-point atomics are not
portable (WGSL has none) and would make results order-dependent.

Group the spikes by `⌊t / (nt + 1)⌋ mod 3`. Peaks are more than `nt` apart, so a block of `nt + 1`
samples holds at most one, and two spikes of the same group are at least two blocks apart: more than
`2·nt` samples, beyond each other's reach. Three launches per round, one per group, subtract every
spike with plain writes, in any order, with the same result every time. The device's peak compaction
does not need to sort either.

The test that pins the convention: after the pursuit, the scores updated incrementally must equal
the scores recomputed from the residual (they agree to 10⁻³). It confirmed a sign worked out on
paper first: a spike at `t₀` changes the score at `t₀ + lag` by the template product at `−lag`.

## Two bugs the checks found

- **Exact ties.** On a coarse HD-EMG grid, neighbouring detection centres computed bit-identical
  responses, and a local-maximum test (`value == max`) keeps every tie: 56 000 of 94 000 detections
  were copies. No kernel was wrong; the comparison was. One host pass over each window's spikes
  (already downloaded) keeps one per tie.
- **A constant-started variable, again** ([Pitfalls](pitfalls.md)): a mutable kernel variable
  initialised from a `const` becomes compile-time. In the split kernel it surfaced as a compile error
  (the radix select's case had failed silently); the variable now starts from a literal.
