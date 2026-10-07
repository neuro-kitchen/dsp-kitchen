# Case study: HDBSCAN

EMUsort removes outlier waveforms with HDBSCAN before it clusters the universal templates
(`dsp_synapse::sorting::hdbscan`). Upstream runs scikit-learn's HDBSCAN on the raw clips (61
samples each), all of them, up to 500 000. That is the one quadratic step of the sorter: scikit-
learn builds the spanning tree with Prim's algorithm over every pair of clips, on one CPU thread.

## The algorithm on the device

1. **Core distances.** Each clip's distance to its `min_samples`-th nearest neighbour (itself
   included). A tiled kernel: a block loads a tile of clips into shared memory, every thread
   compares its own clip (held in registers) against the tile and keeps its nearest list sorted.
2. **Minimum spanning tree** of the mutual-reachability distance `max(core_a, core_b, ‖a − b‖)`
   by **Borůvka**: each round, every component's cheapest edge to another component, then merge.
   Borůvka suits a GPU better than Prim: a round is one parallel pass over all points, and there
   are at most `log₂ n` rounds. Ties are broken by (weight, lower index, higher index), so the
   tree is unique.
3. The hierarchy, its condensed tree and the excess-of-mass selection run on the host (linear).

Each pass is split into launches of measured length (`TARGET_LAUNCH_SECONDS`): long enough that
waiting for each costs little, short enough that a display driver never stops a launch and the
progress bar moves.

## The hang

After the CubeCL 0.11 upgrade, EMUsort froze at "Learning templates" every time; Kilosort4, which
learns templates too but runs no HDBSCAN, was fine. Reading the kernels showed one construct that
no other kernel used: the nearest-list insertion looped **on a flag a branch cleared**
(`while moving { … moving = false … }`), with a data-dependent trip count. Every other loop was a
plain counter. If a compiler change stops the flag's update from reaching the loop condition, the
loop never ends and the GPU never finishes.

The rewrite removes the construct rather than patching it: the new distance replaces the largest
entry and moves down by a **fixed, compile-time-unrolled** sequence of compare-and-swaps. It
cannot hang, keeps the order of equal distances, and, since every index is now a constant, lets
the list live in registers. The same unrolling applies to the clip's 61 features (`load_own`,
`tile_sq_dist`). **Rule: no loop in a kernel may depend on a flag cleared inside it; bound every
loop by a counter or a compile-time constant.**

## Exact shortcuts

A full Borůvka round compares every clip with every other: about 1.5·10¹³ operations at 500 000
clips, twenty times over. Two observations skip most of it **without changing the result**:

- **Neighbour edges.** A clip's mutual-reachability weight to anything is at least its own core
  distance. The nearest-neighbour pass already lists the clips within that distance. If one of
  them is in another component and has a core distance no larger, that edge **reaches the lower
  bound**: it is the clip's cheapest edge. Keeping one more neighbour than `min_samples` proves the
  list holds every clip within the core distance, so the tie rule (lowest index) is honoured too.
  Such clips need no scan (`knn_candidate_kernel`).
- **Component bound.** Every neighbour edge into another component is a real edge, so the lightest
  bounds the component's cheapest edge from above. A clip whose core distance exceeds that bound
  cannot improve it: no scan either.

Only the remaining clips are scanned against all others (`cheapest_edge_tile_kernel` over a list
of active points). The test `pruning_is_exact` builds the tree both ways on clip-like data with
duplicates and requires identical labels.

## Results

Clip-like data, 61 dimensions, `min_cluster_size = 20`, RTX 2070:

| Clips | Time |
|---:|---:|
| 20 000 | 0.54 s |
| 50 000 | 0.79 s |
| 200 000 | 6.6 s |
| 500 000 | 52 s |

## A lesson about tests: ties

The test against a host Prim reference failed on one point of 240. That point's five lightest
edges all had exactly the same weight, its core distance. Ties like this are built into HDBSCAN:
`max(core_i, core_j, d) = core_i` for every close neighbour with a smaller core distance. Prim and
Borůvka then pick different, equally minimal edges, and a point attached through a tie can be noise
in one tree and in a cluster in the other. Both are correct. The test now requires agreement
everywhere except at tied points, which it identifies on the host.
