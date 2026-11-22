# P2 — Observation windows and leakage-free graph features

Implement Phase P2 in the `nn-window-ablation` repository. First read `AGENTS.md`, `docs/01_theory_and_decoder.md`, `docs/02_ablation_protocol.md`, `docs/03_repository_design.md`, `docs/05_implementation_phases.md`, the current configuration files, `prompts/00_shared_requirements.md`, and `prompts/test_assurance_plan.md`. Inspect P1's implementation, oracle coverage, and review record. Implement windows, features, and their scientific tests. Do not implement or train the P3 neural model, publish the repository, or submit cluster jobs.

## 1. Inputs, outputs, and fixed conventions

Implement explicit typed core queries, window views, temporal-slot metadata, graph records, and batching in `windows.py` and `features.py`. Accept resolved parameters and P1 records; do not import Hydra into numerical code. Preserve the scientific pilot $d=5,T=60,c=b_L=b_R=5,W=15,p=0.003$. The later model remains width 256 with four GRU layers and distance-5 $2\times2$ spatial aggregation.

P1 supplies complete physical detector shots, exact core labels, measurement dependencies, and arrival phases. P2 selects the observations for a query; it does not resimulate a closed memory for each window, recompute target ownership, or alter the detector definitions at an artificial boundary. A detector may depend on an earlier measurement outside the left edge: it remains the original computed detector bit when its detector layer is selected.

For core $j$, let $t_j=jc$ and define the allowed extraction-layer range

$$
\max(0,t_{j-1}-b_L)<r\le\min(T,t_j+b_R).
$$

Every nonfinal core releases after ancilla round $\min(t_j+b_R,T)$, before destructive readout. The final core releases at $T^{+}$. At the default setting, core 11 releases at $T^{-}$ and excludes terminal detectors; core 12 includes the actual terminal layer at $T^{+}$. Preserve this policy for all supported buffers, including $b_R=0$.

Availability is derived from the actual measurement-record dependencies and phase ordering. Geometric detector time coordinates alone are insufficient. Retain extraction-round, detector-layer, and instruction-position identities separately. Resolve dynamic buffer candidates as $\{0,\lceil d/2\rceil,d\}$; at $d=5$ they are $\{0,3,5\}$.

## 2. Feature construction and metadata

Restrict raw detector observations first, then construct Lange's overlapping two-layer event graphs and spatial tiles using the pinned source [1, 2]. Preserve the upstream coordinate, local-time, edge-weight, and tile-order conventions where applicable. Document precisely how consecutive allowed layers, genuine endpoint layers, and missing boundary slots form graph groups. Never obtain a window by cropping full-history graphs or embeddings that already consumed an excluded layer.

Event-node features use spatial and local temporal coordinates. For distinct nodes, the upstream edge weight is inverse-square $\ell_\infty$ distance. Construct hand-checkable expected graphs; do not use the production graph builder to manufacture expected tensors. Preserve full-context versus window coordinate conventions and the identity of each physical detector.

Keep observed empty layers in temporal order. Distinguish observed zero events, unavailable/excluded observations, and physical padding in explicit metadata; a zero-event graph is not an absent graph. Reserve the learned empty-graph embedding for P3. Metadata must remain available when there are no nodes, and batching must retain the correct number and order of graph slots.

Build query metadata exclusively from the target interval, permitted observation schedule, and physical boundaries. Boundary flags depend on the observed layers, not only on the target-core index. In the default pilot, core 2's left buffer reaches round 1 and therefore includes preparation-boundary information. An artificial window start or end does not assert a physical initialization or final measurement.

Global and core queries use one feature schema. Global input and full-context core input expose the same physical history but different query metadata. Full-context and window queries for the same core keep the identical target. Neither $Y$, $G_j$, $Y_j$, fault histories, DEM error mechanisms, nor label-derived statistics may enter a model-input record. Keep labels in a separate supervised-batch component.

## 3. Independent selection and graph oracles

Create small reference helpers under `tests/reference/`. Do not import production mask selection, boundary-flag construction, grouping, or graph-building helpers. For a literal fixture, list measurement IDs and arrival phases, detector measurement-dependency sets, layer membership, and expected query boundaries. Derive each detector's availability from the latest required measurement, then intersect with the independently specified layer range and release phase. Assert exact detector-index sets, not only counts.

Use a separate hand-constructed graph fixture with zero, one, and two event nodes. For two nodes with coordinate difference $(1,0,1)$, the expected directed pair weights are one; for $(2,0,0)$ they are $1/4$, subject to the explicitly documented coordinate normalization. Verify node coordinates, absence/presence of self edges according to the upstream convention, temporal membership, and tile indices. Any coordinate normalization must be applied independently in the expected values.

## 4. Required tests and negative controls

Place tests in `tests/unit/test_p2_*.py` and `tests/integration/test_p2_*.py`.

- `test_p2_window_sets_against_reference`: independently enumerate all 12 default pilot queries and supported buffer pairs. Check first, second, bulk, penultimate, and final cores; include asymmetric buffers, zero buffers, and the odd middle buffer three. Verify disjoint target ownership and overlapping observation sets without changing P1 labels.
- `test_p2_terminal_phase_equality`: exact expected sets exclude every destructive-readout detector when a nonfinal deadline equals $T$; the final query includes eligible terminal detectors. An apparent matching geometric coordinate cannot override measurement availability.
- `test_p2_boundary_flags_follow_observations`: core 2 with its left buffer reaches the preparation boundary, whereas a sufficiently interior window does not. Only actual observed terminal information sets the readout flag. Test corresponding full-context flags independently of the selected target.
- `test_p2_raw_crop_precedes_two_layer_features`: place events immediately outside both edges and in neighboring retained layers. Compare exact features with the independent oracle. Include odd core/buffer boundaries where a two-layer group could straddle the cutoff; no excluded event contributes an edge, node, pooled input, or metadata value.
- `test_p2_unavailable_perturbations_do_not_change_features`: arbitrarily change all excluded detector bits, including terminal bits, while holding allowed observations fixed. Window feature records remain identical. Include a physically generated pair of P1 trajectories whose differing late fault occurs after a nonterminal release.
- `test_p2_empty_missing_padding_are_distinct`: observed empty slots persist through batching and unbatching; absent context and physical padding retain distinct metadata. Check all-zero shots, a one-event shot, and variable-length endpoint windows without filtering examples.
- `test_p2_labels_never_enter_features`: hold detector observations and public metadata fixed while changing all label/snapshot columns. Model inputs remain identical. Enforce an explicit field whitelist and verify batch serialization cannot attach teacher columns to feature tensors.
- `test_p2_full_context_and_batch_roundtrip`: independently requested full-context core views reproduce the full-context feature path with matching query metadata. Packing/unpacking and batching/unbatching preserve shot IDs, core IDs, temporal order, detector indices, and labels.

Add controlled negative cases for a deadline comparison that admits $T^{+}$ at $T^{-}$, core-index-only boundary flags, graph construction before cropping, dropped empty slots, and a copied target column. Demonstrate that the relevant oracle or metamorphic assertion rejects each defect. Do not assert that any particular input change must change an untrained network's output; this phase verifies feature dependence, not classification sensitivity.

## 5. Integration, execution, and exported guarantee

Run a bounded P0→P1→P2 integration at $d=5,T=60,p=0.003$. Use genuine joint detector/label batches and all 12 query IDs. Verify zero-event handling in a separately identified $p=0$ fixture; do not wait for a rare zero-event trajectory at $p=0.003$ or mislabel its generating distribution. Verify unchanged P1 targets under every context mask and exact feature/identity round trips. Do not start training or infer logical-error performance from this integration.

Use the shared eight-CPU policy: eight xdist workers, one computational thread per worker, and no nested worker pools. Run the phase suite plus relevant inherited P0/P1 checks using pytest-xdist:

```bash
python -m pytest -n 8 --dist=load --strict-markers --strict-config tests/unit/test_p2_*.py tests/integration/test_p2_*.py
```

At completion, run the cumulative ordinary suite once as specified in the shared requirements. Record claim C2, its C1 prerequisites, exact commands, environments, fixtures, negative controls, results, and limitations in `docs/verification/P2.md`, `docs/verification/assurance_case.md`, and `IMPLEMENTATION_STATUS.md`. P2 exports evidence that feature construction depends only on the permitted observations and label-independent query metadata. P3 must additionally enforce recurrent-state reset and absence of cross-query caches before this becomes a prediction-level no-future-dependence guarantee. No classification accuracy or code-distance claim follows from these tests.

## References

1. Lange et al., modular architecture: https://arxiv.org/abs/2610.03869
2. Pinned upstream graph construction: https://github.com/LangeMoritz/modular-gnn-rnn-decoder/blob/458210ac74ed0047760b8789c169e6b9eff732b8/gnnrnn/graphs.py
3. Zhang et al., independent overlapping windows and core targets: https://arxiv.org/html/2601.09921v1
4. Stim v1.15 gate definitions and detector measurement dependencies: https://github.com/quantumlib/Stim/blob/v1.15.0/doc/gates.md
5. pytest-xdist distribution modes: https://pytest-xdist.readthedocs.io/en/stable/distribution.html
