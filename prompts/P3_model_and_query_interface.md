# P3: Implement the model and shared query interface

Implement P3 in `nn-window-ablation`, including the model adapter, objective, substantive tests, and a bounded CPU profile. Complete the implementation and execute its checks; do not stop at a plan. Do not run a scientific training campaign, publish the repository, or submit PBS jobs.

## Required context and inherited claims

Read repository `AGENTS.md`, `IMPLEMENTATION_STATUS.md`, `docs/01_theory_and_decoder.md`, `docs/02_ablation_protocol.md`, `docs/03_repository_design.md`, the P3 section of `docs/05_implementation_phases.md`, and the current model, window, runtime, and training YAML groups. Also read this prompt bundle's `prompts/00_shared_requirements.md` and `prompts/test_assurance_plan.md`. Preserve their evidence and resource rules.

P0 supplies resolved configuration; P1 supplies physical trajectories and independently checked labels; P2 supplies observation restriction, graph features, query metadata, and availability. Verify those inherited gates rather than assuming their existence. P3 establishes that the specified network consumes those representations, implements the shared objective, and preserves observation/state isolation. It does not establish decoder accuracy, convergence, or a context-transfer effect.

## Implementation contract

Adapt the author implementation at commit `458210ac74ed0047760b8789c169e6b9eff732b8`. Retrieve the needed source during implementation, retain its MIT license and copyright notices, and record incorporated files, hashes, and modifications under `third_party/`. Store the reference needed by differential tests locally; tests must not fetch a moving branch or require network access. [1, 2]

Preserve the backbone: event-node inputs of dimension 3; weighted `GraphConv` layers 3→64→256 with ReLU; mean pooling and a learned empty-observation embedding; overlapping two-detector-layer feature groups; distance-dependent spatial aggregation; and a four-layer GRU with hidden size 256. Distance 5 uses the original 2×2 tile aggregation. Do not replace it with a distance-3 network or a reduced model for the phase's integration tests. Initialize the entire distance-5 study model freshly, without a parent checkpoint; retain and document module initializers, including deterministic initializers where defined upstream. [2]

Add the agreed metadata through $u_r=h_r+P_\theta m_{r,i}$ before the GRU, with one shared scalar logit head for global and core queries. Keep the metadata schema identical across contexts. Return logits for `BCEWithLogitsLoss`; convert to probabilities only where requested. A hard decision at probability $1/2$ corresponds to logit $\ge0$. Labels remain outside `ModelBatch` and outside `forward`. [3]

Initialize recurrent state to zero for each independent query. Respect each sequence's valid length: padding from another batch member cannot influence its final state. Preserve observed empty slots, unavailable context, and actual physical-boundary metadata. Keep the pinned upstream default of no GRU dropout; do not add regularization to obtain a test effect.

Implement the objective, without the long training loop owned by P4:

$$
L=\frac{b(Y,z_*)+\lambda J^{-1}\sum_{j=1}^{J}b(Y_j,z_j)}{1+\lambda},
\qquad J=12,\quad\lambda=1.
$$

The sampled estimator replaces the averaged core term by one uniformly sampled core per trajectory. Separate deterministic loss evaluation from core-index sampling so both are independently testable.

## Required unit tests and independent oracles

Create these files or equivalently scoped files with the same `test_p3` prefix. Expected values must not be obtained by calling the production helper under test.

| Test file | Required oracle and coverage |
|---|---|
| `tests/unit/test_p3_upstream_components.py` | Copy known weights into the adapter and an isolated, unmodified pinned reference. Compare GNN outputs, empty-slot placement, the distance-5 spatial aggregator, and GRU/readout outputs on explicitly specified inputs. Use neutral metadata only where a common input/output contract exists. Independently check a small weighted graph calculation and tile ordering. |
| `tests/unit/test_p3_query_loss.py` | Compute float64 BCE from $b(y,z)=\max(z,0)-yz+\log(1+e^{-|z|})$ and its analytic derivative $\sigma(z)-y$. Check ordinary, zero, and extreme logits. Enumerate all 12 sampled-core choices and compare their exact mean loss and gradients with the full objective. Use a small affine parameterization and analytic chain-rule gradients to verify parameter gradients as well as logit gradients. |

Differential agreement applies only to preserved components with matched inputs and weights. Do not require the modified window decoder to reproduce an upstream whole-history prediction, or require a global query to equal the XOR of marginal core decisions. Those are different computations.

Include controls that detect incorrect normalization, omitted $1/J$, wrong core indexing, swapped tiles, and treating a missing slot as an observed empty slot. A controlled incorrect result must be rejected by the checker; do not leave deliberately failing tests in the regular suite.

## Mandatory full-architecture integration tests

In `tests/integration/test_p3_d5_model_contract.py`, use a genuine P1/P2 batch at $d=5,T=60,p=0.003$, with both global and core queries. Exercise the complete distance-5 architecture, not mocked embeddings or a tiny model. A small trajectory batch is sufficient. Check finite logits, both objective terms, backward gradients, and one bounded optimizer update. Confirm that both losses reach shared trainable components, including the metadata projection. Use deliberately active fixtures when testing nonzero aggregate gradient contributions; do not assert that every ReLU unit or parameter has nonzero gradient.

In evaluation mode, compare a query alone, in a reordered batch, and alongside different-length queries. Compare A→B→A execution with independent A executions. Snapshot parameters and buffers before evaluation and require their immutability afterward. Verify that saving/loading the model preserves outputs and the recorded architecture/query schema. Use justified numerical tolerances for different batching arithmetic; state them explicitly.

In `tests/integration/test_p3_window_prediction_isolation.py`, compose the actual sampler, P2 selector, feature builder, and model. Perturb excluded future detector bits and final-readout bits for an already committed nonfinal core, then rebuild inputs. Its features and logits must remain unchanged. Include the penultimate default window. Check a retained observation against an independently expected feature change, and use a test-local leaky readout that explicitly reads an excluded bit to verify that the prediction-invariance comparator detects leakage. Do not require arbitrary different inputs to produce different learned outputs. Changing target arrays alone must never change logits.

These tests extend P2's feature-level information restriction to the model output. They cannot replace P1's independent physical-label checks. Tiny helper models are allowed for mathematical unit tests only.

## Eight-CPU execution and profiling

Follow the shared eight-CPU policy and select `runtime=test_cpu` for integration/CLI fixtures. Each xdist worker uses one PyTorch intra-operation thread, one inter-operation thread, no data-loader workers, and no nested sampler pools. Apply the shared native-library limits and run the inherited and new ordinary suite with:

```bash
python -m pytest -n 8 --dist=load --strict-markers --strict-config \
  -m "not performance and not cluster" tests
```

Place the timing check in `tests/performance/test_p3_cpu_profile.py`, marked `performance`. Run it separately with xdist `-n 1` and at most eight numerical CPUs; profile thread counts such as 1, 2, 4, and 8 with explicit startup settings. Separate sampling/labels, graph construction, forward, backward/update, and serialization; record warm-up, batch size, trajectories/query examples per second, memory, and hardware. Do not time it while eight xdist workers compete for CPUs. The later 96-resource PBS profile is a separate operation. Do not invent main training or shot budgets from this bounded development profile. [4]

## Completion evidence

Update `IMPLEMENTATION_STATUS.md`, `docs/verification/assurance_case.md`, and `docs/verification/P3.md` with C3's interfaces, assumptions, provenance, commands, test counts, negative controls, tolerances, and observed profile measurements. Report inherited C0–C2 evidence, newly established C3 claims, and unresolved failures separately. Leave unmeasured main budgets unresolved. Provide a concise implementation report and exact reproduction commands; never describe an unexecuted criterion as passed.

## References

1. Lange et al., paper: https://arxiv.org/abs/2610.03869
2. Pinned source and license: https://github.com/LangeMoritz/modular-gnn-rnn-decoder/tree/458210ac74ed0047760b8789c169e6b9eff732b8
3. PyTorch BCEWithLogitsLoss: https://docs.pytorch.org/docs/2.7/generated/torch.nn.BCEWithLogitsLoss.html
4. PyTorch CPU oversubscription: https://docs.pytorch.org/docs/2.7/notes/multiprocessing.html#cpu-in-multiprocessing
