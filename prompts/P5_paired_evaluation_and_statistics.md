# P5: Paired evaluation and statistical analysis

Implement P5 in the existing `nn-window-ablation` repository. Read `AGENTS.md`, `IMPLEMENTATION_STATUS.md`, `docs/01_theory_and_decoder.md`, Sections 5–9 of `docs/02_ablation_protocol.md`, `docs/03_repository_design.md`, P5 of `docs/05_implementation_phases.md`, `prompts/00_shared_requirements.md`, and `prompts/test_assurance_plan.md` before editing. Inspect the actual analysis, evaluation, and runtime YAML groups. P0–P4 must provide working circuits, labels, observation policies, the query model, immutable datasets, and reloadable checkpoints. Inspect their evidence; repair directly blocking interface defects without replacing those components with mocks.

## Implementation and scientific claim

Implement `evaluation.py`, `analysis.py`, and their workflow/CLI integration. Evaluate G, F, and S on identical complete trajectories from one immutable checkpoint. G executes the global query; F executes all 12 full-context core queries; S executes the same targets within their allowed windows. Threshold probabilities at 0.5, XOR hard core decisions, retain shot identities and packed core decisions, and preserve the model's parameters and normalization buffers. Use evaluation mode and disable gradient updates. Apply the established $T^{-}/T^{+}$ endpoint policy.

Record inference mode separately from checkpoint training provenance. `evaluation.id=S` identifies window inference; combined with full-closed, continued-full, fixed-window, or dynamic-window checkpoint provenance, its statistical condition is S, S+, W, or D. A current CLI training preset cannot relabel an existing checkpoint. Companion G/F results from adapted checkpoints must retain that provenance.

Report failure counts, complete-shot LER, Wilson intervals, paired contrasts, and core disagreement diagnostics. Preserve signed differences and the identity

$$
P_S-P_G=(P_F-P_G)+(P_S-P_F).
$$

Do not pool training seeds as independent shots. Support explicitly configured ratio uncertainty, but report an unresolved ratio when the denominator has insufficient failures; never insert a pseudocount silently. An optional PyMatching evaluation must decode the jointly sampled detector rows using the original global-only DEM. Its predictions are a baseline, not target labels.

Record this evidence under claim C5, conditional on C1–C4: correct inference composition and statistical computation on the identified shots and checkpoints. Tiny integration runs establish neither decoder accuracy nor a performance advantage.

## Tests and independent references

Create the following tests, keeping reference calculations independent of production aggregation and interval helpers.

- `tests/unit/test_p5_prediction_composition.py`: hand-write complete-shot target and prediction tables with zero, one, two, and three core disagreements. Include nonzero targets and pad to 12 cores where appropriate. Independently enumerate XOR to verify $E_n=\bigoplus_j(\widehat Y_{nj}\oplus Y_{nj})$ and the odd-disagreement probability. Include cancellation of two local mistakes. Reject malformed, nonbinary, or incomplete core arrays.
- `tests/unit/test_p5_paired_statistics.py`: use, for example, failure vectors A=`[1,1,1,0,0,0]`, B=`[0,0,1,1,0,0]`. Assert $n_{10}=2$, $n_{01}=1$, and $\widehat\Delta_{A-B}=1/6$ from direct counting. Reverse A/B and verify the sign; test complete agreement, complete disagreement, and the three-condition decomposition. Never infer pairing from equal marginal totals. Test detection of missing, duplicate, and permuted shot identities; align by validated identities or fail explicitly before aggregation.
- `tests/unit/test_p5_confidence_intervals.py`: verify Wilson limits against its analytical expression, including $N=10,k=0$ giving approximately $[0,0.2775328]$ and $k=10$ giving $[0.7224672,1]$ at 95%. Reject $N=0$, invalid counts, and invalid confidence levels. Use separately calculated beta/binomial quantiles to check exact intervals.
- `tests/unit/test_p5_bootstrap_units.py`: enumerate all resampling-index tuples for a tiny complete-trajectory table and aggregate the resulting statistic distribution. Compare its probability mass function with the empirical four-category multinomial construction. This verifies equality in distribution; different random algorithms need not return identical samples for the same seed. Test repeatability within each declared algorithm/environment, and ensure one selected trajectory index carries every condition and all its core values together. A deliberately window-resampled reference must disagree on a cancellation fixture.
- `tests/unit/test_p5_condition_provenance.py`: cover the S/S+/W/D mapping, checkpoint identity, mismatched dataset/label schemas, and incompatible configurations. Confirm that changing an irrelevant caller training preset cannot change the saved condition identity.

For sparse differences, implement the protocol exactly: construct two-sided 97.5% Clopper–Pearson intervals $[L_{10},U_{10}]$ and $[L_{01},U_{01}]$, each with denominator $N$, and return

$$
[L_{10}-U_{01},\ U_{10}-L_{01}].
$$

The union bound gives at least 95% coverage for this paired difference. At zero discordances the limits must remain nondegenerate: with $u=1-0.0125^{1/N}$ they are $[-u,u]$. Check this formula independently. Do not clip negative differences to zero or interpret a degenerate ordinary bootstrap as proof of equality.

The existing `sparse_rule` and `bootstrap_replicates` are unresolved. Implement validation that prevents main-study analysis until they are fixed. Record the bootstrap interval construction, quantile convention, RNG/seed, replicate count, and sparse-switch rule in resolved analysis metadata. Use explicit development-only values in tests. These choices must be settled from the protocol and pilot before main-data inspection, not chosen to obtain a preferred interval.

## Integrated execution

In `tests/integration/test_p5_d5_paired_evaluation.py`, generate a tiny genuine $d=5,T=60,p=0.003$ dataset with joint labels, load a real 256-wide/four-layer model checkpoint from the earlier implementation, and execute G/F/S. Use a few shots and a negligible training budget if a fixture checkpoint is needed. Assert output shapes, all 12 core IDs, paired identities, finite logits, parity identities, and unchanged canonical parameter/buffer hashes before and after evaluation. Reorder evaluation calls and batches to detect persistent window state; apply declared floating-point tolerances without demanding inaccurate bitwise portability. Check the real terminal-window metadata and crop path. Do not mock the simulator, graph encoder, recurrent model, or statistics here.

If PyMatching is enabled, add `tests/integration/test_p5_matching_baseline.py` using the actual original DEM and the same detector rows. Verify observable dimensions and record provenance; do not demand agreement with every stochastic label.

Run unit and scientific integration tests using the shared eight-CPU policy:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONHASHSEED=0 python -m pytest -n 8 --dist=load --strict-markers --strict-config -m "not performance and not cluster" tests
```

Use P0's `test_cpu` profile: PyTorch intra/inter-operation threads one, data-loader workers zero, and isolated worker output directories. Run any timing diagnostic separately with `-n 1` and no more than eight PyTorch threads, never concurrently with the suite; timing is not an accuracy assertion.

Update `IMPLEMENTATION_STATUS.md`, `docs/verification/P5.md`, and `docs/verification/assurance_case.md` with changed interfaces, executed commands, counts, failure repairs, unresolved statistical choices, and evidence paths. Report which negative controls fail as intended and which integration components actually ran. Do not label unexecuted cluster or production evaluations complete.

## Primary references

- SciPy binomial intervals: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html
- SciPy paired bootstrap interface: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
- PyMatching batch decoding: https://pymatching.readthedocs.io/en/stable/api.html
- pytest-xdist process selection: https://pytest-xdist.readthedocs.io/en/stable/distribution.html
