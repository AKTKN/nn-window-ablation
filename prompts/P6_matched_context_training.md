# P6: Matched full-context and window-context continuation

Implement P6 in the existing `nn-window-ablation` repository. Read `AGENTS.md`, `IMPLEMENTATION_STATUS.md`, `docs/01_theory_and_decoder.md`, Sections 5 and 8 of `docs/02_ablation_protocol.md`, `docs/03_repository_design.md`, P6 of `docs/05_implementation_phases.md`, `prompts/00_shared_requirements.md`, and `prompts/test_assurance_plan.md`. Inspect all four training presets and the current evaluation configuration. Require the functioning P0–P5 scientific pipeline; inspect its tests and checkpoints before extending it. Repair directly blocking defects without silently changing the accepted physics, architecture, targets, or observation policy.

## Implementation and controlled comparison

Implement S+, W, and D as continuation policies in the existing training/workflow modules. Within a replicate, every arm starts from the same full-context $\theta_F$, including learned empty embeddings, query projection, scalar head, and normalization buffers. Initialize no layer afresh. Reset optimizer and scheduler state for all three new arms, including S+, with identical classes, settings, and schedules. A resumed arm instead restores its own saved optimizer/scheduler and continuation cursor.

All arms retain a full-context global query plus one uniformly sampled core query per complete trajectory, with $\lambda=1$ and normalization by $1+\lambda$. S+ supplies full context to the core query; W supplies the fixed deployment window; D supplies dynamically selected buffers. Each D minibatch receives one left/right pair, shared by its trajectories, drawn uniformly from the nine pairs in

$$
\{0,\lceil d/2\rceil,d\}^2.
$$

The initial $d=5$ choices are $\{0,3,5\}$. Preserve core length five, all 12 label definitions, clean-cut ownership, and terminal-readout timing. Dynamic buffers change observations, not labels. Do not introduce cross-window recurrent state or a second head.

Use one new continuation trajectory/core-query namespace, shared by S+/W/D and distinct from base training. The stream identity must not depend on the arm name. Keep independent named randomness for dynamic buffers, so its consumption cannot shift physical shots, sampled core IDs, or model randomness. Record trajectory order, sampled cores, and buffer choices sufficiently to audit short runs.

`training.max_steps` means $U_1$ new optimizer updates. Preserve the parent's $U_0$ separately, track phase-local and cumulative counts, and resume only the remaining continuation budget. At effective batch size $B$, each completed arm consumes $U_1B$ trajectories and $2U_1B$ query examples. Gradient accumulation must preserve the same effective loss and example budget. Compare arms using the same deployment buffers and P5 condition-provenance rules. Equal budgets constrain inputs and updates; trained weights need not remain equal.

Record the controlled-comparison evidence under C6, conditional on C2–C5. Preserve the interface assertions connecting observation masks, checkpoint/stream state, and paired evaluation.

## Tests and independent references

Implement the following focused tests using actual training components and deterministic, small fixtures.

- `tests/unit/test_p6_continuation_fork.py`: start from a checkpoint with nonzero optimizer moments and an advanced scheduler. Independently inventory and hash named model tensors before forking. Verify exact initial tensor/buffer equality in S+/W/D, fresh optimizer moments/counters in all three, and the prescribed initial learning rate. Use independent copies: mutating one arm must change neither its siblings nor the parent. Negative controls must catch carrying optimizer moments into S+ alone or reinitializing the query head.
- `tests/unit/test_p6_continuation_streams.py`: compare recorded immutable trajectory identities, order, and sampled core IDs across arms. Assert continuation identities differ from base-training identities even when chunk indices restart at zero. Vary arm names and dynamic-buffer draws and verify that the shared physical/core streams are unchanged. Detect repeated/skipped chunks, a wrongly reused base namespace, and an arm-derived sampling seed.
- `tests/unit/test_p6_dynamic_buffer_sampling.py`: inject a controlled integer-draw interface and enumerate all nine left/right combinations. Verify the uniform product mapping, support $\{0,3,5\}^2$, and one pair per logical minibatch of $B$ trajectories, reused across its accumulation microbatches. Document this minibatch convention. Avoid flaky frequency tests. Assert targets and query IDs remain identical across all pairs, observed inputs obey each availability mask, and the penultimate core never receives destructive-readout detectors.
- `tests/unit/test_p6_continuation_objective.py`: compare losses with an independent stable scalar BCE calculation for manually specified logits. Check equal total global/core weights and that the global query remains full-context in every arm. Enumerating all 12 sampled-core choices must recover the full mean-core objective. Inspect real prepared batches to verify that context policy changes only the intended core observations. Detect accidentally windowing the global term, changing targets with buffers, or dropping zero-event trajectories.
- `tests/unit/test_p6_continuation_budget.py`: use a parent counter larger than $U_1$ to expose confusion between total and new steps. Verify exactly $U_1$ additional optimizer steps, trajectory/query counts, fixed checkpoint selection, and no remaining work after completion. Verify an interruption after $k$ new steps executes only $U_1-k$ more. Reject incompatible resume settings and simultaneous initialization/resumption requests.

Use a deterministic degenerate-policy test as a compositional oracle. Force every D buffer draw to $(d,d)$, then run W and D from identical state with identical trajectories, core IDs, model randomness, optimizer settings, and tensor/mask ordering. Their prepared inputs, objective, gradients, and updates must agree under the declared deterministic numerical criterion. This equivalence applies only when the effective masks and loss are identical; it does not assert equality for ordinary dynamic training. Its controlled buffer RNG must not consume model RNG state.

## Integrated execution and resume

Add `tests/integration/test_p6_d5_matched_continuation.py`. Obtain a real tiny $d=5,T=60$ base checkpoint from the implemented pipeline, fork all three arms, and execute a small explicitly configured number of updates with the unchanged 256-dimensional/four-GRU architecture. Use real joint circuit sampling, graph construction, query projection, loss, optimizer, and checkpoint serialization. Assert finite values, matched stream/budget records, common starting hash, preserved labels, and distinct provenance. Do not assert that W or D improves LER after a handful of updates.

Add `tests/integration/test_p6_continuation_resume.py`. Within a recorded deterministic CPU environment, compare uninterrupted continuation with interruption, serialization, reload in a fresh process, and completion. Check remaining update count, exact stream/buffer records, scheduler position, losses, and final model tensors using a justified numerical criterion. Exercise the D buffer RNG explicitly. Verify that the continuation fork resets state while resume preserves it.

Run the shared test suite with eight independent workers and one numerical thread per worker:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONHASHSEED=0 python -m pytest -n 8 --dist=load --strict-markers --strict-config -m "not performance and not cluster" tests
```

Use P0's `test_cpu` profile with PyTorch intra/inter-operation threads one, data-loader workers zero, and worker-specific temporary paths. Timing, if needed, runs separately with `-n 1` and at most eight PyTorch threads. Do not select production budgets or report CPU scaling from this correctness suite.

Update `IMPLEMENTATION_STATUS.md`, `docs/verification/P6.md`, and `docs/verification/assurance_case.md` with evidence, exact commands, implemented stream semantics, numerical comparison policy, and remaining cluster checks. Report whether all three arms and fresh-process resume executed. Record which negative controls detect unmatched initialization, data, objective, or budget. This phase establishes a controlled continuation implementation; scientific claims about adaptation require the later fixed-budget experiment.

## Primary references

- Lange author source: https://github.com/LangeMoritz/modular-gnn-rnn-decoder/tree/458210ac74ed0047760b8789c169e6b9eff732b8
- Li et al., dynamic chunk training motivation: https://arxiv.org/abs/2304.09325
- PyTorch optimizer state: https://docs.pytorch.org/docs/2.7/optim.html
- PyTorch reproducibility: https://docs.pytorch.org/docs/2.7/notes/randomness.html
- pytest-xdist process selection: https://pytest-xdist.readthedocs.io/en/stable/distribution.html
