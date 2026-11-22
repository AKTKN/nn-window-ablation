# P1 — Physical circuits, temporal cuts, and exact labels

Implement Phase P1 in the `nn-window-ablation` repository. First read `AGENTS.md`, `docs/01_theory_and_decoder.md`, `docs/02_ablation_protocol.md`, `docs/03_repository_design.md`, `docs/05_implementation_phases.md`, the current configuration files, `prompts/00_shared_requirements.md`, and `prompts/test_assurance_plan.md`. P0 must already provide the installed package and resolved configuration. Implement this phase and its scientific tests; do not implement the neural model, train a decoder, publish the repository, or submit cluster jobs.

## 1. Physical and software scope

Implement explicit typed circuit, cut, measurement, detector, observable, and sampled-batch records in `simulation.py` or small supporting modules. Keep Hydra outside the scientific implementation. The scientific pilot remains $d=5,T=60,c=5,p=0.003$; optional small $d=3$ circuits are test fixtures only. Preserve the configured width-256, four-GRU-layer model specification without implementing it here.

Build Stim `surface_code:rotated_memory_z` with all four documented generator probabilities equal to $p$. Retain the original global-observable circuit and a separately annotated teacher circuit. Record the Stim version, serialized circuits, hashes, physical logical support, absolute instruction positions, measurement-record dependencies, and detector arrival phases. Resolve repeat blocks without losing their physical instruction order. Distinguish the final ancilla phase $T^{-}$ from destructive data readout $T^{+}$.

Each internal cut is immediately after the completed ancilla measurement/reset instruction and before subsequent reset or idle noise. Measurement-result noise within `MR(p)` belongs to the preceding interval. Noise after that instruction belongs to the following interval. Record the exact position even when intervening annotations or `TICK` instructions occur. Preparation belongs to the first core; destructive-readout faults belong to the last. Do not infer ownership from rounded detector coordinates or merged DEM mechanisms.

## 2. Label and sampling contract

For the original final observable's physical support $Q_L$, use $\overline Z=\prod_{q\in Q_L}Z_q$. At each internal cut add a separate Pauli-target `OBSERVABLE_INCLUDE(j)`; retain the original measurement-record observable at index zero. This annotation is not an `MPP` measurement [1, 2].

For data-frame $X$ components $x(t_j)$, define

$$
G_0=0,\qquad G_j=\bar z^{\mathsf T}x(t_j)\ (1\le j<J),\qquad G_J=Y,
$$

$$
Y_j=G_j\oplus G_{j-1}=L e_{C_j},\qquad Y=\bigoplus_{j=1}^{J}Y_j.
$$

Sample physical detectors and observable columns jointly using the teacher circuit's detector sampler with separate observable output. Return aligned detector bits, global labels, core labels, and shot identities; keep auxiliary snapshots as optional diagnostic data. A measurement-record-only sampler cannot recover these annotations. Expose the original global-only circuit/DEM for later classical decoding; auxiliary observables must never become decoder input.

The XOR identity alone cannot validate the cut locations or fault ownership: arbitrary cumulative bits also telescope. Acceptance therefore requires independently predicted fault signatures and clean-cut preservation of the chosen physical representative.

## 3. Independent reference calculation

Create a deliberately small binary Pauli-frame oracle under `tests/reference/`. It must not import production cut discovery, label construction, logical-support extraction, or detector-dependency helpers. Use reviewed literal fixture specifications for logical support, cuts, and measurement parities. Include primitive tests with hand-calculated expected arrays.

Represent a relative error by binary $(x,z)$. Implement the supported generated-circuit gates explicitly: $H$ swaps $x,z$; $S$ adds $x$ to $z$ if encountered; for `CX(c,t)`, update $x_t\leftarrow x_t\oplus x_c$ and $z_c\leftarrow z_c\oplus z_t$. Pauli noise toggles the corresponding components. A $Z$ measurement has record-flip contribution $x$; a pure measurement-result flip toggles only its record. A $Z$ reset clears both components; `MR` first records the measurement flip, then resets. Handle other encountered basis operations through stated conjugation rules, or fail explicitly. Intended Pauli gates conjugate an error frame and are not automatically injected faults. Specify canonical frame handling after non-reset measurements and ignore only proven irrelevant phases.

Compute detector parities from independently specified measurement dependencies, final $Y$ from the original final measurement parity, and internal signs from the oracle frame. Include an independent propagation check that an ideal complete extraction round preserves the physical $\overline Z$ at the declared cuts. Document the conditions under which stabilizer-frame ambiguity leaves these signs invariant.

Inject deterministic faults as probability-one error channels, such as `X_ERROR(1)`, `Y_ERROR(1)`, or a specified Pauli-product error, with other stochastic channels disabled. Do not substitute intended `X`/`Y` gates: their effects can be absorbed into Stim's noiseless reference.

## 4. Required tests and negative controls

Use named tests in `tests/unit/test_p1_*.py` and `tests/integration/test_p1_*.py`.

- `test_p1_annotations_preserve_physical_circuit`: stripping only added observables restores the flattened original operations, measurement order, and detector definitions exactly. Check the original logical support independently. The classical DEM has one global observable.
- `test_p1_noiseless_reference_and_single_core`: noiseless detector/global/cumulative/core flips are zero; $J=1$ gives $Y_1=Y$. Verify shapes and column ordering without deriving expected values through production label code.
- `test_p1_single_fault_signatures_against_oracle`: exhaustively enumerate allowed single-location Pauli outcomes on a small two-core fixture, for example $d=3,T=6,c=3$, plus measurement-result flips. Compare the complete $(D,Y,G_1,Y_1,Y_2)$ signature to the independent oracle. Enumerate all nonidentity outcomes of one- and two-qubit depolarizing locations, not just convenient data faults. Store a coverage inventory identifying every tested location/outcome.
- `test_p1_clean_cut_ownership`: use explicit data faults on logical support on both sides of a cut. Hand-established expected labels are $(1,0)$ before and $(0,1)$ after for the two-core fixture. Include a fault whose detector effects straddle a cut. Test measurement/reset ordering and post-reset noise ownership separately; ancillary faults can have zero logical contribution.
- `test_p1_preparation_and_readout_ownership`: a preparation $X$ fault on the tracked support belongs to the first core; a final measurement-record flip on that support belongs to the last. Corresponding off-support single faults have zero direct logical parity. Check these expectations with the oracle, including final residual construction.
- `test_p1_composed_fault_signatures`: selected physically realizable multi-location faults produce the XOR of independently computed elementary signatures. Do not combine mutually exclusive outcomes at one channel as independent stochastic events.
- `test_p1_pilot_boundary_integration`: build $d=5,T=60$, verify all eleven internal cut positions, and test representative adjacent faults, preparation, and readout against the independent reference. Joint $p=0.003$ sampling obeys the label identity. Verify retention of zero-event shots using a separately identified $p=0$ fixture; do not require a rare zero-event sample in a bounded noisy batch.

Add negative controls that deliberately shift a cut past its allowed instruction position, alter the logical support, omit preparation ownership, or replace the last residual by $Y$. Also corrupt two core labels while preserving their global XOR: the independent core oracle must reject this case. The corresponding structural or independent-oracle assertion must detect each defect. A displaced cut need not change every injected fault's signature; check the violated ordering directly when appropriate.

## 5. Execution, evidence, and exported guarantee

Follow the shared eight-CPU execution policy: eight xdist workers, one computational thread per worker, no nested sampler/loader pools. Run the phase suite and inherited relevant P0 checks with `pytest-xdist`; for the phase suite use:

```bash
python -m pytest -n 8 --dist=load --strict-markers --strict-config tests/unit/test_p1_*.py tests/integration/test_p1_*.py
```

At completion, run the cumulative ordinary suite once as specified in the shared requirements. Record claim C1, its C0 prerequisites, exact commands, environment, enumerated fault coverage, results, and unresolved cases in `docs/verification/P1.md`, `docs/verification/assurance_case.md`, and `IMPLEMENTATION_STATUS.md`. State separately what follows from finite fixtures and what follows by induction from the supported linear Clifford/frame rules. Linearity of deterministic fault effects does not establish a channel's probability distribution; stochastic sampling relies on the pinned Stim implementation. Do not claim a universal circuit proof or decoder-performance result.

P2 may inherit the verified circuit/label schema, explicit fault ownership, and measurement-dependent arrival metadata. It must still independently validate how windows select observations from that metadata.

## References

1. Stim v1.15 gate reference, `OBSERVABLE_INCLUDE`: https://github.com/quantumlib/Stim/blob/v1.15.0/doc/gates.md#the-observable_include-instruction
2. Stim v1.15 release notes: https://github.com/quantumlib/Stim/releases/tag/v1.15.0
3. Zhang et al., core-label composition: https://arxiv.org/html/2601.09921v1
4. pytest-xdist distribution modes: https://pytest-xdist.readthedocs.io/en/stable/distribution.html
