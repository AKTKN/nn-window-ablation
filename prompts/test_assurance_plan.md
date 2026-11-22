# Compositional verification of the simulation logic

This document specifies the evidence structure to implement with P0–P7. It is a
test design, not a record of tests already executed. The phase prompts require
an executable implementation and a growing `docs/verification/assurance_case.md`.

## 1. Claims and their dependencies

| Claim | Meaning | Depends on | Principal evidence |
|---|---|---|---|
| C0 | Installed commands resolve and validate the intended configuration and resource limits. | Recorded environment | Arithmetic/configuration fixtures, installed-package subprocess tests, thread tests. |
| C1 | Sampled detector/global/core records represent the specified physical circuit and clean-cut contributions. | C0; Stim sampling semantics | Independent Pauli-frame propagation, elementary-fault comparisons, preparation/cut/readout cases. |
| C2 | Each model input is a function only of the declared observations and permitted structural metadata. | C1 metadata | Independent availability-set oracle, future perturbation, empty/missing cases, graph-group boundary tests. |
| C3 | The adapted network realizes the common query interface and objective without hidden cross-window state. | C2 | Upstream component comparisons, analytic loss/gradient calculations, full-architecture distance-5 execution. |
| C4 | Storage and resumable training preserve shot/target correspondence, random streams, and update budgets. | C1–C3 | Exact round trips, independent update oracle, interrupted/uninterrupted comparison, corruption controls. |
| C5 | G/F/S and reported paired statistics refer to the same complete shots and defined checkpoints. | C1–C4 | Truth tables, contingency arithmetic, independent interval checks, integrated inference. |
| C6 | S+/W/D differ in the declared training context while matching their prescribed data/update controls. | C2–C5 | Common-fork traces, phase namespaces, controlled buffer schedules, resume and degenerate-context equivalence. |
| C7 | Execution/export preserves the evidence needed to reproduce the local analysis. | C4–C6 | Scheduler rendering tests, archive integrity, relocation and re-analysis without simulation. |

Each implication requires tests of the interface between the components. Passing
isolated component tests is insufficient if, for example, labels are reordered
when a batch is assembled. Tests at later phases must use real earlier-phase
components in at least one small distance-5 integration path.

## 2. Algebraic basis for the circuit checks

For a fixed Clifford extraction circuit with Pauli faults, represent the relative
data/ancilla frame by $f_s\in\mathbb F_2^{2n}$. At an elementary step, frame
propagation, reset, and measurement-record flips have binary-linear update rules
of the form

$$
f_{s+1}=A_s f_s\oplus B_s e_s,\qquad
r_s=M_s f_s\oplus N_s e_s.
$$

Here $e_s$ describes injected Pauli/measurement faults, and $r_s$ records flips
relative to the ideal reference process. The supported gate set and measurement
semantics must be explicitly stated in the independent oracle. Checking its
elementary rules and composing them gives the expected response for a specified
fault pattern by induction over circuit instructions.

For a fixed finite circuit this produces maps

$$
D=He,\qquad Y=\ell^{\mathsf T}e,\qquad
Y_i=\ell^{\mathsf T}P_i e,
$$

under the stated clean-cut/fault-ownership convention. The $P_i$ form a disjoint
partition. Equality on a complete fault basis, together with established
linearity of the compared maps, implies equality on all binary combinations for
that circuit. Exhaustive checks on a small fixture support this finite-circuit
statement; selected distance-5 cases check that the production construction uses
the same rules. They do not alone quantify over every distance or establish
linearity of arbitrary unexamined code paths.

Categorical branches of depolarizing channels are mutually exclusive at one
noise location. Linear response to a chosen pattern does not make their
probabilities independent. Check physical noise placement and parameters
separately, and use the original Stim circuit's sampler for the joint
distribution. Record that sampling behavior as a trusted dependency [1, 2].

The production target construction uses cumulative logical-frame flips at clean
cuts. The identity

$$
\bigoplus_{i=1}^{J}(G_i\oplus G_{i-1})=G_J\oplus G_0=Y
$$

is an algebraic consistency condition. It would still hold for incorrectly
assigned intermediate $G_i$. Therefore compare individual core contributions
with independently propagated faults before and after each relevant cut.
Include a negative control that changes two core labels while preserving their
global XOR; the independent core oracle must detect it.

## 3. Observation and inference composition

Let $\mathcal A_i$ be the independently specified detector availability set for
query $i$. Verify that feature construction has the form

$$
X_i=\Phi_i(D|_{\mathcal A_i},Q_i,M_i,B_i),
$$

with query/availability/boundary metadata that contain no teacher values or
unavailable measurement outcomes. The reference set must use measurement
dependencies and event phase, rather than call the production window function.
Then two histories agreeing on $\mathcal A_i$ must have identical $X_i$.
Perturb excluded data and check this equality through packing, cropping,
two-round graph construction, batching, and the final evaluation-mode prediction.

State reset and fixed evaluation state are additional premises. If either fails,
the output can depend on prior queries even when the current feature tensor is
correct. Compare query orderings and isolated/batched inference with the same
weights. Do not require different metadata or inputs to yield different neural
outputs: a learned function can legitimately map them to the same value.

Given correct core labels and predictions, whole-shot failure obeys

$$
\widehat Y\oplus Y
=\bigoplus_i(\widehat Y_i\oplus Y_i).
$$

Verify this independently from hand-written bit tables and from a real small
pipeline. This establishes the implemented composition rule; it does not assert
that XORing marginal predictions is the Bayes-optimal global decision.

## 4. Learning, pairing, and statistical interpretation

The sampled-core loss is checked by explicit averaging over every possible core
on a tiny fixed batch, including its gradient. This establishes the unbiased
estimator algebra used in the implementation. It does not require a random
sample average to equal the exact loss after a few draws.

For training state, independently compute at least one optimizer update on an
analytic fixture. Compare a short uninterrupted sequence with a save/reload
sequence using the same fixed environment and operation order. Inspect data/core
IDs and optimizer state as well as final weights. Continuation namespaces must
produce new base-versus-continuation draw identities while matching all three
adaptation arms; matching final weights across different contexts is not required.

For inference, maintain immutable shot IDs and derive each condition from its
checkpoint provenance plus inference mode. Check the signs of $n_{10}-n_{01}$,
the difference decomposition, whole-trajectory resampling, and interval edge
cases. Conditional checkpoint uncertainty and variation across training seeds
remain separate. LER orderings and training benefits are excluded from the
correctness test oracle because they are the study's empirical questions.

Use exact-count interval identities or an independent numerical construction for
small deterministic examples. A bootstrap rerun with the same algorithm and
random state must reproduce its result. Two mathematically equivalent bootstrap
implementations need not consume randomness identically; compare their finite
distributions or appropriately designed reference cases, not their seeded draw
arrays [3, 4].

## 5. Required evidence record

For every C0–C7 claim, record:

1. The precise statement and the configuration domain covered.
2. Assumptions and trusted dependencies.
3. Test IDs, fixtures, and the independent source of expected results.
4. The negative control and the defect it detects.
5. Executed command, environment, result, and numerical tolerance if relevant.
6. The interface assertion connecting this claim to the next one.

Classify each item as an algebraic argument, exhaustive finite verification,
deterministic regression check, statistical experiment, or unexecuted check.
Do not label a finite test suite as a general formal proof. The intended
conclusion is a conditional, traceable argument that the implemented data,
observation, learning, composition, and reporting paths realize the specified
experiment. A failure of an earlier claim invalidates dependent evidence until
the defect is repaired and the relevant tests rerun.

All executable verification follows the eight-CPU policy in
`prompts/00_shared_requirements.md`. Local tests of PBS rendering establish
orchestration behavior; only a recorded execution in an allocated cluster job
establishes that site's operational compatibility.

## References

1. C. Gidney, *Stim: a fast stabilizer circuit simulator*.
   https://quantum-journal.org/papers/q-2021-07-06-497/
2. Stim 1.15 observable annotation semantics.
   https://github.com/quantumlib/Stim/blob/v1.15.0/doc/gates.md#the-observable_include-instruction
3. SciPy, paired bootstrap.
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
4. SciPy, binomial proportion confidence intervals.
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html
5. Zhang et al. https://arxiv.org/html/2601.09921v1
6. Lange et al. https://arxiv.org/abs/2610.03869
