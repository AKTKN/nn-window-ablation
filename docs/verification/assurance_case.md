# Verification assurance case

The claims below follow the P0–P7 [phase contracts](../05_implementation_phases.md).
The shared bootstrap checks support test execution. The implementation and
evidence for each phase determine whether its claim is established.

| Claim | Scope | Depends on | Principal evidence | Status |
|---|---|---|---|---|
| C0 | Installed commands resolve and validate the intended configuration and CPU limits. | Recorded environment | Independent configuration calculations, installed CLI subprocesses, invalid-setting controls, and runtime checks | Verified locally. Allocated-node check pending. [P0 evidence](P0.md) |
| C1 | Detector and label records represent the specified circuit and clean-cut contributions. | C0 and Stim sampling semantics | Independent Pauli-frame propagation, 2,645 elementary signatures, and pilot cut/readout checks | Verified locally on the specified fixtures. [P1 evidence](P1.md) |
| C2 | Features use only the observations and metadata allowed for a query. | C1 metadata | Exact availability sets, excluded-bit and physical-fault perturbations, and graph and batch comparisons | Verified locally and independently reviewed. [P2 evidence](P2.md) |
| C3 | The shared query model and objective preserve the architecture and reset state across independent windows. | C2 | Component and gradient checks, query-order checks, and distance-5 execution | Pending P3 |
| C4 | Data, training, and resume preserve identities, random streams, and update budgets. | C1–C3 | Exact round trips, an independent update calculation, and interruption and corruption checks | Pending P4 |
| C5 | Frozen G/F/S predictions and paired estimates use the same complete trajectories and checkpoint. | C1–C4 | Parity truth tables, signed paired counts, independent interval checks, and integrated inference | Pending P5 |
| C6 | S+/W/D match continuation controls while varying the declared training context. | C2–C5 | Common-fork traces, distinct base/continuation streams, buffer schedules, and resume checks | Pending P6 |
| C7 | Execution and export retain evidence sufficient to reconstruct the analysis. | C4–C6 | Scheduler rendering, archive integrity, relocation and local reconstruction, and allocated-node validation | Pending P7 |

## Recording evidence

Add a `P0.md` through `P7.md` record as each phase is implemented and link it
from its claim above. Record the assumptions, trusted dependencies, supported
configuration domain, exact test IDs, and the independent source of expected
results. Describe the negative control and the defect it detects. Record source
revision, environment, commands, observed results, and any numerical tolerances
with their rationale. State the interface checked for the next phase and identify
pending acceptance checks.

Identify each item as an algebraic argument, finite exhaustive check,
deterministic regression check, statistical experiment, or pending check. A
later claim can rely on a prerequisite only after the relevant interface has
executed evidence. Keep allocated-node evidence with the recorded cluster
configuration.

See the [implementation status](../../IMPLEMENTATION_STATUS.md) for the shared
test bootstrap validation and the [test guide](../../tests/README.md) for the
execution policy.
