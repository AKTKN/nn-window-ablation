# Scientific test plan

No test implementation is included in this scaffold. P0–P7 acceptance criteria
are defined in [05_implementation_phases.md](../docs/05_implementation_phases.md).
Tests should establish the following scientific and operational properties when
the corresponding implementation is added.

| Phase | Required evidence |
|---|---|
| P0 | Installed configuration discovery, valid derived lengths, invalid-setting rejection, and CPU dependency checks. |
| P1 | Noiseless/single-fault labels, clean-cut ownership, unchanged physical detector process, and shotwise global/core XOR identity. |
| P2 | Future-bit perturbation invariance, terminal arrival-phase policy, zero-event windows, and cropping before graph features. |
| P3 | Genuine distance-5 forward/backward step, shared query gradients, independent window state, and immutable G/F/S model state. |
| P4 | Packed-array round trip, split/shot identity, interruption-safe writes, and interrupted/resumed stream equivalence. |
| P5 | Whole-shot pairing, known discordance-count contrasts, final parity identity, and the specified intervals. |
| P6 | Common initial checkpoint, equal trajectories/updates/core-ID streams, optimizer reset, and unchanged deployment windows. |
| P7 | Actual small PBS allocation, complete/incomplete run identification, archive checksums, and local statistical reconstruction. |

Small $d=3$ fixtures may be used for inexpensive P1 algebraic checks. The scientific
pilot, CPU profile, and model integration check use $d=5$. Static parsing of the
scaffold is not evidence that these invariants hold in a simulator.

## References

- Stim observable annotation reference.
  https://github.com/quantumlib/Stim/blob/v1.15.0/doc/gates.md#the-observable_include-instruction
- Zhang et al. https://arxiv.org/html/2601.09921v1
- PyTorch, reproducibility. https://docs.pytorch.org/docs/2.7/notes/randomness.html
- pytest, getting started. https://docs.pytest.org/en/stable/getting-started.html
