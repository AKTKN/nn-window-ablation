# Upstream source provenance

P2 adapts detector graph construction and square tiling from the official
Lange repository at commit `458210ac74ed0047760b8789c169e6b9eff732b8`.
The copyright notice and MIT license ship alongside the adapted Python code
in [LANGE_LICENSE](../src/nn_window_ablation/LANGE_LICENSE). [1, 2, 3]

The adaptation uses `gnnrnn/graphs.py` for spatial tiling, temporal grouping
and edge conventions, and `gnnrnn/data.py` for coordinate normalization and
distance-specific grid choices. It selects raw detector bits before graph
construction, keeps physical detector and query identities, pads metadata
explicitly, and retains empty graphs. Batching returns NumPy arrays for
variable-length queries and keeps supervision separate. Standalone tile
templates use Stim's circuit coordinates. P2 introduces no learned weights.

During P3, record each incorporated model file and retain its applicable
license. Record the query-metadata projection, the common global/core output,
and any changes to data filtering or training curriculum. The study policy
uses direct random initialization at $d=5$.

The upstream repository is a source reference, not a promised stable decoder API
or a package that already implements this study's target/window semantics.
The project-wide publication license has not been selected by this scaffold.
Dependency use and copying upstream source are recorded separately.

## References

1. M. Lange et al. https://arxiv.org/abs/2610.03869
2. Pinned author source.
   https://github.com/LangeMoritz/modular-gnn-rnn-decoder/tree/458210ac74ed0047760b8789c169e6b9eff732b8
3. Source license at the selected snapshot.
   https://github.com/LangeMoritz/modular-gnn-rnn-decoder/blob/458210ac74ed0047760b8789c169e6b9eff732b8/LICENSE
