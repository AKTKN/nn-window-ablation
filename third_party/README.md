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

## P3 model and offline differential reference

P3 adapts `gnnrnn/model.py` into
`src/nn_window_ablation/models/gnn_rnn.py`. It retains the weighted GraphConv
layers, mean pooling, zero-initialized trainable empty vector, CNN tile
aggregation, four-layer GRU without dropout, scalar head, and upstream module
initializers. The adapter accepts P2's routed graph batches and adds a
bias-free projection of public query metadata. Packed sequences give the head
each query's last valid recurrent state. The distance-5 model is constructed
directly with fresh parameters. The P3 payload adds a versioned architecture
and ordered query schema to the model weights.

The files in `third_party/lange_reference/` are byte-for-byte copies from the
pinned commit. Tests can use them without network access.

| Upstream file | Local reference | SHA-256 |
|---|---|---|
| `gnnrnn/model.py` | [model.py](lange_reference/model.py) | `b370c0e399d7f7a652af73c8b966b0a36fdf7cabd03bccf89a20d8fae201c403` |
| `gnnrnn/graphs.py` | [graphs.py](lange_reference/graphs.py) | `53267914dd90825b816216cf0ea2c493a015299e928d7182cbf3f94f28f8a3fe` |
| `LICENSE` | [LICENSE](lange_reference/LICENSE) | `e30f70b235fd0964687ee33b187fc544658b214ea945750abf11d8b276a3d6ee` |

The decoder objective and per-trajectory core choice are study-specific.
Dataset filtering and training schedules belong to later phases.

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
