# Upstream source provenance

The planned decoder source is the official Lange repository at commit
`458210ac74ed0047760b8789c169e6b9eff732b8` [1, 2]. The source snapshot uses the MIT
license. No upstream source, license file, or checkpoint has been copied into
this scaffold.

During P3, record each incorporated file and retain its applicable copyright and
license notices. Record the added query-metadata projection, common global/core
output task, raw-window availability handling, and any change to upstream data
filtering or training curriculum. Direct random initialization at $d=5$ is the
initial study policy; an upstream transfer curriculum is not silently assumed.

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
