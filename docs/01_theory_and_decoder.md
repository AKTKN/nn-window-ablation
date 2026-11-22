# Theory and decoder interface

Design version: 2026-11-22. Status: proposed specification, prior to implementation.

## 1. Scope and status

This document specifies the decoding problem and the model interface for a controlled offline-to-streaming ablation on a rotated surface-code $Z$-memory experiment. The experiment prepares $|0_L\rangle$, performs $T$ noisy syndrome-extraction rounds, and ends with destructive data-qubit measurements in the $Z$ basis. The initial design uses stochastic Pauli circuit noise, a fixed patch, and no conditional corrections or logical gates.

The adopted construction combines the core-parity interface of Zhang et al. [Z26] with the graph-recurrent architecture of Lange et al. [L26]. Joint full-context supervision of global and core queries, subsequent restriction to each core's buffered window, explicit query metadata, and the intermediate-observable label construction below are study extensions. Neither cited paper reports this exact controlled comparison. This is a design specification; no simulator or training implementation is included.

All binary algebra is over $\mathbb F_2$. The initial configuration has

$$
T=Kd,\qquad c=d,\qquad b_{\mathrm L}=b_{\mathrm R}=d,
$$

where $K$ is the number of cores, $c$ their length, and $b_{\mathrm L},b_{\mathrm R}$ the past and future buffer lengths. A bulk decoding window therefore spans $3d$ extraction rounds. Extraction-round indices, detector-layer indices, and instruction positions are distinct objects; the circuit metadata must retain their mapping.

## 2. Logical-observable labels and composition

### 2.1 Definitions adopted for this study

Let $e\in\mathbb F_2^{N_f}$ specify the realized circuit faults. A multi-outcome Pauli channel can be represented by mutually exclusive indicators; independence of these indicators is not needed for the following linear identities. For the specified Clifford circuit, write

$$
s=He,\qquad Y=\ell^{\mathsf T}e,
$$

where $s$ is the detection-event vector, $H$ is the fault-to-detector incidence matrix, and $Y$ is the flip of the final physical logical readout relative to its noiseless reference. The vector $\ell$ specifies which faults flip that observable. $Y$ is the supervised observable-flip label. The decoder's logical failure event is $\widehat Y\ne Y$.

Partition circuit-fault locations into disjoint temporal cores. Let $P_i$ be their diagonal selection matrices, satisfying

$$
\sum_{i=1}^{K}P_i=I,\qquad P_iP_j=0\quad(i\ne j).
$$

The selected core's label and the global identity are

$$
Y_i=\ell^{\mathsf T}P_i e,\qquad
Y=\bigoplus_{i=1}^{K}Y_i.
$$

The first core includes preparation faults and the last includes destructive-readout faults. Place each interior cut immediately after the ancilla measurement/reset instruction that completes the round and before the following reset-noise instruction. All subsequent noise instructions belong to the next interval. Measurement noise represented within the completed measurement/reset instruction belongs to the preceding interval. A buffer enlarges the observed input; it does not enlarge the set of faults whose parity defines $Y_i$.

### 2.2 Correspondence to Zhang et al.

Zhang et al. define $y_i=|\mathcal E\cap E_i^c\cap L|\bmod2$ and its global XOR in Section 3.1, Eqs. (1)–(2) [Z26]. Their overlapping windows predict disjoint core contributions. They use shared weights, reset recurrent state for each window, and pad missing buffers at actual experiment boundaries. Their numerical baseline sets $b=c=d$. Local binary cross-entropy is Eq. (3); Eq. (4) adds intermediate recurrent supervision. These are the paper components adopted here. Their differentiable global aggregation, Eq. (5), is an optional later training extension, not the initial loss.

The paper uses zero boundary padding in its lattice representation. The explicit availability and boundary metadata used with the graph encoder below are a study-specific representation of that boundary information, not the paper's original embedding implementation.

The distinction between a core and its input window is operational. Predictions for the full contents of overlapping windows cannot be XORed under the partition identity above: a buffer fault can contribute to several such outputs. The interface must specify $Y_i$ before defining the network or its training data.

### 2.3 Local ambiguity and the physical evaluation target

The local label is well-defined for a sampled fault history, but need not be determined by its syndrome. For example, suppose a fault difference $\delta$ satisfies

$$
H\delta=0,\qquad \ell^{\mathsf T}\delta=0.
$$

Then $e$ and $e+\delta$ give identical observations and the same global logical label. Individual labels can nevertheless differ by

$$
Y_i(e+\delta)\oplus Y_i(e)=\ell^{\mathsf T}P_i\delta.
$$

The nonzero differences cancel over cores. Thus local label mismatch and global decoding failure are different random variables. Define

$$
\epsilon_i=\widehat Y_i\oplus Y_i.
$$

The complete memory fails exactly when $\bigoplus_i\epsilon_i=1$. Marginal local error rates do not determine this probability without their joint dependence. The principal performance measure is therefore the final memory logical error rate; local mismatch rates and adjacent mismatch correlations are diagnostics of context dependence and composition.

For continuous outputs $q_i$, the expression

$$
q_{\oplus}=\frac{1-\prod_i(1-2q_i)}{2}
$$

reduces to XOR at binary inputs. Interpreting it as an exact parity posterior requires an appropriate conditional-independence assumption. The present study makes no such assumption about overlapping-window predictions. If used later as a differentiable loss construction, it must be documented as that construction rather than an exact posterior calculation.

## 3. Correction-output windows and a bit-only interface

A conventional sequential window decoder returns a correction representative, commits its core part, and updates the retained syndrome for the next window. In a fault-graph representation, if the estimated correction is $\widehat e_i$ and its committed component is $\widehat c_i=P_i\widehat e_i$, its two relevant effects are

$$
\Delta F_i=\ell^{\mathsf T}\widehat c_i,
\qquad
\Delta s_i=H\widehat c_i.
$$

The decoder accumulates the logical-parity change $\Delta F_i$ and applies the appropriate retained-detector restriction of $\Delta s_i$ to the next window. Parallel schemes instead use correction representatives to compute and resolve residual seam syndromes. Skoric et al. describe these constructions in Results and Figs. 2–3, with implementation details in Methods [S23].

A scalar logical prediction supplies $\Delta F_i$, but in general does not determine $\Delta s_i$. Two candidate corrections can have the same logical parity and different boundaries. Accordingly, replacing a correction-output inner decoder with a scalar classifier is not an interface-preserving substitution. One must either add a correction or boundary-state output, or use a composition rule specifically defined for scalar core labels.

The initial study selects the latter. Each window consumes the original detection-event record in its allowed region, predicts $Y_i$, and contributes one committed bit. It does not modify the detector record and does not supply a physical correction to its neighbor. This isolates the scalar-core interface without adding a learned boundary-state representation.

## 4. Adopted independent-window schedule

Let $t_i=id$ for $i=0,\ldots,K$. Core $i$ contains the extraction rounds between cuts $t_{i-1}$ and $t_i$, together with the preparation/readout ownership defined above. Its ancilla-detector input covers rounds $r$ with $\max(0,t_{i-1}-d)<r\le\min(T,t_i+d)$. Treat completion of the last ancilla round as phase $T^{-}$ and completion of destructive data readout as the distinct later phase $T^{+}$. This extraction-round and endpoint-phase convention specifies the present simulator schedule. Artificial cuts remain open. A detector's availability is determined by its constituent measurements and phase, not solely by its geometric time coordinate.

| Window position | Past side | Core target | Future side |
|---|---|---|---|
| Initial | Actual preparation boundary; missing outer slots marked as padding | First core, including preparation faults | Observed future buffer |
| Bulk, away from physical endpoints | Observed past buffer with an open outer edge | Interior core | Observed future buffer with an open outer edge |
| Final | Observed past buffer | Last core, including readout faults | Actual terminal data-readout layer; missing outer slots marked as padding |

Boundary metadata is computed from the actual available layers, independently of the core index. For example, a second core whose left buffer reaches the first extraction layer also receives that layer's preparation-boundary information. Such a window is recorded as touching the preparation boundary even though its target is an interior core.

For every nonfinal core $i<K$, commit its bit after ancilla round $t_i+d$, before any subsequent physical stage. The observation delay relative to the core end is $d$ rounds, before adding computation time. In particular, the penultimate core has deadline $t_{K-1}+d=T$ and is evaluated at $T^{-}$: it must not receive terminal data-readout detectors. Complete the final window at $T^{+}$, after the actual destructive readout, and do not revise previously committed bits. For later buffer configurations that leave additional cores uncommitted at termination, a separately declared terminal-flush policy may expose the readout to those still-uncommitted cores; it does not alter the initial $b_{\mathrm R}=d$ rule.

Each window starts from zero recurrent state and uses the same model parameters. The global accumulator updates as

$$
\widehat F_i=\widehat F_{i-1}\oplus\widehat Y_i,
\qquad \widehat F_0=0,
$$

with $\widehat Y=\widehat F_K$. The mathematical schedule does not require parallel hardware: windows may be evaluated independently in a batch or sequentially after becoming available. Passing hidden states between windows would define a different observation policy and is excluded from the initial comparison.

Padding is represented as absent input, with explicit metadata. An observed detector layer containing no detection events remains an observed layer. The representation must not conflate either case with an artificial measurement asserting a closed internal boundary.

### 4.1 Limits of the supporting theory

Theorem 1 in the theoretical appendix of [Z26] (HTML Section 10.2) concerns MWPM under Assumption 1, which constrains consistency of overlapping corrections. It lower-bounds the total error weight needed to leave a seam syndrome by $w_b/2$. Exponential suppression for $b\ge d$ is stated as Conjecture 1; Section 10.4 provides a further argument. These are not guarantees for the proposed neural decoder. Here $b=d$ is an experimentally motivated baseline, not a proof of preserved distance or equality with global decoding.

## 5. Lange GNN–RNN architecture and study adaptation

### 5.1 Published architecture

The paper studies transfer of reusable local syndrome encoders across surface-code distances and across memory, $X$-type stability, and $ZZ$-merge experiments. It combines simulated pretraining with hardware fine-tuning on superconducting-processor data. Its reported comparisons include improvements over MWPM on IBM distance-3/5 memory data and performance approaching AlphaQubit on Google's memory data [L26]. The present study adopts its modular architecture and public implementation as the backbone for a controlled context-transfer experiment.

Lange et al. use a graph encoder on overlapping $\Delta t=2$ detector layers, followed by a GRU and scalar readout [L26, Section III.1; Appendix B]. Event nodes carry spatial and local temporal coordinates; edges connect distinct nodes with inverse-square $\ell_\infty$ distance weights (Eq. (2)). Two graph-convolution layers and mean pooling produce a 256-dimensional embedding, with a learned embedding for an empty graph (Eqs. (3)–(4)). The GRU integrates the embedding sequence (Eqs. (5)–(8)); the final scalar head and BCE appear in Eqs. (9) and (12). Larger patches reuse overlapping spatial tiles, whose embeddings are aggregated before recurrence (Section III.2; Appendix B.5).

The public repository identifies `gnnrnn/graphs.py`, `model.py`, `data.py`, and `train.py` as the graph construction, model, sampling, and training components [LCODE]. The proposed adaptation starts from commit `458210ac74ed0047760b8789c169e6b9eff732b8` (2026-11-19). Its released models and training interface provide a reproducibility reference. A released whole-memory head is not already a core-conditioned head: adopting its architecture does not supply the core-label semantics required in this study.

The encoder's two-layer temporal support and the outer decoding window serve different purposes. The former groups local events before encoding. The latter restricts all observations available for a core prediction. A GRU retaining state across the full experiment can depend on arbitrarily old encoder outputs even when each graph uses only two detector layers.

### 5.2 Core-conditioned model

The proposed model has the conditional interface

$$
q_\theta(Y_C=1\mid s_V,C,M_V),
$$

where $V$ is the allowed detector region, $C$ specifies the target fault subset, and $M_V$ describes observation availability and physical boundaries. A query can be a core $C_i$ or the full experiment $C_{\mathrm{all}}$, whose target is $Y$. The target is not inferred from the input length. It is explicitly specified so that the same core can be evaluated using full history or its restricted window, and one parameter set supports both global and core queries.

Retain the graph encoder and spatial aggregation structure. Supply a small metadata vector alongside each resulting temporal embedding before the GRU. It identifies each detector-layer slot's position relative to the query's temporal cuts, genuine preparation/readout boundaries, and padding or unavailable slots. These markers specify the target interval; they do not assign individual events to the faults that caused them. The metadata exists even when a graph contains no event nodes. Its projection and the scalar query readout are study additions and must be trained in every compared query-conditioned model.

Symbolically, for encoder interval $r$,

$$
h_r=A_\theta\bigl(E_\theta(G_{r,1}),\ldots,E_\theta(G_{r,J})\bigr),
\qquad
u_r=h_r+P_\theta m_{r,i},
$$

$$
z_r=\operatorname{GRU}_\theta(u_r,z_{r-1}),\qquad z_0=0,
\qquad
q_\theta=\sigma(w_\theta^{\mathsf T}z_R+a_\theta).
$$

Here $G_{r,j}$ is the allowed event graph for spatial tile $j$, $A_\theta$ is the spatial aggregator (the identity for a single tile), and $m_{r,i}$ is the core/boundary metadata. The initial proposed adaptation uses a learned linear projection $P_\theta$ into the 256-dimensional embedding space and adds it before the GRU. The metadata schema and this projection are identical in every ablation arm. This specifies a small study-specific addition to the published backbone, not an architecture attributed to Lange et al.

The full-context predictor processes the complete history separately for the selected core. It can use observations after that core because its readout occurs after the complete supplied sequence. The restricted predictor processes only its buffered window and initializes a fresh state. Encoder graphs must be constructed after observation restriction: cropping already-computed embeddings can retain information from excluded detector layers.

### 5.3 Matched training and inference semantics

Train one offline query-conditioned checkpoint by minimizing the following expected loss using both the global label and the core labels:

$$
\mathcal L_{\mathrm F}(\theta)=
\mathbb E\left[
\frac{1}{1+\lambda}\operatorname{BCE}\bigl(Y,q_\theta(s_{\mathrm{all}},C_{\mathrm{all}},M_{\mathrm{all}})\bigr)
+\frac{\lambda}{(1+\lambda)K}\sum_{i=1}^{K}
\operatorname{BCE}\bigl(Y_i,q_\theta(s_{\mathrm{all}},C_i,M_{\mathrm{all}})\bigr)
\right].
$$

Let $\theta_{\mathrm F}$ denote the checkpoint obtained at the prescribed optimization budget; no exact global minimization is assumed. The proposed initial coefficient is $\lambda=1$. Uniformly sampling one core per trajectory gives an unbiased estimator of the averaged core term, as specified in the ablation protocol. Every term uses the complete closed history; restricted windows are not introduced in this training condition.

The same $\theta_{\mathrm F}$ then defines three inference conditions. The global query is

$$
\widehat Y_{\mathrm G}
=\mathbf 1\{q_{\theta_{\mathrm F}}(s_{\mathrm{all}},C_{\mathrm{all}},M_{\mathrm{all}})\ge1/2\}.
$$

The full-context core decomposition and frozen window transfer are

$$
\widehat Y_{\mathrm F}
=\bigoplus_i
\mathbf 1\{q_{\theta_{\mathrm F}}(s_{\mathrm{all}},C_i,M_{\mathrm{all}})\ge1/2\},
$$

$$
\widehat Y_{\mathrm S}
=\bigoplus_i
\mathbf 1\{q_{\theta_{\mathrm F}}(s_{W_i},C_i,M_{W_i})\ge1/2\}.
$$

The subsequent window-trained or fine-tuned checkpoint changes the training observation policy while retaining this interface. The comparison $\mathrm F$ versus $\mathrm S$ changes context availability without changing the core targets, parameters, threshold, or aggregation rule. The comparison $\mathrm G$ versus $\mathrm F$ measures the effect of replacing the global decision by composed core decisions under complete observation. All three conditions use the same trained checkpoint.

This baseline is a full-context query-conditioned decoder trained on closed trajectories. It is not an unmodified classifier trained only to predict the final whole-memory bit. An unmodified upstream Lange model may be retained as a separate reproducibility reference; it is not condition $\mathrm G$. XOR of marginal core decisions need not equal the Bayes decision for their parity, which is why $\mathrm F$ is retained alongside the direct global query.

## 6. Exact core labels from intermediate Pauli observables

### 6.1 Intermediate parity definition

Choose the physical representative

$$
\overline Z=\prod_{q\in Q_L}Z_q
$$

whose final data-measurement parity defines the original global observable. At the clean cut after the completed ancilla measurement/reset and before the next reset-noise instruction, let $E(t)$ be the data-qubit Pauli frame and let $x(t)$ record its $X$ components, including the $X$ component of $Y$. The ancillas have been reset at this cut; a physical ancilla preparation fault for the next round has not yet been injected. Define the binary commutation parity

$$
E(t)\overline Z=(-1)^{g(t)}\overline Z E(t),
\qquad
g(t)=\bar z^{\mathsf T}x(t).
$$

Define auxiliary variables

$$
G_0=0,\qquad G_i=g(t_i)\quad(1\le i<K),\qquad G_K=Y,
$$

and construct

$$
Y_i=G_i\oplus G_{i-1}.
$$

This includes the $K=1$ case, for which $Y_1=Y$. The value $G_0=0$ is the known ideal reference before noisy preparation; subtracting a frame measured after noisy preparation would incorrectly remove preparation faults from the first core. The final residual $G_K\oplus G_{K-1}$ incorporates all faults in the final interval and the actual destructive readout.

The XOR identity follows by cancellation of all interior $G_i$. It holds for every sampled shot and does not rely on a decoder, an independence assumption, or a matching approximation.

### 6.2 Why the increments are temporal fault contributions

For the standard fixed-patch CSS extraction circuit, an ideal complete round preserves the chosen physical $\overline Z$. Data-to-ancilla CNOTs used for $Z$ checks preserve each data $Z$. Propagating $\overline Z$ through an $X$-check circuit introduces an ancilla $Z$ factor once for every data-qubit overlap. That overlap is even because the check commutes with $\overline Z$, so the ancilla factors cancel over the complete check.

Consequently, in the absence of new faults, the parity of a pre-existing frame is unchanged at consecutive cuts. With stochastic Pauli faults,

$$
g(t_i)=g(t_{i-1})\oplus\eta_i,
$$

where $\eta_i$ is the contribution of faults in the intervening interval. Subsequent ideal rounds preserve that contribution to the final physical logical parity. Under these stated circuit assumptions, the snapshot construction therefore realizes the circuit-location definition $Y_i=\ell^{\mathsf T}P_i e$.

The proof requires these clean completed-round cuts and a preserved physical representative. A geometric time coordinate does not specify the required instruction ordering: the circuit adapter must identify the completed measurement/reset and place the annotation before subsequent noise. The proof does not follow merely from preservation of a logical operator modulo unknown stabilizer signs. It also does not apply unchanged to code deformation, omitted feedback corrections, or a logical Clifford transformation of the tracked observable. Those circuits are outside the initial study.

### 6.3 Stim annotation and sampling contract

Stim supports Pauli targets on `OBSERVABLE_INCLUDE` from version 1.15 [SREL]. At each interior cut, add a separately indexed annotation containing the Pauli terms of $\overline Z$, for example

```text
OBSERVABLE_INCLUDE(j) Zq1 Zq2 ... Zqd
```

where the symbolic qubit names stand for the actual data-qubit indices. Keep the original final observable at its original index. These Pauli annotations do not perform a measurement. They affect detection-event sampling and observable error annotations; they are ignored by ordinary measurement-record sampling [SGATE, `OBSERVABLE_INCLUDE`].

Generate the detector and observable arrays together using the annotated circuit's detector sampler, requesting separate observable outputs. The additional observable columns supply the $G_i$. Computing them solely from sampled measurement records would omit the Pauli annotations. An interactive `FlipSimulator` that exposes frames at the same cuts is an independent reference method, not the required high-throughput sampling path.

Default stabilizer randomization may remain enabled. Stim describes this as random multiplication by newly introduced stabilizers [SFLIP, `FlipSimulator.__init__`]. At the completed-round cuts of this memory circuit, those frame ambiguities commute with $\overline Z$ and do not change $g(t)$. A mid-check cut would not have this property: an ancilla randomizer can have propagated onto only part of a check, with odd overlap with $\overline Z$.

Only the detection-event array is observable input to the decoder. $Y$, the snapshots, and the derived $Y_i$ are labels or evaluation data. Build the matching baseline from the original circuit containing only its original global observable, and decode the same detector shots. The auxiliary labels must not become measured checks available to any decoder.

This construction avoids assigning a merged DEM mechanism to a physical time by guesswork. Multiple circuit faults can have the same global detector/observable signature and be combined in a DEM. Circuit-level snapshots retain the temporal information needed for the chosen core labels without undoing that aggregation.

## 7. Correctness conditions for the implementation phase

The following conditions define acceptance of the eventual simulator/model interface, rather than additional experimental claims.

1. The original and annotated circuits have identical physical operations and detector definitions. Only observable annotations differ.
2. The chosen $\overline Z$ matches the original final observable's data support and noiseless-reference convention.
3. Every interior core boundary names a concrete instruction position immediately after the completed ancilla measurement/reset and before the subsequent reset noise. Preparation, terminal readout, and adjacent noise instructions have explicit ownership; detector coordinates alone do not define these cuts.
4. Noiseless samples have $G_i=0$ and $Y=0$. The shotwise XOR of the derived core labels equals $Y$.
5. Injected individual Pauli faults give the expected observable contribution in their assigned core, including faults that generate detector events on both sides of a cut.
6. Every input detector is available by the window's release phase. In the initial configuration, the penultimate core is committed at $T^{-}$ without terminal detectors, and only the final core receives the destructive-readout layer at $T^{+}$.
7. Restriction occurs before graph construction or any other operation with temporal support. Empty observed layers, missing physical padding, and excluded context have distinct metadata.
8. Full-context and frozen-window evaluation use identical core labels, model weights, threshold, recurrent initialization policy, and XOR rule. Neither labels nor complete-history hidden states enter window inputs.

## References

[Z26] K. Zhang et al., *Learning to Decode in Parallel: Self-Coordinating Neural Network for Real-Time Quantum Error Correction*, arXiv:2601.09921v1 (2026). Relevant locations: Section 3.1, Eqs. (1)–(2); Sections 3.3–3.4, Eqs. (3)–(5); Algorithm 1; architecture appendix, boundary padding and recurrent initialization; theoretical appendix, Assumption 1, Theorem 1, and Conjecture 1. The HTML labels the architecture and theory appendices as Sections 6 and 10. [Full text](https://arxiv.org/html/2601.09921v1).

[S23] L. Skoric et al., *Parallel window decoding enables scalable fault tolerant quantum computation*, Nature Communications **14**, 7040 (2023), DOI: 10.1038/s41467-023-42482-1. [Published article](https://doi.org/10.1038/s41467-023-42482-1); [open manuscript](https://eprints.whiterose.ac.uk/id/eprint/204962/1/s41467-023-42482-1.pdf).

[L26] M. Lange, O. Fjeldså, G. Jonasson Johansson, and M. Granath, *Modular Neural Decoder for Surface Code Memory and Logic on Superconducting Processors*, arXiv:2610.03869v1 (2026). Relevant locations: Sections III.1–III.2; Appendix B.1–B.6, Eqs. (2)–(12). [Abstract](https://arxiv.org/abs/2610.03869); [PDF](https://arxiv.org/pdf/2610.03869v1).

[LCODE] M. Lange et al., *modular-gnn-rnn-decoder*, author implementation, commit `458210ac74ed0047760b8789c169e6b9eff732b8` (2026-11-19). The repository README identifies the graph, model, data, and training modules and links released models. [Versioned source tree](https://github.com/LangeMoritz/modular-gnn-rnn-decoder/tree/458210ac74ed0047760b8789c169e6b9eff732b8).

[SGATE] Stim, *Gates supported by Stim*, `OBSERVABLE_INCLUDE` instruction. This is the primary specification of Pauli-target annotations and their distinction from measurement-record sampling. [Documentation](https://github.com/quantumlib/Stim/blob/main/doc/gates.md#the-observable_include-instruction).

[SREL] Stim, release notes for version 1.15.0, introducing Pauli targets for `OBSERVABLE_INCLUDE`. [Release notes](https://github.com/quantumlib/Stim/releases/tag/v1.15.0).

[SFLIP] Stim, *Python API Reference*, `stim.FlipSimulator.__init__` and `peek_pauli_flips`. [API reference](https://github.com/quantumlib/Stim/wiki/Stim-v1.13-Python-API-Reference#stim.FlipSimulator).
