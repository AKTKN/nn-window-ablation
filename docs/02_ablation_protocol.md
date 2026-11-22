# Ablation protocol for transferring an offline neural decoder to temporal windows

Design version: 2026-11-22. Status: proposed specification, prior to implementation; no simulation or training results are claimed here.

## 1. Objective and scope

The study measures how a neural decoder trained with complete, closed quantum-memory histories performs when its inference input is restricted to finite temporal windows. It then measures the improvement obtained by training on the deployed observation context. The first implementation uses a rotated surface-code $Z$-memory experiment. Color codes are a later extension, conditional on a functioning surface-code pipeline.

The primary questions are:

1. For fixed model parameters, how much does the final logical error rate change when full-history core predictions are replaced by window-conditioned predictions of the same core labels?
2. How much of that degradation is recovered by fixed-window or dynamic-window fine-tuning, compared with additional full-context training of the same duration?
3. How does the answer depend on the available right buffer, with the core definition and output-composition rule held fixed?

The design follows the full-context versus restricted-context comparison in Li et al. [R1], while using the direct core-parity composition approach studied by Zhang et al. [R2]. It does not reproduce either paper unchanged. In particular, the common global/core query interface below is a study extension to the chosen neural backbone; it is not an untouched pretrained Lange checkpoint.

## 2. Physical experiment

Use Stim's `surface_code:rotated_memory_z` circuit generator, with the generated circuit retained as a versioned experimental artifact [R3]. The memory initializes logical $|0\rangle$, performs $T$ syndrome-extraction rounds, and ends with destructive data-qubit measurement in the $Z$ basis.

The proposed default parameters are:

| Quantity | Scientific pilot | Main comparison |
|---|---:|---:|
| Code distance $d$ | 5 | 5 and 7; 9 optional |
| Syndrome-extraction rounds $T$ | 60 | $12d$ |
| Core length $c$ | 5 | $d$ |
| Number of cores $J=T/c$ | 12 | 12 |
| Default left buffer $b_L$ | 5 | $d$ |
| Default right buffer $b_R$ | 5 | $d$ |
| Nominal bulk window width $W$ | 15 | $3d$ |
| Physical noise parameter $p$ | 0.003 | 0.003 initially |
| Independent training seeds | 1 for profiling and development | 3 initially |

The initial configuration selects `pilot_d5`, `runtime=cpu`, and `training.initialization=random`. Small $d=3$ circuits may be used as optional invariant-test fixtures; they are not the scientific pilot and no distance-3 training is required.

After the default comparison works, evaluate the same trained checkpoints at $p\in\{0.001,0.002,0.003,0.005\}$ as a secondary noise-transfer evaluation. The primary attribution to context adaptation is made at the training value $p=0.003$; changing the training noise distribution is outside the initial ablation.

Set all four Stim generator noise parameters to the same value $p$:

```python
stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    distance=d,
    rounds=12 * d,
    after_clifford_depolarization=p,
    before_round_data_depolarization=p,
    before_measure_flip_probability=p,
    after_reset_flip_probability=p,
)
```

This specifies the noise model operationally. The first parameter inserts one- or two-qubit depolarization after Clifford gates; the second inserts data-qubit depolarization at the start of each round; the remaining parameters specify measurement and reset flips [R3]. This is the equal-parameter Stim noise model, not SI1000. The exact circuit, gate ordering, Stim version, and circuit hash are part of every run's metadata.

The choice $T=12d$ provides multiple internal windows of width $3d$ whose predictions cannot use final destructive-readout information. A run with $T=d$ would not exercise this bulk-window regime.

### 2.1 Execution environment and resource decisions

The confirmed execution workflow uses local development, versioned source transferred through GitHub, a clone on the cluster, and PBS jobs submitted with `qsub` for substantial dataset generation, profiling, and training. The initial target is one CPU node with up to 96 PBS CPU resources and 350 GB of memory. Conda is the environment manager; resolve and verify a CPU-compatible environment before recording its package versions and export [R10]. No dependency combination or cluster execution is claimed to have been validated by this document.

Profile the actual $d=5,T=60,W=15,p=0.003$ model on that node before selecting effective batch size, sampler/data-loader concurrency, optimizer-update budgets, or test-shot counts. Keep the 256-dimensional representation and four GRU layers specified in Section 5.1. CPU execution must not implicitly reduce network capacity. Record sampling, feature construction, forward/backward computation, and I/O separately, together with throughput and peak resident memory. Training processes, data-loader workers, and PyTorch/BLAS threads share the same allocated CPU budget; uncontrolled nested parallelism can reduce throughput [R11].

Record the source revision, resolved configuration, environment, actual CPU allocation, and thread/worker settings for each job. Completed result bundles are compressed on the cluster and downloaded with `scp` for local inspection and analysis. Their prediction/label arrays, manifests, and checksums preserve the shot pairing required below. Large generated datasets and checkpoints are separate from the source-code repository.

## 3. Targets, core ownership, and simulator labels

### 3.1 Algebraic definition

All binary algebra is over $\mathbb F_2$. For a sampled fault history $e$, write

$$
D=He,\qquad Y=Le,
$$

where $D$ is the vector of physical detector outcomes and $Y$ is the flip of the final logical $Z$ measurement relative to the noiseless experiment. The columns of $H$ and $L$ describe elementary fault effects. This notation does not assume that all fault indicators are independent.

Partition the physical fault locations into consecutive intervals $C_1,\ldots,C_J$, each containing $c=d$ completed extraction rounds. Preparation faults belong to $C_1$, and destructive-readout faults belong to $C_J$. When an extraction interval ends, place its cut after the completed ancilla measurement/reset operation and before reset noise associated with preparation of the next interval. This makes the ancillas clean at the cut. The schedule manifest, rather than a rounded detector coordinate, defines this ownership.

Define

$$
Y_j=L e_{C_j},\qquad
Y=\bigoplus_{j=1}^{J}Y_j.
$$

The observation windows may overlap, but the target ownership intervals do not. A fault whose detector effects cross a window boundary is still owned once. Its detector effects must not be clipped when generating the full physical trajectory.

### 3.2 Exact cumulative-label construction

For this memory circuit, labels can be generated without inferring fault timestamps from an aggregated detector error model. Fix the physical representative $\overline Z$ used by the final logical measurement. Let $z_j$ be the Pauli-frame sign of that representative at the clean cut after round $jc$, for $1\leq j<J$. Equivalently, for data Pauli frame $E_{jc}$,

$$
E_{jc}\overline Z=(-1)^{z_j}\overline Z E_{jc}.
$$

Use the same representative at all cuts. Set $z_0=0$ and obtain $Y$ from the original final measurement-record observable. Construct

$$
Y_1=z_1,\qquad
Y_j=z_{j-1}\oplus z_j\quad(2\leq j<J),\qquad
Y_J=z_{J-1}\oplus Y.
$$

Their XOR equals $Y$ exactly by cancellation. The final expression includes all faults between the final internal cut and destructive readout. For the stated memory schedule, a complete noiseless extraction round preserves the chosen logical operator; the clean-cut convention therefore makes these differences agree with the interval contribution definition. This preservation and the cut placement are implementation acceptance checks.

Current Stim documentation permits Pauli targets in `OBSERVABLE_INCLUDE`. These annotations affect detector/observable sampling but do not insert a physical measurement [R4]. Retain the original final observable at index 0 and add one distinct auxiliary observable at each internal cut. For a representative with support $\{q_1,\ldots,q_d\}$, the annotation has the form

```text
OBSERVABLE_INCLUDE(j) Zq1 Zq2 ... Zqd
```

Here `j` and each `qk` are placeholders for numeric indices. These targets are a list of Pauli terms; this instruction is not an `MPP` measurement. Sample the circuit once with physical detector outcomes and auxiliary observable flips returned separately. From the same shot, use observable 0 as $Y$ and observables $1,\ldots,J-1$ as the cumulative labels $z_j$.

Pin a Stim release or commit that supports this syntax. The Pauli annotations are simulator-only labels. They are excluded from the NN input, from the observed-detector mask, and from any classical baseline's syndrome vector. A measurement-record-only export cannot reconstruct these auxiliary labels. An independent reference implementation may instead inspect the simulator's Pauli frame at the same cuts.

Do not construct the classical baseline from an augmented DEM that exposes these auxiliary observables as if they were measured checks. Use the original physical circuit and its original logical observable.

### 3.3 Meaning of local errors

A core label is a contribution to a fixed global parity under the stated convention. It need not be uniquely determined by the detector record. Equivalent fault histories can change two neighboring core labels while preserving $Y$. Accordingly, per-core classification error is a diagnostic; final memory failure is the parity of all core prediction errors.

## 4. Observation policy and composition

Let core $j$ end at round $t_j=jc$. Its default detector window contains complete detector layers from the left buffer, the core, and the right buffer. The nominal bulk width is

$$
W=b_L+c+b_R=3d.
$$

Physical start and end boundaries truncate the available buffer. Windows away from the physical endpoints have open artificial temporal boundaries. A detector is available only after all measurement records needed to calculate it have arrived; the implementation records this availability explicitly. Compute physical-boundary flags from the observed layers, not from the target core's index: an interior core's left buffer can include preparation-boundary information.

For $j<J$, commit the prediction after round $\min(t_j+b_R,T)$, before destructive readout. Even if the right buffer reaches round $T$, this prediction cannot use destructive-readout-derived detectors. Only the final core may use those terminal detectors, after they become available. Already committed predictions are never revised at the final readout.

For a fixed context $R$, the common model returns

$$
q_\theta(Y_C=1\mid D_R,Q_C,M_R,B),
$$

where $Q_C$ identifies the target interval, $M_R$ identifies observed detector tokens, and $B$ contains the physical-boundary and token-type metadata. A global query identifies the union of all cores and has target $Y$; a core query identifies $C_j$ and has target $Y_j$. Query metadata and labels are unchanged when the observation context changes.

Use a single parameter set and a single query-conditioned output interface. Missing observations are represented by exclusion or an explicit observation mask, not by an asserted zero syndrome. Preserve the coordinate and position convention across full-context and window inference. A window crop must not silently create a new physical initialization or termination.

With threshold $1/2$,

$$
\widehat Y_j=\mathbf1\{q_\theta(Y_j=1\mid D_{W_j},Q_j,M_j,B)\ge1/2\},
\qquad
\widehat Y_{\mathrm{window}}=\bigoplus_j\widehat Y_j.
$$

The initial study resets the model state for each window and uses explicit buffers. Hidden-state or cache transfer between windows is a later factor because it changes the retained information. No correction representatives or artificial-defect updates are needed for this direct-parity composition. Such updates are required by ordinary correction-commit sliding-window methods [R5, R6].

## 5. Training and inference conditions

### 5.1 Full-context initialization

For the first scientific pilot, construct the distance-5 query-conditioned model directly and randomly initialize all its parameters: GNN, learned empty-graph embedding, $2\times2$ spatial tile aggregator, query-metadata projection, four-layer GRU, and shared scalar readout. Retain hidden width 256 and four GRU layers, and train all parameters. The backbone and distance-dependent aggregation follow the chosen Lange implementation [R9]; the common query interface is the study adaptation. A released checkpoint or a distance-3 model is not required for this initial run.

The default main-study policy is also `training.initialization=random`, with independent initialization and training streams for each distance and seed. Compatible-parameter transfer from a distance-3 model is an optional later policy, based on the upstream transfer mechanism [R9]. If adopted for a separate campaign, fix it before that campaign, record parent checkpoint hashes and copied/fresh parameter names, and use the same preparation policy across all ablation arms. Parent-training cost and dependence between models sharing a parent must be reported. Do not mix random and warm-start initialization when attributing a difference to context training.

For each distance, train a common model $\theta_F$ on complete closed trajectories. Every trajectory has one global label and the fixed set of 12 core labels. Write $p_\theta^*(D)$ for the probability of $Y=1$ under the global query and $p_\theta^j(D)$ for the probability of $Y_j=1$ under core query $j$. The population training objective is

$$
\mathcal L_F(\theta)
=\mathbb E\left[\frac{1}{1+\lambda}\operatorname{BCE}(Y,p_\theta^*(D))
+\frac{\lambda}{(1+\lambda)J}\sum_{j=1}^{J}
\operatorname{BCE}(Y_j,p_\theta^j(D))\right].
$$

Set $\lambda=1$ initially, so the global and core objectives have equal total weight irrespective of $J$. To avoid evaluating 13 full-history queries per trajectory during training, sample one core index $K$ uniformly from $\{1,\ldots,J\}$ independently for each trajectory. Evaluate its global query and only that sampled core query. The per-trajectory stochastic loss

$$
\widehat{\mathcal L}_F
=\frac{\operatorname{BCE}(Y,p_\theta^*(D))
+\lambda\operatorname{BCE}(Y_K,p_\theta^K(D))}{1+\lambda}
$$

is an unbiased estimator of the stated objective. Evaluation still predicts every core. The weight and query-sampling policy are proposed defaults to validate in the pilot and then freeze. The model learns core-query semantics under full context before streaming evaluation, preventing an unseen output task from being mistaken for missing-context degradation.

### 5.2 Primary conditions

| ID | Checkpoint/training | Inference queries and context | Final prediction |
|---|---|---|---|
| G | $\theta_F$, full-context training | One global query; complete closed history | Global prediction |
| F | Exactly the same $\theta_F$ | Core queries; complete closed history for every core | XOR of core predictions |
| S | Exactly the same $\theta_F$, frozen | Core queries; deployment windows | XOR of core predictions |
| S+ | $\theta_+$: additional full-context training from $\theta_F$ | Core queries; deployment windows | XOR of core predictions |
| W | $\theta_W$: fixed-window fine-tuning from $\theta_F$ | Core queries; the same deployment windows | XOR of core predictions |
| D | $\theta_D$: dynamic-window fine-tuning from $\theta_F$ | Core queries; the same deployment windows | XOR of core predictions |

G, F, and S use identical weights, normalization state, feature encoding, and thresholds. F is necessary because replacing one global prediction with XOR-composed core predictions is itself an algorithmic change. The following measured contrasts have distinct interpretations:

$$
\Delta_{\mathrm{total}}=P_S-P_G,\qquad
\Delta_{\mathrm{composition}}=P_F-P_G,\qquad
\Delta_{\mathrm{context}}=P_S-P_F.
$$

They satisfy $\Delta_{\mathrm{total}}=\Delta_{\mathrm{composition}}+\Delta_{\mathrm{context}}$ arithmetically. Neither term is constrained to be positive for a learned model. The composition term includes the difference between global and core-query predictions; it is not a pure estimate of an abstract XOR penalty. In particular, XORing marginal MAP core decisions need not equal a global MAP decision.

The main adaptation contrast is $P_{S+}-P_W$. This controls for additional optimization, unlike $P_S-P_W$ alone. Compare $P_W-P_D$ at the default deployment window and across the predeclared buffer sweep to assess dynamic-context training.

Also evaluate the global and full-context core queries of $\theta_+$, $\theta_W$, and $\theta_D$ on the same test shots. These companion evaluations show whether adaptation changes the offline task or the composition gap; account for their inference cost separately.

### 5.3 Matched additional training

All three additional-training arms start from the same $\theta_F$ checkpoint within a seed replicate. Fix the effective batch size at $B$ complete trajectories per optimizer update. With $U_1$ continuation updates, each arm processes exactly $U_1B$ newly sampled trajectories and $2U_1B$ query examples: one global and one uniformly sampled core query per trajectory. Use the same trajectory stream, minibatch order, and sampled core indices in S+, W, and D. Gradient accumulation may accommodate memory limits, but its effective batch size and update count must remain identical.

In configuration, `training.max_steps` for a continuation denotes $U_1$ new updates, excluding the parent checkpoint's $U_0$. Record phase-local and cumulative counters separately. Use a continuation trajectory/core-query namespace distinct from base training and identical across S+, W, and D; restarting the base-training stream or keying it by the arm name would violate this contract. Resuming an interrupted arm restores its continuation cursor and remaining budget.

At the continuation fork, copy the model parameters and normalization buffers from $\theta_F$. Reset the optimizer state and scheduler in every arm, including S+, using the same optimizer class, hyperparameters, and learning-rate schedule fixed in the pilot. No arm reinitializes model layers. This makes the additional-update and data budgets operational rather than relying on an ambiguous epoch count.

S+ retains $\mathcal L_F$. W retains the global full-context term but replaces the core inputs by windows with $b_L=b_R=d$. D also retains the global term, while independently sampling the left and right core buffers from

$$
\mathcal B_d=\{0,\lceil d/2\rceil,d\}.
$$

For D, draw one buffer pair per minibatch and use it for all shots in that minibatch; keep $c=d$, the core ownership, and the final-readout availability policy fixed. Initially sample the nine buffer pairs uniformly. This is a proposed dynamic-window policy inspired by dynamic chunk training [R1], not the exact ASR training recipe.

All methods are evaluated at the same fixed buffer setting in any one comparison. Dynamic training is not evaluated with a larger window than its comparator. Additional full-context global queries in W and D make the adaptation a mixed full/window objective; this choice is explicit and is identical across those two arms.

Matching updates and trajectories does not match floating-point work when contexts have different lengths. Record processed detector tokens, wall time, allocated CPU-core hours, actual CPU/thread settings, and peak resident memory in addition to the primary data/update budgets. Do not claim equal computational cost solely from equal update counts.

## 6. Experimental stages and minimal matrix

### Stage 0: data and interface validation

Establish the exact label identity, physical observation masks, independent-window inference, and global/core query interface before long training. Validate noiseless trajectories and selected single faults near preparation, internal cuts, spatial boundaries, and final readout. Optional small $d=3$ fixtures may reduce the cost of individual invariant checks, but include a $d=5,T=60$ integration check. This stage requires no distance-3 training and makes no comparative-performance claim.

### Stage 1: pilot at one operating point

Use `pilot_d5`, `runtime=cpu`, and `training.initialization=random`: $d=5$, $T=60$, $c=b_L=b_R=5$, $W=15$, and $p=0.003$. First profile the fixed architecture on the CPU node to determine batch size and execution concurrency. Then train one full-context model and run G/F/S, estimating update costs, validation needs, and failure/discordance rates. Run a short S+/W/D continuation to check that all arms execute and that their objectives receive nonzero gradients. Select the main training and shot budgets from these measurements before inspecting the main test set. Model width and GRU depth remain fixed throughout this resource calibration.

### Stage 2: primary transfer and adaptation study

After the distance-5 pilot establishes feasible budgets, at $d\in\{5,7\}$ and $p=0.003$ train three independent $\theta_F$ replicates per distance under the declared initialization policy. Confirm the execution budget for the longer distance-7 histories before the main runs. Fork each model into S+, W, and D using the matched additional budget. Evaluate all six primary conditions on identical held-out complete trajectories, with $b_L=b_R=d$ for window inference. Within every replicate, G/F/S still share one exact checkpoint and all continuation arms start from it.

### Stage 3: context dependence

Reuse the Stage 2 checkpoints and test trajectories. First fix $b_L=d$ and evaluate $b_R\in\mathcal B_d$. Then fix $b_R=d$ and evaluate $b_L\in\mathcal B_d$. The shared default need only be evaluated once. This produces five distinct buffer settings instead of a full nine-setting inference grid and separates the first examination of future and past context.

### Stage 4: optional extensions

Run $d=9$ only after the preceding stages establish a feasible shot and training budget. The secondary $p\in\{0.001,0.002,0.003,0.005\}$ evaluation uses unchanged checkpoints and is reported as noise-distribution transfer. A change in $T/d$, a second neural backbone, hidden-state transfer, color-code support, or end-to-end composition losses is a separate extension, not part of the initial factorial comparison.

## 7. Evaluation and statistical precision

### 7.1 Primary observable

For decoder condition $a$ on shot $n$, define

$$
E_{a,n}=\mathbf1\{\widehat Y_{a,n}\ne Y_n\},\qquad
\widehat P_a=\frac1N\sum_{n=1}^{N}E_{a,n}.
$$

Report final memory LER per complete shot as the primary metric, together with failure counts, shot counts, and 95% Wilson intervals. Report absolute paired differences and relative ratios with uncertainty; state which rate is used as the denominator. Do not call a change of 0.001 an improvement of 0.1% without specifying whether this means an absolute probability, percentage points, or a relative change.

For the window-composed conditions, let $e_j=\widehat Y_j\oplus Y_j$. Then

$$
E=\bigoplus_j e_j.
$$

Record per-core disagreement rates, the distribution of $K=\sum_j e_j$, and adjacent-core error correlations. Report first, interior, and last core positions separately, and retain flags for whether each observed window touches a physical boundary. These diagnose cancellation and endpoint effects; they do not replace the memory LER. The identity $P(E=1)=P(K\bmod2=1)$ is an exact consistency check.

Keep per-core and global-query cross entropy as secondary probabilistic metrics. Do not turn marginal core probabilities into a claimed exact global probability using an independence assumption that has not been established.

### 7.2 Paired comparisons

All conditions decode the same test trajectories. For conditions A and B, let $n_{10}$ count shots where A fails and B succeeds, and let $n_{01}$ count the converse. Then

$$
\widehat\Delta_{A-B}=\frac{n_{10}-n_{01}}N,
\qquad
\widehat q=\frac{n_{10}+n_{01}}N.
$$

For planning at large $N$, an approximate standard error is

$$
\operatorname{SE}(\widehat\Delta)
\simeq\sqrt{\frac{\widehat q-\widehat\Delta^2}{N}}.
$$

Use paired bootstrap confidence intervals over whole trajectories [R7]. For statistics depending only on the two failure indicators, drawing the four contingency counts from their empirical multinomial distribution is equivalent to a nonparametric paired bootstrap and avoids materializing resampled arrays of millions of shots. Never resample windows as independent observations.

For sparse or zero discordances, use a prespecified conservative interval for the difference. Compute two-sided 97.5% exact binomial intervals $[L_{10},U_{10}]$ and $[L_{01},U_{01}]$ for $p_{10}$ and $p_{01}$, each using $N$ total shots [R8]. Then $[L_{10}-U_{01},U_{10}-L_{01}]$ has at least 95% coverage for $P_A-P_B$ by the union bound, without assuming independence of the counts. This remains nondegenerate when no discordances are observed. If ratios have too few denominator failures, report the counts and interval limits rather than a finite estimate with unjustified precision.

For training randomness, report each seed's result and a prespecified aggregate. The same-shot error bars are conditional on a checkpoint; they do not measure variability across independently trained models. With only three training seeds, do not describe between-seed uncertainty as precisely estimated.

### 7.3 Shot-budget planning

For a small single failure probability, a 95% interval's relative half-width is approximately $1.96/\sqrt{k}$ when $k$ failures are observed. Thus roughly 400 failures support 10% relative precision and 1,500 support 5%, before considering training-seed variation.

Power for a paired difference depends on discordance $q$, not only on either marginal LER. For a target absolute difference $\delta$, a rough planning approximation is

$$
N\approx\frac{(z_{1-\alpha/2}+z_{1-\beta})^2q}{\delta^2},
$$

where $1-\beta$ is desired power and $q$ is estimated from pilot data. The approximation is used to choose a fixed main test budget; it is not a guarantee for very sparse events. Define the practically relevant $\delta$ or relative improvement before the main test. A small nonsignificant difference supports a bound only when the interval is sufficiently narrow.

Use pilot data to select a fixed shot budget and a maximum feasible budget. Do not repeatedly inspect the main test and stop when a significance threshold is crossed. If the maximum budget cannot resolve the chosen difference, report the resulting interval and the unresolved precision.

### 7.4 Interpretation of full versus restricted context

For the same target and nested observations, Bayes classification risk cannot increase when more information is available. At the optimal log-loss predictor,

$$
H(Y_j\mid D_{W_j})-H(Y_j\mid D)
=I(Y_j;D_{\overline{W_j}}\mid D_{W_j})\geq0,
$$

where $D_{\overline{W_j}}$ denotes the complementary detector observations. The empirical frozen-model contrast $P_S-P_F$ includes both restricted information and model mismatch. The improvement after fine-tuning measures recovered performance. The remaining gap is not automatically an irreducible information-theoretic loss.

## 8. Data, seeds, and checkpoint policy

Split complete trajectories into training, validation, and test sets before forming windows. All windows and queries derived from one trajectory stay in the same split. Preserve low-weight and zero-detector shots; the initial study uses no failure oversampling or class-balanced sampling.

Maintain separate seeds for circuit sampling, model initialization, minibatch ordering, sampled core indices, and dynamic buffer sampling. Within each replicate, S+, W, and D use the same new training trajectories, minibatch order, and sampled core indices; only their prescribed context masks differ. Different model replicates use independent initialization and training streams. This pairing of training replicates controls optimization randomness. Separately, the held-out test trajectories are shared across conditions and replicates to support shot-paired statistical comparisons.

Stim seeds alone are not a portable dataset identifier across versions and sampling-call structures. Retain the final test detector arrays, target arrays, shot identifiers, circuit hash, and data hash, or retain a fully specified version-pinned generation manifest and verify the regenerated hash. Storing the compact bit-packed test set is the preferred initial choice.

The primary checkpoint policy is fixed-budget evaluation. Determine $U_0$ full-context updates and $U_1$ continuation updates in the pilot, then freeze them for the main comparison. Use the checkpoint at exactly $U_0$ for G/F/S and at exactly $U_0+U_1$ for each continuation arm. Validation detects failed runs and supports pilot decisions; it is not used to choose the most favorable streaming checkpoint after seeing main-test behavior.

Save model parameters, optimizer and scheduler state, normalization buffers, random-number generator states, source commit, resolved Conda environment and package versions, data-stream position, and the resolved run configuration. Include the PBS job identity, allocated resources, and effective CPU thread/worker settings. Verify short-run interruption and resume on the cluster before long jobs, so a resumed run preserves trajectory/query streams and the prescribed update budget. Apply the optimizer-reset rule in Section 5.3 to all continuation arms. Use the same validation-update schedule for all arms and evaluate their corresponding deployment conditions on validation trajectories, never on the main test trajectories. During evaluation, disable dropout and parameter/state updates. If a normalization layer depends on the test batch, specify and control that behavior identically in all conditions.

The downloaded analysis bundle contains the fixed test labels, aligned predictions, checkpoint/run identities, resolved configurations, environment records, and checksums needed to reconstruct the paired results locally. Archive creation and transfer must preserve these identities; regenerating a different test sample after download would invalidate the existing shot pairing.

## 9. Acceptance checks before production runs

- Every sampled shot satisfies $Y=\bigoplus_jY_j$ exactly, including preparation and final-readout faults.
- Auxiliary simulator observables are absent from every NN feature tensor and every classical syndrome input.
- The noiseless circuit and the inserted Pauli-label annotations produce the intended zero flip labels.
- Selected single faults around cuts agree with an independent Pauli-frame or circuit-propagation calculation.
- Perturbing data outside an internal window's availability set does not change its committed output.
- Final-readout perturbations cannot change any already committed nonterminal-core output.
- With full context restored, the F inference path reproduces the corresponding full-context core-query outputs without a padding or position convention change.
- All paired results use the same trajectory identifiers, and XOR-composed failures agree with the odd parity of core disagreements.
- Any classical reference uses the original physical detector/observable definition. An MWPM reference is a useful benchmark, not a label oracle.

These checks validate the experimental object. They do not assert that the NN matches a published performance curve or that a chosen window preserves code distance.

## 10. Primary references

- [R1] X. Li et al., *Dynamic Chunk Convolution for Unified Streaming and Non-Streaming Conformer ASR* (2023). [Paper](https://arxiv.org/abs/2304.09325). Relevant sections: 2.2–2.4 and Table 2.
- [R2] Zhang et al., *Learning to Decode in Parallel: Self-Coordinating Neural Network for Real-Time Quantum Error Correction*. [Paper](https://arxiv.org/abs/2601.09921). Relevant content: overlapping input windows, core contributions, and correlated local prediction errors.
- [R3] Stim, `Circuit.generated` and sampler APIs. [Python API reference](https://github.com/quantumlib/Stim/blob/main/doc/python_api_reference_vDev.md). The implementation must pin a release or commit rather than depend on moving `main`.
- [R4] Stim, `OBSERVABLE_INCLUDE` gate documentation. [Gate reference](https://github.com/quantumlib/Stim/blob/main/doc/gates.md#the-observable_include-instruction). Relevant feature: Pauli-target annotations for detector/observable sampling.
- [R5] L. Skoric et al., *Parallel window decoding enables scalable fault tolerant quantum computation*, Nature Communications 14, 7040 (2023). [Published PDF](https://eprints.whiterose.ac.uk/id/eprint/204962/1/s41467-023-42482-1.pdf). Relevant content: committed correction edges, artificial defects, and the limitation of a homology-only inner decoder.
- [R6] X. Tan et al., *Scalable surface code decoders with parallelization in time*. [Paper](https://arxiv.org/abs/2209.09219). Relevant sections: supplementary S2.2–S2.4.
- [R7] SciPy, `scipy.stats.bootstrap`. [Official API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html). Relevant option: `paired=True` for aligned samples.
- [R8] SciPy, `BinomTestResult.proportion_ci`. [Official API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html). Relevant methods: `wilson` and `exact` (Clopper–Pearson). The conservative paired-difference interval above is a construction from these marginal bounds.
- [R9] Lange et al., *Modular Neural Decoder for Surface Code Memory and Logic on Superconducting Processors*, [paper](https://arxiv.org/abs/2610.03869), and author implementation at commit `458210ac74ed0047760b8789c169e6b9eff732b8`: [model and `init_from`](https://github.com/LangeMoritz/modular-gnn-rnn-decoder/blob/458210ac74ed0047760b8789c169e6b9eff732b8/gnnrnn/model.py), [training entry point](https://github.com/LangeMoritz/modular-gnn-rnn-decoder/blob/458210ac74ed0047760b8789c169e6b9eff732b8/scripts/train.py).
- [R10] Conda, *Managing environments*. [Official documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Relevant operations: environment creation, export, and package/version recording.
- [R11] PyTorch, *Multiprocessing best practices*, CPU oversubscription. [Official documentation](https://docs.pytorch.org/docs/2.7/notes/multiprocessing.html#cpu-in-multiprocessing). Relevant constraint: coordinate process and per-process thread counts with the actual CPU allocation.
