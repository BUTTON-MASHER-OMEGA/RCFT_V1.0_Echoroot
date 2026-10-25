# Rediscovered LCO-related toys: inspection and evidence review

## Scope and provenance

Inspected all 11 Python files in `projects/lco-related/` on 2026-10-24. These are newly supplied historical inputs, separate from the onboarding record. No source file, RCF/RCMT contract, previous maintenance report or frozen experiment was modified. The external IDE path under `projects/rcft_related/` was not opened; the copied files under theory were sufficient. No other agent's output was consulted.

All 11 files parse. Source hashes and function/line inventory are in `source-manifest.json` and `inventory.json`. The folder supplies code, not historical output logs or a reproducible execution manifest. Copy timestamps do not establish the original dates or a definitive chronological lineage. Conclusions below distinguish static code findings, bounded new runs, and historical interpretation.

Seven small deterministic/enumerative entrypoints completed, each under a five-second hard timeout and each actually under one second. Full gamma/sample sweeps and the two Matthew trajectory defaults were not run. Geometry constructors were separately inspected and evaluated without invoking their sampling loops. The source-integrity check confirms all 11 original files unchanged.

## Inventory

| File | What the code actually does | Inspection/execution |
|---|---|---|
| `v_test.1.py` | Enumerates 64 binary configurations; chooses M1 by minimum summed incident loss; compares mean-plus-variance motif score with global edge penalty change | Full small run |
| `mode_sphere_test.1.py` | Embeds binary demands along three chosen 120° axes, minimizes over eight planar directions, then projects back to a bit using the x component | Full small run |
| `u1-relational-universe-probe.py` | Enumerates 4^6 configurations, chooses M1 from 16 directions, updates the vector directly, and measures ΔV | Full small run |
| `skew_probe.py` | Adds a fixed historical anchor and memory-weighted loss; active gamma is zero | Full small run, default only |
| `isp_phi_probe.py` | Greedily adds an earlier scalar capacity to the latest one under an imposed rational overlap threshold | Full 15-capacity toy |
| `cell-600-test.py` | Generates 120 unit vertices and lists rounded pairwise inner-product classes | Full small static run |
| `constraint_regime_sweep.py` | Computes 600-cell and E8 inner-product spectra; prints assumed dual-demicube descriptions | Full small static run; not an actual gamma sweep |
| `polytope-packing.py` | At gamma=1/phi, snaps random four-vector demand sums to nested 8/24/120-point sets and reports mean-plus-variance costs | Read completely; constructors checked; 10,000-sample default skipped |
| `h3-parameter-sweep.py` | Samples 2,000 fresh random motifs at each of 21 gamma values in [0,1], compares 5/8/24/120-point packings | Read completely; constructors checked; full sweep skipped |
| `Matthew/dynamic_crystalization.py` | Adds a visit-count reinforcement discount to geometric mismatch and chooses minimum cost per packing | Read completely; constructors checked; 10,000-step default skipped |
| `Matthew/test-1.py` | Reinforced random walk with a 50-transition sliding memory window, a fixed observation partition, and first-/second-order empirical observation kernels | Read completely; exact carrier/probability check only; two 20,000-step defaults skipped |

## What these recover from the older research

### 1. Genuine computational substrate for some geometry vocabulary

The static 600-cell probes reproduce three positive nonidentity dot-product tiers: approximately **0.80902, 0.5, 0.30902**. The E8 probe produces **0.5** as its one positive nonidentity tier. These are numerical finite-coordinate calculations, not measured survival rates. The 600-cell generated here has 120 distinct unit vertices and nearest-neighbor degree 12; the other packing constructors supply 8/24-point unit sets with degrees 6/8. The five-point simplex constructor also passes the small count/norm/degree screen. The first comment in `cell-600-test.py` incorrectly calls the axis family “16 vertices”; its loop actually constructs eight, and the final count is correct.

Sources: [projects/lco-related/cell-600-test.py:4](C:/Users/blaur/projects/theory/projects/lco-related/cell-600-test.py:4), [projects/lco-related/cell-600-test.py:53](C:/Users/blaur/projects/theory/projects/lco-related/cell-600-test.py:53), [projects/lco-related/constraint_regime_sweep.py:37](C:/Users/blaur/projects/theory/projects/lco-related/constraint_regime_sweep.py:37); new results: [projects/ops-codex/research/old-toys-review/geometry-checks.json:1](C:/Users/blaur/projects/theory/projects/ops-codex/research/old-toys-review/geometry-checks.json:1) and the corresponding stdout files.

This narrows the earlier provenance gap for the **tier-count statements**. It does not recover the historical gamma-vs-friction table or establish the full survivor hierarchy.

### 2. The dual-demicube result is supplied rather than computed here

`constraint_regime_sweep.py` explicitly comments “Dual 5-Demicube properties injected from RCFT axioms” and prints 26 vertices, irregular spectrum and “slush-dominated” without constructing that geometry or evaluating it. Despite the filename, this file does not scan gamma or evaluate F(G,gamma).

Source: [projects/lco-related/constraint_regime_sweep.py:75](C:/Users/blaur/projects/theory/projects/lco-related/constraint_regime_sweep.py:75). Its printed demicube text cannot be classified as a reproduced geometric calculation. The surrounding 600-cell and E8 spectra are computed and must remain distinguished from that injected middle case.

### 3. A plausible source family for historical friction language, but not the missing table

The packing toys minimize the **sum** of relation losses by using a normalized weighted demand vector and its nearest packing vertex. They then report a different diagnostic, **mean(losses) + variance(losses)**. The reporting diagnostic is not what the argmax/nearest-snap step explicitly minimizes.

Sources: [projects/lco-related/polytope-packing.py:69](C:/Users/blaur/projects/theory/projects/lco-related/polytope-packing.py:69), [projects/lco-related/polytope-packing.py:81](C:/Users/blaur/projects/theory/projects/lco-related/polytope-packing.py:81), [projects/lco-related/h3-parameter-sweep.py:77](C:/Users/blaur/projects/theory/projects/lco-related/h3-parameter-sweep.py:77).

This code can yield nonlinear dependence on gamma: gamma changes the selected direction/vertex, the variance term includes squared weighted memory losses, and the sweep draws fresh random motifs for each gamma rather than reusing one fixed sample set. That supplies a concrete reason not to identify these toy scores automatically with a fixed-coefficient affine F(G,gamma).

It does **not** establish the provenance of `density-regimes.md`: the actual sweep here only uses gamma 0–1 and compares simplex/16-cell/24-cell/600-cell; it neither computes the dual-demicube nor compares E8 friction across gamma 0–5. No supplied output ties it to the table's numbers. No fit or conversion was attempted. The existing historical-provenance caveat remains justified.

### 4. Monotonicity depends on the chosen policy and carrier map

| Small toy | Actual bounded-run outcome | Supported interpretation |
|---|---|---|
| Binary 64-state model | 32 changes; ΔV is −3 for 8 and −1 for 24; correlation 1.0 | This particular coordinate-minimization policy decreases the defined edge penalty |
| Vector-sphere → bit model | 28 changes; ΔV=+1 for 8, −1 for 16, +3 for 4; correlation about 0.7628 | The chosen vector embedding and bit projection do not preserve the binary objective |
| Native U(1) model | 3,456 changes; largest ΔV about −0.2304; correlation about 0.4004 | Minimizing the changing incident terms lowers this particular V |
| Fixed-anchor model at active gamma=0 | 3,456 changes; largest ΔV about −0.2304; minimum reported motif cost zero | Active default disables the anchor's contribution; no epsilon-band confirmation follows |

Sources: [projects/lco-related/v_test.1.py:24](C:/Users/blaur/projects/theory/projects/lco-related/v_test.1.py:24), [projects/lco-related/mode_sphere_test.1.py:28](C:/Users/blaur/projects/theory/projects/lco-related/mode_sphere_test.1.py:28), [projects/lco-related/mode_sphere_test.1.py:66](C:/Users/blaur/projects/theory/projects/lco-related/mode_sphere_test.1.py:66), [projects/lco-related/u1-relational-universe-probe.py:39](C:/Users/blaur/projects/theory/projects/lco-related/u1-relational-universe-probe.py:39), [projects/lco-related/skew_probe.py:38](C:/Users/blaur/projects/theory/projects/lco-related/skew_probe.py:38). Raw stdout is retained alongside this report.

The binary and native-vector cases include the current M1 among the candidates, and their minimized summed loss contains exactly the global terms that change when M1 changes. Their descent is consequently built into this particular coordinate-update construction. The positive-damage vector-to-bit branch is equally valuable: optimization in the auxiliary representation does not ensure improvement after the chosen projection.

This is consistent with the current evaluator/policy distinction: an explicitly selected policy can have a descent property without making it a universal LCO guarantee. No general theory revision is needed to accommodate either outcome. Correlation after choosing an update is not independent validation of the diagnostic, and these runs do not prove an ISP surrogate “perfect.”

### 5. Historical “Sp(8) proxy” and epsilon language is interpretive

`skew_probe.py` calls a two-dimensional fixed-anchor dot-product penalty “Sp(8) drag/proxy.” It does not define an eight-dimensional symplectic carrier, an alternating form or a form-preserving group action. Its fixed anchor is not itself a history-updating process, and the active gamma=0 eliminates that term. The commented values 0.618, 0.05 and 9 are choices/comments, not a documented parameter scan or mapping to the current Temporal Interface.

Source: [projects/lco-related/skew_probe.py:22](C:/Users/blaur/projects/theory/projects/lco-related/skew_probe.py:22), [projects/lco-related/skew_probe.py:38](C:/Users/blaur/projects/theory/projects/lco-related/skew_probe.py:38), [projects/lco-related/skew_probe.py:123](C:/Users/blaur/projects/theory/projects/lco-related/skew_probe.py:123). The conditional “EPSILON-BAND CONFIRMED” print is not reached in the default run; even in another setting a positive finite-sample minimum would require a separate argument before a universal conclusion.

### 6. The phi frontier is a useful recurrence toy, not a recovered determinant theorem

The default scalar frontier produces 1,2,3,5,8,...,987. It seeds 1 and 2, searches prior values from largest downward, and adds the first admissible one. With the default threshold, the preceding capacity is repeatedly admitted, yielding the Fibonacci recurrence and its familiar ratio behavior. The loss function is the supplied rational expression p*c/|p²−c²|; no kernel matrix, determinant, logarithm, graph shortest-path cost or proof of that loss is computed.

Source: [projects/lco-related/isp_phi_probe.py:3](C:/Users/blaur/projects/theory/projects/lco-related/isp_phi_probe.py:3), [projects/lco-related/isp_phi_probe.py:22](C:/Users/blaur/projects/theory/projects/lco-related/isp_phi_probe.py:22). “Log-det” and “Dijkstra-variant” are analogies in these comments, not established implementations. The output is a verified toy recurrence under its assumptions, not an independent derivation of a physical golden-ratio law.

### 7. The older eight-state reinforcement model is related in vocabulary, distinct in structure

The Matthew model uses the same pairwise observation partition {0,1}/{2,3}/{4,5}/{6,7}, but has **ten undirected edges** with degrees 2/3, rather than the frozen OPS eight-cycle. It uses a normalized Boltzmann transition law, projection-boundary cost, directed-edge reinforcement, and a finite sliding window. Those are explicit implementation choices; do not identify it with the later frozen carrier without a map.

Source: [projects/lco-related/Matthew/test-1.py:8](C:/Users/blaur/projects/theory/projects/lco-related/Matthew/test-1.py:8), [projects/lco-related/Matthew/test-1.py:19](C:/Users/blaur/projects/theory/projects/lco-related/Matthew/test-1.py:19), [projects/lco-related/Matthew/test-1.py:40](C:/Users/blaur/projects/theory/projects/lco-related/Matthew/test-1.py:40), [projects/lco-related/Matthew/test-1.py:66](C:/Users/blaur/projects/theory/projects/lco-related/Matthew/test-1.py:66).

At zero reinforcement, two hidden states in observation A already have different aggregate transition probabilities to observations. For state 0 the row is approximately (0.786986,0.106507,0,0.106507); for state 1 it is (0.880797,0.119203,0,0). Thus the partition is not generically strongly lumpable even with lambda=0. A projected first-/second-order discrepancy is therefore not, on its own, a clean demonstration that reinforcing memory created all observable memory. This exact small transition check is recorded in `old-carrier-check.json`; no random walk was run. Special initial laws would need their own analysis.

### 8. Reinforcement-based “crystallization” includes an explicit decreasing cost contribution

The dynamic toy subtracts lambda*(1−exp(−visit_count/tau)) and increments the selected vertex's count. Repeated visits lower that candidate's future discounted cost by construction. Gamma=1/phi is supplied, not discovered. The anchor is freshly randomized at each step; the persistent memory here is the visit ledger. A declining reported score alone would not separate geometric organization from the imposed reinforcement discount.

Source: [projects/lco-related/Matthew/dynamic_crystalization.py:48](C:/Users/blaur/projects/theory/projects/lco-related/Matthew/dynamic_crystalization.py:48), [projects/lco-related/Matthew/dynamic_crystalization.py:73](C:/Users/blaur/projects/theory/projects/lco-related/Matthew/dynamic_crystalization.py:73). Its default trajectory was not executed, so no crystallization outcome is claimed here.

## Research-record implications

These are worthwhile recovered implementation examples. They make the early sum/dispersion scoring, projection sensitivity, packing comparisons, scalar phi recurrence and reinforcing-memory ideas executable and inspectable. They also document how freely earlier prototypes combined evaluation and selection before the current contract separation.

They do not recover a two-lane necessity proof, a symplectic realization, a computed demicube survival result, three generation/offset derivation, density-table provenance, or a physically validated ISP/LCO. No active source doctrine was rewritten from these toys. The useful next step, if assigned, would be a scoped provenance addendum for the recovered tier computations and toy objectives; that work is not initiated by this inspection.

## Verification boundary

All seven executed entrypoints exited zero. No long-running simulation was run; all four sampling/trajectory defaults listed as skipped stayed unrun. Constructors and the small transition-law check do not reproduce historical Monte Carlo outputs. No dependencies were installed. All source bytes remain unchanged, including imperfect comments and over-strong print labels, because this was an inspection rather than a repair request.

Evidence: [projects/ops-codex/research/old-toys-review/bounded-runs.json:1](C:/Users/blaur/projects/theory/projects/ops-codex/research/old-toys-review/bounded-runs.json:1), [projects/ops-codex/research/old-toys-review/geometry-checks.json:1](C:/Users/blaur/projects/theory/projects/ops-codex/research/old-toys-review/geometry-checks.json:1), [projects/ops-codex/research/old-toys-review/old-carrier-check.json:1](C:/Users/blaur/projects/theory/projects/ops-codex/research/old-toys-review/old-carrier-check.json:1), [projects/ops-codex/research/old-toys-review/integrity.json:1](C:/Users/blaur/projects/theory/projects/ops-codex/research/old-toys-review/integrity.json:1).
