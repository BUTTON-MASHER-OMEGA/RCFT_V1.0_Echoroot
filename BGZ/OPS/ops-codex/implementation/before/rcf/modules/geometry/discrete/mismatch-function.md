# mismatch-function.md

## Document Contract

- owner: RCF discrete geometry module
- type: Interpretation
- scope: module-local content of `geometry/discrete/mismatch-function.md`
- authority: tier-7 discrete survivor-recording layer
- upstream dependencies: `/modules/geometry/discrete/interface.md`
- status: active
- purpose: record candidate and surviving discrete geometries without owning projection or downstream continuous behavior
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

## purpose

This document defines the mismatch function \(F(G, \gamma)\), referred to as constraint friction or local mismatch \(C(\mu)\). It provides the analytic form used to score discrete geometries in the survival sieve under varying temporal drag \((\gamma)\) in current evaluations.

This file does not interpret the function physically; it records the cost function used to evaluate structural persistence within the present discrete-geometry formalism.

---

## the functional definition

The total constraint friction (F) for a candidate discrete geometry (G) under temporal drag \(\gamma\) is expressed as the sum of a static quantization term and a dynamic relational drag term:

\(F(G, \gamma)\) = \(\mathcal{E}*{static}(G)\) + \(\gamma \cdot \mathcal{R}*{drag}(G)\)

This decomposition reflects two contributions observed in current evaluations:

* baseline mismatch arising from discretization
* additional mismatch accumulated under recursive updates as (\gamma) increases

---

## 1. static quantization error \((\mathcal{E}_{static})\)

This term measures the baseline mismatch of the geometry at rest \((\gamma = 0)\). It is defined as the expected angular deviation when mapping a continuous isotropic mode space onto the discrete vertex set \(V_G\):

\[
\mathcal{E}*{static}(G) = \int*{S^{n-1}} \left( 1 - \max_{v \in V_G} (\hat{u} \cdot \hat{v}) \right) d\hat{u}
\]

* **Interpretation (structural):** Geometries with higher vertex density and more uniform triangle-compatible packing tend to produce lower values of this integral in current evaluations.

---

## 2. dynamic relational drag \((\mathcal{R}_{drag})\)

This term measures resistance to sequential relational updates. In the current formalism, it is **modeled as inversely proportional** to the cardinality of \(\Lambda^+_G\), the set of strictly positive, non-identity relational distance tiers:

\[
\mathcal{R}_{drag}(G) \propto \frac{1}{|\Lambda^+_G|}
\]

* **Interpretation (structural):** Geometries with multiple relational tiers provide greater transition flexibility and show lower drag contributions in current evaluations. Geometries with fewer tiers exhibit reduced flexibility and correspondingly higher drag contributions.

---

## phase transition mechanics (the ladder)

Because \(F(G, \gamma)\) scales linearly with \(\gamma\) but at different rates for different geometries, candidate geometries **can intersect** as \(\gamma\) (constraint density) increases.

* The survival sieve selects the geometry with minimal observed \(F(G, \gamma)\) at a given \(\gamma\).
* The points where these functions intersect correspond to observed transitions between density regimes in current evaluations.

---

## invariants

* \(F\) is treated as a geometric and statistical penalty function within the current formalism.
* It evaluates quantization mismatch and spectrum-dependent drag without assuming physical mass or energy.
* The observed survival of the 600-cell at low \(\gamma\) and its loss of viability at higher \(\gamma\) are consistent with this functional form in current evaluations.

