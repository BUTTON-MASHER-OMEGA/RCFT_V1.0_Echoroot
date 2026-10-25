# Jacobian Entropy (Local Kinematic Sensitivity)

## Document Contract

- owner: RCF kinematics module
- type: Interpretation
- scope: module-local content of `kinematics/jacobian-entropy.md`
- authority: tier-8 downstream change-representation layer
- upstream dependencies: `/modules/kinematics/interface.md`
- status: active
- purpose: represent admissible change without defining dynamics, causality, or temporal law
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

## **1. Purpose**

Define **Jacobian entropy**: a local kinematic descriptor that measures the *sensitivity* of admissible relational transitions under refinement.

Jacobian entropy is:

- not thermodynamic
- not probabilistic
- not information‑theoretic
- not a physical entropy

It is a **local structural diagnostic** that quantifies how small admissible changes in a relational state propagate through the kinematic space.

---

## **2. Relation to RCD and Kinematics**

RCD defines:

- admissible transitions
- constraint‑driven elimination
- cascade thresholds
- coherence gradients
- non‑Markovian admissibility

Kinematics defines:

- how transitions are represented
- how trajectories are organized
- how local vs global descriptors behave

Jacobian entropy sits at the interface:

> **It measures how “locally unstable” a relational state is under admissible transitions.**

---

## **3. Local Transition Map**

Given an admissible transition:

\[
R \rightarrow R'
\]

define the **local transition map**:

\[
J(R) : \delta R \mapsto \delta R'
\]

where:

- \(\delta R\) is a small admissible perturbation of \(R\)
- \(\delta R'\) is the induced perturbation of \(R'\)

This is not a derivative in the calculus sense.
It is a **relational sensitivity operator**.

---

## **4. Jacobian Entropy (Definition)**

Jacobian entropy \(S_J(R)\) is defined as:

\[
S_J(R) = \log \left( \frac{\|\delta R'\|}{\|\delta R\|} \right)
\]

where:

- \(\|\cdot\|\) is any refinement‑compatible relational norm
- the log ensures additivity across transitions
- the ratio measures local amplification or suppression

Interpretation:

- \(S_J > 0\): local divergence (instability)
- \(S_J = 0\): neutral propagation
- \(S_J < 0\): local convergence (stability)

---

## **5. Refinement Compatibility**

Jacobian entropy must satisfy:

\[
S_J(R) = S_J(\text{Refine}(R))
\]

up to kinematic equivalence.

This ensures:

- no scale‑dependent artifacts
- no hidden structure introduced by refinement
- stability diagnostics remain consistent across levels

If a proposed entropy measure fails refinement compatibility, it is invalid.

---

## **6. Relation to Coherence Gradient**

Jacobian entropy is the **local** diagnostic.
The coherence gradient is the **global** diagnostic.

They relate as:

- \(S_J\) measures local sensitivity
- \(\nabla_C\) measures global coherence change
- high \(S_J\) often precedes cascade
- low \(S_J\) indicates approach to attractor

Thus:

> **Jacobian entropy is the local “slope,”
> coherence gradient is the global “direction.”**

---

## **7. Behavior Under Cascade**

During cascade:

- local perturbations amplify rapidly
- \(S_J\) spikes
- admissible modes collapse
- the resolved state has sharply lower entropy

Formally:

\[
S_J(R_{\text{pre-cascade}}) \gg S_J(R_{\text{resolved}})
\]

Jacobian entropy is therefore a **cascade precursor**.

---

## **8. Non‑Markovian Effects**

Because admissibility is history‑dependent:

- identical states may have different \(S_J\)
- entropy depends on accumulated constraint
- past eliminations shape local sensitivity

Thus:

\[
S_J(R_t) \neq S_J(R_t \mid \text{different history})
\]

Jacobian entropy is inherently non‑Markovian.

---

## **9. Trajectory Integration**

For a trajectory:

\[
\Gamma = (R_0 \rightarrow R_1 \rightarrow \dots \rightarrow R_n)
\]

define cumulative Jacobian entropy:

\[
S_J(\Gamma) = \sum_{i=0}^{n-1} S_J(R_i)
\]

This measures:

- total local instability
- accumulated sensitivity
- divergence along the trajectory

It is not temporal.
It is purely structural.

---

## **10. Scope**

Jacobian entropy does **not** introduce:

- thermodynamic entropy
- probabilistic uncertainty
- physical heat or disorder
- temporal evolution

It is a **local kinematic diagnostic** for relational sensitivity under admissible transitions.

