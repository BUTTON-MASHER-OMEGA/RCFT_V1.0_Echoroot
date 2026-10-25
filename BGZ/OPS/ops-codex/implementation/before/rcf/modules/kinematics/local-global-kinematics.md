# Local and Global Kinematics

## Document Contract

- owner: RCF kinematics module
- type: Interpretation
- scope: module-local content of `kinematics/local-global-kinematics.md`
- authority: tier-8 downstream change-representation layer
- upstream dependencies: `/modules/kinematics/interface.md`
- status: active
- purpose: represent admissible change without defining dynamics, causality, or temporal law
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

## **1. Purpose**

Define the relationship between **local** and **global** kinematic structure within RCF/RCD.

Local kinematics describes **immediate, refinement‑compatible transitions**.
Global kinematics describes **large‑scale coherence, attractor structure, and basin geometry**.

The two must remain consistent:

> **Local descriptors must integrate into global structure without contradiction.
> Global structure must constrain local behavior without imposing dynamics.**

---

## **2. Relation to RCD and Kinematic Space**

RCD provides:

- admissible transitions
- coherence gradients
- attractor basins
- cascade behavior
- non‑Markovian admissibility

Kinematic space provides:

- admissible states
- admissible transitions
- refinement structure
- equivalence classes

Local/global kinematics describes **how these structures interact across scale**.

---

## **3. Local Kinematics**

Local kinematics concerns **infinitesimal or small‑scale admissible transitions**:

\[
R \rightarrow R'
\]

Local descriptors include:

- Jacobian entropy
- local instability
- refinement‑level divergence
- immediate admissibility changes
- local coherence shifts

Local structure is **sensitive**, **history‑dependent**, and **refinement‑compatible**.

### **3.1 Local Neighborhood**

Define the local neighborhood of a state \(R\):

\[
\mathcal{N}(R) = \{ R' \mid R \rightarrow R' \text{ is admissible} \}
\]

This neighborhood is:

- non‑metric
- refinement‑stable
- shaped by constraint geometry
- history‑dependent

---

## **4. Global Kinematics**

Global kinematics concerns **large‑scale relational structure**:

- attractor basins
- coherence gradients
- long‑range refinement paths
- global equivalence classes
- cascade endpoints

Global structure is **coarse**, **stable**, and **refinement‑invariant**.

### **4.1 Global Basin Structure**

Define the basin of an attractor \(R^*\):

\[
\mathcal{B}(R^*) = \{ R \mid R \rightarrow \dots \rightarrow R^* \}
\]

Basins are:

- non‑temporal
- refinement‑compatible
- shaped by constraint accumulation
- stable under relabeling

---

## **5. Local–Global Consistency Condition**

Local and global kinematics must satisfy:

\[
\text{Local}(R) \Rightarrow \text{Global}(R)
\]

and

\[
\text{Global}(R) \Rightarrow \text{Local}(R)
\]

up to refinement‑induced equivalence.

This ensures:

- local instability does not contradict global coherence
- global attractors do not override local admissibility
- cascades remain consistent across scale
- refinement does not introduce scale‑dependent artifacts

---

## **6. Local Diagnostics Feeding Global Structure**

Local descriptors (e.g., Jacobian entropy) integrate into global structure:

- high local entropy → approach to cascade
- repeated local convergence → approach to attractor
- local refinement → global basin sharpening
- local instability → global boundary formation

Thus:

> **Global structure is the accumulation of local behavior under constraint.**

---

## **7. Global Structure Constraining Local Behavior**

Global structure constrains local transitions:

- attractor membership restricts admissible local moves
- basin boundaries limit local divergence
- coherence gradients bias local refinement
- cascade endpoints restrict future admissibility

Thus:

> **Local behavior is shaped by global constraint geometry.**

---

## **8. Refinement Across Scales**

Refinement must preserve both local and global structure:

### **8.1 Local Refinement**
\[
R \prec R' \quad \Rightarrow \quad \mathcal{N}(R) \subseteq \mathcal{N}(R')
\]

### **8.2 Global Refinement**
\[
\mathcal{B}(R^*) = \mathcal{B}(\text{Refine}(R^*))
\]

Refinement cannot:

- create new attractors
- destroy existing basins
- alter global coherence gradients

It only increases resolution.

---

## **9. Non‑Markovian Scale Interaction**

Because admissibility is history‑dependent:

- local transitions depend on global history
- global structure depends on accumulated local eliminations
- identical local states may diverge globally
- global basins may shift under repeated local alignment

Thus:

> **Local and global kinematics are mutually non‑Markovian.**

---

## **10. Scope**

This module does **not** introduce:

- metric geometry
- temporal evolution
- forces or potentials
- probabilistic dynamics
- physical interpretation

It defines only the **scale structure** of kinematic representation.

