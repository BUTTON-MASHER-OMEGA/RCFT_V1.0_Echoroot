# LCO Operator

## Document Contract

- owner: RCF local coherence operator
- type: Contract
- scope: module-local content of `lco/lco-operator.md`
- authority: tier-5 local execution contract
- upstream dependencies: `/modules/lco/interface.md`
- status: active
- purpose: define local admissibility scoring without global selection, scheduling, or geometry ownership
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

### *Canonical Form of the Coherence Scoring Function*

This document defines the **coherence scoring function** \( f \) used by the Local Coherence Operator (LCO). The scoring function combines mismatch, memory mismatch, and dispersion into a single scalar value that quantifies how well a candidate mode preserves local relational structure.

The operator is strictly local, algebraic, and geometry‑neutral. It uses only the primitives defined in `lco-primitives.md` and must satisfy all invariants in `lco-invariants.md`.

---

## **1. Purpose of the Scoring Function**

The scoring function:

- evaluates the quality of a candidate mode
- integrates relational mismatch, memory influence, and dispersion
- provides a scalar value used by the update policy
- supports local coherence descent under admissible update rules

The operator does **not** select updates. It only produces the score that an update policy may interpret.

---

## **2. Inputs to the Scoring Function**

The scoring function receives:

- mismatch values


\[
  \{\ell_i\}_{i=1}^N
  \]



- memory mismatch


\[
  \ell_{\text{mem}}
  \]



- aggregate mismatch


\[
  \bar{\ell}
  \]



- dispersion


\[
  \text{Disp}
  \]



All of these quantities are defined in earlier module files and are constructed solely from local primitives.

---

## **3. Canonical Form of the Operator**

The LCO scoring function has the general form:


\[
C=f(\{\ell_i\}_{i=1}^N \cup\{\ell_{\text{mem}}\})
\]


where \( f \) is constructed from:

- the mismatch terms
- the γ‑weighted memory mismatch
- aggregate mismatch
- dispersion

The operator must remain:

- algebraic
- local
- deterministic
- geometry‑neutral
- invariant‑compatible

`ℓ̄` and `Disp` are admissible **derived quantities** used inside `f`

---

## **4. Admissible Structure of \( f \)**

The scoring function may use:

- linear combinations
- means
- variances
- sums of squares
- other simple algebraic combinations

provided they are:

- expressible solely in terms of mismatch and dispersion
- local (no global statistics or normalization)
- uniform across relations (The scoring function must not introduce any **additional** relation-specific weighting.)
- well‑defined for all admissible inputs

### **Example of an admissible form**



\[
C = \bar{\ell} + \text{Disp}
\]



This is not required, but illustrates the canonical structure:
a combination of average deviation and uniformity of deviation.

### **Prohibited structures**

The scoring function must **not**:

- introduce non‑local normalization
- apply relation‑specific weights
- infer or encode external geometric or topological structure
- apply nonlinear transformations to γ
- depend on any global state or downstream module

---

## **5. Operator Behavior Requirements**

The scoring function must satisfy:

### **Determinism**


For identical admissible inputs, the scoring function must return the same scalar value.



### **Well‑definedness**
No division by zero, undefined operations, or unbounded expressions.

### **Locality**
All computations depend only on:

- \( \{r_i\} \)
- \( H \)
- \( m \)
- γ

### **Compatibility with Local Coherence Descent**
The operator must be designed such that an admissible update policy can use \( C \) to enforce monotonic local coherence descent.

The operator itself does not enforce descent; it must simply not obstruct it.

---

## **6. Separation from Update Policy**

The scoring function defines:



\[
\mathcal{L}(\{r_i\}, H, m, \gamma) \rightarrow C
\]



It does **not**:

- compare candidate modes
- choose updates
- introduce stochasticity
- enforce thresholds
- perform acceptance/rejection logic

All such decisions belong to the update policy defined in `lco-update-policy.md`.

---

## **7. Status**

This document defines the canonical structure and constraints of the LCO scoring function.
Any implementation of the operator must remain expressible solely in terms of mismatch, memory mismatch, aggregate mismatch, and dispersion, and must satisfy all LCO invariants.

