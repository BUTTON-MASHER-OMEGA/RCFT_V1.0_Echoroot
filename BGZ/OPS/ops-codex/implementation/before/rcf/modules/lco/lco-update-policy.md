# LCO Update Policy

## Document Contract

- owner: RCF local coherence operator
- type: Contract
- scope: module-local content of `lco/lco-update-policy.md`
- authority: tier-5 local execution contract
- upstream dependencies: `/modules/lco/interface.md`
- status: active
- purpose: define local admissibility scoring without global selection, scheduling, or geometry ownership
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

### *Definition of Update Selection and Application Logic*

This document defines the **update policy**: the mechanism that interprets coherence scores produced by the Local Coherence Operator (LCO) and decides how candidate updates are selected, accepted, or rejected.

The update policy is **logically separate** from the LCO.
The LCO computes a scalar coherence score.
The update policy determines what to do with it.

The update policy must not alter the operator’s scoring function or introduce any structure that violates the invariants defined in the LCO module.

---

## **1. Purpose of the Update Policy**

The update policy:

- interprets coherence scores
- selects among candidate modes
- determines whether an update is applied
- supports local coherence descent under the chosen acceptance policy
- introduces any stochasticity, thresholds, or admissibility rules

The LCO itself performs **none** of these actions.

---

## **2. Inputs to the Update Policy**

The update policy receives:

- a set of candidate modes \( \{m_j\} \)
- their corresponding coherence scores


\[
  C_j = \mathcal{L}(\{r_i\}, H, m_j, \gamma)
  \]


- any externally defined thresholds or admissibility rules
- any stochasticity parameters (if used)

The update policy does **not** modify mismatch, dispersion, or the scoring function.

---

## **3. Core Responsibilities**

The update policy is responsible for:

### **3.1 Candidate Comparison**
Evaluating multiple candidate modes based on their coherence scores.

### **3.2 Update Selection**
Choosing a candidate mode according to a rule such as:

- minimum coherence score
- thresholded acceptance
- probabilistic selection
- multi‑step descent logic

### **3.3 Enforcement of Local Coherence Descent**
Ensuring that accepted updates do not increase local incoherence.

The policy may enforce strict descent or allow controlled stochastic exploration, as long as invariants are preserved.

### **3.4 Admissibility Rules**
The policy may define:

- constraints on candidate modes
- rejection conditions
- neighborhood‑dependent rules

These rules must remain local and must not introduce global structure.

---

## **4. Prohibited Behavior**

The update policy must **not**:

- modify the LCO scoring function
- introduce non‑local normalization
- reinterpret γ or apply nonlinear transformations to it
- introduce geometric or topological assumptions
- alter mismatch or dispersion definitions
- depend on global state or downstream modules
- introduce selection logic that depends on implicit global ordering or external structure not present in the local candidate set
- use γ independently of the coherence scores produced by the LCO or introduce decision logic that depends directly on γ

The update policy operates strictly on:

- candidate modes
- their coherence scores
- local admissibility rules


---

## **5. Invariant Compatibility**

The update policy must satisfy:

- **Locality:**
  Decisions depend only on local relations, memory, and candidate scores.

- **Determinism (or well-defined stochastic behavior when stochasticity is explicitly specified):**
  Identical inputs must yield identical decisions.

- **Geometry‑neutrality:**
  No external geometric or topological structure may be inferred or introduced.

- **Well‑definedness:**
  All acceptance and comparison rules must be finite and unambiguous.

- **Separation of concerns:**
  The policy must not alter or extend the LCO operator.

---

## **6. Example Policy Structures**

These examples illustrate admissible patterns but are not required:

### **Strict Descent**
Accept the candidate with the lowest coherence score.

### **Thresholded Acceptance**
Accept a candidate only if:


\[
C_j \le C_{\text{current}}
\]



### **Stochastic Descent**
Sample candidates with probability proportional to:


\[
\exp(-\beta C_j)
\]


for some externally supplied \(\beta\).

### **Admissibility‑Filtered Descent**
Reject candidates that violate local constraints before scoring.

All examples remain local, algebraic, and invariant‑compatible.

---

## **7. Status**

The update policy defines how coherence scores are used to select and apply updates.
It is fully separate from the LCO operator and must remain compatible with all invariants and primitives defined in the LCO module.

