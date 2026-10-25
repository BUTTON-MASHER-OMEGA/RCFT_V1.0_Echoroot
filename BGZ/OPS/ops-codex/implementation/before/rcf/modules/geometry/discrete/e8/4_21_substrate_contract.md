# modules/geometry-discrete/4_21_substrate_contract.md

## Document Contract

- owner: RCF discrete geometry module
- type: Contract
- scope: module-local content of `geometry/discrete/e8/4_21_substrate_contract.md`
- authority: tier-7 discrete survivor-recording layer
- upstream dependencies: `/modules/geometry/discrete/interface.md`
- status: active
- purpose: record candidate and surviving discrete geometries without owning projection or downstream continuous behavior
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

## `modules/geometry-discrete/4_21_substrate_contract.md`

### **(Revised with softening, dataset‑scoping, and explicit F(G,γ) definition)**

## **purpose**
To record the structural role of the $4_{21}$ Gosset polytope as a high‑density discrete relational configuration within the current discrete‑geometry formalism.
This file describes observed behavior under evaluated constraint‑density conditions and does not assert completeness or exclusivity.

---

## **1. the 8D master file**
The $4_{21}$ Gosset polytope (the 240‑vertex root polytope of $E_8$) appears in current evaluations as a high‑rigidity relational configuration.

- **Vertices:** 240
- **Role:** Highest‑density candidate tested in the present dataset
- **Property:** Self‑dual adjacency ensures strong algebraic closure under relational updates

This definition is structural only; no physical interpretation is assumed.

---

## **2. the 4D active workspace ($H_4$)**
Under projection into the 4D active workspace, the $4_{21}$ substrate manifests as nested 600‑cell structures with $H_4$ symmetry.

- $H_4$ acts as a **spectral filter** on adjacency patterns
- The smallest stable closure motif observed is the **Chromatic 12**
- These 12 modes form the basis for all downstream survivorship evaluations

This section records observed combinatorial behavior, not theoretical necessity.

---

## **3. the survivorship model and the 36‑mode count**
The Chromatic 12 modes are evaluated under increasing temporal drag ($\gamma$) using the Local Coherence Operator (LCO).

### **Viability Rule**
A geometry $G$ is considered viable at density $\gamma$ when:

\[
F(G,\gamma) \le 1.0
\]

where:

- **F(G,γ)** measures constraint friction / mismatch under LCO updates
- **1.0** marks the recursive‑closure threshold (above this, coherence fails)

### **Observed behavior (current dataset)**
Across the tested geometries {600‑cell, dual 5‑demicube, $4_{21}$}, the Chromatic 12 modes survive across **three** distinct γ‑density plateaus before $F$ exceeds 1.0.

This yields:

\[
12 \times 3 = 36 \text{ discrete survivor modes}
\]

This count reflects observed survivorship behavior under the evaluated candidate set.

---

## **4. scope and dataset boundaries**
The results in this file reflect evaluations over a **limited candidate set**:

\[
\{ \text{600‑cell},\ \text{dual 5‑demicube},\ 4_{21} \}
\]

No claim is made about the full space of possible geometries.

### **High‑density regime**
In current evaluations, $4_{21}$ exhibits the lowest observed $F(G,\gamma)$ in the high‑γ regime.
This is recorded as:

> “lowest observed F among evaluated candidates,”
not as a universal or terminal statement.

### **Transition regions**
Transitions between geometries occur across **ranges**, not sharp boundaries.
For example:

- crossover between 600‑cell and dual 5‑demicube observed around
  \[
  \gamma \approx 2.3–2.7
  \]

These ranges are empirical observations, not theoretical phase transitions.

### **Beyond $4_{21}$**
Geometries beyond $4_{21}$ may exist mathematically, but in current evaluations they either:

- fail the recursive‑closure condition, or
- lose correspondence under dual projection

Thus $4_{21}$ is recorded as the **highest‑rigidity configuration observed within the scope of this framework**, not as a universal endpoint.

