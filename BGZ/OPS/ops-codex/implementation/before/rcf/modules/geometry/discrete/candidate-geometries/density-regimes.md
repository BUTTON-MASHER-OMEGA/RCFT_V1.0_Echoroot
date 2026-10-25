# candidate-geometries/density-regimes.md

## Document Contract

- owner: RCF discrete geometry module
- type: Interpretation
- scope: module-local content of `geometry/discrete/candidate-geometries/density-regimes.md`
- authority: tier-7 discrete survivor-recording layer
- upstream dependencies: `/modules/geometry/discrete/interface.md`
- status: active
- purpose: record candidate and surviving discrete geometries without owning projection or downstream continuous behavior
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

## **density‑dependent mismatch profiles**

---

In current evaluations, each candidate geometry exhibits a characteristic mismatch profile as the constraint‑density parameter γ is varied.
The values below represent **observed** mismatch accumulation under recursive updates within the present formalism.

- **Lower values** → reduced mismatch, greater persistence
- **Values exceeding 1.0** → loss of recursive viability under the current mismatch rule
- **Bracketed values** → near‑failure regimes where survival is marginal

This table summarizes the relative behavior of the **tested candidate set**:

\[
\{ \text{600‑cell},\ \text{dual 5‑demicube},\ 4_{21} \}
\]

across the evaluated γ‑range.

---

### 🏛️ **Observed γ‑Sweep (Current Dataset)**

```text
Gamma     F(600)      F(D-5D)     F(4_21)     | Lowest Observed F(G,γ)
----------------------------------------------------------------------
 0.00     0.472       0.734       0.891       | 600-Cell (baseline)
 1.00     0.550       0.741       0.891       | 600-Cell
 2.00     0.750       0.760       0.892       | 600-Cell (near crossover)
 2.3–2.7  [0.82–0.88] [0.76–0.79]  0.892       | Dual 5-Demicube (crossover region)
 3.00     0.980       0.820       0.893       | Dual 5-Demicube (intermediate γ)
 4.00    (1.200)*     [0.900]     [0.894]     | 4₍₂₁₎ (high-γ regime)
 5.00    (non-viable) 0.980       0.895       | 4₍₂₁₎ (highest γ tested)
----------------------------------------------------------------------
*Values > 1.0 indicate loss of recursive viability under the current mismatch rule.
```

---

## **summary**

Across the evaluated γ‑range, each geometry exhibits a distinct survivorship profile:

- The **600‑cell** shows the lowest observed mismatch at low and moderate γ.
- The **dual 5‑demicube** shows lower mismatch in the intermediate γ‑range.
- The **4₍₂₁₎** configuration shows the lowest observed mismatch in the high‑γ regime within the current dataset.

These transitions reflect **observed behavior** under the present mismatch function and update algebra.
No physical interpretation of γ or the transitions is assigned here.

---

## **invariants**

- The γ‑sweep records **observed** mismatch values under the current mismatch function.
- Transitions correspond to **intersections** of mismatch curves in the evaluated dataset.
- The survival sieve selects the geometry with **lowest observed F(G,γ)** at each γ.
- Results apply only to the tested candidate set and the present formalism.

