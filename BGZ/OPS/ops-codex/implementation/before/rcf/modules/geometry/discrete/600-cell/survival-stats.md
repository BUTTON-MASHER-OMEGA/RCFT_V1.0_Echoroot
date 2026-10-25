# 600-cell/survival-stats.md

## Document Contract

- owner: RCF discrete geometry module
- type: Observation
- scope: module-local content of `geometry/discrete/600-cell/survival-stats.md`
- authority: tier-7 discrete survivor-recording layer
- upstream dependencies: `/modules/geometry/discrete/interface.md`
- status: active
- purpose: record candidate and surviving discrete geometries without owning projection or downstream continuous behavior
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

## purpose
This document records the survival behavior of the 600‑cell under the RCF survival sieve. It explains why the 600‑cell remains stable in standard constraint‑density regimes and documents the structural features that support its persistence.

---

## survival profile (standard density)

### 1. triangle‑compatibility
The 600‑cell is fully simplicial:
every local neighborhood decomposes into stable 3‑point motifs.
This supports stable LCO evaluation with reduced mismatch spikes.

### 2. mode‑space completeness
The adjacency graph contains no missing relational modes (“holes”).
Recursive updates do not force snap‑displacement events, reducing mismatch accumulation.

### 3. φ‑structured absorption
The 600-cell’s coordinate families contain intrinsic φ-ratios.
In current evaluations, this structure is associated with reduced mismatch under γ-scaled temporal drag.

### 4. density‑regime stability
At standard constraint densities:
- the 600-cell shows stable behavior
- lower densities collapse into sparse simplices
- higher densities transition toward dual 5‑demicube or E₈ ($4_{21}$)behavior

---

## invariants
- survival is statistical, not prescriptive
- stability arises from adjacency and closure, not coordinate embedding
- φ‑structured behavior is recorded as an empirical invariant, not an interpretation

