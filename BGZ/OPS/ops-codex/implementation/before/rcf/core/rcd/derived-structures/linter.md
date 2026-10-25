# Constraint Linter (RCD)

## Document Contract

- owner: Relational Coherence Dynamics
- type: Contract
- scope: optional derived consistency linter
- authority: eliminable tier-2 RCD tooling contract
- upstream dependencies: `../README.md` and `spectral-structures.md`
- status: formal; eliminable
- purpose: define higher-order consistency checks over declared RCD records
- anti-backflow: the linter may reject inconsistent RCD records but does not replace LCO, add invariants, or define domain admissibility

---

## 1. Purpose

Define a meta-constraint operator that evaluates relational admissibility under higher-order consistency requirements.

The Linter removes inconsistent structures.
It does not construct new ones.

---

## 2. Linter Operator

    L : R → R'

L applies additional admissibility checks beyond base constraints.

---

## 3. Loop Set

Given a relational state R, define a loop set:

    C(R) = { C_k }

Each loop C_k may be assigned a representational tension τ(C_k).

High-tension loops are penalized; low-tension loops are retained.

---

## 4. Spectral Penalties

Given a spectrum { λ_i }, define degeneracy pressure:

    P_deg = Σ_i 1 / Δλ_i

where Δλ_i = λ_{i+1} − λ_i.

High degeneracy pressure indicates unstable clustering.

---

## 5. Ratio Preservation

Persistent structures preserve relational ratios under global rescaling:

    λ_i / λ_j invariant under λ_i → α λ_i

---

## 6. Total Penalty

    P_total = w_L Σ_k f(τ(C_k)) + w_D P_deg

A structure is admissible if:

    P_total ≤ P_max

---

## 7. Eliminability

The Linter is optional.
Removing it does not affect RCD invariants.
