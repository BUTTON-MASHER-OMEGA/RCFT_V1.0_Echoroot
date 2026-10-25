# candidate-geometries.md

## Document Contract

- owner: RCF discrete geometry module
- type: Interpretation
- scope: module-local content of `geometry/discrete/candidate-geometries.md`
- authority: tier-7 discrete survivor-recording layer
- upstream dependencies: `/modules/geometry/discrete/interface.md`
- status: active
- purpose: record candidate and surviving discrete geometries without owning projection or downstream continuous behavior
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

## **purpose**
This document records the discrete geometries evaluated under the survival sieve and summarizes why most fail. It provides a transparent account of the elimination process, ensuring that the surviving geometries are understood as statistical outcomes rather than assumptions.

---

## **scope**
This file includes:

### **1. evaluated candidates**
In current evaluations, 5D bulk realizations incur higher mismatch over the relevant γ regime than 4D realizations. The discrete-geometry formalism therefore operates on the surviving 4D low-friction bulk dimension for recursive update analysis.

A list of 4D simplicial and near‑simplicial topologies tested under the sieve, including but not limited to:
- 16‑cell
- 24‑cell
- snub 24‑cell
- grand antiprism
- rectified 600‑cell
- 600‑cell
- dual 5‑demicube
- E₈ lattice

### **2. common failure modes**
Most candidates fail due to:
- high quantization error
- incomplete mode coverage
- algebraic instability
- spinor‑closure failure
- poor mismatch behavior under recursion

### **3. density‑dependent survivors**
In current evaluations, a small subset of geometries shows sustained viability across their respective density regimes:
- **600‑cell** (standard density)
- **dual 5‑demicube** (intermediate density)
- **E₈ ($4_{21}$)** (extreme density)

These survivors are documented in their own subdirectories.

---

### local closure requirement

A candidate geometry must support local triangle packing sufficient to realize
the minimum viable relational neighborhood.

In the current sieve profile, viable local closure requires at least three
mutually supporting relations. Two-relation neighborhoods do not satisfy the
closure requirement under local evaluation.

This criterion is structural and does not imply downstream physical meaning.

---

### H4 compatibility requirement

Candidate geometries must either:
- realize H4 structure directly,
- be isomorphic to an allowed H4 representation,
- or admit a documented compression/projection into an H4-compatible representation.

---

## **invariants**
- No geometry is assumed; all are tested.
- Failure is structural, not aesthetic.
- Survivors are those that remain stable under the update rules in current tests.
- This file does not interpret survivors; it only records outcomes.

