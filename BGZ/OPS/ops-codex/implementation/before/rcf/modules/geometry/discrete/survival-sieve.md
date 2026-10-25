# survival-sieve.md

## Document Contract

- owner: RCF discrete geometry module
- type: Interpretation
- scope: module-local content of `geometry/discrete/survival-sieve.md`
- authority: tier-7 discrete survivor-recording layer
- upstream dependencies: `/modules/geometry/discrete/interface.md`
- status: active
- purpose: record candidate and surviving discrete geometries without owning projection or downstream continuous behavior
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

## **purpose**
This document defines the admissibility criteria that any discrete relational geometry must satisfy in order to persist under the Indivisible Stochastic Process (ISP). The sieve is not a selector of “preferred shapes,” but a statistical filter: geometries that accumulate mismatch, fail closure, or lose coherence under recursion simply fall out of the distribution.

The sieve exists to formalize the upstream constraints that determine which discrete structures can serve as viable substrates for downstream projection.

---

## **scope**
The survival sieve evaluates candidate geometries according to:

### **1. local mismatch behavior**
Each geometry must minimize the mismatch function \(C(\mu)\) under varying constraint densities. Persistent local spikes indicate quantization error or incomplete mode coverage.

### **2. mode‑space completeness**
Geometries with incomplete mode coverage show increased mismatch and reduced persistence. Missing modes (“holes”) cause snap‑displacement events that destabilize recursion.

### **3. spinor embedding**
In current representations where sequential updates are modeled via spin covers (e.g., SU(2)), candidate sets must embed cleanly into that representation to pass the sieve, as sets that do not embed cleanly show reduced persistence under recursive composition.

### **4. recursive closure**
A geometry must remain stable under repeated LCO-evaluated updates as summarized by GCO. Under the adopted spinor profile, a composition that leaves the candidate set fails that profile's closure check.

### **5. density‑regime stability**
A geometry must remain viable across the constraint densities relevant to its regime. Some geometries survive only at standard density; others only at extreme density.

### **6. triangle‑compatibility**
A candidate geometry must support consistent triangle‑based local packing.

Since the Local Coherence Operator (LCO) evaluates updates over 3‑point relational motifs, any geometry that cannot be decomposed into stable triangle neighborhoods is not admissible. Structures lacking triangle‑compatible adjacency patterns cannot be evaluated under the execution rules and are eliminated before mismatch analysis.

---

## **invariants**
- The sieve does not “choose” geometries; it eliminates those that fail.
- Survival is statistical, not prescriptive.
- Among declared candidates, only geometries that remain stable under the current recursion, mismatch, and closure checks are recorded as survivors.
- No continuous assumptions appear here; this layer is purely discrete.
