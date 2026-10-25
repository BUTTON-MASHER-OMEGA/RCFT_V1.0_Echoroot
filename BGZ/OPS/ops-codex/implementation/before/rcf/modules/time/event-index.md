# Event Index

## Document Contract

- owner: RCF time module
- type: Interpretation
- scope: module-local content of `time/event-index.md`
- authority: tier-4 temporal ordering layer
- upstream dependencies: `/modules/time/interface.md`
- status: active
- purpose: define event index and refinement order without importing downstream rates, geometry, or dynamics
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

### *Discrete Temporal Ordering and Indexed Change*

The event index \( n \) is the primitive temporal ordering relation in RCF.
It is the minimal Time-owned structure for indexing refinement and other downstream records.

---

## 1. Definition



\[
n \in \mathbb{N}, \quad n \ge 1
\]



Each \( n \) labels an indivisible event.
There is no event \( n = 0 \) and no continuous time prior to the first event.

---

## 2. Ordering Relation

The event index induces a discrete successor ordering:



\[
n \rightarrow n+1
\]



This ordering is:

- discrete
- acyclic
- independent of relational structure
- available to downstream modules that index updates

---

## 3. Forward Difference Operator

An adopting module may express indexed differences as:



\[
\Delta_n X = X_{n+1} - X_n
\]



This does not define dynamics. A continuous limit requires an explicit downstream limiting construction.

---

## 4. Predecessor Order

The index records the predecessor relation:



\[
n \rightarrow n+1
\]



not an external timeline. A causal interpretation requires an additional downstream contract.

Eventwise order supplies RCF's minimal “before” and “after” relation.

---

## 5. Independence from Relations

The event index:

- does not depend on relational structure
- does not depend on geometry
- does not depend on constraint density
- does not encode local pacing

It is a pure temporal ordering relation.

---

## 6. Status

The event index provides the discrete backbone of RCF time.
It is the minimal Time-owned structure for indexing refinement and downstream change records.
