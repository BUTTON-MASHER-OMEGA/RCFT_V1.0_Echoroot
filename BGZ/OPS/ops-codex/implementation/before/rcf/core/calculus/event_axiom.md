# **Event Axiom**

## Document Contract

- owner: RCF discrete calculus substrate
- type: Contract
- scope: non-temporal transition-granularity convention
- authority: tier-3 calculus module-local convention
- upstream dependencies: `README.md`
- status: active; subordinate to the Time module
- purpose: define an indivisible transition record at a declared active resolution
- anti-backflow: this convention does not define RCF temporal events, event indices, refinement time, or temporal ontology

`/core/calculus/event_axiom.md`

## Purpose

Define the minimal calculus transition convention used to describe admissible transitions.

This axiom constrains refinement without introducing ontology.

The Time module owns RCF temporal events and ordering. The term *event* here names only transition granularity at an explicitly declared resolution.

---

## 1. Event Definition

An event is an indivisible admissible transition between configurations.

It cannot be decomposed within the active resolution scale.

---

## 2. Minimality Principle

If a transition can be decomposed into admissible sub-transitions, it is not an event.

Events are refinement atoms.

---

## 3. Structural Consequence

Events:

* compose sequentially
* preserve algebraic invariants
* cannot violate operator identities

They define the minimal ordering units of refinement.

---

## 4. Non-Ontological Constraint

The event axiom:

* introduces no physical claim
* assigns no causal privilege
* specifies only admissible granularity

It is a bookkeeping rule for refinement structure.

---

## 5. Role in RCF

The event axiom guarantees:

* discrete refinement consistency
* termination of infinite regress
* stable operator domains

It defines the minimal transition unit for this calculus convention.
