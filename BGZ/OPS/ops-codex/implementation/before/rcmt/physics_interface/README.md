# `physics_interface/README.md`

## purpose

The `physics_interface/` layer defines the **public interface contract** for the physics project.

It exists to make clear:

* what downstream physics modules may import
* what they are expected to expose
* what boundaries they must respect
* how the physics project compiles against RCF

This layer defines **structure**, not interpretation.

---

## role

The physics interface sits between:

* the RCF upstream core
* the project-local physics layer
* downstream physics modules

It provides the minimal contract required to build physics modules safely and consistently.

---

## allowed inputs

Physics modules may access only:

* upstream RCF constraints and interfaces
* project-local invariants, constraints, and adapters
* explicitly declared interface records

Physics modules may **not**:

* bypass the local layer
* import undeclared sibling internals
* introduce undeclared primitives
* modify upstream ownership or invariants

---

## expected outputs

Physics modules should expose only explicit downstream records such as:

* state records
* update records
* boundary records
* handoff records
* compatibility records

All outputs must be:

* explicit
* typed
* non-interpretive
* upstream-safe

---

## boundary rules

Physics modules must respect the following:

* no Physics Time before `time_bridge`
* no metric structure before `emergent_metric_geometry`
* no hidden state
* no hidden scheduler behavior
* no hidden interpolation
* no backflow into RCF core, Time, Temporal Interface, or geometry ownership

---

## compilation rule

A physics module is valid only if it compiles against:

1. the RCF upstream core
2. the project-local physics layer
3. this interface contract

If a module exceeds these boundaries, it does not compile cleanly.

---

## guidance

Use this layer to keep the physics project:

* modular
* explicit
* maintainable
* downstream-safe

The physics interface is intentionally minimal.
It exists to keep implementations compatible, not to define physics itself.
