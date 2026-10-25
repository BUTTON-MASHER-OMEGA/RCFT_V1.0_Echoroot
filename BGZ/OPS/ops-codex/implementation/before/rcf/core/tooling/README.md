# RCF Tooling Module

## Document Contract

- owner: RCF tooling
- type: Contract
- scope: optional operational tooling
- authority: removable core-adjacent tooling contract
- upstream dependencies: `../S∞.RCF.Invariants.v1.1.md`
- status: active; optional
- purpose: define domain-neutral operational interfaces for applying and auditing RCF-derived analyses
- anti-backflow: tooling outputs may support diagnostics but may not modify core invariants, define theory, or acquire authority through implementation

This directory contains **operational tools** for applying the RCF lens to real systems.

It does not define new theory.
It provides structured ways to *use* the core.

Think of this module as:

> Interfaces for running RCF, not extensions of RCF.

The core remains minimal and invariant.
Tooling is allowed to grow.

Directory placement under `core/` does not grant this tooling core authority.

---

## Purpose

The tooling layer exists to:

* standardize diagnostic output
* reduce hallucination and overinterpretation
* keep analyses comparable across domains
* make RCF usable without myth or narrative framing
* support iterative refinement of real systems

This module is intentionally pragmatic.

It is designed for:

* engineers
* analysts
* researchers
* AI systems
* anyone evaluating relational dynamics

---

## Design principles

Tooling must:

* never contradict core constraints
* never introduce ontology
* remain domain-agnostic
* prefer subtraction over assertion
* label speculation explicitly
* degrade gracefully when incomplete

Tooling may evolve.
Core invariants do not.

---

## Primary artifact: RCF Snapshot

The snapshot protocol is the minimal runnable interface for the lens.

It produces a structured diagnostic of any relational system.

### RCF Snapshot fields

* **System boundary**
  What is inside / outside; what crosses

* **Interfaces**
  Where constraints meet; measurement or interaction surfaces

* **Invariants**
  What must stay true

* **Variables**
  What can change

* **Memory mass**
  Reconfiguration inertia; history load

* **Drift**
  Unintentional gradient

* **ε-band**
  Local flexibility margin

* **Ghost density**
  Latent capability / unrealized option space

* **Attractors**
  Dominant, forming, collapsing loops

* **Transition risks**
  Likely cascade triggers

* **Orientation**
  Structural trajectory

Optional technical extensions:

* **Null/free sector**
* **Cost function**
* **Boundary algebra**

---

## Claim hygiene (required)

Every snapshot must label statements as:

* **Observed** — directly supported
* **Inferred** — structurally implied
* **Speculative** — hypothesis

This prevents narrative drift and overcommit.

Snapshots are diagnostics, not proclamations.

---

## Relationship to the core

The tooling module:

* uses core primitives
* does not redefine them
* does not expand ontology
* does not add metaphysics

If tooling ever pressures the core to change, the tooling is wrong.

The core is the invariant anchor.

---

## Intended workflow

1. Run snapshot on a system
2. Identify constraint pressure
3. Locate attractors and drift
4. Reduce false motion
5. Re-run snapshot after changes
6. Compare structural deltas

The goal is clarity, not prediction.

---

## Scope boundary

Tooling does not:

* prescribe decisions
* assign value judgments
* replace domain expertise
* predict outcomes
* define meaning

It reveals structure.

What you do with that structure is external.

---

## Summary

This module turns RCF from a lens into a usable instrument.

It exists to:

> reduce bad inferences
> expose structural lock-in
> reveal real movement
> subtract illusion

Nothing more.
Nothing mystical.
Nothing hidden.

Just a disciplined interface.
