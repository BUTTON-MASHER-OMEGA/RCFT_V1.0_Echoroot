# Interface for group_structures

This document defines the **interface contract** between:

- the upstream abstract module: `rcf/modules/group_structures/`, and
- downstream physics modules that consume group structure labels.

The goal is to make the usage of group structure labels:

- explicit,
- constrained,
- composable,
- and safe with respect to upstream invariants.

---

## 1. Upstream-provided artifacts

The upstream `group_structures` module provides (schematically):

- `group-state-label`
- `projection-operator-label`
- `phase-orientation-band`
- `continuous-limit-map`
- other structurally similar labels and maps

These are **non-operator**, **non-metric**, **non-dynamical** structural artifacts.

They are:

- identifiers,
- partitions,
- classification handles,
- maps between structural regimes.

They are **not**:

- physical operators,
- generators,
- metrics,
- fields,
- curvature objects,
- conserved quantities.

---

## 2. Physics-interface responsibilities

This module:

- assigns **physics-usable interpretations** to the abstract labels,
- defines what physics modules **may** assume about them,
- defines what physics modules **may not** assume,
- ensures that all usage remains compatible with upstream constraints.

It does **not**:

- introduce new upstream structure,
- modify upstream invariants,
- reinterpret labels in a way that contradicts the abstract module.

---

## 3. Downstream consumer contract

Physics modules that consume group structure labels must:

- treat labels as **structural classifiers**, not as physical objects,
- respect the non-operator, non-metric, non-dynamical nature of the labels,
- use labels only in ways listed in `allowed_usage.md`,
- avoid all patterns listed in `forbidden_usage.md`,
- remain compatible with upstream eliminability and modularity constraints.

If a physics module needs additional structure, it must:

- introduce that structure **locally**,
- avoid retrofitting semantics onto upstream labels,
- avoid pushing new assumptions upstream into RCF core.
