# Allowed usage patterns for group_structures labels

This document specifies **allowed usage patterns** for physics modules that consume group structure labels.

All usage must remain compatible with:

- the upstream `rcf/modules/group_structures/` constraints,
- the interpretations in `interpretations.md`,
- the interface contract in `interface.md`.

---

## 1. Sector partitioning

Physics modules may:

- use `group-state-label` to partition states, fields, or configurations into symmetry-related sectors,
- treat different `group-state-label` values as indexing distinct but structurally related regimes,
- define selection rules or branching structures based on `group-state-label` values.

Example (schematic):

- SMG module partitions field content into sectors labeled by `group-state-label`,
- each sector is treated as a distinct symmetry regime.

---

## 2. Transformation family classification

Physics modules may:

- use `projection-operator-label` to classify families of projection-like transformations,
- use these labels to select which transformation family is admissible in a given construction,
- define local operators that are associated with a given `projection-operator-label`, as long as the label itself is not treated as the operator.

---

## 3. Orientation regime classification

Physics modules may:

- use `phase-orientation-band` to classify orientation or phase-related regimes,
- distinguish between different orientation bands when defining local physics,
- use these bands as structural tags in configuration spaces.

---

## 4. Continuous parameterization regimes

Physics modules may:

- use `continuous-limit-map` to indicate where continuous or smooth parameterizations are admissible,
- define families of configurations indexed by continuous parameters in regimes marked by `continuous-limit-map`,
- treat these maps as structural guides for where continuum methods are appropriate.

---

## 5. Structural tagging and bookkeeping

Physics modules may:

- attach group structure labels to their own objects (fields, states, configurations) as metadata,
- use labels for structural bookkeeping, indexing, and partitioning,
- propagate labels through constructions as long as their structural meaning is preserved.

Labels remain:

- non-operator,
- non-metric,
- non-dynamical.

They are **tags**, not physical entities.
