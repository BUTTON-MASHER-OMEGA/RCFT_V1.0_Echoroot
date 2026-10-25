# Default interpretations of group_structures labels

This document provides **default physics-facing interpretations** of the abstract group structure labels.

These are **interpretations**, not redefinitions. They must remain compatible with the upstream abstract module.

---

## 1. group-state-label

**Interpretation:**
A `group-state-label` is a **symmetry sector identifier**.

Physics modules may treat it as:

- a handle for partitioning states, fields, or configurations into symmetry-related sectors,
- a way to distinguish between different group-related regimes (e.g., different representation classes, sectors, or orbits),
- a structural tag that can be attached to physical objects without becoming a physical object itself.

It is **not**:

- a generator,
- a representation object,
- a conserved charge,
- an operator.

---

## 2. projection-operator-label

**Interpretation:**
A `projection-operator-label` is an **allowed transformation class label**.

Physics modules may treat it as:

- a way to classify which kinds of projection-like transformations are admissible in a given construction,
- a structural tag that indicates which projection family a given operation belongs to,
- a handle for selecting between different projection regimes.

It is **not**:

- a projection operator,
- a linear map,
- a Hilbert-space object,
- a physical observable.

---

## 3. phase-orientation-band

**Interpretation:**
A `phase-orientation-band` is an **orientation class for phase-like or orientation-sensitive structures**.

Physics modules may treat it as:

- a way to classify orientation regimes (e.g., phase sectors, orientation bands, sign/orientation classes),
- a structural tag that can be used to distinguish between different orientation-related behaviors.

It is **not**:

- a phase field,
- a complex phase factor,
- a metric signature,
- a curvature object.

---

## 4. continuous-limit-map

**Interpretation:**
A `continuous-limit-map` is a **smooth parameterization domain handle**.

Physics modules may treat it as:

- a way to indicate that a given structural regime admits a continuous or smooth parameterization,
- a handle for defining families of configurations indexed by continuous parameters,
- a structural tag that marks where continuous limits are admissible.

It is **not**:

- a coordinate chart,
- a manifold,
- a metric,
- a dynamical evolution map.

---

## 5. Other labels

Any additional labels defined in the upstream `group_structures` module must be interpreted:

- as **structural classifiers**,
- as **non-operator**, **non-metric**, **non-dynamical** handles,
- in ways that do not introduce new upstream semantics.

If a physics module needs richer semantics, it must introduce them locally without overloading the meaning of upstream labels.
