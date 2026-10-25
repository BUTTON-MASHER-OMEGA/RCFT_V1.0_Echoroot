# Forbidden usage patterns for group_structures labels

This document specifies **forbidden usage patterns** for physics modules that consume group structure labels.

These patterns are disallowed because they:

- violate upstream invariants,
- overload structural labels with physical semantics,
- risk contaminating RCF core with domain-specific assumptions.

---

## 1. Treating labels as operators

Physics modules must **not**:

- treat `projection-operator-label` as a projection operator,
- treat `group-state-label` as a generator or representation object,
- treat any label as a linear map, operator, or observable.

If an operator is needed, it must be:

- defined locally in the physics module,
- associated with labels only as metadata,
- not identified with the labels themselves.

---

## 2. Inferring metric structure

Physics modules must **not**:

- infer metric structure from `phase-orientation-band`,
- infer metric structure from `continuous-limit-map`,
- treat any label as implying a specific metric, signature, or curvature.

Metric structure, if needed, must be:

- introduced explicitly in the relevant physics module,
- not retrofitted onto upstream labels.

---

## 3. Inferring conserved quantities or charges

Physics modules must **not**:

- treat `group-state-label` as a conserved charge,
- infer conservation laws directly from labels,
- treat labels as Noether charges or similar objects.

Conservation laws must arise from:

- explicit dynamical or symmetry constructions,
- not from the mere presence of structural labels.

---

## 4. Inferring gauge groups or representation content

Physics modules must **not**:

- treat any group structure label as directly specifying a gauge group,
- infer representation content from labels alone,
- overload labels with detailed group-theoretic semantics.

Gauge groups and representations must be:

- defined explicitly in physics modules,
- not encoded implicitly in upstream labels.

---

## 5. Using labels as dynamical variables

Physics modules must **not**:

- treat labels as dynamical degrees of freedom,
- evolve labels in time as if they were fields or coordinates,
- assign equations of motion to labels.

Labels are:

- structural,
- static with respect to dynamics,
- used for classification and partitioning only.

---

## 6. Pushing new semantics upstream

Physics modules must **not**:

- reinterpret upstream labels in ways that contradict `rcf/modules/group_structures/`,
- modify the meaning of labels at the RCF core level,
- introduce domain-specific semantics into upstream modules.

All domain-specific semantics must remain:

- local to physics modules,
- downstream of this interface layer.
