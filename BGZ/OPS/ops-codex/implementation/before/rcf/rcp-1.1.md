# **RCF Compression Protocol (RCP-1.1)**

### *Typed English for Canon-Safe Compression and Validation*

---

## **Purpose**

RCP-1.1 defines a stricter compression protocol for transforming RCF module content into a form that is:

- structurally justified,
- canon-safe,
- validator-friendly,
- and resistant to upstream backflow during compression.

This version extends RCP-1.0 by adding a lightweight **type system for English statements**.

The goal is not only to compress documents, but to make them easier for agents and validators to read without inventing missing structure.

---

## **Core Principle**

> **If a statement cannot survive removal of interpretation, ownership ambiguity, and directional ambiguity, it must be rewritten or removed.**

---

## **Primary Design Goal**

RCP-1.1 exists to prevent two specific compression failures:

1. downstream modules being treated as if they can redefine upstream structure,
2. compressed models using downstream language to retroactively repair or reinterpret upstream layers.

The protocol therefore treats English statements as if they were typed program elements:

- each statement has an **owner**,
- each statement has a **type**,
- each statement has a **scope**,
- each statement has a **direction**,
- and each statement must remain valid when stripped of narrative explanation.

---

## **Statement Type System**

Every retained statement in a compressed or compression-ready document should be classifiable as one of the following:

- **Invariant**
  - defines a non-negotiable constraint owned by the correct canonical layer
- **Contract**
  - defines a boundary, interface, permission, or prohibition local to a layer or module
- **Observation**
  - records behavior seen in the current formalism or current evaluations
- **Dependency**
  - records directional structural reliance
- **Permission**
  - states an allowed input, output, or use
- **Prohibition**
  - states a forbidden input, output, inference, or dependency
- **Interpretation**
  - records a downstream reading that is explicitly non-canonical
- **Status**
  - records maturity, revision state, or implementation status

If a sentence cannot be typed, it is likely too loose for canonical compression.

---

## **Ownership Rule**

Every strong statement must be owned by the correct layer.

### Required question

> **Which module or layer is allowed to say this?**

### Examples

```text
Projection defines continuous-shadow language.
```

```text
LCO defines local admissibility scoring.
```

```text
Geometry-discrete records survivors observed under the current stack.
```

### Invalid pattern

```text
Geometry-discrete explains how projection smooths survivors.
```

This fails because the statement is not owned by the geometry-discrete layer.

---

## **Directionality Rule**

Dependency direction must remain explicit and one-way unless formal reciprocity is documented upstream.

### Allowed directional forms

- upstream constrains downstream
- downstream consumes upstream
- downstream may interpret upstream outputs
- downstream may not redefine upstream contracts

### Forbidden directional forms

- downstream repairs upstream
- downstream explains upstream ownership
- downstream widens upstream admissibility
- downstream treats its own outputs as upstream causes

### Required anti-backflow test

For each sentence:

> **Could a model read this as permission to go backward and modify upstream meaning?**

If yes, rewrite it.

---

## **Scope Rule**

All non-core statements must be explicitly scoped.

### Preferred scope labels

- **core-canonical**
- **current formalism**
- **current evaluations**
- **module-local contract**
- **downstream interpretation**

### Preferred phrases

- “within the present formalism”
- “in current evaluations”
- “as documented upstream”
- “for this module”
- “downstream modules may interpret this as…”

Unscoped claims tend to be over-promoted during compression.

---

## **Modal Verb Discipline**

RCP-1.1 assigns stronger meaning to common verbs.

### High-trust verbs

- **defines**
  - use only for owner layers
- **consumes**
  - use for allowed upstream inputs
- **exposes**
  - use for interface-visible outputs
- **records**
  - use for observational content
- **bounds**
  - use for range restriction without generation
- **selects**
  - use for sealed-parameter choice within allowed ranges
- **may interpret downstream as**
  - use for non-canonical extension

### Caution verbs

- **provides**
- **supports**
- **grounds**
- **underlies**
- **drives**
- **yields**

These are allowed only when the ownership and direction are already obvious.

### Flagged verbs

- **causes**
- **explains**
- **produces** (when used mechanism-like rather than interface-like)
- **generates** (unless formal generation is owned and defined)
- **enforces** (unless invariant ownership is explicit)

If a flagged verb appears, the statement should be reviewed under the causality and mechanism passes.

---

## **Compression Passes**

Each document undergoing compression must pass the following stages.

---

## **Pass 1 — Causality Elimination**

### Rule

Remove or neutralize causal or explanatory language unless explicitly derived upstream.

### Flagged Patterns

- “because”
- “due to”
- “results from”
- “caused by”
- “consistent with X enforcing Y”

### Transform

```text
X occurs because Y
→
X is observed under conditions where Y is present
```

---

## **Pass 2 — Mechanism → Observation**

### Rule

Replace inferred mechanisms with observed behavior.

### Transform

```text
LCO reduces symmetry
→
Reduced symmetry is observed under LCO-constrained evaluations
```

---

## **Pass 3 — Invariant Hardening Check**

### Rule

Downgrade implicit invariants unless formally owned by the correct layer.

### Flagged Terms

- “must”
- “ensures”
- “guarantees”
- “always”
- “cannot”

### Transform

```text
Survivors must maintain X
→
Survivors are observed to maintain X in current evaluations
```

---

## **Pass 4 — Label vs Ontology Separation**

### Rule

Ensure all structural terms remain labels, not claims of physical or ontological reality.

### Test

For each term:

> Is this being used as a label or as a claim about what exists?

### Transform

```text
modes represent physical structure
→
mode labels index recurring structural patterns
```

---

## **Pass 5 — Explanation Removal**

### Rule

Remove “why” statements unless supported by formal derivation.

### Transform

```text
This occurs because of X
→
This is observed under X
```

---

## **Pass 6 — Downstream Leakage Check**

### Rule

Remove references to downstream constructs unless explicitly framed as downstream interpretation or downstream dependency.

### Flagged Concepts

- metric
- curvature
- flow
- fields
- dynamics

### Transform

```text
produces curvature
→
provides structure that downstream modules may interpret as curvature-like variation
```

---

## **Pass 7 — Scope Binding**

### Rule

All statements must be explicitly scoped to the present formalism, current evaluations, or owner-layer contract.

### Required Language Patterns

- “in current evaluations”
- “within the present formalism”
- “as observed in upstream modules”
- “for this interface”
- “for this module”

---

## **Pass 8 — Load-Bearing Sentence Test**

### Rule

Every sentence must pass:

> **If all interpretation is removed, is this still true?**

If:

- YES → keep
- NO → rewrite or remove

---

## **Pass 9 — Ownership Check**

### Rule

Every strong statement must be attributed to the layer that is allowed to make it.

### Validator questions

- Is this invariant owned by the correct layer?
- Is this contract owned by the correct interface?
- Is this observational claim being presented as if it were canonical?
- Is this module claiming authority over language it only consumes?

### Transform

```text
The metric module defines projection behavior
→
The metric module consumes projection-documented structure
```

---

## **Pass 10 — Directionality / Anti-Backflow Check**

### Rule

Reject any sentence that permits downstream interpretation to move backward and redefine upstream meaning.

### Flagged patterns

- “therefore upstream…”
- “which implies the core really means…”
- “this shows projection must…”
- “thus the invariant should be read as…”

### Transform

```text
This module shows what the invariant really means
→
This module is downstream-compatible with the invariant as currently stated
```

---

## **Pass 11 — Interface Narrowing Check**

### Rule

If a module has an interface, the interface contract outranks freeform prose during compression.

### Required checks

- Are allowed inputs narrower than the prose suggests?
- Are allowed outputs narrower than the prose suggests?
- Does the prose imply hidden dependencies not listed in the interface?
- Does the prose imply outputs the interface does not authorize?

If yes, compress to the interface contract, not to the prose.

---

## **Pass 12 — Statement Type Mismatch Check**

### Rule

Reject statements whose language and type disagree.

### Examples

- an **Observation** written like an **Invariant**
- a **Permission** written like an **Ontology claim**
- a **Dependency** written like a **causal mechanism**
- an **Interpretation** written like a **core fact**

### Transform

```text
Plateau structure guarantees X
→
Within the present formalism, plateau structure is observed alongside X
```

---

## **Module Compression Normal Form**

When compressed, a module should preferably reduce to the following fields:

- **Purpose**
- **Owner**
- **Statement Types Present**
- **Dependencies**
- **Allowed Inputs**
- **Allowed Outputs**
- **Prohibitions**
- **Observations in Current Formalism**
- **Downstream Interpretations**
- **Status**

If the document cannot be reduced into this shape, it likely still contains mixed ownership or interpretive drift.

---

## **Canonical Sentence Patterns**

These sentence forms are preferred for compression-safe writing.

### Invariant

```text
This layer defines X.
```

### Dependency

```text
This module consumes X from Y.
```

### Interface Output

```text
This interface exposes X.
```

### Observation

```text
Within the present formalism, X is observed under Y.
```

### Prohibition

```text
This module does not define X.
```

### Downstream interpretation

```text
Downstream modules may interpret X as Y.
```

These forms are intentionally narrow. Narrow language is a feature, not a bug.

---

## **Compression Output Standard**

A fully compressed document:

- contains no implicit causality,
- contains no unowned invariants,
- contains no ontology claims outside their owner layer,
- contains no downstream leakage,
- preserves directionality,
- preserves interface narrowing,
- uses observational language where ownership is absent,
- remains fully compatible with upstream constraints.

---

## **Validator Directive**

During validation, reject or rewrite any statement that:

- introduces mechanism without derivation,
- asserts invariants outside its layer,
- implies physical meaning without owned scope,
- depends on explanation rather than observation,
- lets downstream language reinterpret upstream contracts,
- widens allowed inputs or outputs beyond the interface,
- uses ambiguous verbs where typed verbs are available.

---

## **Meta Principle**

> **Compression does not weaken the system.**
>
> **It reveals what survives once ownership, scope, and direction are made explicit.**

---

## **Status**

- Protocol: **DRAFT PROPOSED**
- Version: **RCP-1.1**
- Applicability: **All modules prior to canonical freeze**
- Intended use: **Compression, linting, validator passes, and build-time drift control**
