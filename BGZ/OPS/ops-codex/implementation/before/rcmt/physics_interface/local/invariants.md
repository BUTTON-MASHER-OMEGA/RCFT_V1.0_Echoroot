# Project‑Specific Invariants (local/invariants.md)

## Purpose
This file defines the upstream‑adjacent invariants specific to this project.
These invariants refine RCF core constraints and shape all downstream physics modules.
They are loaded before any module in `physics/`.

These invariants:
- do not redefine RCF core invariants
- do not introduce interpretive semantics
- do not introduce hidden state
- do not introduce downstream assumptions
- refine only what this project requires

---

## 1. Adjacency Refinement Invariants

### 1.1 Allowed Adjacency Forms
- adjacency must remain discrete and explicit
- adjacency refinements must preserve RCF adjacency axioms
- adjacency transitions must be locally defined
- no module may introduce non‑local adjacency

### 1.2 Forbidden Adjacency Forms
- no global adjacency operators
- no adjacency shortcuts
- no implicit adjacency inference

---

## 2. Containment Refinement Invariants

### 2.1 Allowed Containment Forms
- containment refinements must preserve RCF containment axioms
- containment transitions must be explicit
- containment must remain upstream‑compatible

### 2.2 Forbidden Containment Forms
- no implicit containment
- no containment‑based hidden state
- no containment rewrites of upstream geometry

---

## 3. Time‑Related Invariants

### 3.1 Time Ownership

- The RCF Time Core is the sole upstream owner of temporal ontology for the physics project.
- The RCF Temporal Interface is the sole upstream owner of exposed local temporal parameters for the physics project.
- The term “Upstream Time Index” is an alias referring exclusively to the upstream RCF temporal structure imported through Time Core and Temporal Interface contracts.
- No module may redefine RCF Time Core temporal ontology.
- No module may redefine or bypass Temporal Interface parameter boundaries.
- No module may introduce alternative upstream time indices or parallel upstream time constructs.

### 3.2 Physics Time Boundary
- Physics Time appears only after `time_bridge`
- no pre‑bridge physics‑time usage is permitted
- no module may introduce hidden temporal interpolation

---

## 4. Geometry‑Related Invariants

### 4.1 Geometry Ownership
- geometry‑continuous and geometry‑discrete remain upstream
- no downstream module may redefine geometry ownership
- no module may introduce alternative geometry substrates

### 4.2 Metric Boundary
- metric structure appears only after `emergent_metric_geometry`
- no pre‑metric geometry claims are permitted
- no hidden metric state is allowed

---

## 5. Local Primitive Invariants

### 5.1 Primitive Declaration Rule
All project‑specific primitives must be declared in `local/primitives.md`.

### 5.2 Primitive Usage Rule
Downstream modules may only use primitives declared here or in RCF core.
No module may introduce undeclared primitives.

---

## 6. Local Interface Invariants

### 6.1 Import Boundary
Downstream modules may import only from:
- RCF core
- local/interfaces.md
- their declared dependencies

### 6.2 Forbidden Imports
- no module may import from sibling modules
- no module may import from undeclared upstream sources
- no module may bypass the local layer

---

## 7. Downstream Safety Invariants

### 7.1 No Hidden State
All state must be explicit and locally defined.

### 7.2 No Interpretive Semantics
Modules must not introduce narrative, ontology, or interpretation.

### 7.3 Adjacency‑Only Propagation
All propagation must be adjacency‑local and explicit.

---

## 8. Project‑Specific Commitments (Fill‑In Section)

This section is intentionally left open for project‑specific invariants that do not belong in RCF core but are required for this project.

Examples of what *could* go here (categories only):
- additional adjacency constraints
- additional containment constraints
- project‑specific time dilation rules
- project‑specific geometry refinements
- project‑specific operator restrictions

Populate this section as the project evolves.

---

# RCP‑1.1 Compliance
- All invariants canonical
- No interpretive semantics
- No hidden state
- All constraints explicit
- Ready for downstream loading
