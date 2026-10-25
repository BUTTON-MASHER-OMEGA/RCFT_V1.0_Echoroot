# Project‑Specific Primitives (local/primitives.md)

## Purpose
This file declares all project‑specific primitives that downstream modules may use.
Primitives defined here extend the RCF core primitive set without redefining it.

All primitives must:
- be explicit
- be non‑interpretive
- be adjacency‑compatible
- be safe for downstream propagation
- avoid introducing hidden state

No downstream module may introduce new primitives outside this file.

No project‑specific local primitives are currently declared.
This file exists as a placeholder for future expansion.

---

## 1. Discrete Support Primitives

These primitives describe discrete structures that downstream physics modules may reference.

Examples (categories only):
- discrete support token
- adjacency‑local field carrier
- local update marker

Populate with explicit definitions as needed.

---

## 2. Continuous Shadow Primitives

These primitives describe continuous structures that appear downstream of geometry‑continuous.

Examples (categories only):
- continuous shadow coordinate
- metric‑adjacent placeholder
- continuous field support token

Populate with explicit definitions as needed.

---

## 3. Temporal Primitives

These primitives describe time‑related structures that downstream modules may use.

Examples (categories only):
- physics‑time index token
- time‑bridge handoff marker
- local dilation placeholder

Populate with explicit definitions as needed.

---

## 4. Conversion Primitives

These primitives support conversion operations in `conversion_laws`.

Examples (categories only):
- discrete‑to‑continuous lift token
- continuous‑to‑metric projection token
- metric‑to‑quantum compatibility marker

Populate with explicit definitions as needed.

---

## 5. Primitive Declaration Rules

- all primitives must be explicitly declared
- no implicit primitive creation is allowed
- no primitive may redefine an RCF core primitive
- primitives must remain non‑interpretive and adjacency‑safe

---

# RCP‑1.1 Compliance
- All primitives explicit
- No interpretive semantics
- No hidden state
- All declarations canonical
- Ready for downstream usage
