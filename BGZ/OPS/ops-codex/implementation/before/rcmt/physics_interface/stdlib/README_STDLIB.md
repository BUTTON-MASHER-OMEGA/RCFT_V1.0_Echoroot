# Standard Library Layer — Optional Shared Utilities

The `stdlib/` directory contains **optional shared utilities** that support module development.
This layer is downstream‑adjacent: it is not upstream, but it is not domain‑specific either.

The stdlib provides reusable patterns, helpers, and structural components that multiple modules may depend on.

---

## What Belongs in `stdlib/`

- Reusable structural patterns
- Helper definitions used by multiple modules
- Common idioms expressed in Typed English
- Shared utilities that simplify module construction
- Components that are not domain‑specific but support domain work

The stdlib is a **support layer**, not a theory layer.

---

## What Does *Not* Belong in `stdlib/`

- Upstream or upstream‑adjacent invariants (`local/invariants.md` is the correct location)
- Domain‑specific content (place that in `modules/`)
- Project‑level primitives or constraints
- Any modification of the RCF core
- One‑off definitions used by a single module

If a definition is only used by one module, it belongs in that module.
If it defines project‑level structure, it belongs in `local/`.

---

## Interfaces Available From Upstream

The stdlib may import:

- All invariants and primitives from `rcf_compressed.md`
- All project‑specific invariants from `local/invariants.md`

The stdlib may **not**:

- Override upstream or local invariants
- Introduce new upstream‑adjacent structure
- Collapse upstream and downstream boundaries

The stdlib is strictly downstream‑adjacent.

---

## Role of the Standard Library

The stdlib exists to:

- Reduce duplication across modules
- Provide consistent structural patterns
- Offer reusable components that maintain coherence
- Support module authors without imposing domain assumptions

This directory is optional.
Projects may leave it empty.

---

## Compiler Expectations

The compiler treats `stdlib/` as:

- A collection of optional shared utilities
- A downstream‑adjacent layer loaded before modules
- A non‑mandatory component of the project

The compiler does not require this directory to contain any files.

---

## Guidance

Humans do not need to memorize stdlib conventions.
Models can guide users through:

- deciding whether a utility belongs here
- creating reusable patterns
- maintaining clean boundaries
- avoiding upstream contamination

The stdlib is a convenience layer that supports clean, modular project design.
