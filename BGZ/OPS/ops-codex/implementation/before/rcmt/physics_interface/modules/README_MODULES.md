# Modules Layer — Downstream Domain Content

The `modules/` directory contains the **downstream components** of the project.
Modules define the actual domain content of the theory being built on top of RCF.

This layer is where users express concepts, structures, and relationships specific to their project’s domain.

---

## What Belongs in `modules/`

- Domain modules written in Markdown
- Definitions, claims, and structures expressed in Typed English
- Components that depend on:
  - the RCF upstream core (`rcf_compressed.md`)
  - the project’s local invariants (`local/invariants.md`)
- Any downstream logic, examples, or domain‑specific reasoning

Modules are the **working space** of the project.

---

## What Does *Not* Belong in `modules/`

- Upstream invariants (place those in `local/invariants.md`)
- Project‑level primitives or constraints
- Shared utilities intended for reuse across projects (`stdlib/`)
- Any modification of the RCF core

Modules must not redefine or override upstream or local invariants.

---

## Interfaces Available From Upstream

Modules may import:

- All invariants and primitives defined in `rcf_compressed.md`
- All project‑specific invariants defined in `local/invariants.md`
- Any shared utilities placed in `stdlib/`

Modules may **not**:

- Modify upstream invariants
- Introduce new upstream‑adjacent structure
- Collapse upstream and downstream boundaries

Modules operate strictly in the **downstream** layer.

---

## Module Structure

Each module should:

- Be a single `.md` file
- Contain Typed English definitions, claims, or structures
- Reference upstream and local invariants as needed
- Maintain internal coherence
- Respect all project‑level constraints

Modules may be organized by domain, theme, or dependency, but must remain within this directory.

---

## Compiler Expectations

The compiler treats `modules/` as:

- The downstream content layer
- A collection of domain modules to be validated
- The final stage of the build process

The compiler loads modules **after**:

1. The RCF core
2. The project’s local invariants

Modules must compile cleanly against both.

---

## Guidance

Humans do not need to memorize module structure.
Models can guide users through:

- creating new modules
- extending existing ones
- ensuring compatibility with invariants
- maintaining clean downstream boundaries

This directory is the primary workspace for building the project’s theory.
