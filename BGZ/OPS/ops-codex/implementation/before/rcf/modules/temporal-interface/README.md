# Temporal Interface

## Document Contract

- owner: RCF temporal interface
- type: Status
- scope: module-local content of `temporal-interface/README.md`
- authority: tier-5 temporal parameter bridge
- upstream dependencies: `/modules/temporal-interface/interface.md`
- status: active
- purpose: expose sealed temporal parameters without inferring geometry or retuning upstream parameters
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

---

### *Local Constraint Density and Temporal Weighting*

The Temporal Interface is the first layer in RCF where **local relational structure** influences temporal behavior.
It sits between the pure Time Core and all operator modules (e.g., LCO, GCO).

This layer does **not** modify the temporal ontology defined in the Time Core.
Instead, it provides **local temporal parameters** that modulate how hard it is for a system to advance its refinement sequence.

The Temporal Interface provides:
- a definition of **local constraint density**, derived from relational admissibility,
- a mechanism for **bounding** the admissible temporal‑weighting regime (γ, ε),
- optional local pacing terms that contextualize refinement difficulty.

This layer exposes only **temporal parameters**, never geometric or global structural judgments, and does not define, reference, or require global time.

---

## 1. Purpose

The Temporal Interface answers:

> *Given the local relational neighborhood, how constrained is the system,
> and what temporal‑weighting regime is admissible for this context?*

It provides the lawful bridge between:
- **pure time** (relation‑independent), and
- **operators** (relation‑dependent).

Constraint density informs **temporal admissibility**, but does **not** generate γ or ε.

---

## 2. Architectural Position

The Temporal Interface is:

- **downstream** of the Time Core,
- **upstream** of all operator modules,
- **local** rather than global,
- **relation‑dependent** but **geometry‑neutral**,
- **parameter‑bounding**, not **parameter‑deriving**.

It does not:
- define time,
- modify event indices,
- alter refinement time,
- infer geometry,
- identify basins or attractors,
- compute global coherence.

---

## 3. Interface Contract Condition

The Temporal Interface may consume **local relational structure**,
but must expose only **temporal parameters** selected from sealed admissible ranges:

- γ (memory drag),
- ε (freedom to move),
- optional local pacing terms.

These parameters are **sealed**:
- γ and ε are **not derived** from geometry, mismatch, or operator behavior,
- γ and ε are **not computed** from constraint density,
- constraint density **bounds** the admissible ranges but does **not** generate them.

This firewall prevents the interface from becoming a proto‑GCO or a hidden dynamics layer.

---

## 4. Local and Scale‑Relative Temporal Behavior

Temporal behavior in RCF is **local and scale‑relative**:

- Each system carries its own refinement index.
- Subsystems carry their own local indices.
- A containing system carries an index over inter‑system relations.
- Constraint density is computed per system or region.
- Temporal parameters are **selected** locally from admissible ranges.

Time is not a single chain.
It is a hierarchy of lawful event orderings stitched together by relations across scales.

---

## 5. Files in This Module

- **local-constraint-density.md**
  Defines how relational structure produces a scalar constraint density that bounds temporal weighting.

- **temporal-weights-adapter.md**
  Uses constraint density to restrict the admissible ranges of γ, ε, and pacing terms without deriving them.

---

## 6. Status

The Temporal Interface is the lawful bridge between the Time Core and all operator modules.
It introduces relational dependence into temporal behavior without altering temporal ontology and without deriving sealed temporal parameters.
