# Geometry-Dynamics Module (GDM)

## Document Contract

- owner: RCF geometry-dynamics module
- type: Status
- scope: module-local content of `geometry/dynamics/README.md`
- authority: tier-8 downstream continuous-variation layer
- upstream dependencies: `/modules/geometry/dynamics/interface.md`
- status: active
- purpose: describe continuous relational variation without introducing forces, fields, or upstream primitives
- anti-backflow: this document may refine its module-local contract but may not redefine cited upstream structures, transfer downstream assumptions upstream, or exceed the module authority declared here

## purpose
The Geometry‑Dynamics Module (GDM) records the **continuous relational structure** that appears after projection by Π.
It introduces **no new primitives**.
All operators in this module are **continuous shadows** of upstream discrete structure:

- mismatch → local mismatch density
- curvature → second‑order relational variation
- flow → relational characterization of mismatch-related variation under Π-consistent smoothing
- least‑action → cumulative mismatch represented continuously

The GDM is:

- coordinate‑free,
- metric‑optional,
- interpretation‑neutral,
- and fully Π‑determined.

It sits downstream of geometry‑continuous and upstream of any metric‑aware or domain‑specific modules.

```
geometry‑discrete → Π → geometry‑continuous → geometry‑dynamics → (metrics, kinematics, etc.)
```

---

## scope
The GDM documents:

- mismatch as a continuous functional \(L\),
- curvature as second‑order relational variation,
- flow as Π‑consistent relational smoothing,
- the action functional as cumulative mismatch,
- stationary action (δS = 0) as a **retroactive summary** of survivor structure,
- survivorship bias as the structural context for δS = 0.

The GDM **does not**:

- assume a metric,
- assume coordinates,
- introduce forces, fields, or potentials,
- assign numerical values,
- interpret any operator physically.

---

## inputs

### 1. continuous shadow
From `continuous-shadow.md`:

> “a smoothed representation of discrete adjacency relations”
> and
> “second-order relational variation corresponding to slush and mismatch in the discrete layer.”

### 2. sector structure and tags
From the GCM interface module…:

- γ‑plateau assignments,
- ordering structure provided upstream,
- adjacency structure,
- dual‑closure tags.

### 3. discrete mismatch
The upstream mismatch functional \(F(G,\gamma)\), which:

> “evaluates the structural penalty of a local geometry.”

---

## internal components

### 1. mismatch (`mismatch/`)
Defines the continuous mismatch functional \(L\) as a relational density derived from discrete mismatch.
No coordinates, metrics, or external structure are introduced.

### 2. curvature (`curvature/`)
Describes curvature as second‑order relational variation of mismatch gradients.
Curvature is coordinate‑free and metric‑optional.

### 3. flow (`flow/`)
Describes flow as a relational characterization of mismatch‑related variation under Π‑consistent smoothing.
Flow is parameter‑agnostic and does not assume time.

### 4. least‑action (`least-action/`)
Describes the action functional as cumulative mismatch and records δS = 0 as a **retroactive structural observation**.

### 5. survivorship bias (`survivorship-bias.md`)
Records how survivor‑consistent configurations correspond to stationary action in current evaluations, without implying optimization or causality.

---

## outputs

### 1. relational variation descriptions
- mismatch density,
- curvature‑related variation,
- flow‑consistent relational patterns.

### 2. variational structure
- Lagrangian \(L\),
- action functional \(S\),
- stationary action (δS = 0) as a survivor‑consistent summary.

### 3. structural stability
All outputs reflect the survivor‑only nature of Π and remain interpretation‑neutral.

---

## invariants
The GDM enforces:

- **No new primitives**
  All operators are shadows of mismatch, slush, closure, and Π’s smoothing rules.

- **Coordinate‑free formulation**
  No coordinate charts or basis choices.

- **Metric‑optional behavior**
  Metrics may appear downstream but are not required here.

- **Π‑determinism**
  All structure is determined by mismatch, survivor sequence, and Π.

- **Locality**
  All densities and variations are locally described.

- **Evolution without time**
  Flow is parameter‑agnostic; τ appears only as a Π-induced parameterization in least-action.

- **Retroactive optimality**
  δS = 0 is a summary of survivor structure, not a governing rule.

- **No physical interpretation**
  All operators are structural and relational.

---

## status
The Geometry‑Dynamics Module is structurally complete and aligned with all upstream and downstream constraints.
It is ready for integration with metric‑aware, kinematic, or domain‑specific modules.
