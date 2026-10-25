# 24-Cell Translation Contract

---

## Purpose

This document defines the translation interface consumed by SMGv2.1 when interpreting projected routing records.

It establishes the minimal routing identities required for SMGv2.1 semantic interpretation.

This document does not define projection mechanics, particle dynamics, Local Coherence Operator behavior, or upstream RCMT structure.

Unless explicitly imported through an upstream binding, the routing identities described here are SMGv2.1-local translation constructs.

---

## Translation Principle

Within SMGv2.1, imported substrate routing records are treated as the computational source used to construct projected Standard Model interpretations.

SMGv2.1 operates only on the relational information exposed through the translation interface.

Explicit substrate geometry is neither reconstructed nor required.

---

## 24-Cell Routing Interpretation

Within SMGv2.1, 24-cell routing classes are interpreted as stable routing identities that distinguish protected mode families.

SMGv2.1 does not require explicit reconstruction of the underlying 24-cell geometry.

Only the routing identity associated with the projected record is consumed.

---

## Hopf Routing Interpretation

SMGv2.1 interprets Hopf routing identities as stable transport descriptors associated with projected routing records.

The implementation of the projection operator is outside the scope of this document.

Only the resulting routing identity is required for downstream semantic interpretation.

---

## Mode Translation

Within SMGv2.1, projected routing identities are translated into projected mode labels.

Conceptually,

    substrate routing record
            ↓
      routing identity
            ↓
       projected mode label
            ↓
       SMG interpretation

SMGv2.1 preserves routing identity rather than explicit geometric realization.

---

## Separation of Responsibilities

This document intentionally excludes:

- projection implementation
- particle dynamics
- coherence evolution
- gravity
- mass hierarchy
- electric charge
- spin dynamics
- propagation

These behaviors are owned by their respective RCMT modules.

SMGv2.1 consumes translation records produced elsewhere.

---

## Architect's Addendum

### Computation Analogy

SMGv2.1 adopts the following architectural interpretation:

**Computation Layer**

- 600-cell substrate
- Local Coherence Operator (LCO)

This layer evaluates relational tension across imported substrate adjacency and exposes results from which routing records may be constructed.

**Translation Layer**

Projection and routing translation expose stable routing identities suitable for downstream interpretation.

The translation layer is treated as an interface contract rather than a physical object.

**Interpretation Layer**

SMGv2.1 consumes projected routing identities and renders them into observable Standard Model semantic forms.

---

## Implementation Note

SMGv2.1 accepts projected routing records only when the routing identities required by this translation contract are preserved.

Multiple projection implementations are therefore admissible provided they expose routing records satisfying the SMGv2.1 translation interface.