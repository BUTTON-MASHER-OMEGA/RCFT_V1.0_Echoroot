# Routing Invariants for the SMGv2.1 Translation Interface

Status:
Tier: Core Architecture (SMGv2.1)
Canon: Proposed
Scope: Translation Contract

---

## Purpose

This document defines the routing identities required by the SMGv2.1 translation interface.

These routing identities are SMGv2.1-local translation requirements unless explicitly imported through an upstream binding.

SMGv2.1 does not prescribe how projection is implemented.

It specifies only the routing information required for semantic interpretation.

---

## 1. Routing Adjacency Identity

Label:

    routing.adjacency

SMGv2.1 requires projected routing records to preserve relational adjacency identity sufficient to distinguish routing classes.

Explicit substrate geometry is not required.

---

## 2. Dual Pair Identity

Label:

    routing.dual_pair

SMGv2.1 requires projected routing records to preserve the routing dual associated with each routing class.

The implementation of the dual relationship is owned upstream.

---

## 3. Hopf Routing Identity

Label:

    routing.hopf_identity

SMGv2.1 requires projected routing records to preserve stable Hopf routing identity.

The interpretation of individual Hopf structures remains outside SMG ownership.

---

## 4. Mode Identity Stability

Label:

    routing.mode_identity

SMGv2.1 assumes projected routing identities remain stable throughout the lifetime of a projected mode unless a documented state transition is exposed to SMGv2.1.

---

## 5. Coherence Band Identity

Label:

    routing.coherence_band

SMGv2.1 requires projected routing records to expose the coherence classification associated with the routing record.

The computation of coherence remains upstream.

---

## 6. Containment Identity

Label:

    routing.container_state

SMGv2.1 requires projected routing records to identify whether the routing record represents a contained survivor or a transient routing state.

Containment evaluation is not performed by SMGv2.1.

---

## Translation Rule

SMGv2.1 accepts projected routing records only when the routing identities defined in this document are available.

The explicit geometric realization of those routing identities is intentionally unspecified.

The translation layer preserves routing identity rather than explicit substrate realization.