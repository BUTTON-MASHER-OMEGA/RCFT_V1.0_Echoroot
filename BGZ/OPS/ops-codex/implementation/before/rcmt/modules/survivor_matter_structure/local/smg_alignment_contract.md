# SMG Alignment Contract (Candidate RCMT Interpretation)

## Document Contract

- owner: RCMT Survivor Matter Structure
- type: Contract
- scope: read-only adoption of SMGv2.1 semantic mappings
- depends_on: `../README.md` and `../../../smg_v2.1/smg_contract.md`
- status: active exploratory contract
- anti-backflow: this contract cannot modify SMGv2.1 or upstream projection ownership

## 1. Purpose

This file defines the alignment boundary between:

- SMGv2.1 (semantic projection), and
- the Matter Structure Module (candidate survivor identity interpretation).

The purpose of this contract is to preserve ownership separation while allowing downstream matter-structure interpretations to consume SMGv2.1 semantic mappings.

This file is downstream-scoped and does not modify RCF core, `physics_interface`, `gcm_interface`, or SMGv2.1 ownership.

---

## 2. Imported Structures

The Matter Structure Module consumes the following SMGv2.1 structures read-only:

- Chromatic-12 semantic partition,
- Gamma plateau mappings,
- 36-mode semantic indexing,
- spinor translation labels,
- gauge-translation labels,
- and semantic generation labels.

These imported structures remain SMGv2.1-owned.

The Matter Structure Module may not redefine:

- the 36-mode dictionary,
- Chromatic-12 partition semantics,
- gamma plateau ownership,
- or SMGv2.1 translation bindings.

---

## 3. Matter Structure Responsibilities

The Matter Structure Module is responsible for proposing candidate interpretations of:

- survivor offset regimes,
- phase-cycle structure,
- baryon-like composite stability,
- meson-like transient conflicts,
- flavor-like renegotiation,
- and LCO work profiles associated with survivor persistence.

These interpretations are downstream and provisional.

---

## 4. Projection Interpretation Boundary

SMGv2.1 defines model-local semantic translation structure. Upstream geometry and projection modules retain geometric projection ownership.

The Matter Structure Module may interpret:

- correspondences between imported 600-cell orientation, 120-cell representation, and survivor-regime records,
- how those regimes align with SMG semantic labels,
- and how LCO refinement work varies across those regimes.

The Matter Structure Module may not:

- redefine projection ownership,
- assert new upstream geometry,
- or replace SMG semantic mappings.

---

## 5. LCO Work Alignment

Within this module, LCO refinement work is treated as a candidate structural correlate for:

- mass-like behavior,
- survivor persistence cost,
- offset depth,
- and mismatch production.

This interpretation remains RCMT-local.

No claim is made that SMGv2.1 imports or requires this interpretation.

---

## 6. Semantic Label Usage

Physical labels such as:

- electron,
- neutrino,
- quark,
- baryon,
- meson,
- color,
- and flavor

are treated as downstream semantic approximations inherited from SMGv2.1.

The Matter Structure Module does not treat these labels as upstream ontology.

---

## 7. Boundary with Geometry-Behavior

The Matter Structure Module defines candidate identity structures only.

It does not define:

- mismatch propagation,
- echoon dynamics,
- reinjection behavior,
- collapse windows,
- or curvature-like gradients.

Those responsibilities belong to the Geometry-Behavior Module.

---

## 8. Open Status

This contract remains exploratory.

The following areas remain unresolved:

- precise 600-cell → 120-cell orientation mappings,
- admissible survivor orientation classes,
- detailed LCO work distributions,
- and stable baryon-like configuration rules.

These unresolved areas remain open to refinement.

---

## 9. Intent

This contract ensures that:

- SMGv2.1 retains semantic ownership,
- Matter Structure retains survivor-identity interpretation freedom,
- and downstream RCMT development remains modular, replaceable, and compatible with RCP-1.1 directionality discipline.
