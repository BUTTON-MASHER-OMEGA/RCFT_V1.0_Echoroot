# SMGv2.1 Standard Model Kinematics Export

## Purpose

This file provides a downstream-readable SMGv2.1 export payload.

It initializes Standard Model-facing continuous forms as SMGv2.1-authorized downstream approximations.
It is not a contract file and not a policy file.

---

## 1. Export Status

- export type: downstream payload
- authority scope: SMGv2.1 adopters
- required contract: `smg_contract.md`
- required policy: `policies/limitations_and_boundaries.md`
- required dictionaries:
  - `dictionary/mode_dictionary.md`
  - `dictionary/gamma_generations.md`

---

## 2. 36-Mode State Vector

SMGv2.1 exports the 36-mode semantic state vector:

```text
Psi_36 = Plateau[A,B,C] x BaseMode[1..12]
```

This is an SMGv2.1-authorized downstream approximation.

The labels attached to this vector are semantic labels from the dictionary, not upstream particle objects.

---

## 3. Generation Blocks

SMGv2.1 exports:

- Plateau A block as first-generation downstream labels
- Plateau B block as second-generation downstream labels
- Plateau C block as third-generation downstream labels

No fourth-generation block is exported by SMGv2.1.

---

## 4. Authorized Continuous Forms

For downstream models adopting SMGv2.1, the following may be used as SMGv2.1-authorized downstream approximations:

- Dirac-style free propagation form
- gauge-interaction vertex form
- Standard Model-style kinematic Lagrangian form

These forms are descriptive exports.
They are not upstream laws.

---

## 5. Boundary Requirements

Any downstream use of this export must remain within:

- `B_T`
- `B_g`
- `B_alpha`
- `B_final`

Use outside these boundaries is invalid under SMGv2.1.

---

## 6. Export Non-Smuggling Clause

This export does not introduce:

- new upstream primitives
- new SMGv2.1 thesis claims
- policy rules not already stated in `policies/`
- binding rules not already stated in `bindings/`

---

## RCP-1.1 Compatibility

- exported forms are downstream approximations
- no hidden premises are introduced
- no upstream law is rewritten
