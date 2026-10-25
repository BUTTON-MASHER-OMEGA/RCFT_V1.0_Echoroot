# SignatureForge and ValidationDriftGates Glue Note

## Document Contract

- owner: RCF tooling documentation
- type: Permission
- scope: optional interface compatibility note
- authority: nonbinding tooling documentation
- upstream dependencies: `../signature-forge.md` and `../validation-gates.md`
- status: active documentation
- purpose: document when a SignatureForge vector may be supplied to ValidationDriftGates G4
- anti-backflow: this note creates no module dependency, guarantee, semantic coupling, or RCF-level requirement

**Document ID:** `S∞.RCF.Tooling.GlueNote`

---

## **0. Non‑Binding Relationship**

`ValidationDriftGates` and `SignatureForge` are **independent tooling modules**.

- Neither imports the other
- Neither assumes the other exists
- Neither defines meaning for the other’s outputs
- Both remain fully eliminable

The glue note exists only to clarify **how** their interfaces align *if* both are present.

---

## **1. What G4 Expects (Interface‑Only)**

`ValidationDriftGates.G4` requires two objects:

```
S^(A) : signature_vector
S^(B) : signature_vector
```

with the following minimal properties:

- both are numeric vectors
- both have equal dimensionality
- both are ordered consistently
- both are stable under perturbation

G4 does **not** require:

- knowledge of how the vectors were constructed
- knowledge of hashing
- knowledge of schema versions
- knowledge of encoding or rounding modes
- any semantic interpretation

G4 treats signatures as **opaque numeric arrays**.

---

## **2. How SignatureForge Satisfies That Interface (Optionally)**

If a caller uses `SignatureForge`, then:

- `signature_vector` is already canonicalized
- ordering is deterministic
- encoding is stable
- schema version ensures comparability
- perturbation stability remains subject to G5 evaluation

Thus, a `signature_vector` produced by SignatureForge is a candidate G4 input when the caller verifies the compatibility conditions below.

But this is **not required**.
Any module may produce its own signature vectors.

---

## **3. Compatibility Conditions (Non‑Enforced)**

Two signatures are *comparable* under G4 if:

- they have the same dimensionality
- they share the same ordering convention
- they use compatible numeric encodings

If the caller uses SignatureForge, these conditions are satisfied when:

- `signature_schema_id` matches
- `ordering_mode` matches
- `float_encoding` matches
- `rounding_mode` matches

G4 does **not** check these conditions.
It is the caller’s responsibility.

---

## **4. No Cross‑Module Guarantees**

This glue note explicitly states:

- SignatureForge does **not** guarantee G4 compliance
- G4 does **not** require SignatureForge
- No module is privileged
- No dependency is implied

This preserves RCF’s eliminability and modularity.

---

## **5. Caller Responsibilities**

If a caller uses both modules:

- they MAY pass `signature_vector` from SignatureForge into G4
- they MUST ensure schema compatibility
- they MUST handle schema mismatches
- they MUST treat `signature_hash` as irrelevant to G4

G4 ignores hashes entirely.

---

## **6. Summary**

**SignatureForge produces canonical numeric vectors.
ValidationDriftGates consumes canonical numeric vectors.
Neither module knows the other exists.**

This glue note documents the *shape* of the handshake without creating a dependency.
