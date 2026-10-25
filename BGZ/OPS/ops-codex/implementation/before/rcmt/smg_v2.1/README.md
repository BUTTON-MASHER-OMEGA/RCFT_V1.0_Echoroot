# SMGv2.1 - Standard Model of Geometry

## Status

SMGv2.1 is a downstream, theory-facing domain model.

It is not neutral prose.
It is not RCF canon.
It is not an upstream physics-interface or gcm-interface module.

SMGv2.1 is a scoped translation bridge that consumes read-only upstream structural records and maps them into a forceful Standard Model-facing semantic frame.

---

## Purpose

SMGv2.1 takes upstream structural records from `gcm_interface` and `physics_interface` and translates them into downstream-readable Standard Model geometry claims.

The model keeps its thesis energy:

- SMGv2.1 asserts Dual Closure as its central model-local thesis.
- SMGv2.1 maps survivor-mode structure into Standard Model semantic labels.
- SMGv2.1 treats continuous gauge, spinor, mass, and resonance forms as downstream shadows of discrete structural patterns.

Those premises are SMGv2.1-local.
Downstream modules that adopt SMGv2.1 inherit them.
Upstream RCF, `gcm_interface`, and `physics_interface` do not inherit them.

---

## Architectural Position

```text
RCF / geometry stack
        |
gcm_interface + physics_interface
        |
SMGv2.1
        |
downstream Standard Model-facing models
```

SMGv2.1 is downstream of:

- `gcm_interface`
- `physics_interface`
- RCF geometry, time, metric, and conversion contracts

SMGv2.1 may consume upstream records.
SMGv2.1 may not repair, reinterpret, or rewrite upstream ownership.

---

## Directory Roles

### `bindings/`

Pure read-only imports from upstream interfaces.
These files contain no thesis claims, no downstream enforcement, and no physical commentary.

### `claims/`

SMGv2.1-local thesis documents.
These files may be bold, but their claims remain scoped to SMGv2.1.

### `dictionary/`

Semantic lookup tables and naming maps.
Physical names are downstream semantic labels, not upstream facts.

### `translations/`

SMGv2.1-local translation bridges from upstream structural patterns to downstream semantic or continuous forms.
These files may be theory-forward, but they remain downstream-scoped.

### `policies/`

Rules for downstream models that choose to adopt SMGv2.1.
Policy failures are invalid under SMGv2.1, not universally invalid.

### `exports/`

Downstream-readable payloads.
Exports may initialize structures for adopters, but they must not smuggle in new premises.

---

## Reading Order

1. `smg_contract.md`
2. `bindings/gcm_bindings.md`
3. `bindings/physics_bindings.md`
4. `claims/dual_closure.md`
5. `dictionary/mode_dictionary.md`
6. `dictionary/gamma_generations.md`
7. `translations/gauge_translation.md`
8. `translations/spinor_mapping.md`
9. `translations/zeta_resonance.md`
10. `policies/limitations_and_boundaries.md`
11. `exports/standard_model_kinematics.md`

---

## Anti-Backflow Rule

SMGv2.1 may say:

- within SMGv2.1
- SMGv2.1 asserts
- SMGv2.1 maps
- invalid under SMGv2.1
- for downstream models adopting SMGv2.1

SMGv2.1 may not say or imply that its thesis claims are upstream-proven RCF facts unless a statement is explicitly imported from upstream.

---

## RCP-1.1 Compatibility

- upstream imports remain read-only
- thesis claims are explicitly SMGv2.1-local
- downstream obligations apply only to SMGv2.1 adopters
- no backflow into RCF, `gcm_interface`, or `physics_interface`
