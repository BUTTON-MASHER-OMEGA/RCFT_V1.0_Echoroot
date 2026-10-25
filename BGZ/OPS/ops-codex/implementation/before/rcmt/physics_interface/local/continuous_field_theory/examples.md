# Examples: continuous_field_theory

## Example 1: Field Lift

Given:
- discrete support: {A, B}
- field values: vA, vB

Apply:
`lift_field({A, B}, {vA, vB})`

Result:
- continuous support
- continuous assignment record

---

## Example 2: Continuous Update

Given:
- continuous field assignment: Φ
- physics‑time index: τ0

Apply:
`update_continuous_field(Φ, τ0)`

Result:
- continuous field update record
- no PDEs used
