# Examples: conversion_laws

## Example 1: Discrete → Continuous

Given:
- discrete field Fd

Apply:
`discrete_to_continuous(Fd) → Fc`

Result:
- explicit continuous field Fc

---

## Example 2: Continuous → Metric

Given:
- continuous field Fc
- metric assignment g

Apply:
`continuous_to_metric(Fc, g) → Fg`

Result:
- metric‑aware field Fg

---

## Example 3: Metric → Amplitude

Given:
- metric‑aware field Fg
- amplitude structure α

Apply:
`metric_to_amplitude(Fg, α)`

Result:
- amplitude‑compatible record
- ready for downstream use
