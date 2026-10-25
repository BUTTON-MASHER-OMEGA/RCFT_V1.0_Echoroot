# Interface: conversion_laws

## Purpose
Defines the explicit interface for converting between physics representations.

---

## Inputs

- discrete field structures
- continuous field structures
- metric assignment records
- relational amplitude records

---

## Outputs

- conversion records
- compatibility record
- final boundary record (temporal, metric, amplitude)

---

## Allowed Transformations
- explicit discrete ↔ continuous conversions
- explicit metric‑aware conversions
- explicit amplitude‑aware conversions

---

## Forbidden Transformations
- no lossy hidden conversions
- no reinterpretation of geometry
- no reinterpretation of time
- no reinterpretation of amplitudes

---

# RCP‑1.1 Compliance
- All interfaces explicit
- No interpretive semantics
- No hidden state
