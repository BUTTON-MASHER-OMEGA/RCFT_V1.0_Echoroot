# Operators: conversion_laws

## Purpose
Defines canonical conversion operators across representations.

---

## 1. `discrete_to_continuous`
**Inputs:** discrete field, support
**Outputs:** continuous field, support

Rules:
- explicit mapping
- no hidden interpolation

---

## 2. `continuous_to_metric`
**Inputs:** continuous field, metric assignment
**Outputs:** metric‑aware field record

Rules:
- no reinterpretation of metric

---

## 3. `metric_to_amplitude`
**Inputs:** metric‑aware field, amplitude structure
**Outputs:** amplitude‑compatible record

Rules:
- no probability semantics
- no Hilbert spaces

---

# RCP‑1.1 Compliance
- All operators explicit
- No interpretive semantics
- No hidden state
