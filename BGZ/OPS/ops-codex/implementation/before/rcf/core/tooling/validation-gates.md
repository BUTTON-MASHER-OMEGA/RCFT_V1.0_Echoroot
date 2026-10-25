# ValidationDriftGates v0.1

## Document Contract

- owner: RCF tooling / ValidationDriftGates
- type: Contract
- scope: optional numeric validation gates
- authority: removable tooling-local contract
- upstream dependencies: `README.md`
- status: complete; optional; version 0.1
- purpose: define numeric gates, tolerances, failure codes, and audit records for callers that adopt this interface
- anti-backflow: gate results are tooling-local and do not define RCF invariants, semantic validity, or upstream admissibility

**Document ID:** `S∞.RCF.Tooling.ValidationDriftGates.v0.1`

---

## 0. Module Contract

This module:

* **Does**: define validation gates, tolerances, interfaces, and failure conditions
* **Does not**: define meaning, interpretation, ontology, intent, or pedagogy
* **Does not**: privilege any representation, domain, or substrate
* **Does not**: introduce new invariants

Removing ValidationDriftGates does not invalidate RCF or RCD; it removes enforcement only.

---

## 1. Inputs / Outputs

### Inputs

Any upstream module MAY provide the following numeric streams:

* Normalized state vectors
* Conserved totals
* Monotone metrics
* Checkpointed signatures
* Perturbed variants of the above

All inputs are treated as opaque numeric data.

### Outputs

* `gate_status ∈ {PASS, FAIL}`
* `failure_codes ⊆ {FM-1 … FM-5}`
* `drift_report = (metric_id, max_deviation)`
* `audit_record` (structured, machine-readable)

---

## 2. Parameters (default values)

These defaults MAY be overridden by the calling module.

* `ε_norm = 1e-6`
* `tol_C = 1e-9`
* `tol_M = 1e-12`
* `tol_drift = 1e-9`
* `ε_perturb = 1e-8`

---

## 3. Gates (formal definitions)

### **G1 — Normalization Span Gate**

Given a normalized quantity
\[
G_{\text{norm}} = \frac{G - G_{\min}}{G_{\max} - G_{\min}}
\]

Reject if:

1. \( G_{\max} - G_{\min} < \varepsilon_{\text{norm}} \)
2. \( G_{\text{norm}} \notin [-\varepsilon_{\text{norm}}, 1+\varepsilon_{\text{norm}}] \)

---

### **G2 — Conservation Gate**

Given a conserved scalar \( C_t \):

Reject if:
\[
|C_{t+1} - C_t| > tol_C
\]

---

### **G3 — Monotonicity Gate**

Given a monotone scalar \( M_t \):

Reject if:
\[
M_{t+1} - M_t < -tol_M
\]

---

### **G4 — Cross-Phase Drift Gate**

Given two independently computed metric signatures
\( S^{(A)}, S^{(B)} \):

Reject if:
\[
\max_i |S^{(A)}_i - S^{(B)}_i| > tol_{\text{drift}}
\]

---

### **G5 — Perturbation Stability Gate**

Given a perturbed input
\( x' = x + \delta \), with \( |\delta| \le \varepsilon_{\text{perturb}} \):

Re-evaluate G1–G4.

Reject if any gate fails under perturbation.

---

## 4. Failure Modes

| Code     | Condition                              |
| -------- | -------------------------------------- |
| **FM-1** | Normalization span or bounds violation |
| **FM-2** | Conservation violation                 |
| **FM-3** | Monotonicity violation                 |
| **FM-4** | Cross-phase drift violation            |
| **FM-5** | Perturbation instability               |

Failure codes are **non-hierarchical** and may co-occur.

---

## 5. Audit Record (schema)

All evaluations emit an audit record:

```yaml
timestamp: ISO-8601
module_id: <caller>
gate_status: PASS | FAIL
failure_codes: [FM-x, ...]
metrics:
  normalization: {min, max, epsilon}
  conservation: {C_t, C_t1, tol_C}
  monotonicity: {M_t, M_t1, tol_M}
  drift: {max_deviation, tol_drift}
perturbation:
  epsilon: ε_perturb
  triggered: true|false
hash: <stable_signature>
```

No semantic fields permitted.

---

## 6. Interface Requirements (caller-side)

Any module invoking ValidationDriftGates MUST expose:

* `metrics.normalization`
* `metrics.conservation`
* `metrics.monotonicity`
* `metrics.signature`
* `checkpoint.serialize()`

ValidationDriftGates does not inspect internal structure beyond these interfaces.

---

## 7. Non-Goals (explicit exclusions)

This module intentionally excludes:

* semantic drift detection
* language analysis
* witness protocols
* clarity floors
* interpretation of failure meaning

Those belong in **downstream modules**.

---

## 8. Termination Condition

ValidationDriftGates evaluates all enabled gates for the current batch, records every violation, and then halts downstream continuation when the aggregate status is `FAIL`.

---

### **Status**

**ValidationDriftGates — complete, formal-only, invariant-safe**
