# SignatureForge v0.1

## Document Contract

- owner: RCF tooling / SignatureForge
- type: Contract
- scope: optional numeric canonicalization and hashing
- authority: removable tooling-local contract
- upstream dependencies: `README.md`
- status: complete; optional; version 0.1
- purpose: define deterministic encoding and signature records for declared numeric bundles
- anti-backflow: signatures are opaque tooling records and do not define metric meaning, RCF admissibility, semantics, or invariants

**Document ID:** `S∞.RCF.Tooling.SignatureForge.v0.1`

---

## 0. Module contract

This module:

- **Does:**
  **canonicalize**, **order**, **encode**, and **hash** numeric metric bundles into stable signatures.
- **Does not:**
  define meaning, ontology, interpretation, or “what the metrics represent.”
- **Does not:**
  introduce new invariants or privilege any domain, representation, or substrate.

Removing SignatureForge does not invalidate RCF, RCD, or ValidationDriftGates; it only removes standardized signature construction.

---

## 1. Inputs / outputs

### Inputs

Caller MAY provide:

- **Raw metric bundles:**
  - `metrics.normalization` \(min, max, ε_norm\)
  - `metrics.conservation` \(C_t, C_t1, tol_C\)
  - `metrics.monotonicity` \(M_t, M_t1, tol_M\)
  - `metrics.drift` \(max_deviation, tol_drift\)
  - optional extra numeric fields (opaque)
- **Metadata:**
  - `module_id`
  - `timestamp`
  - `phase_id` (optional)
  - `run_id` (optional)

All inputs are treated as opaque numeric or scalar data.

### Outputs

- `signature_vector ∈ ℝ^n` (canonical numeric representation)
- `signature_hash ∈ {0,1}^k` (fixed-length bitstring)
- `signature_schema_id` (versioned schema tag)

---

## 2. Parameters (default values)

Caller MAY override:

- `ordering_mode ∈ {LEX, STABLE_TAG, CUSTOM}` (default: `LEX`)
- `hash_fn ∈ {BLAKE2s, SHA-256, XXH}` (default: implementation-defined, but fixed per schema)
- `float_encoding ∈ {IEEE754_64, IEEE754_32, FIXED_Q}` (default: IEEE754_64)
- `rounding_mode ∈ {NEAREST_EVEN, FLOOR, CEIL}` (default: NEAREST_EVEN)
- `schema_version = "SF.v0.1"`

---

## 3. Canonicalization pipeline

SignatureForge applies the following steps in order:

### **P1 — Field selection**

Construct a flat field list:

- required:
  - `normalization.min`
  - `normalization.max`
  - `normalization.epsilon`
  - `conservation.C_t`
  - `conservation.C_t1`
  - `conservation.tol_C`
  - `monotonicity.M_t`
  - `monotonicity.M_t1`
  - `monotonicity.tol_M`
  - `drift.max_deviation`
  - `drift.tol_drift`
- optional:
  - any additional numeric fields provided by caller (tagged)

Non-numeric fields are ignored.

---

### **P2 — Ordering**

Apply a deterministic ordering over fields:

- if `ordering_mode = LEX`:
  sort by fully-qualified field name (ASCII lexicographic)
- if `ordering_mode = STABLE_TAG`:
  sort by a pre-defined tag index table
- if `ordering_mode = CUSTOM`:
  caller MUST provide an ordering function or index map

Result: ordered list
\[
[(k_1, v_1), (k_2, v_2), \dots, (k_n, v_n)]
\]

---

### **P3 — Numeric encoding**

Each \( v_i \) is encoded as a fixed representation:

- apply `float_encoding`
- apply `rounding_mode`
- produce a byte sequence \( b_i \) of fixed length per value

Result: concatenated byte stream
\[
B = b_1 \Vert b_2 \Vert \dots \Vert b_n
\]

---

### **P4 — Signature vector construction**

In parallel, construct a numeric signature vector:

- map each \( v_i \) into a canonical numeric space (e.g., ℝ) after encoding
- produce:
  \[
  \mathbf{S} = (s_1, s_2, \dots, s_n)
  \]
- this is the `signature_vector`

This vector is what ValidationDriftGates uses for cross-phase comparison (G4).

Signature_vector entries are the post-rounding, post-encoding numeric values (not raw floats).

---

### **P5 — Hashing**

Apply `hash_fn` to the byte stream \( B \):

\[
H = \text{hash\_fn}(B)
\]

Output:

- `signature_hash = H`
- `signature_schema_id = schema_version`

---

## 4. Cross-phase alignment

To ensure comparability across phases:

- **Requirement:**
  Two signatures are comparable iff:
  - `signature_schema_id` matches
  - `ordering_mode` matches
  - `float_encoding` matches
  - `rounding_mode` matches

- **Alignment rule:**
  If any of the above differ, caller MUST:
  - either re-encode prior signatures under the new schema
  - or treat them as non-comparable and skip G4 for that pair

SignatureForge does not auto-align across schema changes.

---

## 5. Failure modes

SignatureForge MAY emit internal failures (tooling-level, not RCF-level):

- **SF-1:** non-numeric field in required slot
- **SF-2:** missing required metric field
- **SF-3:** unsupported encoding or hash function
- **SF-4:** inconsistent schema parameters across a batch
- **SF-5:** overflow/NaN during encoding

On failure:

- no `signature_hash` is emitted
- `signature_vector` MAY be partial or omitted
- a tooling-level error record is produced (for downstream logging)

These failures do **not** propagate as RCF invariants; they are purely tooling concerns.

---

## 6. Audit record (schema)

```yaml
timestamp: ISO-8601
module_id: <caller>
schema_version: SF.v0.1
ordering_mode: LEX | STABLE_TAG | CUSTOM
float_encoding: IEEE754_64 | IEEE754_32 | FIXED_Q
rounding_mode: NEAREST_EVEN | FLOOR | CEIL
fields:
  - name: <string>
    value: <numeric>
    encoded: <hex-bytes>
signature:
  vector_dim: n
  hash: <hex>
  hash_fn: <id>
errors: [SF-x, ...]
```

No semantic interpretation fields permitted.

---

## 7. Interface requirements (caller-side)

Caller MUST:

- provide required metric fields if they want a full signature
- ensure numeric types are representable under chosen `float_encoding`
- treat `signature_hash` and `signature_vector` as **opaque identifiers**, not as semantic objects

SignatureForge does not inspect or enforce meaning.

---

## 8. Non-goals

This module explicitly excludes:

- semantic drift detection
- anomaly interpretation
- clustering, classification, or labeling
- witness protocols
- human-readable summaries

Those belong to **downstream analysis modules**.

---

### **Status**

**SignatureForge — complete, eliminable, numerically canonical, validator-ready.**
