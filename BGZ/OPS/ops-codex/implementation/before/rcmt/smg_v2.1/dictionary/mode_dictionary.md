# SMGv2.1 Mode Dictionary

## Purpose

This file defines the SMGv2.1 semantic lookup table from upstream structural indices to downstream Standard Model-facing labels.

This is a dictionary file.
It maps names.
It does not prove that the names are upstream facts.

---

## 1. Source Bindings

This dictionary consumes:

- `BaseMode_Index` from `bindings/gcm_bindings.md`
- `Gamma_Plateaus` from `bindings/gcm_bindings.md`
- `Survivor_Modes` from `bindings/gcm_bindings.md`

The imported structures remain upstream-owned.
The labels below are SMGv2.1 downstream semantic labels.

---

## 2. Chromatic-12 Semantic Partition

SMGv2.1 maps the twelve base modes into the downstream semantic partition:

- modes `1..8`: `Octet`
- modes `9..11`: `Triplet`
- mode `12`: `Singlet`

SMGv2.1 maps these names to the Standard Model-facing labels:

- `Octet`: downstream color-routing label
- `Triplet`: downstream weak-routing label
- `Singlet`: downstream neutral-phase label

These names are SMGv2.1 dictionary labels, not upstream ontology.

---

## 3. Generation Labels

SMGv2.1 maps:

- Plateau A to Generation 1 labels
- Plateau B to Generation 2 labels
- Plateau C to Generation 3 labels

The plateau records are imported structurally.
The generation names are downstream semantic labels.

---

## 4. Canonical Semantic Table

### Plateau A

| SMGv2.1 structural input | downstream semantic label | subset |
| --- | --- | --- |
| `Mode(1..8, A)` | Up / Down Quark labels | Octet |
| `Mode(9..11, A)` | Electron-family charged label | Triplet |
| `Mode(12, A)` | Electron-neutrino label | Singlet |

### Plateau B

| SMGv2.1 structural input | downstream semantic label | subset |
| --- | --- | --- |
| `Mode(1..8, B)` | Charm / Strange Quark labels | Octet |
| `Mode(9..11, B)` | Muon-family charged label | Triplet |
| `Mode(12, B)` | Muon-neutrino label | Singlet |

### Plateau C

| SMGv2.1 structural input | downstream semantic label | subset |
| --- | --- | --- |
| `Mode(1..8, C)` | Top / Bottom Quark labels | Octet |
| `Mode(9..11, C)` | Tau-family charged label | Triplet |
| `Mode(12, C)` | Tau-neutrino label | Singlet |

---

## 5. Anti-Ontology Clause

Within SMGv2.1:

- particle names are downstream semantic labels
- color charge is a downstream semantic label
- mass is a downstream metric-shadow label

SMGv2.1 does not treat these labels as fundamental upstream objects.

For downstream models adopting SMGv2.1, treating these labels as generative upstream objects is invalid under the SMGv2.1 contract.

---

## RCP-1.1 Compatibility

- physical names are classified as downstream semantic labels
- upstream structure remains upstream-owned
- no new ontology is introduced
