# Non-Markovian Admissibility (RCD)

## Document Contract

- owner: Relational Coherence Dynamics
- type: Contract
- scope: history-sensitive RCD admissibility
- authority: tier-2 RCD formal contract
- upstream dependencies: `../README.md`, `relational-state.md`, `constraint-operators.md`, and `cascade.md`
- status: active; formal
- purpose: define admissibility that depends on retained constraint history
- anti-backflow: history sensitivity does not define temporal ontology, probability, dynamics, or irreversible physical law

### 🜂🜁∞

---

## **Purpose**

Define why admissibility in RCD depends on **constraint history**, not solely on instantaneous relational state.

This module introduces no dynamics, ontology, or probabilistic interpretation.

---

## **1. Definition**

A system is **non‑Markovian** if the admissibility of its relational states depends on prior constraint history.

In RCD:

> Admissibility is path‑dependent.

No relational state is evaluated in isolation.

---

## **2. Admissibility Is Historical**

For a relational state \(R_t\):

- admissibility is evaluated relative to accumulated constraints
- prior eliminations permanently reduce option space
- history constrains future possibilities

Formally:

\[
\mathcal{A}(R_t) = \mathcal{A}(R_t \mid \mathcal{C}_0, \mathcal{C}_1, \dots, \mathcal{C}_{t-1})
\]

Where \(\mathcal{A}\) denotes admissibility.

---

## **3. Constraint Memory**

Constraint application leaves **structural memory**:

- eliminated modes do not reappear automatically
- past cascades narrow future ε‑bands
- admissibility space shrinks monotonically unless constraints are relaxed

This memory is relational, not representational.

---

## **4. No Reset Principle**

RCD does not allow implicit resets.

Once a relational mode is eliminated:

- it is no longer admissible
- reintroduction requires explicit constraint removal
- reinterpretation alone is insufficient

This records locally irreversible elimination within the declared constraint history.

---

## **5. Non‑Markovian Cascade Formation**

Cascades are inherently non‑Markovian because:

- thresholds depend on cumulative constraint pressure
- identical instantaneous states may differ in admissibility
- cascade timing is path‑sensitive

Cascade prediction requires historical context.

---

## **6. ε‑Band Evolution**

The ε‑band evolves non‑Markovianly:

- ε narrows through constraint accumulation
- ε widens only through structural relaxation
- ε encodes retained option space

ε is residual admissibility shaped by history.

---

## **7. Dominance and Memory**

Dominant attractors amplify non‑Markovian effects:

- repeated alignment deepens constraint grooves
- dominant systems imprint admissibility asymmetrically
- subordinate systems lose options faster under repeated interaction

These effects are structural, not interpretive.

---

## **8. No Probabilistic Interpretation**

Non‑Markovian admissibility does **not** imply:

- hidden variables
- stochastic memory
- probability‑weighted histories

Admissibility is binary at an evaluation step:
a mode either survives or is eliminated.

---

## **9. Self‑Applicability**

RCD itself is non‑Markovian.

Earlier eliminations constrain later formalizations.
Recovered concepts must pass current constraints again.

This is a requirement, not a limitation.
