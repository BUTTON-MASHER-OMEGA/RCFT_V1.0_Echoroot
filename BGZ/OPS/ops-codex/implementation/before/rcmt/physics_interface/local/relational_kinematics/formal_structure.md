# formal_structure.md
Relational Kinematics — Formal Structure

## Purpose
This file defines the mathematical structures permitted in `relational_kinematics`.
All definitions are adjacency‑local, invariant‑safe, and non‑interpretive.
No metric, no physics‑time, no PDEs, no hidden state.

---

# 1. Underlying Discrete Structure

## 1.1 Adjacency Graph
Let the kinematic substrate be a finite or countable graph:



\[
G = (V, E)
\]



where:
- \(V\) is the set of nodes (discrete supports),
- \(E \subseteq V \times V\) is the explicit adjacency relation.

Adjacency is the only structural relation available.

## 1.2 Neighborhood
For each node \(v \in V\):



\[
N(v) = \{ u \in V \mid (u, v) \in E \}
\]



All kinematic updates must be expressible using only \(v\) and \(N(v)\).

---

# 2. Local State and Global Configuration

## 2.1 Local State Space
Each node carries a state from a finite or structured set:



\[
\Sigma
\]



No interpretation is assigned to elements of \(\Sigma\).

## 2.2 Global Configuration
A configuration is a function:



\[
s : V \to \Sigma
\]



The system’s kinematic state is fully determined by \(s\).

---

# 3. Local View and Locality Constraint

## 3.1 Local View
The local view at node \(v\) under configuration \(s\) is:



\[
\text{view}_s(v) = \big( s(v), (s(u))_{u \in N(v)} \big)
\]



This is the maximal information any update at \(v\) may depend on.

## 3.2 Locality Constraint
All admissible updates must satisfy:



\[
s'(v) = F_v\big(\text{view}_s(v)\big)
\]



for some local update function \(F_v\).

No update may depend on non‑adjacent nodes.

---

# 4. Local Update Operator

Define a **local update operator**:



\[
U : \Sigma \times \Sigma^{N(v)} \to \Sigma
\]



The updated configuration is:



\[
s'(v) = U\big(s(v), (s(u))_{u \in N(v)}\big)
\]



This operator:
- is adjacency‑local,
- is explicit,
- introduces no hidden state,
- uses no metric or time.

\(U\) may be node-dependent \(U_v\) but must remain adjacency-local and explicit.

---

# 5. Kinematic Transition

A kinematic transition is a pair:



\[
T = (s, s')
\]



subject to:



\[
\forall v \in V,\quad s'(v) = U\big(s(v), (s(u))_{u \in N(v)}\big)
\]



## 5.1 Support of Change
The set of nodes whose state changes:



\[
\text{supp}(T) = \{ v \in V \mid s'(v) \neq s(v) \}
\]



This support is always finite for any admissible transition.

---

# 6. Transition Descriptor

Define the **transition descriptor**:



\[
D(T) = \big( \text{supp}(T), (s(v))_{v \in \text{supp}(T)}, (s'(v))_{v \in \text{supp}(T)} \big)
\]



This descriptor:
- captures exactly what changed,
- is adjacency‑safe,
- contains no global information.

---

# 7. Kinematic Boundary Record

The boundary record is:



\[
B(T) = \big( D(T), (N(v))_{v \in \text{supp}(T)} \big)
\]



This is the only data passed to `relational_dynamics`.

It contains:
- the local changes,
- the adjacency neighborhoods of changed nodes,
- no metric,
- no time,
- no hidden state.

---

# 8. Invariant Compliance

All structures in this file satisfy:

- adjacency‑only propagation
- explicit local updates
- no metric usage
- no physics‑time usage
- no PDEs
- no interpretive semantics
- no hidden state

This file is RCP‑1.1 compliant.
