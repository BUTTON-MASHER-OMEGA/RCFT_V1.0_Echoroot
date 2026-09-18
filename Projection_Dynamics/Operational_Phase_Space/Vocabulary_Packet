I think SIM14.0–14.3 has accumulated enough terminology that we should freeze a **SIM14 vocabulary packet** before going farther.

I’d separate each entry into **standard mathematics**, **RCFT/SIM terminology**, and **status**, so we never accidentally turn a computational observation into an established physical object. The standard mathematical background is consistent with the usual definitions: a symplectic form is a nondegenerate alternating bilinear form and \(\mathrm{Sp}(2n,\mathbb R)\) preserves it; \(PG(2,2)\) is the seven-point/seven-line Fano plane, and its points can be identified with the seven nonidentity elements of \((\mathbb Z_2)^3\), with its lines corresponding to order-4 \(V_4\) subgroups. ([Armin Straub][1])

# RCFT SIM14 Vocabulary Packet

### SIM14.0–SIM14.3 — Symplectic \(C_8\) Carrier Program

## 1. \(X_8\) — Microscopic Eight-State Carrier

**Definition**

$$
\boxed{X_8=\{0,1,2,3,4,5,6,7\}}
$$

The finite microscopic state set used by the SIM14 laboratory.

**RCFT role**

\(X_8\) is the discrete carrier on which the frozen graph, projection operation, symplectic conjugation candidates, and permutation groups act.

It should **not** presently be called physical eight-dimensional spacetime or an octonionic basis. SIM14 only establishes an eight-state finite carrier.

**Status:** Frozen SIM architecture.

---

## 2. \(C_8\) — Eight-Cycle Graph

**Standard definition**

\(C_8\) is the cycle graph on eight vertices. Every vertex has degree two.

**Frozen SIM14 realization**

The cycle order is

$$
(0,1,3,5,7,6,4,2),
$$

giving edges

$$
\boxed{
E(C_8)=
\{
01,02,13,24,35,46,57,67
\}.
}
$$

This exact graph was already frozen before the symplectic carrier was introduced. 

**RCFT role**

The \(C_8\) graph supplies the existing ordinary adjacency structure. It acts as an **admissibility filter** in SIM14 rather than generating the orbit-bridge law itself.

That distinction became important in SIM14.3.

**Status:** Frozen carrier graph.

---

## 3. \(\Pi\) — Projection Partition

The frozen coarse-graining/projection partition is

$$
\boxed{
\Pi=
\{
\{0,1\},
\{2,3\},
\{4,5\},
\{6,7\}
\}.
}
$$

It organizes eight microscopic states into four two-state projection basins. 

**Important distinction**

\(\Pi\) is a partition of \(X_8\). It should not be conflated with the symplectic conjugate pairing \(s_\Omega\).

**Status:** Frozen RCFT toy-model structure.

---

## 4. \(p\) — Projection-Pairing Involution

Associated with \(\Pi\) is

$$
\boxed{
p=(01)(23)(45)(67).
}
$$

It swaps the two microscopic states within each projection pair.

Since

$$
p^2=e,
$$

\(p\) is an **involution**.

**RCFT role**

\(p\) encodes the binary pairing already supplied by projection.

**Status:** Frozen permutation.

---

## 5. \(h\) — \(C_8\) Half-Turn

The graph half-turn is

$$
\boxed{
h=(07)(16)(25)(34).
}
$$

It is an order-two automorphism of the frozen \(C_8\).

Thus

$$
h^2=e.
$$

**RCFT role**

Where \(p\) originates from projection structure, \(h\) originates from the intrinsic symmetry of the carrier graph.

This difference in origin matters even though both are mathematically involutions.

**Status:** Frozen graph symmetry.

---

# Symplectic vocabulary

## 6. \(\Omega\) — Symplectic Form

A symplectic form is a nondegenerate alternating bilinear form

$$
\Omega:V\times V\rightarrow\mathbb R.
$$

For the SIM14 carrier,

$$
V=\mathbb R^8
$$

with canonical coordinate order

$$
(q_1,q_2,q_3,q_4,p_1,p_2,p_3,p_4).
$$

Its matrix takes canonical form

$$
\boxed{
\Omega=
\begin{pmatrix}
0&I_4\\
-I_4&0
\end{pmatrix}.
}
$$

Consequently,

$$
\Omega^T=-\Omega.
$$

This is **antisymmetry**, not noncommutativity.

The standard symplectic group is precisely the collection of invertible linear transformations preserving this form. ([Columbia Mathematics][2])

**Status:** Standard mathematics instantiated in SIM14.

---

## 7. \(\mathrm{Sp}(8,\mathbb R)\) — Symplectic Group

$$
\boxed{
\mathrm{Sp}(8,\mathbb R)
=
\{M\in GL(8,\mathbb R):
M^T\Omega M=\Omega\}.
}
$$

It is the group of real linear transformations preserving the SIM14 canonical symplectic form.

Consequently,

$$
\Omega(Mu,Mv)=\Omega(u,v).
$$

SIM14.0 numerically verified this invariance under random symplectic transformations to numerical precision.

**RCFT terminology recommendation**

Call \(\mathrm{Sp}(8,\mathbb R)\) the **continuous symplectic carrier symmetry** when discussing the SIM architecture.

Do **not** identify the finite group \((\mathbb Z_2)^3\) discovered later with \(\mathrm{Sp}(8,\mathbb R)\). They occur at completely different mathematical layers.

**Status:** Standard group; RCFT uses it as a carrier constraint.

---

## 8. \(S\) — Signed Symplectic Pairing Matrix

Given carrier vectors \(v_i\),

$$
\boxed{
S_{ij}=v_i^T\Omega v_j.
}
$$

\(S\) records the symplectic relationship between every ordered pair of microscopic carrier vectors.

Because \(\Omega\) is antisymmetric,

$$
\boxed{S_{ij}=-S_{ji}.}
$$

Hence \(S\) contains **oriented pairing information**.

For a canonical conjugate pair,

$$
S_{ij}=+1,
\qquad
S_{ji}=-1.
$$

**Informational meaning**

\(S\) remembers:

* whether two states are symplectically paired,
* pairing magnitude,
* pairing orientation/sign.

**Status:** Derived carrier object.

---

## 9. \(|S|\) — Unsigned Symplectic Support

Taking

$$
\boxed{|S|_{ij}=|S_{ij}|}
$$

forgets orientation.

Thus

$$
+1,-1\longmapsto1.
$$

The information map is therefore

$$
\boxed{
S
\longrightarrow
|S|
}
$$

with a genuine information loss:

$$
\text{signed/oriented pairing}
\rightarrow
\text{unsigned conjugacy relation}.
$$

It retains **who is paired with whom** but loses **which orientation the symplectic form assigns to the ordered pair**.

This is not renormalization and is not a failure of commutativity. It is an explicit coarse-graining of an antisymmetric relation.

**Status:** Defined information-reduction operation.

---

## 10. Symplectic Conjugate Pair

For the canonical basis, the nonzero support of \(\Omega\) produces four pairs.

Abstractly we write

$$
\boxed{
s_\Omega=(a_1b_1)(a_2b_2)(a_3b_3)(a_4b_4).
}
$$

Here **conjugate** means paired by the nonzero canonical symplectic form.

It does **not** currently mean:

* complex conjugation,
* antiparticle conjugation,
* Dirac conjugation,
* spin-up/spin-down,
* quantum superposition.

Those possible physical interpretations have not been established by SIM14.

**Status:** RCFT/SIM terminology built from standard symplectic pairing.

---

## 11. \(s\) or \(s_\Omega\) — Symplectic Conjugation Involution

Once the four unsigned conjugate pairs are regarded as a permutation of \(X_8\),

$$
\boxed{s_\Omega^2=e}
$$

and \(s_\Omega\) has no fixed points.

SIM14.0 discovered two \(C_8\)-contained possibilities:

$$
M_1=\{01,24,35,67\}
$$

and

$$
M_2=\{02,13,46,57\}.
$$

In permutation form,

$$
s_1=(01)(24)(35)(67),
$$

$$
s_2=(02)(13)(46)(57).
$$

SIM14.3 preserves these explicitly. 

**Status:** Candidate discrete realization of symplectic conjugacy.

---

# Finite-group vocabulary

## 12. Involution

An **involution** is a group element satisfying

$$
\boxed{g^2=e}.
$$

Equivalently, it has order two.

The SIM14 objects

$$
p,\qquad h,\qquad s
$$

are involutions.

A fixed-point-free involution on eight states consists of exactly four disjoint transpositions.

**Status:** Standard mathematics.

---

## 13. Commutation

Two operations \(a,b\) commute when

$$
\boxed{ab=ba.}
$$

Equivalently their commutator is the identity:

$$
[a,b]=aba^{-1}b^{-1}=e.
$$

SIM14 examines candidate \(s\) satisfying

$$
[s,p]=[s,h]=e.
$$

Again, this is separate from the antisymmetry

$$
S_{ij}=-S_{ji}.
$$

That vocabulary distinction is worth freezing permanently:

$$
\boxed{
\text{antisymmetry of }\Omega
\neq
\text{noncommutativity of transformations}.
}
$$

---

## 14. \(K=\langle p,h\rangle\) — Pre-Symplectic Base Group

SIM14.3 gives a useful name to the structure existing **before \(s\) is added**:

$$
\boxed{K=\langle p,h\rangle.}
$$

The run establishes

$$
|K|=4,
$$

all three nonidentity elements have order two, and

$$
\boxed{K\cong V_4.}
$$



**Recommended RCFT term**

**Pre-symplectic base group** is good terminology *within the SIM14 laboratory*: not because \(V_4\) itself is intrinsically “pre-symplectic,” but because it is the finite permutation structure generated before the candidate symplectic conjugation \(s\) is introduced.

---

## 15. \(V_4\) — Klein Four Group

The Klein four group is

$$
\boxed{
V_4\cong
\mathbb Z_2\times\mathbb Z_2.
}
$$

It contains

$$
\{e,a,b,ab\},
$$

with

$$
a^2=b^2=(ab)^2=e
$$

and all elements commuting.

In SIM14,

$$
K=\langle p,h\rangle\cong V_4.
$$

**Status:** Standard finite group, discovered as the structure of the frozen \(p,h\) action.

---

## 16. \(K\)-Orbit

For \(x\in X_8\),

$$
\boxed{
Kx=\{k(x):k\in K\}
}
$$

is the orbit of \(x\) under \(K\).

SIM14.3 derives exactly two:

$$
\boxed{
O_0=\{0,1,6,7\},
\qquad
O_1=\{2,3,4,5\}.
}
$$

Each contains four states. 

This gives the important pre-existing decomposition

$$
\boxed{X_8=O_0\sqcup O_1.}
$$

---

## 17. Stabilizer

For a group \(G\) acting on \(X\), the stabilizer of \(x\) is

$$
\boxed{
\operatorname{Stab}_G(x)
=
\{g\in G:g(x)=x\}.
}
$$

Orbit-stabilizer gives

$$
|G|
=
|\operatorname{Orb}_G(x)|
\,|\operatorname{Stab}_G(x)|.
$$

For the \(K\)-action,

$$
4=4\times1.
$$

Thus \(K\) acts freely on each individual four-state orbit.

**Status:** Standard group-action concept.

---

## 18. Free Action

A group action is **free** if no nonidentity group element fixes a point:

$$
g(x)=x
\Rightarrow
g=e.
$$

Equivalently,

$$
\operatorname{Stab}(x)=\{e\}
$$

for every \(x\).

---

## 19. Transitive Action

An action is **transitive** if any state can be moved to any other state by some group element.

Equivalently, \(X\) consists of one group orbit.

---

## 20. Regular Action

An action is **regular** when it is simultaneously

$$
\boxed{\text{free}+\text{transitive}.}
$$

For a finite group acting on an equally sized finite set,

$$
|G|=|X|,
$$

transitivity already forces trivial stabilizers and hence regularity.

For M2, SIM14.3 finds

$$
|\langle K,s_2\rangle|=8,
\qquad
|\operatorname{Orb}(x)|=8,
\qquad
|\operatorname{Stab}(x)|=1.
$$

Thus its action on \(X_8\) is regular. M1 instead retains two four-state orbits. 

---

## 21. Torsor

A **\(G\)-torsor** is a set on which \(G\) acts freely and transitively.

It resembles the group itself but has no distinguished identity element until one chooses an origin.

Therefore, when

$$
G\cong(\mathbb Z_2)^3
$$

acts regularly on \(X_8\),

$$
\boxed{X_8\text{ becomes an affine }(\mathbb Z_2)^3\text{-torsor}.}
$$

Choosing one microscopic state as \(000\) then lets the others receive binary coordinates.

This occurs for the M2 branch but not globally for M1.

**Status:** Standard mathematics applied to the SIM14 result.

---

# The binary/Fano layer

## 22. \((\mathbb Z_2)^3\) / \(\mathbb F_2^3\)

The elementary abelian group

$$
\boxed{
(\mathbb Z_2)^3
}
$$

contains eight elements.

As an additive group it can be identified with

$$
\boxed{\mathbb F_2^3}.
$$

Its elements can be represented as

$$
000,001,010,011,100,101,110,111.
$$

Every nonzero element has order two.

Thus there are exactly

$$
\boxed{7}
$$

nonidentity binary directions.

**SIM14 result**

For appropriate \(s\notin K\),

$$
\langle p,h,s\rangle
\cong
(\mathbb Z_2)^3.
$$

But SIM14.2/14.3 showed that **abstract group isomorphism alone does not imply a regular action on \(X_8\)**.

That distinction is now fundamental.

---

## 23. Binary Direction

Once a regular \(\mathbb F_2^3\)-torsor structure exists, each nonzero

$$
d\in\mathbb F_2^3
$$

defines a translation

$$
\boxed{x\mapsto x+d.}
$$

SIM14 terminology calls the seven nonzero \(d\)'s the **seven binary directions**.

This is an RCFT/SIM convenience term; mathematically they are simply the seven nonzero vectors/elements of \(\mathbb F_2^3\).

---

## 24. \(PG(2,2)\) — Fano Plane

The finite projective plane over \(\mathbb F_2\) is

$$
\boxed{PG(2,2)}.
$$

It contains

$$
7\text{ points},\qquad7\text{ lines},
$$

with three points per line and three lines through each point. ([Wikipedia][3])

For the SIM14 group-theoretic construction:

$$
\boxed{
\text{points}
\leftrightarrow
7\text{ nonidentity elements of }(\mathbb Z_2)^3
}
$$

and

$$
\boxed{
\text{lines}
\leftrightarrow
7\text{ order-4 }V_4\text{ subgroups}.
}
$$

This group-theoretic realization is standard mathematics, not a new RCFT construction. ([Wikipedia][3])

**Critical SIM14 distinction**

SIM14.2 taught us to distinguish:

$$
\boxed{\text{abstract }PG(2,2)}
$$

from

$$
\boxed{\text{a Fano/torsor organization realized on }X_8.}
$$

M1 can retain the abstract group incidence without making the eight microscopic states one regular \(\mathbb F_2^3\)-torsor.

M2 does both.

---

## 25. Fano Line

Within

$$
PG(2,2),
$$

a line consists of three nonzero vectors

$$
\boxed{\{a,b,a+b\}.}
$$

Equivalently, adjoining zero gives

$$
\{0,a,b,a+b\},
$$

which is a subgroup

$$
\cong V_4.
$$

Hence SIM14's seven \(V_4\) subgroups reproduce the seven Fano lines at the **abstract finite-group level**.

That does **not yet establish an octonionic Fano plane physically inside RCFT**.

---

# SIM14.3-specific vocabulary

## 26. \(S_K\) — Commuting Fixed-Point-Free Candidate Class

SIM14.3 defines

$$
\boxed{
S_K=
\{
s:
s^2=e,\;
\operatorname{Fix}(s)=\varnothing,\;
[s,p]=[s,h]=e
\}.
}
$$

So \(S_K\) is the complete tested class of four-pair involutions compatible with the pre-existing \(K\)-action through commutation.

Exhaustive enumeration gives

$$
\boxed{|S_K|=13.}
$$



**Status:** SIM14-defined candidate space.

---

## 27. \(B(s)\) — Orbit-Transition Matrix

SIM14.3 introduces

$$
\boxed{
B_{ab}(s)
=
\#\{x\in O_a:s(x)\in O_b\}.
}
$$

It records how \(s\) moves states relative to the two pre-existing \(K\)-orbits.

Two types occur:

$$
B_{\rm internal}
=
\begin{pmatrix}
4&0\\
0&4
\end{pmatrix}
$$

and

$$
B_{\rm bridge}
=
\begin{pmatrix}
0&4\\
4&0
\end{pmatrix}.
$$

M1 has the first; M2 has the second. 

**Recommended RCFT term:** **orbit-transition matrix**.

---

## 28. Orbit Preservation / Internal Pairing

If

$$
s(O_0)=O_0,
\qquad
s(O_1)=O_1,
$$

then \(s\) is **orbit-preserving** relative to \(K\).

Its transition matrix is

$$
B(s)=
\begin{pmatrix}
4&0\\
0&4
\end{pmatrix}.
$$

M1 belongs to this class.

Such candidates do not make \(\langle K,s\rangle\) transitive across \(X_8\).

---

## 29. Complete Orbit Exchange / Orbit Bridge

If

$$
\boxed{
s(O_0)=O_1,
\qquad
s(O_1)=O_0,
}
$$

then \(s\) is a **complete orbit bridge**.

Its transition matrix is

$$
B(s)=
\begin{pmatrix}
0&4\\
4&0
\end{pmatrix}.
$$

M2 is such a bridge. 

This gives us one of the most important new RCFT/SIM terms:

> **Orbit bridge:** a candidate conjugation that exchanges the two pre-existing \(K\)-orbits completely.

---

## 30. Orbit-Bridge Law

SIM14.3's computational result is:

$$
\boxed{
\text{complete }K\text{-orbit exchange}
\iff
\langle K,s\rangle
\text{ acts regularly on }X_8
}
$$

for every

$$
s\in S_K.
$$

All four complete-exchange candidates were regular; all nine non-exchange candidates were nonregular, with no counterexamples in the exhaustive tested class. 

**Recommended formal name**

$$
\boxed{\textbf{\(V_4\) Orbit-Bridge Regularity Criterion}}
$$

until we write and verify the analytic theorem.

**Status:** Exhaustively established computationally within \(S_K\); analytic proof is the next step.

---

## 31. \(G_s=\langle K,s\rangle\)

Define

$$
\boxed{
G_s=\langle K,s\rangle
=
\langle p,h,s\rangle.
}
$$

This is the finite permutation group obtained after adding a candidate conjugation \(s\) to the pre-symplectic base group.

For \(s\notin K\) in the tested class,

$$
|G_s|=8
$$

and the elements have order at most two. 

The crucial lesson is:

$$
\boxed{
\text{abstract structure of }G_s
\neq
\text{action type of }G_s\curvearrowright X_8.
}
$$

That distinction explains M1 versus M2.

---

## 32. C8-Contained Matching

A candidate symplectic matching is **C8-contained** when all four of its unordered conjugate pairs are edges of the frozen \(C_8\).

Within \(S_K\), only

$$
M_1,\quad M_2
$$

satisfy this.

Of these, exactly one—M2—is a complete bridge and hence regular. 

Thus SIM14.3 gives the precise separation:

$$
\boxed{
K+s\quad\text{controls the orbit-bridge/regularity relation}
}
$$

while

$$
\boxed{
C_8\quad\text{acts as an additional admissibility restriction}.
}
$$

---

# Information hierarchy

I think this is the cleanest vocabulary diagram for everything we've learned:

$$
\boxed{
\begin{array}{c}
\mathbb R^8,\Omega\\
\downarrow\\
S_{ij}=v_i^T\Omega v_j\\
\downarrow\;|\cdot|\\
|S|:\text{ unsigned conjugate support}\\
\downarrow\\
s_\Omega:\text{ four conjugate 2-cycles}\\[1mm]
\hline\\[-3mm]
X_8,\ C_8,\ \Pi\\
\downarrow\\
p,\ h\\
\downarrow\\
K=\langle p,h\rangle\cong V_4\\
\downarrow\\
X_8=O_0\sqcup O_1\\
\downarrow\quad +s_\Omega\\
G_s=\langle p,h,s_\Omega\rangle\\
\downarrow\\
\text{internal or orbit-bridging action}\\
\downarrow\\
\text{nonregular or regular action}\\
\downarrow\\
\text{if regular: }X_8\text{ is an }\mathbb F_2^3\text{-torsor}\\
\downarrow\\
7\text{ nonzero directions}\\
\downarrow\\
7\ V_4\text{ subgroups}\\
\downarrow\\
PG(2,2)\text{ incidence}
\end{array}
}
$$

The important thing is that **not every arrow has the same epistemic status**. Some are definitions, some are standard theorems, some are computational SIM results, and some require the regularity condition.

---

# Terms I would explicitly reserve for later

For the moment I would **not** put any of the following into the formal SIM14 vocabulary as synonyms for what we've found:

**octonionic multiplication, physical Fano plane, spinor, fermion, quantum superposition, nonassociative dynamics, \(H_4\), \(E_8/RE_8\), \(4_{21}\), spacetime, or physical conjugate trajectory.**

Those remain hypotheses or targets elsewhere in RCFT. SIM14.3 itself explicitly imposes essentially this interpretive ceiling: finite carrier mathematics only, with no downstream physical identification licensed. 

What we **can** now say rigorously is already plenty:

$$
\boxed{
\begin{gathered}
\text{canonical symplectic pairing}\\
+\;\text{frozen }C_8\text{ carrier}\\
+\;\text{projection involution }p\\
+\;\text{graph half-turn }h\\
\\
\Longrightarrow\\
\\
K=\langle p,h\rangle\cong V_4,\qquad
X_8=4+4,
\end{gathered}}
$$

and within the complete commuting fixed-point-free class,

$$
\boxed{
s\text{ bridges the }4+4\text{ decomposition}
\iff
\langle p,h,s\rangle
\text{ acts regularly on all eight states}.
}
$$

Then, **for that regular realization**, the eight-state carrier acquires the affine \(\mathbb F_2^3\) torsor structure whose seven nonzero directions carry the familiar \(PG(2,2)\) incidence.

That is probably the cleanest vocabulary boundary we've had anywhere in the SIM program so far.

[1]: https://arminstraub.com/downloads/pub/diploma.pdf?utm_source=chatgpt.com "1 Introduction                                                                 11"
[2]: https://www.math.columbia.edu/~woit/QM/qmbook.pdf?utm_source=chatgpt.com "16.1 The symplectic group"
[3]: https://en.wikipedia.org/wiki/Fano_plane?utm_source=chatgpt.com "Fano plane"
