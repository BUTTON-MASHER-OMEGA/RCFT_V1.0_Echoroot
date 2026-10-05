
SIM14.8 — AFFINE–PARTNER STABILIZER EXTENSION MAP
=================================================

Purpose
-------
Explain the internal algebraic origin of the already-established

    H192 := Aut_aff(|Omega|)
          = Aut_aff(s_Omega),

with |H192| = 192.

This SIM asks:

1. How does H192 act by conjugation on the translation group

       G ~= C2^3 ?

2. Is the conjugation image exactly

       Stab_GL(3,2)(001) ?

3. What is the common fixed subspace of that action?

4. Does the affine zero-translation section give a constructive
   splitting

       H192 ~= G semidirect Q0 ?

5. How does that section depend on the chosen affine origin?

Interpretive ceiling
--------------------
Finite permutation / affine / partner-structure mathematics only.

No dynamics.
No LCO / ISP.
No J.
No metric.
No physical interpretation.
No D4.
No Weyl-group identification.
No 24-cell.
No F4.
No E8.

Important epistemic distinction
-------------------------------
The following are FROZEN SIM14.6 consequences / regression checks:

    |G| = 8
    |H192| = 192
    G normal in H192
    C_H192(G) = G

Therefore, for the conjugation action

    Psi : H192 -> Aut(G),

we already know mathematically that

    ker(Psi) = G
    |im(Psi)| = 24.

A failure of those checks indicates an implementation inconsistency,
not a surprising new group-theoretic result.

The genuinely new SIM14.8 targets are:

    im(Psi) ?= Stab_GL(3,2)(001)

    Fix_G(im(Psi)) ?= <001>

    im(Psi) ?~= S4

    zero-translation section Q0 ?
    H192 ?= G semidirect Q0

and the dependence of the affine section on origin choice.
"""

import itertools
from collections import Counter

import numpy as np


# ============================================================
# 0. PERMUTATION UTILITIES
# ============================================================

NPTS = 8
ID = tuple(range(NPTS))


def compose(a, b):
    """a o b: apply b first, then a."""
    return tuple(
        a[b[i]]
        for i in range(NPTS)
    )


def inv(a):

    q = [0] * NPTS

    for i, j in enumerate(a):
        q[j] = i

    return tuple(q)


def order(a):

    x = ID

    for n in range(1, 1000):

        x = compose(a, x)

        if x == ID:
            return n

    raise RuntimeError(
        "order bound exceeded"
    )


def cycle_string(a):

    seen = set()
    out = []

    for i in range(NPTS):

        if (
            i not in seen
            and
            a[i] != i
        ):

            cyc = []
            j = i

            while j not in seen:

                seen.add(j)
                cyc.append(j)
                j = a[j]

            out.append(
                "("
                +
                " ".join(
                    map(str, cyc)
                )
                +
                ")"
            )

    return "".join(out) if out else "()"


def generated_group(gens):

    S = {ID}
    Q = [ID]

    gens = list(gens)

    while Q:

        x = Q.pop()

        for g in gens:

            for y in (
                compose(g, x),
                compose(x, g)
            ):

                if y not in S:

                    S.add(y)
                    Q.append(y)

    return frozenset(S)


def commutes(a, b):

    return (
        compose(a, b)
        ==
        compose(b, a)
    )


def conjugate(g, x):

    return compose(
        compose(g, x),
        inv(g)
    )


def intersection(A, B):

    return frozenset(
        set(A) & set(B)
    )


def is_normal(A, H):

    A = set(A)

    return all(

        {
            conjugate(g, a)
            for a in A
        } == A

        for g in H
    )


def centralizer(H, A):

    return frozenset(

        g

        for g in H

        if all(
            commutes(g, a)
            for a in A
        )
    )


def element_order_hist(H):

    return dict(
        sorted(
            Counter(
                order(g)
                for g in H
            ).items()
        )
    )


def perm_from_cycles(cycles):

    q = list(range(NPTS))

    for cyc in cycles:

        for a, b in zip(
            cyc,
            cyc[1:] + cyc[:1]
        ):

            q[a] = b

    return tuple(q)


# ============================================================
# 1. FROZEN SIM14 p,h,s / G
# ============================================================

p = perm_from_cycles(
    [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7)
    ]
)


h = perm_from_cycles(
    [
        (0, 7),
        (1, 6),
        (2, 5),
        (3, 4)
    ]
)


s = perm_from_cycles(
    [
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7)
    ]
)


G = generated_group(
    [p, h, s]
)


# ============================================================
# 2. FROZEN AFFINE COORDINATES
#
# Basis:
#
#     p = 100
#     h = 010
#     s = 001
# ============================================================

coord_of = {}
point_of = {}


for bits in range(8):

    g = ID

    if bits & 4:
        g = compose(p, g)

    if bits & 2:
        g = compose(h, g)

    if bits & 1:
        g = compose(s, g)

    x = g[0]

    coord_of[x] = bits
    point_of[bits] = x


assert len(coord_of) == 8


def bits3(x):

    return format(
        int(x),
        "03b"
    )


def bits_to_vec(x):

    return np.array(
        [
            (x >> 2) & 1,
            (x >> 1) & 1,
            x & 1
        ],
        dtype=int
    )


def vec_to_bits(v):

    return (
        (int(v[0]) << 2)
        |
        (int(v[1]) << 1)
        |
        int(v[2])
    )


def xor_point(x, d):

    return point_of[
        coord_of[x] ^ d
    ]


def translation_perm(d):

    return tuple(

        xor_point(x, d)

        for x in range(NPTS)
    )


translations = {

    d:
        translation_perm(d)

    for d in range(8)
}


translation_label = {

    t:
        d

    for d, t in translations.items()
}


assert (
    frozenset(
        translations.values()
    )
    ==
    G
)


# ============================================================
# 3. FROZEN SIGNED OMEGA
# ============================================================

Omega = np.zeros(
    (8, 8),
    dtype=int
)


for a, b in [

    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7)

]:

    Omega[a, b] = 1
    Omega[b, a] = -1


ABS_OMEGA = np.abs(
    Omega
)


def Pmat(g):

    P = np.zeros(
        (8, 8),
        dtype=int
    )

    for i, j in enumerate(g):

        P[j, i] = 1

    return P


def preserves_abs_omega(g):

    P = Pmat(g)

    A = (
        P.T
        @ ABS_OMEGA
        @ P
    )

    return np.array_equal(
        A,
        ABS_OMEGA
    )


# ============================================================
# 4. GL(3,2) AND AGL(3,2)
# ============================================================

def rank_mod2(A):

    A = (
        A.copy()
        %
        2
    ).astype(int)

    rows, cols = A.shape

    r = 0

    for c in range(cols):

        pivot = next(

            (
                i
                for i in range(r, rows)
                if A[i, c]
            ),

            None
        )

        if pivot is None:
            continue

        A[[r, pivot]] = A[[pivot, r]]

        for i in range(rows):

            if (
                i != r
                and
                A[i, c]
            ):

                A[i] ^= A[r]

        r += 1

    return r


GL32 = []


for entries in itertools.product(
    (0, 1),
    repeat=9
):

    A = np.array(
        entries,
        dtype=int
    ).reshape(
        3,
        3
    )

    if rank_mod2(A) == 3:

        GL32.append(A)


assert len(GL32) == 168


def matrix_key(A):

    return tuple(
        int(x)
        for x in A.reshape(-1)
    )


def key_matrix(k):

    return np.array(
        k,
        dtype=int
    ).reshape(
        3,
        3
    )


I3 = np.eye(
    3,
    dtype=int
)


I3_KEY = matrix_key(
    I3
)


def matmul_key(Ak, Bk):

    A = key_matrix(Ak)
    B = key_matrix(Bk)

    return matrix_key(
        (A @ B) % 2
    )


def matrix_order(Ak):

    x = I3_KEY

    for n in range(1, 100):

        x = matmul_key(
            Ak,
            x
        )

        if x == I3_KEY:
            return n

    raise RuntimeError(
        "matrix order bound exceeded"
    )


def apply_matrix_bits(Ak, x):

    A = key_matrix(Ak)

    y = (
        A
        @ bits_to_vec(x)
    ) % 2

    return vec_to_bits(y)


def matrix_string(Ak):

    A = key_matrix(Ak)

    rows = [

        "".join(
            str(int(x))
            for x in row
        )

        for row in A
    ]

    return "[" + ";".join(rows) + "]"


def affine_perm(A, b):

    out = []

    for x in range(NPTS):

        vx = bits_to_vec(
            coord_of[x]
        )

        vy = (
            A @ vx
            +
            b
        ) % 2

        ybits = vec_to_bits(
            vy
        )

        out.append(
            point_of[ybits]
        )

    return tuple(out)


# Store the unique affine decomposition
#
#     g <-> (A,b)
#
# for every affine permutation.

AFFINE_DATA = {}


for A in GL32:

    Ak = matrix_key(A)

    for bbits in range(8):

        b = bits_to_vec(
            bbits
        )

        g = affine_perm(
            A,
            b
        )

        AFFINE_DATA[g] = (
            Ak,
            bbits
        )


AUT_AFF = frozenset(
    AFFINE_DATA.keys()
)


assert len(AUT_AFF) == 1344
assert len(AFFINE_DATA) == 1344


# ============================================================
# 5. RECONSTRUCT H192
#
# H192 = Aut_aff(|Omega|)
# ============================================================

H192 = frozenset(

    g

    for g in AUT_AFF

    if preserves_abs_omega(g)
)


# ============================================================
# 6. FROZEN H192 REGRESSION OBJECTS
# ============================================================

C_H_G = centralizer(
    H192,
    G
)


G_NORMAL_IN_H = is_normal(
    G,
    H192
)


# ============================================================
# 7. CONJUGATION ACTION
#
# Psi : H192 -> Aut(G) ~= GL(3,2)
#
# Basis order:
#
#     e1 = p = 100
#     e2 = h = 010
#     e3 = s = 001
# ============================================================

BASIS_BITS = (
    4,  # 100 = p
    2,  # 010 = h
    1   # 001 = s
)


BASIS_TRANSLATIONS = tuple(

    translations[d]

    for d in BASIS_BITS
)


def psi_matrix(g):

    cols = []

    for t in BASIS_TRANSLATIONS:

        tp = conjugate(
            g,
            t
        )

        if tp not in translation_label:

            raise RuntimeError(
                "Conjugation left translation group G"
            )

        d = translation_label[tp]

        cols.append(
            bits_to_vec(d)
        )

    A = np.column_stack(
        cols
    ) % 2

    return matrix_key(A)


PSI_BY_ELEMENT = {

    g:
        psi_matrix(g)

    for g in H192
}


PSI_IMAGE = frozenset(
    PSI_BY_ELEMENT.values()
)


PSI_KERNEL = frozenset(

    g

    for g in H192

    if PSI_BY_ELEMENT[g] == I3_KEY
)


# ============================================================
# 8. MATRIX-ACTION UTILITIES
# ============================================================

D7 = tuple(
    range(1, 8)
)


def direction_perm(Ak):

    images = [

        apply_matrix_bits(
            Ak,
            d
        )

        for d in D7
    ]

    return tuple(
        D7.index(y)
        for y in images
    )


def direction_orbits(matrix_group):

    unseen = set(D7)
    orbits = []

    while unseen:

        seed = min(unseen)

        orb = {

            apply_matrix_bits(
                A,
                seed
            )

            for A in matrix_group
        }

        # Close defensively under repeated action.
        changed = True

        while changed:

            changed = False
            new = set(orb)

            for A in matrix_group:

                for x in orb:

                    new.add(
                        apply_matrix_bits(
                            A,
                            x
                        )
                    )

            if new != orb:

                orb = new
                changed = True

        orbits.append(
            tuple(sorted(orb))
        )

        unseen -= orb

    return tuple(
        sorted(
            orbits,
            key=lambda O:
                (len(O), O)
        )
    )


# ============================================================
# 9. INDEPENDENT STABILIZER OF 001 IN GL(3,2)
# ============================================================

S_OMEGA_BITS = 1  # 001


STAB_001 = frozenset(

    matrix_key(A)

    for A in GL32

    if apply_matrix_bits(
        matrix_key(A),
        S_OMEGA_BITS
    )
    ==
    S_OMEGA_BITS
)


PSI_SUBSET_STAB = (
    PSI_IMAGE.issubset(
        STAB_001
    )
)


PSI_EQUALS_STAB = (
    PSI_IMAGE
    ==
    STAB_001
)


# ============================================================
# 10. BLIND CLASSIFICATION OF Q_ACT = im(Psi)
# ============================================================

Q_ACT = PSI_IMAGE


Q_ACT_ORDER_HIST = dict(
    sorted(
        Counter(
            matrix_order(A)
            for A in Q_ACT
        ).items()
    )
)


def matrix_commutes(A, B):

    return (
        matmul_key(A, B)
        ==
        matmul_key(B, A)
    )


Q_ACT_CENTER = frozenset(

    A

    for A in Q_ACT

    if all(
        matrix_commutes(A, B)
        for B in Q_ACT
    )
)


def matrix_inv_key(Ak):

    A = key_matrix(Ak)

    # Brute force inside GL(3,2), tiny and exact.
    for B in GL32:

        Bk = matrix_key(B)

        if (
            matmul_key(Ak, Bk) == I3_KEY
            and
            matmul_key(Bk, Ak) == I3_KEY
        ):

            return Bk

    raise RuntimeError(
        "matrix inverse not found"
    )


def matrix_commutator(A, B):

    return matmul_key(
        matmul_key(
            matmul_key(
                A,
                B
            ),
            matrix_inv_key(A)
        ),
        matrix_inv_key(B)
    )


def generated_matrix_group(gens):

    S = {I3_KEY}
    Q = [I3_KEY]

    gens = list(gens)

    while Q:

        x = Q.pop()

        for g in gens:

            for y in (
                matmul_key(g, x),
                matmul_key(x, g)
            ):

                if y not in S:

                    S.add(y)
                    Q.append(y)

    return frozenset(S)


Q_ACT_DERIVED = generated_matrix_group(

    matrix_commutator(A, B)

    for A in Q_ACT
    for B in Q_ACT
)


# ============================================================
# 11. CONSTRUCTIVE 4-OBJECT ACTION
#
# To avoid identifying S4 from a histogram alone:
#
# There are seven 2D linear subspaces of F2^3.
# Exactly four do NOT contain the fixed direction 001.
#
# If Q_ACT acts faithfully on those four objects with image
# of order 24, this gives a concrete realization as S4.
# ============================================================

def span2(a, b):

    return frozenset(
        [
            0,
            a,
            b,
            a ^ b
        ]
    )


ALL_PLANES = set()


for a in range(1, 8):

    for b in range(
        a + 1,
        8
    ):

        if a != b:

            P = span2(
                a,
                b
            )

            if len(P) == 4:

                ALL_PLANES.add(P)


ALL_PLANES = tuple(
    sorted(
        ALL_PLANES,
        key=lambda P:
            tuple(sorted(P))
    )
)


PLANES_NOT_CONTAINING_001 = tuple(

    P

    for P in ALL_PLANES

    if S_OMEGA_BITS not in P
)


PLANE4_INDEX = {

    P:
        i

    for i, P in enumerate(
        PLANES_NOT_CONTAINING_001
    )
}


def act_on_plane(Ak, P):

    return frozenset(

        apply_matrix_bits(
            Ak,
            x
        )

        for x in P
    )


def plane4_perm(Ak):

    out = []

    for P in PLANES_NOT_CONTAINING_001:

        AP = act_on_plane(
            Ak,
            P
        )

        if AP not in PLANE4_INDEX:

            raise RuntimeError(
                "Q_ACT did not preserve the 4-plane family"
            )

        out.append(
            PLANE4_INDEX[AP]
        )

    return tuple(out)


Q_ACT_ON_4 = frozenset(

    plane4_perm(A)

    for A in Q_ACT
)


ID4 = (
    0,
    1,
    2,
    3
)


Q_ACT_ON_4_KERNEL = frozenset(

    A

    for A in Q_ACT

    if plane4_perm(A) == ID4
)


Q_ACT_S4_CONSTRUCTIVE = (

    len(PLANES_NOT_CONTAINING_001) == 4

    and

    len(Q_ACT_ON_4) == 24

    and

    len(Q_ACT_ON_4_KERNEL) == 1
)


# ============================================================
# 12. COMMON FIXED SUBSPACE
# ============================================================

FIXED_G_BITS = tuple(

    d

    for d in range(8)

    if all(

        apply_matrix_bits(
            A,
            d
        )
        ==
        d

        for A in Q_ACT
    )
)


FIXED_G_EXPECTED = (
    0,
    S_OMEGA_BITS
)


# ============================================================
# 13. ZERO-TRANSLATION SECTION
#
# Q0 = elements of H192 whose affine translation part b is zero.
# ============================================================

Q0 = frozenset(

    g

    for g in H192

    if AFFINE_DATA[g][1] == 0
)


Q0_CAP_G = intersection(
    Q0,
    G
)


GQ0 = frozenset(

    compose(t, q)

    for t in G
    for q in Q0
)


Q0_PSI_IMAGE = frozenset(

    PSI_BY_ELEMENT[q]

    for q in Q0
)


Q0_SECTION_CRITERION = (

    len(Q0) == len(Q_ACT)

    and

    len(Q0_CAP_G) == 1

    and

    GQ0 == H192

    and

    Q0_PSI_IMAGE == Q_ACT
)


# ============================================================
# 14. GENERATORS FOR Q0 AND THEIR ACTUAL ACTION
#
# Search for a small generating pair if possible.
# ============================================================

Q0_GENERATORS = None


if len(Q0) > 0:

    qlist = sorted(
        Q0,
        key=cycle_string
    )

    for a in qlist:

        for b in qlist:

            if generated_group(
                [a, b]
            ) == Q0:

                Q0_GENERATORS = (
                    a,
                    b
                )

                break

        if Q0_GENERATORS is not None:
            break


# ============================================================
# 15. ORIGIN-CHANGE AUDIT
#
# For each affine origin a in G, conjugate Q0 by translation t_a.
#
# This asks how the zero-translation section changes when the
# affine origin is moved.
# ============================================================

ORIGIN_SECTIONS = {}


for a in range(8):

    ta = translations[a]

    Qa = frozenset(

        conjugate(
            ta,
            q
        )

        for q in Q0
    )

    ORIGIN_SECTIONS[a] = Qa


UNIQUE_ORIGIN_SECTIONS = []


for a in range(8):

    Qa = ORIGIN_SECTIONS[a]

    if Qa not in UNIQUE_ORIGIN_SECTIONS:

        UNIQUE_ORIGIN_SECTIONS.append(
            Qa
        )


ORIGIN_SECTION_CLASSES = {}


for i, Qx in enumerate(
    UNIQUE_ORIGIN_SECTIONS
):

    origins = [

        a

        for a, Qa
        in ORIGIN_SECTIONS.items()

        if Qa == Qx
    ]

    ORIGIN_SECTION_CLASSES[i] = origins


ORIGIN_SECTIONS_ALL_COMPLEMENTS = all(

    len(Qa) == 24

    and

    len(
        intersection(
            Qa,
            G
        )
    ) == 1

    and

    frozenset(

        compose(t, q)

        for t in G
        for q in Qa
    )
    == H192

    for Qa in ORIGIN_SECTIONS.values()
)


# ============================================================
# 16. CENTER CHECK
# ============================================================

Z_H192 = frozenset(

    g

    for g in H192

    if all(
        commutes(g, x)
        for x in H192
    )
)


CENTER_CAP_G = intersection(
    Z_H192,
    G
)


FIXED_TRANSLATIONS = frozenset(

    translations[d]

    for d in FIXED_G_BITS
)


# ============================================================
# 17. OUTPUT
# ============================================================

print(
    "=" * 86
)

print(
    "SIM14.8 — AFFINE–PARTNER STABILIZER EXTENSION MAP"
)

print(
    "=" * 86
)


# ------------------------------------------------------------
# A
# ------------------------------------------------------------

print(
    "\nA. FROZEN / REGRESSION SANITY"
)


print(
    "|G|",
    len(G)
)


print(
    "|GL(3,2)|",
    len(GL32)
)


print(
    "|AGL(3,2)|",
    len(AUT_AFF)
)


print(
    "|H192|",
    len(H192)
)


print(
    "G normal in H192",
    G_NORMAL_IN_H
)


print(
    "|C_H192(G)|",
    len(C_H_G)
)


print(
    "C_H192(G) == G",
    C_H_G == G
)


print(
    "coordinates",
    {
        x:
            bits3(
                coord_of[x]
            )
        for x in range(8)
    }
)


print(
    "Omega antisymmetric",
    np.array_equal(
        Omega.T,
        -Omega
    ),
    "rank",
    np.linalg.matrix_rank(
        Omega
    ),
    "det",
    round(
        np.linalg.det(
            Omega
        )
    )
)


# ------------------------------------------------------------
# B
# ------------------------------------------------------------

print(
    "\nB. CONJUGATION ACTION Psi : H192 -> GL(3,2)"
)


print(
    "|im Psi|",
    len(PSI_IMAGE)
)


print(
    "|ker Psi|",
    len(PSI_KERNEL)
)


print(
    "ker Psi == G",
    PSI_KERNEL == G
)


print(
    "quotient order |H192|/|G|",
    len(H192) // len(G)
)


print(
    "distinct Psi matrices:"
)


for Ak in sorted(
    PSI_IMAGE
):

    print(
        " ",
        matrix_string(Ak),
        "order",
        matrix_order(Ak),
        "D7 action",
        tuple(
            bits3(
                apply_matrix_bits(
                    Ak,
                    d
                )
            )
            for d in D7
        )
    )


# ------------------------------------------------------------
# C
# ------------------------------------------------------------

print(
    "\nC. STABILIZER TEST FOR 001 = s_Omega"
)


print(
    "|Stab_GL(001)|",
    len(STAB_001)
)


print(
    "im Psi subset Stab_GL(001)",
    PSI_SUBSET_STAB
)


print(
    "im Psi == Stab_GL(001)",
    PSI_EQUALS_STAB
)


print(
    "[GL(3,2):Stab_GL(001)]",
    len(GL32) // len(STAB_001)
)


print(
    "[AGL(3,2):H192]",
    len(AUT_AFF) // len(H192)
)


print(
    "Q_act direction orbits",
    tuple(
        tuple(
            bits3(x)
            for x in O
        )
        for O in direction_orbits(
            Q_ACT
        )
    )
)


# ------------------------------------------------------------
# D
# ------------------------------------------------------------

print(
    "\nD. BLIND CLASSIFICATION OF Q_ACT = im Psi"
)


print(
    "|Q_act|",
    len(Q_ACT)
)


print(
    "Q_act order histogram",
    Q_ACT_ORDER_HIST
)


print(
    "|Z(Q_act)|",
    len(Q_ACT_CENTER)
)


print(
    "|[Q_act,Q_act]|",
    len(Q_ACT_DERIVED)
)


print(
    "abelianization order",
    len(Q_ACT) // len(Q_ACT_DERIVED)
)


print(
    "2D subspaces of F2^3",
    len(ALL_PLANES)
)


print(
    "2D subspaces not containing 001",
    len(PLANES_NOT_CONTAINING_001)
)


print(
    "four-object family:"
)


for i, P in enumerate(
    PLANES_NOT_CONTAINING_001
):

    print(
        " ",
        i,
        [
            bits3(x)
            for x in sorted(P)
        ]
    )


print(
    "|Q_act image on four objects|",
    len(Q_ACT_ON_4)
)


print(
    "|kernel of four-object action|",
    len(Q_ACT_ON_4_KERNEL)
)


print(
    "faithful four-object action",
    len(Q_ACT_ON_4_KERNEL) == 1
)


print(
    "Q_act ~= S4 constructive criterion",
    Q_ACT_S4_CONSTRUCTIVE
)


# ------------------------------------------------------------
# E
# ------------------------------------------------------------

print(
    "\nE. COMMON FIXED SUBSPACE"
)


print(
    "Fix_G(Q_act)",
    [
        bits3(d)
        for d in FIXED_G_BITS
    ]
)


print(
    "Fix_G(Q_act) == <001>",
    FIXED_G_BITS == FIXED_G_EXPECTED
)


print(
    "nonzero common fixed directions",
    [
        bits3(d)
        for d in FIXED_G_BITS
        if d != 0
    ]
)


# ------------------------------------------------------------
# F
# ------------------------------------------------------------

print(
    "\nF. ZERO-TRANSLATION SECTION Q0"
)


print(
    "|Q0|",
    len(Q0)
)


print(
    "Q0 order histogram",
    element_order_hist(
        Q0
    )
)


print(
    "|Q0 cap G|",
    len(Q0_CAP_G)
)


print(
    "|G Q0|",
    len(GQ0)
)


print(
    "G Q0 == H192",
    GQ0 == H192
)


print(
    "Psi(Q0) == Q_act",
    Q0_PSI_IMAGE == Q_ACT
)


print(
    "zero-translation section criterion",
    Q0_SECTION_CRITERION
)


if Q0_GENERATORS is not None:

    print(
        "Q0 generating pair:"
    )

    for name, q in zip(
        ("q1", "q2"),
        Q0_GENERATORS
    ):

        Ak = PSI_BY_ELEMENT[q]

        print(
            " ",
            name,
            "=",
            cycle_string(q),
            "order",
            order(q)
        )

        print(
            "    Psi =",
            matrix_string(Ak)
        )

        print(
            "    directions:",
            {
                bits3(d):
                    bits3(
                        apply_matrix_bits(
                            Ak,
                            d
                        )
                    )
                for d in D7
            }
        )


# ------------------------------------------------------------
# G
# ------------------------------------------------------------

print(
    "\nG. ORIGIN-CHANGE AUDIT"
)


print(
    "origins tested",
    8
)


print(
    "distinct origin-induced sections",
    len(UNIQUE_ORIGIN_SECTIONS)
)


print(
    "origin classes producing same section:"
)


for i, origins in ORIGIN_SECTION_CLASSES.items():

    print(
        " ",
        i,
        [
            bits3(a)
            for a in origins
        ]
    )


print(
    "all origin-induced sections are complements",
    ORIGIN_SECTIONS_ALL_COMPLEMENTS
)


print(
    "Q0 invariant under every origin change",
    len(UNIQUE_ORIGIN_SECTIONS) == 1
)


# ------------------------------------------------------------
# H
# ------------------------------------------------------------

print(
    "\nH. CENTER / FIXED-DIRECTION CLOSURE"
)


print(
    "|Z(H192)|",
    len(Z_H192)
)


print(
    "Z(H192) elements",
    [
        (
            bits3(
                translation_label[g]
            )
            if g in translation_label
            else "---",
            cycle_string(g)
        )
        for g in sorted(
            Z_H192,
            key=cycle_string
        )
    ]
)


print(
    "Z(H192) cap G == fixed translations",
    CENTER_CAP_G == FIXED_TRANSLATIONS
)


print(
    "Z(H192) cap G labels",
    [
        bits3(
            translation_label[g]
        )
        for g in sorted(
            CENTER_CAP_G,
            key=lambda x:
                translation_label[x]
        )
    ]
)


print(
    "singleton direction 001 fixed by all Q_act",
    all(
        apply_matrix_bits(
            A,
            S_OMEGA_BITS
        )
        ==
        S_OMEGA_BITS
        for A in Q_ACT
    )
)


# ------------------------------------------------------------
# I
# ------------------------------------------------------------

print(
    "\nI. MACHINE TRUTH PACKET"
)


frozen_checks = {

    "|G|=8":
        len(G) == 8,

    "|H192|=192":
        len(H192) == 192,

    "G normal in H192":
        G_NORMAL_IN_H,

    "C_H192(G)=G":
        C_H_G == G,

    "ker Psi=G":
        PSI_KERNEL == G,

    "|im Psi|=24":
        len(PSI_IMAGE) == 24
}


print(
    "FROZEN / THEOREM-CONSEQUENCE CHECKS"
)


for key, value in frozen_checks.items():

    print(
        key,
        value
    )


new_targets = {

    "|Stab_GL(001)|=24":
        len(STAB_001) == 24,

    "im Psi subset Stab_GL(001)":
        PSI_SUBSET_STAB,

    "im Psi = Stab_GL(001)":
        PSI_EQUALS_STAB,

    "Q_act ~= S4 constructively":
        Q_ACT_S4_CONSTRUCTIVE,

    "Fix_G(Q_act)=<001>":
        FIXED_G_BITS == FIXED_G_EXPECTED,

    "|Q0|=24":
        len(Q0) == 24,

    "Q0 cap G={e}":
        Q0_CAP_G == frozenset([ID]),

    "Psi(Q0)=Q_act":
        Q0_PSI_IMAGE == Q_ACT,

    "G Q0=H192":
        GQ0 == H192,

    "split extension criterion":
        Q0_SECTION_CRITERION,

    "center/fixed-translation closure":
        CENTER_CAP_G == FIXED_TRANSLATIONS,

    "all origin sections are complements":
        ORIGIN_SECTIONS_ALL_COMPLEMENTS
}


print(
    "\nNEW SIM14.8 TARGETS"
)


for key, value in new_targets.items():

    print(
        key,
        value
    )


print(
    "\nALL FROZEN CHECKS PASS",
    all(
        frozen_checks.values()
    )
)


print(
    "ALL SIM14.8 TARGET CHECKS PASS",
    all(
        new_targets.values()
    )
)


print(
    "\nNo dynamics, LCO/ISP, J, metric, physical interpretation,"
)

print(
    "D4, Weyl-group, 24-cell, F4, or E8 identification added."
)


# ============================================================
# 18. HARD SANITY ASSERTIONS
#
# Only frozen/pre-SIM14.8 structure is asserted.
#
# New SIM14.8 targets remain measured booleans.
# ============================================================

assert len(G) == 8

assert (
    frozenset(
        translations.values()
    )
    ==
    G
)

assert np.array_equal(
    Omega.T,
    -Omega
)

assert (
    np.linalg.matrix_rank(
        Omega
    )
    ==
    8
)

assert (
    round(
        np.linalg.det(
            Omega
        )
    )
    ==
    1
)

assert len(GL32) == 168
assert len(AUT_AFF) == 1344

# Frozen SIM14.6 facts:
assert len(H192) == 192
assert G_NORMAL_IN_H
assert C_H_G == G

# Mathematical consequences of those frozen facts:
assert PSI_KERNEL == G
assert len(PSI_IMAGE) == 24


print(
    "\nSIM14.8 COMPLETE — FROZEN SANITY CHECKS PASS"
)





~~~~~~~~~~~~~~~~~~~~~~~~





RESULTS:




======================================================================================
SIM14.8 — AFFINE–PARTNER STABILIZER EXTENSION MAP
======================================================================================

A. FROZEN / REGRESSION SANITY|G| 8
|GL(3,2)| 168
|AGL(3,2)| 1344
|H192| 192
G normal in H192 True
|C_H192(G)| 8
C_H192(G) == G True
coordinates {0: '000', 1: '100', 2: '001', 3: '101', 4: '111', 5: '011', 6: '110', 7: '010'}
Omega antisymmetric True rank 8 det 1

B. CONJUGATION ACTION Psi : H192 -> GL(3,2)|im Psi| 24
|ker Psi| 8
ker Psi == G True
quotient order |H192|/|G| 24
distinct Psi matrices:
  [010;100;001] order 2 D7 action ('001', '100', '101', '010', '011', '110', '111')
  [010;100;011] order 4 D7 action ('001', '101', '100', '010', '011', '111', '110')
  [010;100;101] order 4 D7 action ('001', '100', '101', '011', '010', '111', '110')
  [010;100;111] order 2 D7 action ('001', '101', '100', '011', '010', '110', '111')
  [010;110;001] order 3 D7 action ('001', '110', '111', '010', '011', '100', '101')
  [010;110;011] order 3 D7 action ('001', '111', '110', '010', '011', '101', '100')
  [010;110;101] order 3 D7 action ('001', '110', '111', '011', '010', '101', '100')
  [010;110;111] order 3 D7 action ('001', '111', '110', '011', '010', '100', '101')
  [100;010;001] order 1 D7 action ('001', '010', '011', '100', '101', '110', '111')
  [100;010;011] order 2 D7 action ('001', '011', '010', '100', '101', '111', '110')
  [100;010;101] order 2 D7 action ('001', '010', '011', '101', '100', '111', '110')
  [100;010;111] order 2 D7 action ('001', '011', '010', '101', '100', '110', '111')
  [100;110;001] order 2 D7 action ('001', '010', '011', '110', '111', '100', '101')
  [100;110;011] order 4 D7 action ('001', '011', '010', '110', '111', '101', '100')
  [100;110;101] order 2 D7 action ('001', '010', '011', '111', '110', '101', '100')
  [100;110;111] order 4 D7 action ('001', '011', '010', '111', '110', '100', '101')
  [110;010;001] order 2 D7 action ('001', '110', '111', '100', '101', '010', '011')
  [110;010;011] order 2 D7 action ('001', '111', '110', '100', '101', '011', '010')
  [110;010;101] order 4 D7 action ('001', '110', '111', '101', '100', '011', '010')
  [110;010;111] order 4 D7 action ('001', '111', '110', '101', '100', '010', '011')
  [110;100;001] order 3 D7 action ('001', '100', '101', '110', '111', '010', '011')
  [110;100;011] order 3 D7 action ('001', '101', '100', '110', '111', '011', '010')
  [110;100;101] order 3 D7 action ('001', '100', '101', '111', '110', '011', '010')
  [110;100;111] order 3 D7 action ('001', '101', '100', '111', '110', '010', '011')

C. STABILIZER TEST FOR 001 = s_Omega|Stab_GL(001)| 24
im Psi subset Stab_GL(001) True
im Psi == Stab_GL(001) True
[GL(3,2):Stab_GL(001)] 7
[AGL(3,2):H192] 7
Q_act direction orbits (('001',), ('010', '011', '100', '101', '110', '111'))

D. BLIND CLASSIFICATION OF Q_ACT = im Psi|Q_act| 24
Q_act order histogram {1: 1, 2: 9, 3: 8, 4: 6}
|Z(Q_act)| 1
|[Q_act,Q_act]| 12
abelianization order 2
2D subspaces of F2^3 7
2D subspaces not containing 001 4
four-object family:
  0 ['000', '010', '100', '110']
  1 ['000', '010', '101', '111']
  2 ['000', '011', '100', '111']
  3 ['000', '011', '101', '110']
|Q_act image on four objects| 24
|kernel of four-object action| 1
faithful four-object action True
Q_act ~= S4 constructive criterion True

E. COMMON FIXED SUBSPACEFix_G(Q_act) ['000', '001']
Fix_G(Q_act) == <001> True
nonzero common fixed directions ['001']

F. ZERO-TRANSLATION SECTION Q0|Q0| 24
Q0 order histogram {1: 1, 2: 9, 3: 8, 4: 6}
|Q0 cap G| 1
|G Q0| 192
G Q0 == H192 True
Psi(Q0) == Q_act True
zero-translation section criterion True
Q0 generating pair:
  q1 = (1 3)(4 5 6 7) order 4
    Psi = [110;010;111]
    directions: {'001': '001', '010': '111', '011': '110', '100': '101', '101': '100', '110': '010', '111': '011'}
  q2 = (1 4 3 6)(5 7) order 4
    Psi = [100;110;111]
    directions: {'001': '001', '010': '011', '011': '010', '100': '111', '101': '110', '110': '100', '111': '101'}

G. ORIGIN-CHANGE AUDITorigins tested 8
distinct origin-induced sections 4
origin classes producing same section:
  0 ['000', '001']
  1 ['010', '011']
  2 ['100', '101']
  3 ['110', '111']
all origin-induced sections are complements True
Q0 invariant under every origin change False

H. CENTER / FIXED-DIRECTION CLOSURE|Z(H192)| 2
Z(H192) elements [('000', '()'), ('001', '(0 2)(1 3)(4 6)(5 7)')]
Z(H192) cap G == fixed translations True
Z(H192) cap G labels ['000', '001']
singleton direction 001 fixed by all Q_act True

I. MACHINE TRUTH PACKETFROZEN / THEOREM-CONSEQUENCE CHECKS
|G|=8 True
|H192|=192 True
G normal in H192 True
C_H192(G)=G True
ker Psi=G True
|im Psi|=24 True

NEW SIM14.8 TARGETS|Stab_GL(001)|=24 True
im Psi subset Stab_GL(001) True
im Psi = Stab_GL(001) True
Q_act ~= S4 constructively True
Fix_G(Q_act)=<001> True
|Q0|=24 True
Q0 cap G={e} True
Psi(Q0)=Q_act True
G Q0=H192 True
split extension criterion True
center/fixed-translation closure True
all origin sections are complements True

ALL FROZEN CHECKS PASS True
ALL SIM14.8 TARGET CHECKS PASS True

No dynamics, LCO/ISP, J, metric, physical interpretation,D4, Weyl-group, 24-cell, F4, or E8 identification added.

SIM14.8 COMPLETE — FROZEN SANITY CHECKS PASS
