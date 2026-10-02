SIM14.7 — TRANSLATION–SYMPLECTIC INTERSECTION MAP
==================================================

No new geometry. No dynamics. No J. No metric.
No physical interpretation.

Questions
---------
1. What exactly is

       V_Omega := G ∩ N+  ?

   - print its elements as permutations and F2^3 translations
   - compare it with K = <p,h>
   - measure K ∩ V_Omega
   - test normality, normalizer, and centralizer
   - test whether V_Omega is the unique normal V4 in N+

2. How does N+ act by conjugation on V_Omega^×?

       Phi : N+ -> Sym(V_Omega^×)

   - measure image and kernel
   - test ker(Phi) = V_Omega
   - test im(Phi) ~= S3
   - print an explicit noncommuting conjugation witness

3. Does the extension split constructively?

   Search for Q <= N+ such that

       |Q| = 6
       Q ∩ V_Omega = {e}
       V_Omega Q = N+

   Then test

       Q ~= S3

   and find generators r,t satisfying

       r^2 = e
       t^3 = e
       r t r = t^{-1}.

Interpretive ceiling
--------------------
Finite permutation / affine / signed-symplectic carrier
mathematics only.

K and V_Omega are kept explicitly distinct throughout.

Expected textbook structures are TESTED, not inserted.
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


def normalizer(H, A):

    A = set(A)

    return frozenset(

        g

        for g in H

        if {
            conjugate(g, a)
            for a in A
        } == A
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


def is_normal(A, H):

    return (
        normalizer(H, A)
        ==
        frozenset(H)
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


def all_subgroups(H):
    """
    Complete closure-based subgroup enumeration.

    Safe here because N+ has order 24.
    """

    H = frozenset(H)

    seen = {
        frozenset([ID])
    }

    Q = [
        frozenset([ID])
    ]

    HL = list(H)

    while Q:

        A = Q.pop()

        for g in HL:

            if g not in A:

                B = generated_group(
                    list(A) + [g]
                )

                if (
                    B.issubset(H)
                    and
                    B not in seen
                ):

                    seen.add(B)
                    Q.append(B)

    return seen


def is_normal_subgroup(A, H):

    AA = set(A)

    return all(

        {
            conjugate(g, a)
            for a in A
        } == AA

        for g in H
    )


# ============================================================
# 1. FROZEN SIM14.6 p,h,s / K,G
# ============================================================

def perm_from_cycles(cycles):

    q = list(range(NPTS))

    for cyc in cycles:

        for a, b in zip(
            cyc,
            cyc[1:] + cyc[:1]
        ):

            q[a] = b

    return tuple(q)


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


K_STRUCTURAL = generated_group(
    [p, h]
)


G = generated_group(
    [p, h, s]
)


# ============================================================
# 2. FROZEN AFFINE COORDINATES FROM REGULAR G
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
        x,
        "03b"
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


def Pmat(g):

    P = np.zeros(
        (8, 8),
        dtype=int
    )

    for i, j in enumerate(g):

        P[j, i] = 1

    return P


def omega_sign(g):

    P = Pmat(g)

    A = (
        P.T
        @ Omega
        @ P
    )

    if np.array_equal(
        A,
        Omega
    ):

        return 1

    if np.array_equal(
        A,
        -Omega
    ):

        return -1

    return 0


# ============================================================
# 4. FROZEN AGL(3,2)
#
# Direct construction:
#
#       x -> A x + b
#
# with A in GL(3,2), b in F2^3.
#
# This is the same 1344-element affine group used in SIM14.6,
# constructed directly rather than by scanning all of S8.
# ============================================================

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


AUT_AFF = frozenset(

    affine_perm(
        A,
        bits_to_vec(b)
    )

    for A in GL32
    for b in range(8)
)


NPLUS = frozenset(

    g

    for g in AUT_AFF

    if omega_sign(g) == 1
)


assert len(AUT_AFF) == 1344
assert len(NPLUS) == 24


# ============================================================
# 5. PRIMARY SIM14.7 OBJECT
# ============================================================

V_OMEGA = intersection(
    G,
    NPLUS
)


K_CAP_V = intersection(
    K_STRUCTURAL,
    V_OMEGA
)


def describe_translation_group(H):

    rows = []

    for g in sorted(

        H,

        key=lambda q:
            translation_label.get(
                q,
                999
            )
    ):

        d = translation_label.get(
            g,
            None
        )

        rows.append(
            (
                bits3(d)
                if d is not None
                else "---",

                cycle_string(g),

                order(g)
            )
        )

    return rows


# ============================================================
# 6. S3 PERMUTATION UTILITIES
# ============================================================

ID3 = (
    0,
    1,
    2
)


def compose3(a, b):

    return tuple(
        a[b[i]]
        for i in range(3)
    )


def order3perm(a):

    x = ID3

    for n in range(1, 7):

        x = compose3(
            a,
            x
        )

        if x == ID3:
            return n

    raise RuntimeError(
        "S3 order bound exceeded"
    )


def perm3_cycle_string(q):

    seen = set()
    out = []

    for i in range(3):

        if (
            i not in seen
            and
            q[i] != i
        ):

            cyc = []
            j = i

            while j not in seen:

                seen.add(j)
                cyc.append(j + 1)
                j = q[j]

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


# ============================================================
# 7. CONJUGATION ACTION
#
# Phi : N+ -> Sym(V_Omega^x)
# ============================================================

V_NONZERO = tuple(

    sorted(

        (
            g
            for g in V_OMEGA
            if g != ID
        ),

        key=lambda q:
            translation_label[q]
    )
)


V_INDEX = {

    v:
        i

    for i, v in enumerate(
        V_NONZERO
    )
}


def phi_perm(g):

    return tuple(

        V_INDEX[
            conjugate(g, v)
        ]

        for v in V_NONZERO
    )


PHI_IMAGE = frozenset(

    phi_perm(g)

    for g in NPLUS
)


PHI_KERNEL = frozenset(

    g

    for g in NPLUS

    if phi_perm(g) == ID3
)


PHI_ORDER_HIST = dict(

    sorted(

        Counter(

            order3perm(q)

            for q in PHI_IMAGE
        ).items()
    )
)


# Explicit noncommuting witness

NONCOMM_WITNESS = None


for g in sorted(
    NPLUS,
    key=cycle_string
):

    for v in V_NONZERO:

        vp = conjugate(
            g,
            v
        )

        if vp != v:

            NONCOMM_WITNESS = (
                g,
                v,
                vp
            )

            break

    if NONCOMM_WITNESS is not None:
        break


# ============================================================
# 8. NORMAL SUBGROUP / NORMALIZER / CENTRALIZER
# ============================================================

SUBS_NPLUS = all_subgroups(
    NPLUS
)


NORMAL_ORDER4 = [

    A

    for A in SUBS_NPLUS

    if (
        len(A) == 4

        and

        is_normal_subgroup(
            A,
            NPLUS
        )
    )
]


N_V = normalizer(
    NPLUS,
    V_OMEGA
)


C_V = centralizer(
    NPLUS,
    V_OMEGA
)


# ============================================================
# 9. CONSTRUCTIVE S3 COMPLEMENT SEARCH
# ============================================================

ORDER6_SUBGROUPS = sorted(

    [

        A

        for A in SUBS_NPLUS

        if len(A) == 6
    ],

    key=lambda A:
        sorted(
            cycle_string(x)
            for x in A
        )
)


COMPLEMENTS = []


for Q in ORDER6_SUBGROUPS:

    cap = intersection(
        Q,
        V_OMEGA
    )

    product = frozenset(

        compose(v, q)

        for v in V_OMEGA
        for q in Q
    )

    if (
        len(cap) == 1
        and
        product == NPLUS
    ):

        COMPLEMENTS.append(Q)


Q_WITNESS = (

    COMPLEMENTS[0]

    if COMPLEMENTS

    else None
)


RT_WITNESS = None


if Q_WITNESS is not None:

    involutions = sorted(

        [

            q

            for q in Q_WITNESS

            if order(q) == 2
        ],

        key=cycle_string
    )


    order3s = sorted(

        [

            q

            for q in Q_WITNESS

            if order(q) == 3
        ],

        key=cycle_string
    )


    for r in involutions:

        for t in order3s:

            relation = (

                compose(
                    compose(r, t),
                    r
                )

                ==

                inv(t)
            )


            generates = (

                generated_group(
                    [r, t]
                )

                ==

                Q_WITNESS
            )


            if (
                relation
                and
                generates
            ):

                RT_WITNESS = (
                    r,
                    t
                )

                break

        if RT_WITNESS is not None:
            break


# ============================================================
# 10. OUTPUT
# ============================================================

print(
    "=" * 82
)

print(
    "SIM14.7 — TRANSLATION–SYMPLECTIC INTERSECTION MAP"
)

print(
    "=" * 82
)


# ------------------------------------------------------------
# A
# ------------------------------------------------------------

print(
    "\nA. FROZEN SANITY"
)


print(
    "|K_structural|",
    len(K_STRUCTURAL)
)


print(
    "|G|",
    len(G)
)


print(
    "|Aut AG(3,2)|",
    len(AUT_AFF)
)


print(
    "|N+|",
    len(NPLUS)
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
    "\nB. V_OMEGA = G cap N+"
)


print(
    "|V_Omega|",
    len(V_OMEGA)
)


print(
    "V_Omega elements: "
    "(translation, permutation, order)"
)


for row in describe_translation_group(
    V_OMEGA
):

    print(
        " ",
        row
    )


print(

    "V_Omega order histogram",

    element_order_hist(
        V_OMEGA
    )
)


V_OMEGA_ABELIAN = all(

    commutes(a, b)

    for a in V_OMEGA
    for b in V_OMEGA
)


print(
    "V_Omega abelian",
    V_OMEGA_ABELIAN
)


V4_CRITERION = (

    len(V_OMEGA) == 4

    and

    element_order_hist(
        V_OMEGA
    )
    ==
    {
        1: 1,
        2: 3
    }

    and

    V_OMEGA_ABELIAN
)


print(
    "V_Omega ~= V4 criterion",
    V4_CRITERION
)


# ------------------------------------------------------------
# C
# ------------------------------------------------------------

print(
    "\nC. K_STRUCTURAL VERSUS V_OMEGA"
)


print(
    "K_structural elements: "
    "(translation, permutation, order)"
)


for row in describe_translation_group(
    K_STRUCTURAL
):

    print(
        " ",
        row
    )


print(
    "|K_structural cap V_Omega|",
    len(K_CAP_V)
)


print(
    "K_structural cap V_Omega elements:"
)


for row in describe_translation_group(
    K_CAP_V
):

    print(
        " ",
        row
    )


print(
    "K_structural == V_Omega",
    K_STRUCTURAL == V_OMEGA
)


# ------------------------------------------------------------
# D
# ------------------------------------------------------------

print(
    "\nD. NORMALITY / NORMALIZER / CENTRALIZER"
)


print(

    "V_Omega normal in N+",

    is_normal(
        V_OMEGA,
        NPLUS
    )
)


print(
    "|N_N+(V_Omega)|",
    len(N_V)
)


print(
    "N_N+(V_Omega) == N+",
    N_V == NPLUS
)


print(
    "|C_N+(V_Omega)|",
    len(C_V)
)


print(
    "C_N+(V_Omega) == V_Omega",
    C_V == V_OMEGA
)


print(
    "normal order-4 subgroups in N+",
    len(NORMAL_ORDER4)
)


UNIQUE_NORMAL_V4 = (

    len(NORMAL_ORDER4) == 1

    and

    NORMAL_ORDER4[0]
    ==
    V_OMEGA
)


print(
    "V_Omega is unique normal order-4 subgroup",
    UNIQUE_NORMAL_V4
)


# ------------------------------------------------------------
# E
# ------------------------------------------------------------

print(
    "\nE. CONJUGATION ACTION Phi ON V_OMEGA^x"
)


print(
    "ordered V_Omega^x labels:"
)


for i, v in enumerate(
    V_NONZERO,
    start=1
):

    print(

        " v%d =" % i,

        bits3(
            translation_label[v]
        ),

        cycle_string(v)
    )


print(
    "|im Phi|",
    len(PHI_IMAGE)
)


print(
    "|ker Phi|",
    len(PHI_KERNEL)
)


print(
    "ker Phi == V_Omega",
    PHI_KERNEL == V_OMEGA
)


print(
    "image order histogram",
    PHI_ORDER_HIST
)


IMAGE_S3_CRITERION = (

    len(PHI_IMAGE) == 6

    and

    PHI_ORDER_HIST
    ==
    {
        1: 1,
        2: 3,
        3: 2
    }
)


print(
    "im Phi ~= S3 criterion",
    IMAGE_S3_CRITERION
)


print(
    "Phi image on (v1,v2,v3):"
)


for q in sorted(
    PHI_IMAGE
):

    print(

        " ",

        q,

        perm3_cycle_string(q)
    )


if NONCOMM_WITNESS is not None:

    g, v, vp = NONCOMM_WITNESS

    print(
        "explicit conjugation witness:"
    )

    print(
        " g  =",
        cycle_string(g)
    )

    print(

        " v  =",

        bits3(
            translation_label[v]
        ),

        cycle_string(v)
    )

    print(

        " g v g^-1 =",

        bits3(
            translation_label[vp]
        ),

        cycle_string(vp)
    )

    print(
        "g v != v g",
        not commutes(g, v)
    )


# ------------------------------------------------------------
# F
# ------------------------------------------------------------

print(
    "\nF. CONSTRUCTIVE S3 COMPLEMENT"
)


print(
    "order-6 subgroups in N+",
    len(ORDER6_SUBGROUPS)
)


print(
    "complements found",
    len(COMPLEMENTS)
)


if Q_WITNESS is not None:

    print(

        "Q witness",

        sorted(

            cycle_string(q)

            for q in Q_WITNESS
        )
    )


    print(
        "|Q|",
        len(Q_WITNESS)
    )


    print(

        "Q order histogram",

        element_order_hist(
            Q_WITNESS
        )
    )


    print(

        "|Q cap V_Omega|",

        len(
            intersection(
                Q_WITNESS,
                V_OMEGA
            )
        )
    )


    product = frozenset(

        compose(v, q)

        for v in V_OMEGA
        for q in Q_WITNESS
    )


    print(
        "|V_Omega Q|",
        len(product)
    )


    print(
        "V_Omega Q == N+",
        product == NPLUS
    )


    Q_S3_CRITERION = (

        len(Q_WITNESS) == 6

        and

        element_order_hist(
            Q_WITNESS
        )
        ==
        {
            1: 1,
            2: 3,
            3: 2
        }
    )


    print(
        "Q ~= S3 criterion",
        Q_S3_CRITERION
    )


    if RT_WITNESS is not None:

        r, t = RT_WITNESS

        print(
            "S3 generators:"
        )


        print(

            " r =",

            cycle_string(r),

            "order",

            order(r),

            "Phi(r)",

            phi_perm(r),

            perm3_cycle_string(
                phi_perm(r)
            )
        )


        print(

            " t =",

            cycle_string(t),

            "order",

            order(t),

            "Phi(t)",

            phi_perm(t),

            perm3_cycle_string(
                phi_perm(t)
            )
        )


        print(

            "r^2 = e",

            compose(
                r,
                r
            )
            ==
            ID
        )


        print(

            "t^3 = e",

            compose(
                t,
                compose(t, t)
            )
            ==
            ID
        )


        print(

            "r t r = t^-1",

            compose(
                compose(r, t),
                r
            )
            ==
            inv(t)
        )


        print(

            "<r,t> == Q",

            generated_group(
                [r, t]
            )
            ==
            Q_WITNESS
        )


        print(
            "conjugation action on V_Omega^x:"
        )


        for name, q in [
            ("r", r),
            ("t", t)
        ]:

            print(
                " ",
                name
            )

            for v in V_NONZERO:

                vp = conjugate(
                    q,
                    v
                )

                print(

                    "   ",

                    bits3(
                        translation_label[v]
                    ),

                    "->",

                    bits3(
                        translation_label[vp]
                    )
                )


# ------------------------------------------------------------
# G
# ------------------------------------------------------------

print(
    "\nG. MACHINE TRUTH PACKET"
)


SEMIDIRECT_CRITERION = (

    Q_WITNESS is not None

    and

    len(
        intersection(
            Q_WITNESS,
            V_OMEGA
        )
    )
    ==
    1

    and

    frozenset(

        compose(v, q)

        for v in V_OMEGA
        for q in Q_WITNESS
    )
    ==
    NPLUS
)


truth = {

    "|V_Omega|=4":
        len(V_OMEGA) == 4,

    "V_Omega~=V4":
        V4_CRITERION,

    "|K cap V_Omega|=2":
        len(K_CAP_V) == 2,

    "K != V_Omega":
        K_STRUCTURAL != V_OMEGA,

    "V_Omega normal in N+":
        is_normal(
            V_OMEGA,
            NPLUS
        ),

    "normalizer=N+":
        N_V == NPLUS,

    "centralizer=V_Omega":
        C_V == V_OMEGA,

    "unique normal V4":
        UNIQUE_NORMAL_V4,

    "|im Phi|=6":
        len(PHI_IMAGE) == 6,

    "|ker Phi|=4":
        len(PHI_KERNEL) == 4,

    "ker Phi=V_Omega":
        PHI_KERNEL == V_OMEGA,

    "im Phi~=S3":
        IMAGE_S3_CRITERION,

    "S3 complement exists":
        Q_WITNESS is not None,

    "semidirect product criterion":
        SEMIDIRECT_CRITERION
}


for key, value in truth.items():

    print(
        key,
        value
    )


print(

    "ALL SIM14.7 TARGET CHECKS PASS",

    all(
        truth.values()
    )
)


print(
    "No geometry, dynamics, J, metric, "
    "or physical interpretation added."
)


# ============================================================
# 11. HARD SANITY ASSERTIONS
#
# Only frozen SIM14.6 facts are asserted.
# New SIM14.7 target claims remain measured booleans.
# ============================================================

assert len(K_STRUCTURAL) == 4
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
assert len(NPLUS) == 24


print(
    "\nSIM14.7 COMPLETE — FROZEN SANITY CHECKS PASS"
)





#####################





RESULTS:




==================================================================================
SIM14.7 — TRANSLATION–SYMPLECTIC INTERSECTION MAP
==================================================================================

A. FROZEN SANITY|K_structural| 4
|G| 8
|Aut AG(3,2)| 1344
|N+| 24
coordinates {0: '000', 1: '100', 2: '001', 3: '101', 4: '111', 5: '011', 6: '110', 7: '010'}
Omega antisymmetric True rank 8 det 1

B. V_OMEGA = G cap N+|V_Omega| 4
V_Omega elements: (translation, permutation, order)
  ('000', '()', 1)
  ('011', '(0 5)(1 4)(2 7)(3 6)', 2)
  ('100', '(0 1)(2 3)(4 5)(6 7)', 2)
  ('111', '(0 4)(1 5)(2 6)(3 7)', 2)
V_Omega order histogram {1: 1, 2: 3}
V_Omega abelian True
V_Omega ~= V4 criterion True

C. K_STRUCTURAL VERSUS V_OMEGAK_structural elements: (translation, permutation, order)
  ('000', '()', 1)
  ('010', '(0 7)(1 6)(2 5)(3 4)', 2)
  ('100', '(0 1)(2 3)(4 5)(6 7)', 2)
  ('110', '(0 6)(1 7)(2 4)(3 5)', 2)
|K_structural cap V_Omega| 2
K_structural cap V_Omega elements:
  ('000', '()', 1)
  ('100', '(0 1)(2 3)(4 5)(6 7)', 2)
K_structural == V_Omega False

D. NORMALITY / NORMALIZER / CENTRALIZERV_Omega normal in N+ True
|N_N+(V_Omega)| 24
N_N+(V_Omega) == N+ True
|C_N+(V_Omega)| 4
C_N+(V_Omega) == V_Omega True
normal order-4 subgroups in N+ 1
V_Omega is unique normal order-4 subgroup True

E. CONJUGATION ACTION Phi ON V_OMEGA^xordered V_Omega^x labels:
 v1 = 011 (0 5)(1 4)(2 7)(3 6)
 v2 = 100 (0 1)(2 3)(4 5)(6 7)
 v3 = 111 (0 4)(1 5)(2 6)(3 7)
|im Phi| 6
|ker Phi| 4
ker Phi == V_Omega True
image order histogram {1: 1, 2: 3, 3: 2}
im Phi ~= S3 criterion True
Phi image on (v1,v2,v3):
  (0, 1, 2) ()
  (0, 2, 1) (2 3)
  (1, 0, 2) (1 2)
  (1, 2, 0) (1 2 3)
  (2, 0, 1) (1 3 2)
  (2, 1, 0) (1 3)
explicit conjugation witness:
 g  = (0 1 4 5)(2 3 6 7)
 v  = 011 (0 5)(1 4)(2 7)(3 6)
 g v g^-1 = 100 (0 1)(2 3)(4 5)(6 7)
g v != v g True

F. CONSTRUCTIVE S3 COMPLEMENTorder-6 subgroups in N+ 4
complements found 4
Q witness ['()', '(0 1 4)(2 3 6)', '(0 1)(2 3)', '(0 4 1)(2 6 3)', '(0 4)(2 6)', '(1 4)(3 6)']
|Q| 6
Q order histogram {1: 1, 2: 3, 3: 2}
|Q cap V_Omega| 1
|V_Omega Q| 24
V_Omega Q == N+ True
Q ~= S3 criterion True
S3 generators:
 r = (0 1)(2 3) order 2 Phi(r) (2, 1, 0) (1 3)
 t = (0 1 4)(2 3 6) order 3 Phi(t) (2, 0, 1) (1 3 2)
r^2 = e True
t^3 = e True
r t r = t^-1 True
<r,t> == Q True
conjugation action on V_Omega^x:
  r
    011 -> 111
    100 -> 100
    111 -> 011
  t
    011 -> 111
    100 -> 011
    111 -> 100

G. MACHINE TRUTH PACKET|V_Omega|=4 True
V_Omega~=V4 True
|K cap V_Omega|=2 True
K != V_Omega True
V_Omega normal in N+ True
normalizer=N+ True
centralizer=V_Omega True
unique normal V4 True
|im Phi|=6 True
|ker Phi|=4 True
ker Phi=V_Omega True
im Phi~=S3 True
S3 complement exists True
semidirect product criterion True
ALL SIM14.7 TARGET CHECKS PASS True
No geometry, dynamics, J, metric, or physical interpretation added.
