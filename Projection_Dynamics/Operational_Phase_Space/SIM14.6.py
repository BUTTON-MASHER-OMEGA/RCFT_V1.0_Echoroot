#!/usr/bin/env python3
"""
SIM14.6 — OPERATIONAL PHASE-SPACE SYMMETRY CLOSURE
===================================================

No new geometry. No dynamics. No J search.

Questions:
1. Classify the 192-element unsigned-pairing stabilizer.
2. Test Aut_aff(|Omega|) == Aut_aff(s_Omega).
3. Locate G=C2^3 and K=V4 by intersections, normalizers, centralizers.
4. Constructively verify N+ ~= S4 and N± ~= S4 x C2.
5. Map orbit signatures on X8, D7, V7, P14, C7.
6. Test D7 <-> C7 equivariance.
7. Compute End_{N+}(R^8) and decompose the 8D real permutation
   representation. This prepares, but does NOT perform, the J experiment.

CORRECTION FROM INITIAL SIM14.6
-------------------------------
The complete subgroup census for the 192-element group has been disabled.
The naive closure-based subgroup enumerator is appropriate for N+ (order 24)
but too expensive for H192 in the GPT console environment.

The H192 classification still computes:
    - order
    - element-order histogram
    - center
    - derived subgroup
    - abelianization order
    - conjugacy-class sizes

The complete subgroup census remains enabled for N+.

Interpretive ceiling
--------------------
Finite permutation/affine/symplectic carrier mathematics only.

No F4, 24-cell, E8, H4, octonionic multiplication, J, g, U(4),
ISP, LCO, or physical interpretation is inserted.
"""

import itertools
from collections import Counter
import numpy as np


# ============================================================
# 0. PERMUTATION UTILITIES
# ============================================================

NPTS = 8
ID = tuple(range(8))


def compose(a, b):
    """a o b: apply b first, then a."""
    return tuple(a[b[i]] for i in range(8))


def inv(a):
    q = [0] * 8

    for i, j in enumerate(a):
        q[j] = i

    return tuple(q)


def order(a):
    x = ID

    for n in range(1, 1000):
        x = compose(a, x)

        if x == ID:
            return n

    raise RuntimeError("order bound exceeded")


def cycle_string(a):
    seen = set()
    out = []

    for i in range(8):

        if i not in seen and a[i] != i:

            c = []
            j = i

            while j not in seen:
                seen.add(j)
                c.append(j)
                j = a[j]

            out.append(
                "(" + " ".join(map(str, c)) + ")"
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
    return compose(a, b) == compose(b, a)


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


def orbit_of(H, x, action):

    return frozenset(
        action(g, x)
        for g in H
    )


def orbits(H, objects, action):

    rem = set(objects)
    ans = []

    while rem:

        x = next(iter(rem))

        O = orbit_of(
            H,
            x,
            action
        )

        ans.append(O)

        rem -= set(O)

    return sorted(
        ans,
        key=lambda O: (
            len(O),
            str(sorted(map(str, O)))
        )
    )


def stabilizer(H, x, action):

    return frozenset(
        g
        for g in H
        if action(g, x) == x
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


def center(H):

    return frozenset(
        g
        for g in H
        if all(
            commutes(g, h)
            for h in H
        )
    )


def commutator(a, b):

    return compose(
        compose(
            compose(a, b),
            inv(a)
        ),
        inv(b)
    )


def derived_subgroup(H):

    return generated_group(
        [
            commutator(a, b)
            for a in H
            for b in H
        ]
    )


def conjugacy_classes(H):

    rem = set(H)
    ans = []

    while rem:

        x = next(iter(rem))

        C = frozenset(
            conjugate(g, x)
            for g in H
        )

        ans.append(C)

        rem -= set(C)

    return sorted(
        ans,
        key=lambda C: (
            len(C),
            str(
                sorted(
                    cycle_string(x)
                    for x in C
                )
            )
        )
    )


def all_subgroups(H):
    """
    Complete closure-based subgroup enumeration.

    IMPORTANT:
    Use only for relatively small groups in this script.

    It remains enabled for N+ (order 24), but SIM14.6 deliberately
    does NOT call it for H192 (order 192).
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
                    and B not in seen
                ):

                    seen.add(B)
                    Q.append(B)

    return seen


def normal_subgroups(H, subs=None):

    if subs is None:
        subs = all_subgroups(H)

    ans = []

    for A in subs:

        AA = set(A)

        if all(
            {
                conjugate(g, a)
                for a in A
            } == AA
            for g in H
        ):

            ans.append(A)

    return ans


def group_summary(H, census=False):

    D = derived_subgroup(H)

    S = {

        "order":
            len(H),

        "element_orders":
            element_order_hist(H),

        "center_order":
            len(center(H)),

        "center_orders":
            element_order_hist(
                center(H)
            ),

        "derived_order":
            len(D),

        "derived_orders":
            element_order_hist(D),

        "abelianization_order":
            len(H) // len(D),

        "conjugacy_class_sizes":
            sorted(
                len(C)
                for C in conjugacy_classes(H)
            )
    }

    if census:

        subs = all_subgroups(H)

        norms = normal_subgroups(
            H,
            subs
        )

        S["subgroup_orders"] = dict(
            sorted(
                Counter(
                    len(A)
                    for A in subs
                ).items()
            )
        )

        S["subgroups_total"] = len(subs)

        S["normal_subgroup_orders"] = dict(
            sorted(
                Counter(
                    len(A)
                    for A in norms
                ).items()
            )
        )

        S["normal_subgroups_total"] = len(norms)

    return S


def print_summary(name, H, census=False):

    print(name)

    for k, v in group_summary(
        H,
        census
    ).items():

        print(
            " ",
            k,
            "=",
            v
        )


# ============================================================
# 1. FROZEN C8 / p,h,s / K,G
# ============================================================

C8_EDGES = {

    frozenset(e)

    for e in [

        (0, 1),
        (0, 2),

        (1, 3),
        (2, 4),

        (3, 5),
        (4, 6),

        (5, 7),
        (6, 7)
    ]
}


def perm_from_cycles(cycles):

    p = list(range(8))

    for cyc in cycles:

        for a, b in zip(
            cyc,
            cyc[1:] + cyc[:1]
        ):

            p[a] = b

    return tuple(p)


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


K = generated_group(
    [p, h]
)


G = generated_group(
    [p, h, s]
)


def maps_edges_to_self(g, edges):

    return {

        frozenset(
            (
                g[min(e)],
                g[max(e)]
            )
        )

        for e in edges

    } == edges


# ============================================================
# 2. AFFINE COORDINATES FROM REGULAR G
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
    return format(x, "03b")


def xor_point(x, d):

    return point_of[
        coord_of[x] ^ d
    ]


def translation_perm(d):

    return tuple(
        xor_point(x, d)
        for x in range(8)
    )


translations = {

    d:
        translation_perm(d)

    for d in range(8)
}


D7 = tuple(
    range(1, 8)
)


assert frozenset(
    translations.values()
) == G


# ============================================================
# 3. V7 AND P14
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


V7 = sorted(

    {
        span2(a, b)

        for a in D7
        for b in D7

        if a != b
    },

    key=lambda V:
        sorted(V)
)


assert len(V7) == 7


P14_set = set()
plane_dir = {}


for V in V7:

    for x in range(8):

        C = frozenset(
            x ^ v
            for v in V
        )

        P = frozenset(
            point_of[b]
            for b in C
        )

        P14_set.add(P)

        plane_dir[P] = V


P14 = sorted(
    P14_set,
    key=lambda P:
        sorted(P)
)


assert len(P14) == 14


# ============================================================
# 4. SEVEN PARALLEL CLASSES
# ============================================================

C7 = []


for V in V7:

    planes = [
        P
        for P in P14
        if plane_dir[P] == V
    ]

    assert len(planes) == 2

    C7.append(
        frozenset(planes)
    )


assert len(C7) == 7


# ============================================================
# 5. SIGNED OMEGA WITH M2 SUPPORT
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


AbsOmega = np.abs(
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


def preserves_absomega(g):

    P = Pmat(g)

    return np.array_equal(

        P.T
        @ AbsOmega
        @ P,

        AbsOmega
    )


def preserves_partner(g):

    return (
        conjugate(g, s)
        == s
    )


# ============================================================
# 6. EXHAUSTIVE S8 CARTOGRAPHY
# ============================================================

def map_set(g, S):

    return frozenset(
        g[x]
        for x in S
    )


P14_FROZEN = frozenset(
    P14
)


def preserves_affine(g):

    return frozenset(

        map_set(g, P)
        for P in P14

    ) == P14_FROZEN


ALL = list(
    itertools.permutations(
        range(8)
    )
)


AUT_AFF = frozenset(

    g
    for g in ALL

    if preserves_affine(g)
)


AUT_C8 = frozenset(

    g
    for g in ALL

    if maps_edges_to_self(
        g,
        C8_EDGES
    )
)


NPLUS = frozenset(

    g
    for g in AUT_AFF

    if omega_sign(g) == 1
)


NMINUS = frozenset(

    g
    for g in AUT_AFF

    if omega_sign(g) == -1
)


NPM = frozenset(

    set(NPLUS)
    |
    set(NMINUS)
)


AFFABS = frozenset(

    g
    for g in AUT_AFF

    if preserves_absomega(g)
)


AFFS = frozenset(

    g
    for g in AUT_AFF

    if preserves_partner(g)
)


# ============================================================
# 7. ACTIONS
# ============================================================

def act_point(g, x):

    return g[x]


def direction_image(g, d):

    q = conjugate(
        g,
        translations[d]
    )

    for e, t in translations.items():

        if q == t:
            return e

    raise RuntimeError(
        "translation normalization failed"
    )


def act_direction(g, d):

    return direction_image(
        g,
        d
    )


def act_V(g, V):

    return frozenset(

        direction_image(g, d)

        for d in V
    )


def act_plane(g, P):

    return map_set(
        g,
        P
    )


def act_class(g, C):

    return frozenset(

        act_plane(g, P)

        for P in C
    )


def orbit_signature(H):

    return {

        "X8":
            sorted(
                len(O)
                for O in orbits(
                    H,
                    range(8),
                    act_point
                )
            ),

        "D7":
            sorted(
                len(O)
                for O in orbits(
                    H,
                    D7,
                    act_direction
                )
            ),

        "V7":
            sorted(
                len(O)
                for O in orbits(
                    H,
                    V7,
                    act_V
                )
            ),

        "P14":
            sorted(
                len(O)
                for O in orbits(
                    H,
                    P14,
                    act_plane
                )
            ),

        "C7":
            sorted(
                len(O)
                for O in orbits(
                    H,
                    C7,
                    act_class
                )
            )
    }


# ============================================================
# 8. EQUIVARIANT BIJECTION SEARCH
# ============================================================

def equivariant_bijections(
    H,
    A,
    actA,
    B,
    actB,
    cap=50
):

    A = list(A)
    B = list(B)

    OA = orbits(
        H,
        A,
        actA
    )

    OB = orbits(
        H,
        B,
        actB
    )

    if sorted(
        map(len, OA)
    ) != sorted(
        map(len, OB)
    ):

        return []


    solutions = []


    def map_from_reps(
        a0,
        b0
    ):

        f = {}

        for g in H:

            a = actA(
                g,
                a0
            )

            b = actB(
                g,
                b0
            )

            if (
                a in f
                and f[a] != b
            ):

                return None

            f[a] = b

        return f


    def rec(
        i,
        used,
        current
    ):

        if len(solutions) >= cap:
            return

        if i == len(OA):

            if (
                len(current) == len(A)
                and
                len(
                    set(
                        current.values()
                    )
                ) == len(B)
            ):

                solutions.append(
                    dict(current)
                )

            return


        Oa = OA[i]

        a0 = next(
            iter(Oa)
        )


        for j, Ob in enumerate(OB):

            if (
                j in used
                or
                len(Ob) != len(Oa)
            ):

                continue


            for b0 in Ob:

                part = map_from_reps(
                    a0,
                    b0
                )

                if part is None:
                    continue

                if set(part) != set(Oa):
                    continue

                if (
                    set(part.values())
                    != set(Ob)
                ):
                    continue

                new = dict(current)

                new.update(part)

                rec(
                    i + 1,
                    used | {j},
                    new
                )


    rec(
        0,
        set(),
        {}
    )

    return solutions


# ============================================================
# 9. COSET ACTION / CONSTRUCTIVE S4
# ============================================================

def cosets(H, A):

    rem = set(H)
    ans = []

    while rem:

        g = next(
            iter(rem)
        )

        C = frozenset(

            compose(g, a)

            for a in A
        )

        ans.append(C)

        rem -= set(C)

    return ans


def induced_perm(
    g,
    objects,
    action
):

    idx = {
        x: i
        for i, x in enumerate(objects)
    }

    return tuple(

        idx[
            action(g, x)
        ]

        for x in objects
    )


# ============================================================
# 10. REAL COMMUTANT
# ============================================================

def commutant_basis(
    H,
    tol=1e-9
):

    rows = []


    for g in H:

        P = Pmat(g).astype(float)


        for i in range(8):

            for j in range(8):

                row = np.zeros(64)


                # (A P)_ij

                for k in range(8):

                    row[
                        8 * i + k
                    ] += P[k, j]


                # -(P A)_ij

                for k in range(8):

                    row[
                        8 * k + j
                    ] -= P[i, k]


                rows.append(row)


    M = np.vstack(rows)


    U, S, Vh = np.linalg.svd(
        M,
        full_matrices=False
    )


    rank = int(
        np.sum(
            S > tol
        )
    )


    null = Vh[rank:]


    basis = [

        v.reshape(
            8,
            8
        )

        for v in null
    ]


    residual = max(

        (
            np.linalg.norm(

                A @ Pmat(g)
                -
                Pmat(g) @ A
            )

            for A in basis
            for g in H
        ),

        default=0
    )


    return (
        basis,
        rank,
        residual
    )


# ============================================================
# 11. S4 CHARACTER DECOMPOSITION
# ============================================================

S4_IRREPS = {

    "trivial":
        [1, 1, 1, 1, 1],

    "sign":
        [1, -1, 1, 1, -1],

    "std3":
        [3, 1, -1, 0, -1],

    "std3sign":
        [3, -1, -1, 0, 1],

    "two":
        [2, 0, 2, -1, 0]
}


S4_SIZES = [
    1,
    6,
    3,
    8,
    6
]


def s4_class_index(
    ordr,
    size
):

    lookup = {

        (1, 1): 0,

        (2, 6): 1,

        (2, 3): 2,

        (3, 8): 3,

        (4, 6): 4
    }

    return lookup.get(
        (ordr, size)
    )


def decompose_NPLUS():

    classes = conjugacy_classes(
        NPLUS
    )

    chi = [None] * 5

    details = []


    for C in classes:

        r = next(
            iter(C)
        )

        idx = s4_class_index(
            order(r),
            len(C)
        )

        if idx is None:

            return (
                None,
                []
            )


        tr = int(
            np.trace(
                Pmat(r)
            )
        )


        chi[idx] = tr


        details.append(

            (
                idx,
                len(C),
                order(r),
                tr,
                cycle_string(r)
            )
        )


    mults = {}


    for name, irr in S4_IRREPS.items():

        m = sum(

            sz * c * r

            for sz, c, r
            in zip(
                S4_SIZES,
                chi,
                irr
            )

        ) / 24


        mults[name] = m


    return (

        {
            "character":
                chi,

            "multiplicities":
                mults
        },

        sorted(details)
    )


# ============================================================
# 12. REPORT
# ============================================================

print(
    "=" * 90
)

print(
    "SIM14.6 — OPERATIONAL PHASE-SPACE SYMMETRY CLOSURE"
)

print(
    "=" * 90
)


# ------------------------------------------------------------
# A
# ------------------------------------------------------------

print(
    "\nA. FROZEN SANITY"
)


print(
    "|K|",
    len(K),
    "|G|",
    len(G)
)


print(
    "G regular",
    all(
        len(
            orbit_of(
                G,
                x,
                act_point
            )
        ) == 8

        for x in range(8)
    )
)


print(
    "V7",
    len(V7),
    "P14",
    len(P14),
    "C7",
    len(C7)
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


# ------------------------------------------------------------
# B
# ------------------------------------------------------------

print(
    "\nB. MASTER GROUP ORDERS"
)


print(
    "|S8|",
    len(ALL)
)


print(
    "|Aut AG(3,2)|",
    len(AUT_AFF)
)


print(
    "|Aut_aff(|Omega|)|",
    len(AFFABS)
)


print(
    "|Aut_aff(s)|",
    len(AFFS)
)


print(
    "AFFABS == AFFS",
    AFFABS == AFFS
)


print(
    "|N+|",
    len(NPLUS)
)


print(
    "|N-|",
    len(NMINUS)
)


print(
    "|N±|",
    len(NPM)
)


print(
    "|Aut(C8)|",
    len(AUT_C8)
)


# ------------------------------------------------------------
# C
# ------------------------------------------------------------

print(
    "\nC. 192-ELEMENT GROUP"
)


# IMPORTANT CORRECTION:
#
# census=False prevents the naive exhaustive subgroup-lattice
# enumeration of the 192-element group.
#
# We still calculate its complete basic fingerprint.

print_summary(

    "H192 = Aut_aff(|Omega|)",

    AFFABS,

    census=False
)


# ------------------------------------------------------------
# D
# ------------------------------------------------------------

print(
    "\nD. G / K INTERSECTIONS"
)


for name, H in [

    ("AUT_AFF", AUT_AFF),

    ("H192", AFFABS),

    ("N±", NPM),

    ("N+", NPLUS),

    ("Aut(C8)", AUT_C8)

]:

    print(

        name,

        "|cap G|",

        len(
            intersection(
                H,
                G
            )
        ),

        "|cap K|",

        len(
            intersection(
                H,
                K
            )
        )
    )


print(

    "Aut(C8) cap H192",

    len(
        intersection(
            AUT_C8,
            AFFABS
        )
    )
)


print(

    "Aut(C8) cap N±",

    len(
        intersection(
            AUT_C8,
            NPM
        )
    )
)


print(

    "Aut(C8) cap N+",

    len(
        intersection(
            AUT_C8,
            NPLUS
        )
    )
)


# ------------------------------------------------------------
# E
# ------------------------------------------------------------

print(
    "\nE. NORMALIZERS / CENTRALIZERS"
)


for name, H in [

    ("AUT_AFF", AUT_AFF),

    ("H192", AFFABS),

    ("N±", NPM),

    ("N+", NPLUS),

    ("Aut(C8)", AUT_C8)

]:

    print(

        name,

        "|N_H(G)|",

        len(
            normalizer(
                H,
                G
            )
        ),

        "|C_H(G)|",

        len(
            centralizer(
                H,
                G
            )
        ),

        "|N_H(K)|",

        len(
            normalizer(
                H,
                K
            )
        ),

        "|C_H(K)|",

        len(
            centralizer(
                H,
                K
            )
        )
    )


# ------------------------------------------------------------
# F
# ------------------------------------------------------------

print(
    "\nF. CONSTRUCTIVE N+ -> S4"
)


# N+ has only 24 elements.
# Full subgroup census remains intentionally enabled here.

print_summary(

    "N+",

    NPLUS,

    census=True
)


subs = all_subgroups(
    NPLUS
)


index4 = [

    A

    for A in subs

    if (
        len(NPLUS)
        //
        len(A)
        ==
        4
    )
]


faithful = []


for A in index4:

    Cs = cosets(
        NPLUS,
        A
    )


    def act_coset(g, C):

        return frozenset(

            compose(
                g,
                x
            )

            for x in C
        )


    ker = frozenset(

        g

        for g in NPLUS

        if all(

            act_coset(
                g,
                C
            )
            ==
            C

            for C in Cs
        )
    )


    image = {

        induced_perm(
            g,
            Cs,
            act_coset
        )

        for g in NPLUS
    }


    if len(ker) == 1:

        faithful.append(

            (
                A,
                Cs,
                ker,
                image
            )
        )


print(
    "index-4 subgroups",
    len(index4)
)


print(
    "faithful 4-coset actions",
    len(faithful)
)


if faithful:

    A, Cs, ker, image = faithful[0]


    print(
        "witness subgroup order",
        len(A)
    )


    print(

        "witness subgroup",

        sorted(
            cycle_string(x)
            for x in A
        )
    )


    print(
        "kernel order",
        len(ker)
    )


    print(
        "image order",
        len(image)
    )


    print(

        "constructive S4 criterion PASS",

        (
            len(NPLUS) == 24

            and

            len(image) == 24

            and

            len(ker) == 1
        )
    )


# ------------------------------------------------------------
# G
# ------------------------------------------------------------

print(
    "\nG. CONSTRUCTIVE N± -> N+ x C2"
)


central_reversers = [

    r

    for r in NMINUS

    if all(
        commutes(
            r,
            g
        )

        for g in NPLUS
    )
]


print(

    "central reversing elements",

    len(
        central_reversers
    )
)


for r in central_reversers:

    print(

        "r",

        cycle_string(r),

        "order",

        order(r)
    )


if central_reversers:

    r = central_reversers[0]


    R = generated_group(
        [r]
    )


    product = frozenset(

        compose(
            g,
            q
        )

        for g in NPLUS
        for q in R
    )


    print(
        "<r> order",
        len(R)
    )


    print(

        "N+ cap <r>",

        len(
            intersection(
                NPLUS,
                R
            )
        )
    )


    print(

        "N+<r> order",

        len(product)
    )


    print(

        "equals N±",

        product == NPM
    )


    print(

        "direct product PASS",

        (
            order(r) == 2

            and

            len(
                intersection(
                    NPLUS,
                    R
                )
            ) == 1

            and

            product == NPM

            and

            all(
                commutes(
                    r,
                    g
                )

                for g in NPLUS
            )
        )
    )


# ------------------------------------------------------------
# H
# ------------------------------------------------------------

print(
    "\nH. ORBIT SIGNATURES"
)


for name, H in [

    ("AUT_AFF", AUT_AFF),

    ("H192", AFFABS),

    ("N±", NPM),

    ("N+", NPLUS),

    ("G", G),

    ("K", K)

]:

    print(

        name,

        orbit_signature(H)
    )


# ------------------------------------------------------------
# I
# ------------------------------------------------------------

print(
    "\nI. EXPLICIT D7 ORBITS"
)


for name, H in [

    ("AUT_AFF", AUT_AFF),

    ("H192", AFFABS),

    ("N±", NPM),

    ("N+", NPLUS),

    ("G", G),

    ("K", K)

]:

    OO = orbits(
        H,
        D7,
        act_direction
    )


    print(

        name,

        [

            [
                bits3(d)
                for d in sorted(O)
            ]

            for O in OO
        ]
    )


# ------------------------------------------------------------
# J
# ------------------------------------------------------------

print(
    "\nJ. D7 <-> C7 EQUIVARIANT BIJECTION"
)


for name, H in [

    ("AUT_AFF", AUT_AFF),

    ("H192", AFFABS),

    ("N±", NPM),

    ("N+", NPLUS),

    ("G", G)

]:

    sols = equivariant_bijections(

        H,

        D7,
        act_direction,

        C7,
        act_class,

        cap=50
    )


    print(

        name,

        "solutions found (cap 50)",

        len(sols)
    )


    if sols:

        f = sols[0]


        print(
            " witness"
        )


        for d in sorted(f):

            C = f[d]


            print(

                " ",

                bits3(d),

                "->",

                [

                    sorted(P)

                    for P in sorted(

                        C,

                        key=lambda P:
                            sorted(P)
                    )
                ]
            )


# ------------------------------------------------------------
# K
# ------------------------------------------------------------

print(
    "\nK. REAL COMMUTANT End_N+(R8)"
)


CB, rank, residual = commutant_basis(
    NPLUS
)


print(
    "linear-system rank",
    rank
)


print(
    "commutant dimension",
    len(CB)
)


print(
    "max commutator residual",
    residual
)


if CB:

    Bmat = np.column_stack(

        [
            A.reshape(-1)
            for A in CB
        ]
    )


    max_mult_res = 0.0


    for A in CB:

        for B in CB:

            target = (
                A @ B
            ).reshape(-1)


            coeff, *_ = np.linalg.lstsq(

                Bmat,

                target,

                rcond=None
            )


            err = np.linalg.norm(

                Bmat @ coeff

                -

                target
            )


            max_mult_res = max(

                max_mult_res,

                err
            )


    print(

        "multiplication closure residual",

        max_mult_res
    )


# ------------------------------------------------------------
# L
# ------------------------------------------------------------

print(
    "\nL. N+ 8D REPRESENTATION DECOMPOSITION"
)


dec, details = decompose_NPLUS()


print(

    "class rows:",

    "(S4-index,size,order,trace,carrier-cycle)"
)


for row in details:

    print(
        " ",
        row
    )


print(
    "decomposition",
    dec
)


if dec:

    m = dec[
        "multiplicities"
    ]


    dimcheck = (

        m["trivial"]

        +

        m["sign"]

        +

        3 * m["std3"]

        +

        3 * m["std3sign"]

        +

        2 * m["two"]
    )


    predicted = sum(

        float(x) ** 2

        for x in m.values()
    )


    print(

        "representation dimension",

        dimcheck
    )


    print(

        "predicted commutant dimension",

        predicted
    )


    print(

        "matches numerical commutant",

        abs(
            predicted
            -
            len(CB)
        )
        <
        1e-8
    )


# ------------------------------------------------------------
# M
# ------------------------------------------------------------

print(
    "\nM. H192 STRUCTURAL TESTS"
)


print(

    "center",

    sorted(

        cycle_string(x)

        for x in center(
            AFFABS
        )
    )
)


print(

    "derived order",

    len(
        derived_subgroup(
            AFFABS
        )
    )
)


print(

    "orbit signature",

    orbit_signature(
        AFFABS
    )
)


print(

    "G normal in H192",

    normalizer(
        AFFABS,
        G
    )
    ==
    AFFABS
)


print(

    "K normal in H192",

    normalizer(
        AFFABS,
        K
    )
    ==
    AFFABS
)


print(

    "N± normal in H192",

    normalizer(
        AFFABS,
        NPM
    )
    ==
    AFFABS
)


print(

    "N+ normal in H192",

    normalizer(
        AFFABS,
        NPLUS
    )
    ==
    AFFABS
)


# ------------------------------------------------------------
# N
# ------------------------------------------------------------

print(
    "\nN. MACHINE TRUTH PACKET"
)


print(

    "|Aut AG(3,2)|",

    len(AUT_AFF)
)


print(

    "|H192|",

    len(AFFABS),

    "AFFABS=AFFS",

    AFFABS == AFFS
)


print(

    "|N±|",

    len(NPM),

    "|N+|",

    len(NPLUS)
)


print(

    "|G|",

    len(G),

    "|K|",

    len(K)
)


print(

    "G intersections",

    {

        "H192":
            len(
                intersection(
                    G,
                    AFFABS
                )
            ),

        "N±":
            len(
                intersection(
                    G,
                    NPM
                )
            ),

        "N+":
            len(
                intersection(
                    G,
                    NPLUS
                )
            )
    }
)


print(

    "H192 signature",

    orbit_signature(
        AFFABS
    )
)


print(

    "N± signature",

    orbit_signature(
        NPM
    )
)


print(

    "N+ signature",

    orbit_signature(
        NPLUS
    )
)


print(

    "commutant dimension N+",

    len(CB)
)


print(
    "No J searched for or imposed."
)


# ============================================================
# 13. HARD SANITY ASSERTIONS
# ============================================================

assert len(K) == 4

assert len(G) == 8


assert len(V7) == 7

assert len(P14) == 14

assert len(C7) == 7


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


assert len(AUT_AFF) == 1344


assert len(AFFABS) == 192


assert len(NPLUS) == 24

assert len(NMINUS) == 24

assert len(NPM) == 48


assert len(AUT_C8) == 16


assert set(
    NPLUS
).issubset(
    NPM
)


assert set(
    NPM
).issubset(
    AFFABS
)


assert set(
    AFFABS
).issubset(
    AUT_AFF
)


assert frozenset(
    translations.values()
) == G


print(
    "\nSIM14.6 COMPLETE — SANITY CHECKS PASS"
)






~~~~~~~~~~~~~~~~~~~~~







Results:


==========================================================================================

SIM14.6 — OPERATIONAL PHASE-SPACE SYMMETRY CLOSURE

==========================================================================================


A. FROZEN SANITY
|K| 4 |G| 8

G regular True

V7 7 P14 14 C7 7

Omega antisymmetric True rank 8 det 1

coordinates {0: '000', 1: '100', 2: '001', 3: '101', 4: '111', 5: '011', 6: '110', 7: '010'}


B. MASTER GROUP ORDERS
|S8| 40320

|Aut AG(3,2)| 1344

|Aut_aff(|Omega|)| 192

|Aut_aff(s)| 192

AFFABS == AFFS True

|N+| 24

|N-| 24

|N±| 48

|Aut(C8)| 16


C. 192-ELEMENT GROUP
H192 = Aut_aff(|Omega|)

  order = 192

  element_orders = {1: 1, 2: 43, 3: 32, 4: 84, 6: 32}

  center_order = 2

  center_orders = {1: 1, 2: 1}

  derived_order = 96

  derived_orders = {1: 1, 2: 19, 3: 32, 4: 12, 6: 32}

  abelianization_order = 2

  conjugacy_class_sizes = [1, 1, 6, 6, 6, 12, 12, 12, 24, 24, 24, 32, 32]


D. G / K INTERSECTIONS
AUT_AFF |cap G| 8 |cap K| 4

H192 |cap G| 8 |cap K| 4

N± |cap G| 8 |cap K| 4

N+ |cap G| 4 |cap K| 2

Aut(C8) |cap G| 4 |cap K| 4

Aut(C8) cap H192 8

Aut(C8) cap N± 4

Aut(C8) cap N+ 2


E. NORMALIZERS / CENTRALIZERS
AUT_AFF |N_H(G)| 1344 |C_H(G)| 8 |N_H(K)| 192 |C_H(K)| 32

H192 |N_H(G)| 192 |C_H(G)| 8 |N_H(K)| 48 |C_H(K)| 8

N± |N_H(G)| 48 |C_H(G)| 8 |N_H(K)| 16 |C_H(K)| 8

N+ |N_H(G)| 24 |C_H(G)| 4 |N_H(K)| 8 |C_H(K)| 4

Aut(C8) |N_H(G)| 8 |C_H(G)| 4 |N_H(K)| 8 |C_H(K)| 4


F. CONSTRUCTIVE N+ -> S4
N+

  order = 24

  element_orders = {1: 1, 2: 9, 3: 8, 4: 6}

  center_order = 1

  center_orders = {1: 1}

  derived_order = 12

  derived_orders = {1: 1, 2: 3, 3: 8}

  abelianization_order = 2

  conjugacy_class_sizes = [1, 3, 6, 6, 8]

  subgroup_orders = {1: 1, 2: 9, 3: 4, 4: 7, 6: 4, 8: 3, 12: 1, 24: 1}

  subgroups_total = 30

  normal_subgroup_orders = {1: 1, 4: 1, 12: 1, 24: 1}

  normal_subgroups_total = 4

index-4 subgroups 4

faithful 4-coset actions 4

witness subgroup order 6

witness subgroup ['()', '(1 4 5)(3 6 7)', '(1 4)(3 6)', '(1 5 4)(3 7 6)', '(1 5)(3 7)', '(4 5)(6 7)']

kernel order 1

image order 24

constructive S4 criterion PASS True


G. CONSTRUCTIVE N± -> N+ x C2
central reversing elements 1

r (0 2)(1 3)(4 6)(5 7) order 2

<r> order 2

N+ cap <r> 1

N+<r> order 48

equals N± True

direct product PASS True


H. ORBIT SIGNATURES
AUT_AFF {'X8': [8], 'D7': [7], 'V7': [7], 'P14': [14], 'C7': [7]}

H192 {'X8': [8], 'D7': [1, 6], 'V7': [3, 4], 'P14': [6, 8], 'C7': [3, 4]}

N± {'X8': [8], 'D7': [1, 3, 3], 'V7': [1, 3, 3], 'P14': [2, 6, 6], 'C7': [1, 3, 3]}

N+ {'X8': [4, 4], 'D7': [1, 3, 3], 'V7': [1, 3, 3], 'P14': [1, 1, 6, 6], 'C7': [1, 3, 3]}

G {'X8': [8], 'D7': [1, 1, 1, 1, 1, 1, 1], 'V7': [1, 1, 1, 1, 1, 1, 1], 'P14': [2, 2, 2, 2, 2, 2, 2], 'C7': [1, 1, 1, 1, 1, 1, 1]}

K {'X8': [4, 4], 'D7': [1, 1, 1, 1, 1, 1, 1], 'V7': [1, 1, 1, 1, 1, 1, 1], 'P14': [1, 1, 2, 2, 2, 2, 2, 2], 'C7': [1, 1, 1, 1, 1, 1, 1]}


I. EXPLICIT D7 ORBITS
AUT_AFF [['001', '010', '011', '100', '101', '110', '111']]

H192 [['001'], ['010', '011', '100', '101', '110', '111']]

N± [['001'], ['010', '101', '110'], ['011', '100', '111']]

N+ [['001'], ['010', '101', '110'], ['011', '100', '111']]

G [['001'], ['010'], ['011'], ['100'], ['101'], ['110'], ['111']]

K [['001'], ['010'], ['011'], ['100'], ['101'], ['110'], ['111']]


J. D7 <-> C7 EQUIVARIANT BIJECTION
AUT_AFF solutions found (cap 50) 0

H192 solutions found (cap 50) 0

N± solutions found (cap 50) 2

 witness

  001 -> [[0, 1, 4, 5], [2, 3, 6, 7]]

  010 -> [[0, 2, 5, 7], [1, 3, 4, 6]]

  011 -> [[0, 3, 5, 6], [1, 2, 4, 7]]

  100 -> [[0, 1, 6, 7], [2, 3, 4, 5]]

  101 -> [[0, 1, 2, 3], [4, 5, 6, 7]]

  110 -> [[0, 2, 4, 6], [1, 3, 5, 7]]

  111 -> [[0, 3, 4, 7], [1, 2, 5, 6]]

N+ solutions found (cap 50) 2

 witness

  001 -> [[0, 1, 4, 5], [2, 3, 6, 7]]

  010 -> [[0, 2, 5, 7], [1, 3, 4, 6]]

  011 -> [[0, 3, 5, 6], [1, 2, 4, 7]]

  100 -> [[0, 1, 6, 7], [2, 3, 4, 5]]

  101 -> [[0, 1, 2, 3], [4, 5, 6, 7]]

  110 -> [[0, 2, 4, 6], [1, 3, 5, 7]]

  111 -> [[0, 3, 4, 7], [1, 2, 5, 6]]

G solutions found (cap 50) 50

 witness

  001 -> [[0, 1, 2, 3], [4, 5, 6, 7]]

  010 -> [[0, 1, 6, 7], [2, 3, 4, 5]]

  011 -> [[0, 2, 4, 6], [1, 3, 5, 7]]

  100 -> [[0, 3, 4, 7], [1, 2, 5, 6]]

  101 -> [[0, 3, 5, 6], [1, 2, 4, 7]]

  110 -> [[0, 2, 5, 7], [1, 3, 4, 6]]

  111 -> [[0, 1, 4, 5], [2, 3, 6, 7]]


K. REAL COMMUTANT End_N+(R8)
linear-system rank 56

commutant dimension 8

max commutator residual 2.867440710767085e-15

multiplication closure residual 1.171626331461472e-15


L. N+ 8D REPRESENTATION DECOMPOSITION
class rows: (S4-index,size,order,trace,carrier-cycle)

  (0, 1, 1, 8, '()')

  (1, 6, 2, 4, '(1 5)(3 7)')

  (2, 3, 2, 0, '(0 5)(1 4)(2 7)(3 6)')

  (3, 8, 3, 2, '(1 4 5)(3 6 7)')

  (4, 6, 4, 0, '(0 1 4 5)(2 3 6 7)')

decomposition {'character': [8, 4, 0, 2, 0], 'multiplicities': {'trivial': 2.0, 'sign': 0.0, 'std3': 2.0, 'std3sign': 0.0, 'two': 0.0}}

representation dimension 8.0

predicted commutant dimension 8.0

matches numerical commutant True


M. H192 STRUCTURAL TESTS
center ['()', '(0 2)(1 3)(4 6)(5 7)']

derived order 96

orbit signature {'X8': [8], 'D7': [1, 6], 'V7': [3, 4], 'P14': [6, 8], 'C7': [3, 4]}

G normal in H192 True

K normal in H192 False

N± normal in H192 False

N+ normal in H192 False


N. MACHINE TRUTH PACKET
|Aut AG(3,2)| 1344

|H192| 192 AFFABS=AFFS True

|N±| 48 |N+| 24

|G| 8 |K| 4

G intersections {'H192': 8, 'N±': 8, 'N+': 4}

H192 signature {'X8': [8], 'D7': [1, 6], 'V7': [3, 4], 'P14': [6, 8], 'C7': [3, 4]}

N± signature {'X8': [8], 'D7': [1, 3, 3], 'V7': [1, 3, 3], 'P14': [2, 6, 6], 'C7': [1, 3, 3]}

N+ signature {'X8': [4, 4], 'D7': [1, 3, 3], 'V7': [1, 3, 3], 'P14': [1, 1, 6, 6], 'C7': [1, 3, 3]}

commutant dimension N+ 8

No J searched for or imposed.


SIM14.6 COMPLETE — SANITY CHECKS PASS
