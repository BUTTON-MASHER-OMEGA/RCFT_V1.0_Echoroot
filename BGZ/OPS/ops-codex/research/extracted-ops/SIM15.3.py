#!/usr/bin/env python3
"""
SIM15.3 — KERNEL–ARCHITECTURE INCIDENCE AND TRIALITY-FIBER MAP
===============================================================

CENTRAL QUESTION
----------------
Can the rank-3 geometry on the 25 compatible W(D4)-type architectures
be reconstructed intrinsically from the successful C2^3 kernels and
their incidence/intersection structure?

FROZEN INPUT FROM SIM15.0–15.2
------------------------------
We independently reconstruct W(H4), then rediscover:

    T ~= C2^3
    L = N_W(T), |L| = 192
    N = N_W(L), |N| = 576

with the successful T-class defined by the full SIM14/15 fingerprint.

SIM15.2 found:
    L ~= W(D4)
    N/L ~= C3 triality
    |conjugacy class of L| = 25
    L_25 rank = 3
    subdegrees = [1,8,16]
    |Li ∩ Lj| in {8,12}

NO assumption is made here that:
    - the successful T-family has size 25 or 75;
    - every T lies in one L;
    - every L contains three T's;
    - containment forms a (25_3) configuration;
    - the 75 kernels split into triality triples;
    - the rank-3 graph is reconstructible from T-data;
    - the 25-set is F5^2;
    - the 25-set is a 5x5 lattice;
    - C5 pentads are rows, columns, blocks, etc.

Everything below is enumerated.

HARD GATES
----------
A. Enumerate the complete successful T-family and compatible L-family,
   then construct the native containment matrix

       B[T,L] = 1 iff T <= L.

B. Determine the actual bipartite incidence geometry:
       sizes,
       degree distributions,
       total incidences,
       connected components,
       BB^T,
       B^TB,
       W(H4) equivariance.

C. If containment defines fibers T -> L, determine whether those fibers
   are exactly the local triality triples and whether N/L ~= C3 rotates
   them.

D. For each pair Li,Lj, construct the cross-fiber matrix

       M_ij[a,b] = |T_ia ∩ T_jb|,

   canonically classified up to independent row/column permutations.

E. Test whether those intrinsic matrix types reconstruct exactly the
   SIM15.2 rank-3 relation

       |Li ∩ Lj| = 8  versus  12.

F. Compare representative C3 and C5 actions on:
       T-family,
       L-family,
       incidence relation,
       triality fibers,
       cross-fiber relation types.

DEPENDENCIES
------------
Python 3
numpy

No GAP / Sage required.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict, deque

import numpy as np


# ============================================================
# 0. PERMUTATION UTILITIES
# ============================================================

def pid(n):
    return tuple(range(n))


def pmul(a, b):
    """Composition a o b."""
    return tuple(a[b[i]] for i in range(len(a)))


def pinv(a):
    out = [0] * len(a)
    for i, j in enumerate(a):
        out[j] = i
    return tuple(out)


def ppow(a, k):
    r = pid(len(a))
    x = a

    while k:
        if k & 1:
            r = pmul(r, x)
        x = pmul(x, x)
        k >>= 1

    return r


def pconj(g, x):
    return pmul(pmul(g, x), pinv(g))


def porder(p):
    seen = [False] * len(p)
    ans = 1

    for i in range(len(p)):
        if seen[i]:
            continue

        j = i
        L = 0

        while not seen[j]:
            seen[j] = True
            j = p[j]
            L += 1

        if L:
            ans = math.lcm(ans, L)

    return ans


def pcycles(p, include_fixed=False):
    seen = [False] * len(p)
    out = []

    for i in range(len(p)):
        if seen[i]:
            continue

        c = []
        j = i

        while not seen[j]:
            seen[j] = True
            c.append(j)
            j = p[j]

        if include_fixed or len(c) > 1:
            out.append(tuple(c))

    return tuple(out)


def cycle_signature(p):
    return dict(
        sorted(
            Counter(
                len(c) for c in pcycles(p, True)
            ).items()
        )
    )


def generated_group(gens, n=None):
    if not gens:
        if n is None:
            raise ValueError("Need n for empty generator list.")
        return {pid(n)}

    n = len(gens[0])
    e = pid(n)

    G = {e}
    q = deque([e])

    while q:
        x = q.popleft()

        for g in gens:
            y = pmul(g, x)

            if y not in G:
                G.add(y)
                q.append(y)

    return G


def subgroup_generated(gens, n):
    return generated_group(list(gens), n=n)


def greedy_generators(G):
    G = set(G)
    n = len(next(iter(G)))
    e = pid(n)

    gens = []
    H = {e}

    while H != G:
        x = next(g for g in G if g not in H)
        gens.append(x)
        H = subgroup_generated(gens, n)

    return gens


def center(G):
    G = set(G)

    return {
        z for z in G
        if all(
            pmul(z, g) == pmul(g, z)
            for g in G
        )
    }


def centralizer_in(W, H):
    H = set(H)

    return {
        g for g in W
        if all(
            pmul(g, h) == pmul(h, g)
            for h in H
        )
    }


def normalizer_in(W, H):
    H = set(H)

    return {
        g for g in W
        if {
            pconj(g, h) for h in H
        } == H
    }


def is_normal(H, G):
    H = set(H)

    return all(
        {
            pconj(g, h) for h in H
        } == H
        for g in G
    )


def order_hist(G):
    return dict(
        sorted(
            Counter(
                porder(g) for g in G
            ).items()
        )
    )


def conjugate_subgroup(g, H):
    return frozenset(
        pconj(g, h) for h in H
    )


def conjugacy_orbit_subgroup(W, H):
    return {
        conjugate_subgroup(w, H)
        for w in W
    }


def point_orbits(perms, n):
    unseen = set(range(n))
    out = []

    while unseen:
        x = min(unseen)

        O = {
            p[x]
            for p in perms
        }

        # If perms is a group, this is the orbit.
        out.append(O)
        unseen -= O

    return sorted(
        out,
        key=lambda O: (len(O), sorted(O))
    )


# ============================================================
# 1. H4 ROOT SYSTEM
# ============================================================

PHI = (1.0 + math.sqrt(5.0)) / 2.0
IPHI = 1.0 / PHI
SQRT2 = math.sqrt(2.0)
TOL = 1e-8


def permutation_parity(p):
    inv = 0

    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] > p[j]:
                inv += 1

    return inv & 1


def unique_vectors(vectors, decimals=10):
    seen = {}

    for v in vectors:
        key = tuple(np.round(v, decimals))
        seen[key] = np.asarray(v, dtype=float)

    return np.array(
        list(seen.values()),
        dtype=float
    )


def build_h4_roots():
    roots = []

    # 8 coordinate roots.
    for i in range(4):
        for s in (-1.0, 1.0):
            v = np.zeros(4)
            v[i] = 2.0 * s
            roots.append(v / SQRT2)

    # 16 hypercube roots.
    for signs in itertools.product(
        (-1.0, 1.0), repeat=4
    ):
        roots.append(
            np.array(signs, dtype=float) / SQRT2
        )

    # 96 golden roots.
    base = (0.0, 1.0, PHI, IPHI)

    even_perms = [
        p for p in itertools.permutations(range(4))
        if permutation_parity(p) == 0
    ]

    for p in even_perms:
        vals = np.array(
            [base[p[i]] for i in range(4)],
            dtype=float
        )

        nz = [
            i for i, x in enumerate(vals)
            if abs(x) > TOL
        ]

        for signs in itertools.product(
            (-1.0, 1.0), repeat=3
        ):
            v = vals.copy()

            for j, s in zip(nz, signs):
                v[j] *= s

            roots.append(v / SQRT2)

    R = unique_vectors(roots)

    if len(R) != 120:
        raise RuntimeError(
            f"Expected 120 H4 roots, got {len(R)}."
        )

    norms = np.sum(R * R, axis=1)

    if not np.allclose(
        norms, 2.0, atol=1e-8
    ):
        raise RuntimeError(
            "H4 roots do not all have norm^2=2."
        )

    return R


def vec_key(v, decimals=9):
    return tuple(np.round(v, decimals))


def build_root_lookup(R):
    return {
        vec_key(v): i
        for i, v in enumerate(R)
    }


def reflection_matrix(alpha):
    alpha = np.asarray(alpha, dtype=float)

    # alpha.alpha = 2.
    return np.eye(4) - np.outer(alpha, alpha)


def linear_map_to_perm(M, R, lookup):
    p = []

    for v in R:
        w = M @ v
        key = vec_key(w)

        if key in lookup:
            p.append(lookup[key])
            continue

        d = np.linalg.norm(
            R - w[None, :],
            axis=1
        )

        j = int(np.argmin(d))

        if d[j] > 1e-6:
            raise RuntimeError(
                "Linear map failed to permute roots."
            )

        p.append(j)

    return tuple(p)


def find_simple_h4_roots(R):
    dots = R @ R.T

    def inds(i, target):
        return [
            j for j in range(len(R))
            if j != i
            and abs(dots[i, j] - target) < 1e-7
        ]

    a1 = 0

    for a2 in inds(a1, -PHI):

        cand3 = [
            a3 for a3 in inds(a2, -1.0)
            if abs(dots[a1, a3]) < 1e-7
        ]

        for a3 in cand3:

            cand4 = [
                a4 for a4 in inds(a3, -1.0)
                if abs(dots[a2, a4]) < 1e-7
                and abs(dots[a1, a4]) < 1e-7
            ]

            if cand4:
                return [
                    a1,
                    a2,
                    a3,
                    cand4[0]
                ]

    raise RuntimeError(
        "Could not find H4 simple roots."
    )


def build_W_H4(R):
    lookup = build_root_lookup(R)

    simple_idx = find_simple_h4_roots(R)

    refl_perms = [
        linear_map_to_perm(
            reflection_matrix(R[i]),
            R,
            lookup
        )
        for i in simple_idx
    ]

    W = generated_group(refl_perms)

    if len(W) != 14400:
        raise RuntimeError(
            f"|W(H4)|={len(W)}, expected 14400."
        )

    return W, refl_perms, simple_idx


# ============================================================
# 2. ENUMERATE C2^3 SUBGROUPS
# ============================================================

def commute(a, b):
    return pmul(a, b) == pmul(b, a)


def enumerate_C2_3_subgroups(W):
    Wlist = list(W)
    e = pid(len(Wlist[0]))

    invol = [
        g for g in Wlist
        if g != e and porder(g) == 2
    ]

    print(
        "involutions in W(H4):",
        len(invol)
    )

    groups = set()

    for ia, a in enumerate(invol):

        for ib in range(ia + 1, len(invol)):
            b = invol[ib]

            if not commute(a, b):
                continue

            H4 = subgroup_generated(
                [a, b],
                len(a)
            )

            if len(H4) != 4:
                continue

            for c in invol:

                if c in H4:
                    continue

                if (
                    not commute(a, c)
                    or not commute(b, c)
                ):
                    continue

                H8 = subgroup_generated(
                    [a, b, c],
                    len(a)
                )

                if (
                    len(H8) == 8
                    and all(
                        porder(x) in (1, 2)
                        for x in H8
                    )
                ):
                    groups.add(
                        frozenset(H8)
                    )

    return [
        set(H)
        for H in groups
    ]


def subgroup_conjugacy_classes(W, subs):
    unseen = {
        frozenset(H)
        for H in subs
    }

    classes = []

    while unseen:

        H = next(iter(unseen))

        orb = {
            conjugate_subgroup(g, H)
            for g in W
        }

        cls = orb & unseen

        classes.append(cls)
        unseen -= orb

    return classes


# ============================================================
# 3. F2^3 FINGERPRINT
# ============================================================

F2V = [
    (0,0,0),
    (0,0,1),
    (0,1,0),
    (0,1,1),
    (1,0,0),
    (1,0,1),
    (1,1,0),
    (1,1,1),
]

I3 = (
    (1,0,0),
    (0,1,0),
    (0,0,1),
)


def choose_T_basis(T):
    e = pid(len(next(iter(T))))

    nonzero = [
        x for x in T
        if x != e
    ]

    for a, b, c in itertools.combinations(
        nonzero, 3
    ):
        H = subgroup_generated(
            [a, b, c],
            len(a)
        )

        if H == set(T):
            return a, b, c

    raise RuntimeError(
        "Could not choose T basis."
    )


def T_coordinate_map(T):
    e = pid(len(next(iter(T))))
    a, b, c = choose_T_basis(T)

    coord_to_elem = {}
    elem_to_coord = {}

    for v in F2V:
        x = e

        if v[0]:
            x = pmul(a, x)

        if v[1]:
            x = pmul(b, x)

        if v[2]:
            x = pmul(c, x)

        coord_to_elem[v] = x
        elem_to_coord[x] = v

    return elem_to_coord, coord_to_elem


def mat_apply(M, v):
    return tuple(
        sum(
            M[i][j] * v[j]
            for j in range(3)
        ) % 2
        for i in range(3)
    )


def mat_mul(A, B):
    return tuple(
        tuple(
            sum(
                A[i][k] * B[k][j]
                for k in range(3)
            ) % 2
            for j in range(3)
        )
        for i in range(3)
    )


def mat_order(M):
    x = I3

    for k in range(1, 100):
        x = mat_mul(M, x)

        if x == I3:
            return k

    raise RuntimeError(
        "Matrix order failure."
    )


def mat_from_action_on_T(
    g,
    elem_to_coord,
    coord_to_elem
):
    basis = [
        (1,0,0),
        (0,1,0),
        (0,0,1),
    ]

    cols = []

    for v in basis:
        x = coord_to_elem[v]
        y = pconj(g, x)
        cols.append(
            elem_to_coord[y]
        )

    return tuple(
        tuple(
            cols[j][i]
            for j in range(3)
        )
        for i in range(3)
    )


TARGET_GL_HIST = {
    1: 1,
    2: 9,
    3: 8,
    4: 6,
}


def full_fingerprint(W, T):
    N = normalizer_in(W, T)
    C = centralizer_in(W, T)

    elem_to_coord, coord_to_elem = \
        T_coordinate_map(T)

    image = {
        mat_from_action_on_T(
            g,
            elem_to_coord,
            coord_to_elem
        )
        for g in N
    }

    fixed = [
        v for v in F2V
        if all(
            mat_apply(M, v) == v
            for M in image
        )
    ]

    gl_hist = dict(
        sorted(
            Counter(
                mat_order(M)
                for M in image
            ).items()
        )
    )

    zN = center(N)
    e = pid(len(next(iter(T))))

    z = None

    if len(zN) == 2:
        z = next(
            x for x in zN
            if x != e
        )

    nonzero_fixed = [
        v for v in fixed
        if v != (0,0,0)
    ]

    center_fixed = False

    if (
        z is not None
        and len(nonzero_fixed) == 1
    ):
        center_fixed = (
            z ==
            coord_to_elem[
                nonzero_fixed[0]
            ]
        )

    checks = {
        "T8": len(T) == 8,
        "N192": len(N) == 192,
        "centralizer_T": C == set(T),
        "GL24": len(image) == 24,
        "GL_S4_hist":
            gl_hist == TARGET_GL_HIST,
        "one_nonzero_fixed":
            len(nonzero_fixed) == 1,
        "center2":
            len(zN) == 2,
        "center_fixed":
            center_fixed,
    }

    return {
        "full": all(checks.values()),
        "N": N,
        "C": C,
        "image": image,
        "fixed": fixed,
        "z": z,
        "checks": checks,
    }


# ============================================================
# 4. FIND SUCCESSFUL CLASS
# ============================================================

def find_successful_structure(W):
    Tsubs = enumerate_C2_3_subgroups(W)

    print(
        "C2^3 subgroups:",
        len(Tsubs)
    )

    classes = subgroup_conjugacy_classes(
        W,
        Tsubs
    )

    print(
        "C2^3 conjugacy classes:",
        len(classes)
    )

    print(
        "class sizes:",
        sorted(len(C) for C in classes)
    )

    successful = []

    for ci, C in enumerate(classes):

        T = set(next(iter(C)))
        L = normalizer_in(W, T)

        print(
            f"class {ci}: "
            f"orbit={len(C)} "
            f"normalizer={len(L)}"
        )

        if len(L) != 192:
            continue

        fp = full_fingerprint(W, T)

        print(
            "  fingerprint:",
            fp["checks"],
            "FULL=",
            fp["full"]
        )

        if fp["full"]:
            successful.append(
                (ci, C, T, fp)
            )

    if len(successful) != 1:
        raise RuntimeError(
            "Expected exactly one successful "
            f"C2^3 class; found {len(successful)}."
        )

    ci, successful_class, T, fp = \
        successful[0]

    L = fp["N"]
    z = fp["z"]
    N = normalizer_in(W, L)

    return {
        "class_id": ci,
        "successful_class":
            successful_class,
        "T": T,
        "L": L,
        "N": N,
        "z": z,
    }


# ============================================================
# 5. C3 COMPLEMENT
# ============================================================

def find_C3_complement(N, L):
    e = pid(len(next(iter(N))))

    candidates = [
        g for g in N
        if g not in L
        and porder(g) == 3
    ]

    witnesses = []

    Lgens = greedy_generators(L)

    for c in candidates:

        C3 = subgroup_generated(
            [c],
            len(c)
        )

        if C3 & set(L) != {e}:
            continue

        H = subgroup_generated(
            Lgens + [c],
            len(c)
        )

        if H == set(N):
            witnesses.append(c)

    return witnesses


# ============================================================
# 6. FAMILY ACTIONS
# ============================================================

def action_on_subgroup_family(
    g,
    family,
    index
):
    return tuple(
        index[
            conjugate_subgroup(g, H)
        ]
        for H in family
    )


def build_family_action(W, family):
    family = list(family)

    index = {
        H: i
        for i, H in enumerate(family)
    }

    action = {
        g: action_on_subgroup_family(
            g,
            family,
            index
        )
        for g in W
    }

    return family, index, action


def action_kernel(action):
    n = len(
        next(iter(action.values()))
    )

    e = pid(n)

    return {
        g for g, p in action.items()
        if p == e
    }


# ============================================================
# 7. INCIDENCE MATRIX
# ============================================================

def containment_matrix(Tfamily, Lfamily):
    B = np.zeros(
        (
            len(Tfamily),
            len(Lfamily)
        ),
        dtype=int
    )

    for i, T in enumerate(Tfamily):
        Ts = set(T)

        for j, L in enumerate(Lfamily):
            if Ts <= set(L):
                B[i, j] = 1

    return B


def incidence_edges(B):
    rows, cols = B.shape

    return [
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if B[i, j]
    ]


def bipartite_components(B):
    nT, nL = B.shape
    n = nT + nL

    adj = [
        set()
        for _ in range(n)
    ]

    for i, j in incidence_edges(B):
        a = i
        b = nT + j

        adj[a].add(b)
        adj[b].add(a)

    unseen = set(range(n))
    comps = []

    while unseen:
        s = min(unseen)

        C = {s}
        q = deque([s])

        while q:
            x = q.popleft()

            for y in adj[x]:
                if y not in C:
                    C.add(y)
                    q.append(y)

        comps.append(C)
        unseen -= C

    return sorted(
        comps,
        key=lambda C:
            (len(C), sorted(C))
    )


def incidence_spectrum(B):
    """
    Spectrum of Levi adjacency matrix:
        [0 B]
        [B^T 0]
    """
    nT, nL = B.shape

    A = np.zeros(
        (nT + nL, nT + nL),
        dtype=float
    )

    A[:nT, nT:] = B
    A[nT:, :nT] = B.T

    vals = np.linalg.eigvalsh(A)
    vals = np.round(vals, 9)

    return [
        (float(k), int(v))
        for k, v in sorted(
            Counter(vals).items()
        )
    ]


# ============================================================
# 8. INCIDENCE EQUIVARIANCE
# ============================================================

def verify_incidence_equivariance(
    B,
    W,
    actionT,
    actionL
):
    nT, nL = B.shape

    for g in W:
        pT = actionT[g]
        pL = actionL[g]

        for i in range(nT):
            for j in range(nL):

                if (
                    B[i, j]
                    != B[pT[i], pL[j]]
                ):
                    return False, (
                        g, i, j,
                        pT[i], pL[j]
                    )

    return True, None


# ============================================================
# 9. FIBERS
# ============================================================

def fibers_from_B(B):
    """
    For each L_j return T indices incident with it.
    """
    return [
        tuple(
            int(i)
            for i in np.flatnonzero(
                B[:, j]
            )
        )
        for j in range(B.shape[1])
    ]


def containing_Ls_from_B(B):
    """
    For each T_i return L indices containing it.
    """
    return [
        tuple(
            int(j)
            for j in np.flatnonzero(
                B[i, :]
            )
        )
        for i in range(B.shape[0])
    ]


# ============================================================
# 10. TRIALITY ACTION ON FIBERS
# ============================================================

def restricted_action_on_subset(
    p,
    subset
):
    subset = tuple(subset)

    idx = {
        x: i
        for i, x in enumerate(subset)
    }

    image = [
        p[x]
        for x in subset
    ]

    if set(image) != set(subset):
        return None

    return tuple(
        idx[y]
        for y in image
    )


def fiber_action_signature(
    g,
    actionT,
    actionL,
    fibers
):
    pT = actionT[g]
    pL = actionL[g]

    data = []

    for j, F in enumerate(fibers):
        j2 = pL[j]
        F2 = fibers[j2]

        image = tuple(
            pT[i]
            for i in F
        )

        data.append(
            (
                j,
                j2,
                tuple(sorted(image))
                    == tuple(sorted(F2))
            )
        )

    return data


# ============================================================
# 11. CROSS-FIBER INTERSECTION MATRICES
# ============================================================

def subgroup_intersection_size(H, K):
    return len(
        set(H) & set(K)
    )


def cross_fiber_matrix(
    Fi,
    Fj,
    Tfamily
):
    return tuple(
        tuple(
            subgroup_intersection_size(
                Tfamily[a],
                Tfamily[b]
            )
            for b in Fj
        )
        for a in Fi
    )


def canonical_matrix_rowcol(M):
    """
    Canonicalize a small integer matrix under independent
    row and column permutations.

    This intentionally forgets arbitrary labels inside the
    triality fibers.
    """
    nr = len(M)
    nc = len(M[0])

    best = None

    for rp in itertools.permutations(
        range(nr)
    ):
        for cp in itertools.permutations(
            range(nc)
        ):
            X = tuple(
                tuple(
                    M[rp[i]][cp[j]]
                    for j in range(nc)
                )
                for i in range(nr)
            )

            if best is None or X < best:
                best = X

    return best


def matrix_pretty(M):
    return "[" + "; ".join(
        " ".join(map(str, row))
        for row in M
    ) + "]"


# ============================================================
# 12. L-PAIR DATA
# ============================================================

def classify_L_pairs(
    Lfamily,
    fibers,
    Tfamily
):
    """
    For every unordered Li,Lj compute:
      - |Li ∩ Lj|
      - raw 3x3 T-intersection matrix
      - canonical row/column type
    """
    records = []

    for i in range(len(Lfamily)):
        for j in range(i + 1, len(Lfamily)):

            Lint = subgroup_intersection_size(
                Lfamily[i],
                Lfamily[j]
            )

            M = cross_fiber_matrix(
                fibers[i],
                fibers[j],
                Tfamily
            )

            C = canonical_matrix_rowcol(M)

            records.append({
                "i": i,
                "j": j,
                "L_intersection": Lint,
                "matrix": M,
                "canonical": C,
            })

    return records


# ============================================================
# 13. RECONSTRUCTION TEST
# ============================================================

def reconstruction_report(records):
    """
    Determine:
      canonical T-matrix type -> L-intersection types
      L-intersection type -> canonical T-matrix types
    """
    matrix_to_L = defaultdict(Counter)
    L_to_matrix = defaultdict(Counter)

    for r in records:
        C = r["canonical"]
        s = r["L_intersection"]

        matrix_to_L[C][s] += 1
        L_to_matrix[s][C] += 1

    return matrix_to_L, L_to_matrix


# ============================================================
# 14. BB^T / B^TB SUMMARIES
# ============================================================

def offdiag_hist(M):
    n = M.shape[0]
    C = Counter()

    for i in range(n):
        for j in range(i + 1, n):
            C[int(M[i, j])] += 1

    return dict(sorted(C.items()))


def diagonal_hist(M):
    return dict(
        sorted(
            Counter(
                int(M[i, i])
                for i in range(M.shape[0])
            ).items()
        )
    )


# ============================================================
# 15. PAIR ORBITS UNDER W
# ============================================================

def pair_orbit_partition(
    action,
    n
):
    """
    W-orbits on unordered pairs of n points.
    """
    perms = set(action.values())

    unseen = {
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
    }

    orbits = []

    while unseen:
        pair = next(iter(unseen))
        a, b = pair

        O = set()

        for p in perms:
            x, y = p[a], p[b]

            if x > y:
                x, y = y, x

            O.add((x, y))

        orbits.append(O)
        unseen -= O

    return sorted(
        orbits,
        key=lambda O:
            (len(O), sorted(O))
    )


# ============================================================
# 16. C5
# ============================================================

def element_conjugacy_class(W, x):
    return {
        pconj(g, x)
        for g in W
    }


def conjugacy_classes_selected(
    W,
    elements
):
    unseen = set(elements)
    classes = []

    while unseen:
        x = next(iter(unseen))

        C = (
            element_conjugacy_class(W, x)
            & unseen
        )

        classes.append(C)
        unseen -= C

    return classes


# ============================================================
# 17. OPTIONAL GRAPH FROM RECONSTRUCTED RELATION
# ============================================================

def graph_signature_from_edges(
    n,
    edges
):
    adj = [
        set()
        for _ in range(n)
    ]

    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)

    degs = [
        len(A)
        for A in adj
    ]

    A = np.zeros(
        (n, n),
        dtype=float
    )

    for i in range(n):
        for j in adj[i]:
            A[i, j] = 1.0

    vals = np.linalg.eigvalsh(A)
    vals = np.round(vals, 9)

    spectrum = [
        (float(k), int(v))
        for k, v in sorted(
            Counter(vals).items()
        )
    ]

    common = Counter()

    triangles = 0

    for i in range(n):
        for j in range(i + 1, n):

            cn = len(
                adj[i] & adj[j]
            )

            common[
                (
                    "adj"
                    if j in adj[i]
                    else "nonadj",
                    cn
                )
            ] += 1

            if j in adj[i]:
                triangles += cn

    triangles //= 3

    return {
        "degree_hist":
            dict(
                sorted(
                    Counter(degs).items()
                )
            ),
        "edges":
            sum(degs) // 2,
        "triangles":
            triangles,
        "spectrum":
            spectrum,
        "common_neighbor_hist":
            dict(
                sorted(
                    common.items(),
                    key=repr
                )
            ),
    }


# ============================================================
# 18. MAIN
# ============================================================

def main():

    print("=" * 84)
    print(
        "SIM15.3 — KERNEL–ARCHITECTURE INCIDENCE "
        "AND TRIALITY-FIBER MAP"
    )
    print("=" * 84)

    # --------------------------------------------------------
    # A. INDEPENDENT H4 RECONSTRUCTION
    # --------------------------------------------------------

    print(
        "\nA) INDEPENDENT H4 REGRESSION"
    )
    print("-" * 84)

    R = build_h4_roots()

    W, Wgens, simple_idx = \
        build_W_H4(R)

    print("H4 roots =", len(R))
    print("|W(H4)| =", len(W))
    print(
        "H4 simple-root indices =",
        simple_idx
    )

    # --------------------------------------------------------
    # B. REDISCOVER SUCCESSFUL CLASS
    # --------------------------------------------------------

    print(
        "\nB) REDISCOVER SUCCESSFUL C2^3 CLASS"
    )
    print("-" * 84)

    S = find_successful_structure(W)

    T0 = S["T"]
    L0 = S["L"]
    N0 = S["N"]
    z = S["z"]

    successful_class = \
        S["successful_class"]

    print("\nFROZEN CHAIN")
    print("|T0| =", len(T0))
    print("|L0| =", len(L0))
    print("|N0| =", len(N0))
    print("|W| =", len(W))

    print(
        "[L0:T0] =",
        len(L0) // len(T0)
    )

    print(
        "[N0:L0] =",
        len(N0) // len(L0)
    )

    print(
        "[W:N0] =",
        len(W) // len(N0)
    )

    # --------------------------------------------------------
    # C. COMPLETE T AND L FAMILIES
    # --------------------------------------------------------

    print(
        "\nC) COMPLETE SUCCESSFUL T / COMPATIBLE L FAMILIES"
    )
    print("-" * 84)

    Tfamily_set = set(
        successful_class
    )

    Lfamily_set = \
        conjugacy_orbit_subgroup(
            W,
            L0
        )

    print(
        "|successful T family| =",
        len(Tfamily_set)
    )

    print(
        "|compatible L family| =",
        len(Lfamily_set)
    )

    # Freeze representatives first for readable output.
    T0f = frozenset(T0)
    L0f = frozenset(L0)

    Trest = [
        H for H in Tfamily_set
        if H != T0f
    ]

    Lrest = [
        H for H in Lfamily_set
        if H != L0f
    ]

    # Sorting frozensets of permutation tuples is valid.
    Trest.sort(
        key=lambda H:
            tuple(sorted(H))
    )

    Lrest.sort(
        key=lambda H:
            tuple(sorted(H))
    )

    Tfamily = [
        T0f
    ] + Trest

    Lfamily = [
        L0f
    ] + Lrest

    Tindex = {
        H: i
        for i, H in enumerate(Tfamily)
    }

    Lindex = {
        H: i
        for i, H in enumerate(Lfamily)
    }

    print("chosen T0 index =", 0)
    print("chosen L0 index =", 0)

    # --------------------------------------------------------
    # D. W ACTIONS ON BOTH FAMILIES
    # --------------------------------------------------------

    print(
        "\nD) W(H4) ACTIONS ON T AND L FAMILIES"
    )
    print("-" * 84)

    print(
        "building action on successful T family..."
    )

    actionT = {
        g: action_on_subgroup_family(
            g,
            Tfamily,
            Tindex
        )
        for g in W
    }

    print(
        "building action on compatible L family..."
    )

    actionL = {
        g: action_on_subgroup_family(
            g,
            Lfamily,
            Lindex
        )
        for g in W
    }

    imageT = set(
        actionT.values()
    )

    imageL = set(
        actionL.values()
    )

    kernelT = action_kernel(
        actionT
    )

    kernelL = action_kernel(
        actionL
    )

    print(
        "|T-action image| =",
        len(imageT)
    )

    print(
        "|T-action kernel| =",
        len(kernelT)
    )

    print(
        "|L-action image| =",
        len(imageL)
    )

    print(
        "|L-action kernel| =",
        len(kernelL)
    )

    print(
        "z in T-action kernel =",
        z in kernelT
    )

    print(
        "z in L-action kernel =",
        z in kernelL
    )

    # --------------------------------------------------------
    # E. NATIVE CONTAINMENT MATRIX
    # --------------------------------------------------------

    print(
        "\nE) NATIVE T <= L INCIDENCE MATRIX"
    )
    print("-" * 84)

    B = containment_matrix(
        Tfamily,
        Lfamily
    )

    nT, nL = B.shape

    row_degrees = [
        int(x)
        for x in B.sum(axis=1)
    ]

    col_degrees = [
        int(x)
        for x in B.sum(axis=0)
    ]

    total_inc = int(B.sum())

    print(
        "B shape =",
        B.shape
    )

    print(
        "T-side degree histogram =",
        dict(
            sorted(
                Counter(
                    row_degrees
                ).items()
            )
        )
    )

    print(
        "L-side degree histogram =",
        dict(
            sorted(
                Counter(
                    col_degrees
                ).items()
            )
        )
    )

    print(
        "total incidences =",
        total_inc
    )

    print(
        "T-side degree sum =",
        sum(row_degrees)
    )

    print(
        "L-side degree sum =",
        sum(col_degrees)
    )

    # --------------------------------------------------------
    # F. LEVI GRAPH
    # --------------------------------------------------------

    print(
        "\nF) LEVI GRAPH"
    )
    print("-" * 84)

    comps = bipartite_components(B)

    print(
        "Levi vertices =",
        nT + nL
    )

    print(
        "Levi edges =",
        total_inc
    )

    print(
        "Levi component count =",
        len(comps)
    )

    print(
        "Levi component sizes =",
        [
            len(C)
            for C in comps
        ]
    )

    print(
        "Levi spectrum =",
        incidence_spectrum(B)
    )

    # --------------------------------------------------------
    # G. BB^T / B^TB
    # --------------------------------------------------------

    print(
        "\nG) INCIDENCE GRAM MATRICES"
    )
    print("-" * 84)

    BBT = B @ B.T
    BTB = B.T @ B

    print(
        "BB^T diagonal histogram =",
        diagonal_hist(BBT)
    )

    print(
        "BB^T off-diagonal histogram =",
        offdiag_hist(BBT)
    )

    print(
        "B^TB diagonal histogram =",
        diagonal_hist(BTB)
    )

    print(
        "B^TB off-diagonal histogram =",
        offdiag_hist(BTB)
    )

    # --------------------------------------------------------
    # H. INCIDENCE EQUIVARIANCE
    # --------------------------------------------------------

    print(
        "\nH) W(H4)-EQUIVARIANCE OF INCIDENCE"
    )
    print("-" * 84)

    equiv, witness = \
        verify_incidence_equivariance(
            B,
            W,
            actionT,
            actionL
        )

    print(
        "incidence W-equivariant =",
        equiv
    )

    if not equiv:
        print(
            "failure witness =",
            witness
        )

    # --------------------------------------------------------
    # I. DISCOVER FIBERS
    # --------------------------------------------------------

    print(
        "\nI) CONTAINMENT FIBERS"
    )
    print("-" * 84)

    fibers = fibers_from_B(B)

    containingLs = \
        containing_Ls_from_B(B)

    unique_T_parent = all(
        len(X) == 1
        for X in containingLs
    )

    uniform_L_fiber = (
        len(set(
            len(F)
            for F in fibers
        )) == 1
    )

    print(
        "every T belongs to exactly one L =",
        unique_T_parent
    )

    print(
        "L fiber-size histogram =",
        dict(
            sorted(
                Counter(
                    len(F)
                    for F in fibers
                ).items()
            )
        )
    )

    print(
        "uniform L fibers =",
        uniform_L_fiber
    )

    if unique_T_parent:
        parent = {
            i: containingLs[i][0]
            for i in range(nT)
        }

        print(
            "containment defines map "
            "pi: T -> L = True"
        )

        print(
            "pi surjective =",
            set(parent.values())
            == set(range(nL))
        )

    else:
        parent = None

        print(
            "containment defines map "
            "pi: T -> L = False"
        )

    # Print all fibers explicitly.
    for j, F in enumerate(fibers):
        print(
            f"  L[{j:2d}] fiber "
            f"size={len(F)} "
            f"T-indices={F}"
        )

    # --------------------------------------------------------
    # J. LOCAL C3 / TRIALITY FIBER ACTION
    # --------------------------------------------------------

    print(
        "\nJ) LOCAL C3 ACTION ON CONTAINMENT FIBER"
    )
    print("-" * 84)

    c_witnesses = \
        find_C3_complement(
            N0,
            L0
        )

    print(
        "C3 complement witnesses =",
        len(c_witnesses)
    )

    if not c_witnesses:
        raise RuntimeError(
            "No C3 complement witness."
        )

    c = c_witnesses[0]

    print(
        "chosen c order =",
        porder(c)
    )

    print(
        "c fixes L0 =",
        actionL[c][0] == 0
    )

    F0 = fibers[0]

    print(
        "L0 fiber =",
        F0
    )

    c_on_F0 = \
        restricted_action_on_subset(
            actionT[c],
            F0
        )

    print(
        "c action on L0 fiber =",
        c_on_F0
    )

    if c_on_F0 is not None:
        print(
            "c fiber cycle signature =",
            cycle_signature(
                c_on_F0
            )
        )

    # Check all complement witnesses.
    local_c3_fiber_sigs = Counter()

    for x in c_witnesses:

        px = restricted_action_on_subset(
            actionT[x],
            F0
        )

        if px is None:
            sig = "does_not_preserve_fiber"
        else:
            sig = tuple(
                sorted(
                    cycle_signature(px).items()
                )
            )

        local_c3_fiber_sigs[sig] += 1

    print(
        "C3 complement fiber-action signatures ="
    )

    for sig, count in \
        local_c3_fiber_sigs.items():

        print(
            "  ",
            sig,
            "count",
            count
        )

    # --------------------------------------------------------
    # K. DO FIBERS FORM THE EXPECTED TRIALITY TRIPLES?
    # --------------------------------------------------------

    print(
        "\nK) TRIALITY-FIBER GATE"
    )
    print("-" * 84)

    THREE_FIBERS = all(
        len(F) == 3
        for F in fibers
    )

    C3_ROTATES_F0 = (
        c_on_F0 is not None
        and cycle_signature(
            c_on_F0
        ) == {3: 1}
    )

    print(
        "all L fibers have size 3 =",
        THREE_FIBERS
    )

    print(
        "chosen triality C3 rotates "
        "the L0 fiber as a 3-cycle =",
        C3_ROTATES_F0
    )

    print(
        "T75 -> L25 triality-fiber "
        "picture supported =",
        (
            nT == 75
            and nL == 25
            and unique_T_parent
            and THREE_FIBERS
            and C3_ROTATES_F0
        )
    )

    # --------------------------------------------------------
    # L. WITHIN-FIBER T INTERSECTIONS
    # --------------------------------------------------------

    print(
        "\nL) WITHIN-FIBER T INTERSECTION STRUCTURE"
    )
    print("-" * 84)

    within_pair_hist = Counter()
    within_triple_hist = Counter()

    for F in fibers:

        if len(F) >= 2:
            for a, b in itertools.combinations(
                F, 2
            ):
                within_pair_hist[
                    subgroup_intersection_size(
                        Tfamily[a],
                        Tfamily[b]
                    )
                ] += 1

        if len(F) >= 3:
            H = set(
                Tfamily[F[0]]
            )

            for a in F[1:]:
                H &= set(
                    Tfamily[a]
                )

            within_triple_hist[
                len(H)
            ] += 1

    print(
        "within-fiber pairwise "
        "|Ta ∩ Tb| histogram =",
        dict(
            sorted(
                within_pair_hist.items()
            )
        )
    )

    print(
        "within-fiber total-intersection "
        "histogram =",
        dict(
            sorted(
                within_triple_hist.items()
            )
        )
    )

    print(
        "z belongs to every successful T =",
        all(
            z in T
            for T in Tfamily
        )
    )

    # --------------------------------------------------------
    # M. CROSS-FIBER INTERSECTION MATRICES
    # --------------------------------------------------------

    print(
        "\nM) CROSS-FIBER T-INTERSECTION MATRICES"
    )
    print("-" * 84)

    records = classify_L_pairs(
        Lfamily,
        fibers,
        Tfamily
    )

    matrix_type_counts = Counter(
        r["canonical"]
        for r in records
    )

    print(
        "number of unordered L pairs =",
        len(records)
    )

    print(
        "distinct canonical cross-fiber "
        "matrix types =",
        len(matrix_type_counts)
    )

    ordered_matrix_types = sorted(
        matrix_type_counts.items(),
        key=lambda kv:
            (
                matrix_pretty(kv[0]),
                kv[1]
            )
    )

    matrix_type_id = {
        M: i
        for i, (M, count)
        in enumerate(
            ordered_matrix_types
        )
    }

    for M, count in \
        ordered_matrix_types:

        print(
            f"  TYPE "
            f"{matrix_type_id[M]}: "
            f"count={count}"
        )

        print(
            "    canonical matrix =",
            matrix_pretty(M)
        )

        print(
            "    row sums =",
            [
                sum(row)
                for row in M
            ]
        )

        print(
            "    column sums =",
            [
                sum(
                    M[i][j]
                    for i in range(len(M))
                )
                for j in range(
                    len(M[0])
                )
            ]
        )

    # --------------------------------------------------------
    # N. COMPARE MATRIX TYPES TO |Li ∩ Lj|
    # --------------------------------------------------------

    print(
        "\nN) DOES T-FIBER GEOMETRY RECONSTRUCT "
        "THE L-PAIR TYPES?"
    )
    print("-" * 84)

    matrix_to_L, L_to_matrix = \
        reconstruction_report(
            records
        )

    print(
        "matrix type -> L intersection:"
    )

    for M in sorted(
        matrix_to_L,
        key=lambda X:
            matrix_type_id[X]
    ):
        print(
            f"  TYPE "
            f"{matrix_type_id[M]} ->",
            dict(
                sorted(
                    matrix_to_L[M].items()
                )
            )
        )

    print(
        "\nL intersection -> matrix types:"
    )

    for s in sorted(L_to_matrix):

        pretty = {
            matrix_type_id[M]:
                count
            for M, count
            in L_to_matrix[s].items()
        }

        print(
            f"  |Li ∩ Lj|={s} ->",
            dict(sorted(pretty.items()))
        )

    MATRIX_DETERMINES_L = all(
        len(C) == 1
        for C in matrix_to_L.values()
    )

    L_DETERMINES_MATRIX = all(
        len(C) == 1
        for C in L_to_matrix.values()
    )

    EXACT_RECONSTRUCTION = (
        MATRIX_DETERMINES_L
        and L_DETERMINES_MATRIX
    )

    print(
        "\nmatrix type uniquely determines "
        "|Li ∩ Lj| =",
        MATRIX_DETERMINES_L
    )

    print(
        "|Li ∩ Lj| uniquely determines "
        "matrix type =",
        L_DETERMINES_MATRIX
    )

    print(
        "EXACT TWO-WAY RECONSTRUCTION =",
        EXACT_RECONSTRUCTION
    )

    # --------------------------------------------------------
    # O. RECONSTRUCT THE 25-POINT GRAPH FROM T DATA ONLY
    # --------------------------------------------------------

    print(
        "\nO) RECONSTRUCT 25-POINT GRAPH "
        "FROM T-FIBER DATA"
    )
    print("-" * 84)

    L_intersection_values = sorted(
        set(
            r["L_intersection"]
            for r in records
        )
    )

    print(
        "observed L intersection values =",
        L_intersection_values
    )

    # Identify matrix types associated with the smaller
    # L-intersection relation, if exact reconstruction exists.
    reconstructed_edges = []

    if (
        EXACT_RECONSTRUCTION
        and len(L_intersection_values) == 2
    ):
        small = min(
            L_intersection_values
        )

        small_types = {
            M
            for M, C
            in matrix_to_L.items()
            if set(C.keys()) == {small}
        }

        for r in records:
            if r["canonical"] in small_types:
                reconstructed_edges.append(
                    (r["i"], r["j"])
                )

        sig = graph_signature_from_edges(
            nL,
            reconstructed_edges
        )

        print(
            "reconstructed relation chosen by "
            "T-matrix type associated with "
            f"|Li∩Lj|={small}"
        )

        print(
            "reconstructed edges =",
            sig["edges"]
        )

        print(
            "degree histogram =",
            sig["degree_hist"]
        )

        print(
            "triangles =",
            sig["triangles"]
        )

        print(
            "spectrum =",
            sig["spectrum"]
        )

        print(
            "common-neighbor histogram =",
            sig[
                "common_neighbor_hist"
            ]
        )

        TARGET_SRG_25_8_3_2 = (
            sig["degree_hist"] == {8:25}
            and sig["edges"] == 100
            and sig[
                "common_neighbor_hist"
            ].get(("adj",3), 0) == 100
            and sig[
                "common_neighbor_hist"
            ].get(("nonadj",2), 0) == 200
        )

    else:
        sig = None
        TARGET_SRG_25_8_3_2 = False

        print(
            "rank-3 graph reconstruction "
            "not available under exact gate."
        )

    print(
        "T-data reconstructs "
        "srg(25,8,3,2) signature =",
        TARGET_SRG_25_8_3_2
    )

    # --------------------------------------------------------
    # P. PAIR ORBITS ON L_25
    # --------------------------------------------------------

    print(
        "\nP) W(H4) PAIR ORBITS ON L_25"
    )
    print("-" * 84)

    pair_orbits_L = \
        pair_orbit_partition(
            actionL,
            nL
        )

    print(
        "unordered-pair orbit count =",
        len(pair_orbits_L)
    )

    print(
        "unordered-pair orbit sizes =",
        [
            len(O)
            for O in pair_orbits_L
        ]
    )

    # Compare each pair orbit to intersection/matrix type.
    rec_by_pair = {
        (r["i"], r["j"]):
            r
        for r in records
    }

    for oi, O in enumerate(
        pair_orbits_L
    ):
        Lhist = Counter()
        Mhist = Counter()

        for pair in O:
            r = rec_by_pair[pair]

            Lhist[
                r["L_intersection"]
            ] += 1

            Mhist[
                matrix_type_id[
                    r["canonical"]
                ]
            ] += 1

        print(
            f"  pair orbit {oi}: "
            f"size={len(O)}"
        )

        print(
            "    L-intersections =",
            dict(
                sorted(
                    Lhist.items()
                )
            )
        )

        print(
            "    T-matrix types =",
            dict(
                sorted(
                    Mhist.items()
                )
            )
        )

    # --------------------------------------------------------
    # Q. REPRESENTATIVE C5
    # --------------------------------------------------------

    print(
        "\nQ) REPRESENTATIVE C5 ACTION"
    )
    print("-" * 84)

    order5 = [
        g for g in W
        if porder(g) == 5
    ]

    print(
        "order-5 elements =",
        len(order5)
    )

    inN5 = [
        g for g in order5
        if g in N0
    ]

    print(
        "order-5 elements in N0 =",
        len(inN5)
    )

    f = order5[0]

    print(
        "chosen f order =",
        porder(f)
    )

    print(
        "f L-action cycle signature =",
        cycle_signature(
            actionL[f]
        )
    )

    print(
        "f T-action cycle signature =",
        cycle_signature(
            actionT[f]
        )
    )

    # --------------------------------------------------------
    # R. C5 FIBER TRANSPORT
    # --------------------------------------------------------

    print(
        "\nR) C5 TRANSPORT OF TRIALITY FIBERS"
    )
    print("-" * 84)

    fiber_transport = \
        fiber_action_signature(
            f,
            actionT,
            actionL,
            fibers
        )

    FIBER_EQUIVARIANT_C5 = all(
        ok
        for _, _, ok
        in fiber_transport
    )

    print(
        "f transports every T-fiber "
        "onto target L-fiber =",
        FIBER_EQUIVARIANT_C5
    )

    for j, j2, ok in \
        fiber_transport:

        print(
            f"  L[{j:2d}] -> "
            f"L[{j2:2d}] "
            f"fiber preserved={ok}"
        )

    # --------------------------------------------------------
    # S. C3 FIBER TRANSPORT
    # --------------------------------------------------------

    print(
        "\nS) C3 TRANSPORT OF FIBERS"
    )
    print("-" * 84)

    c_fiber_transport = \
        fiber_action_signature(
            c,
            actionT,
            actionL,
            fibers
        )

    FIBER_EQUIVARIANT_C3 = all(
        ok
        for _, _, ok
        in c_fiber_transport
    )

    print(
        "c respects every fiber over its "
        "induced L action =",
        FIBER_EQUIVARIANT_C3
    )

    # --------------------------------------------------------
    # T. DO C3 / C5 PRESERVE CROSS-FIBER TYPES?
    # --------------------------------------------------------

    print(
        "\nT) C3 / C5 PRESERVATION OF "
        "CROSS-FIBER RELATION TYPES"
    )
    print("-" * 84)

    record_lookup = {
        (r["i"], r["j"]):
            r["canonical"]
        for r in records
    }

    def canon_pair(i, j):
        if i > j:
            i, j = j, i
        return i, j

    def preserves_pair_types(g):
        p = actionL[g]

        for (i, j), C in \
            record_lookup.items():

            a, b = canon_pair(
                p[i],
                p[j]
            )

            if (
                record_lookup[(a,b)]
                != C
            ):
                return False, (
                    i, j,
                    a, b,
                    C,
                    record_lookup[(a,b)]
                )

        return True, None

    c_pres, c_fail = \
        preserves_pair_types(c)

    f_pres, f_fail = \
        preserves_pair_types(f)

    print(
        "C3 preserves all canonical "
        "cross-fiber types =",
        c_pres
    )

    if not c_pres:
        print(
            "  C3 failure witness =",
            c_fail
        )

    print(
        "C5 preserves all canonical "
        "cross-fiber types =",
        f_pres
    )

    if not f_pres:
        print(
            "  C5 failure witness =",
            f_fail
        )

    # --------------------------------------------------------
    # U. COMMON CENTER ACROSS T AND L
    # --------------------------------------------------------

    print(
        "\nU) COMMON CENTRAL C2"
    )
    print("-" * 84)

    z_in_all_T = all(
        z in T
        for T in Tfamily
    )

    z_in_all_L = all(
        z in L
        for L in Lfamily
    )

    print(
        "z in every successful T =",
        z_in_all_T
    )

    print(
        "z in every compatible L =",
        z_in_all_L
    )

    common_T = set(
        Tfamily[0]
    )

    for T in Tfamily[1:]:
        common_T &= set(T)

    common_L = set(
        Lfamily[0]
    )

    for L in Lfamily[1:]:
        common_L &= set(L)

    print(
        "intersection of all successful "
        "T's order =",
        len(common_T)
    )

    print(
        "intersection of all compatible "
        "L's order =",
        len(common_L)
    )

    print(
        "all-T intersection order histogram =",
        order_hist(common_T)
    )

    print(
        "all-L intersection order histogram =",
        order_hist(common_L)
    )

    # --------------------------------------------------------
    # V. HARD GATES
    # --------------------------------------------------------

    print(
        "\n" + "=" * 84
    )
    print(
        "V) SIM15.3 HARD GATES"
    )
    print("=" * 84)

    GATE_A = (
        nT > 0
        and nL > 0
        and total_inc > 0
    )

    GATE_B = equiv

    GATE_C = (
        unique_T_parent
        and THREE_FIBERS
        and C3_ROTATES_F0
    )

    GATE_D = (
        len(records)
        == math.comb(nL, 2)
    )

    GATE_E = \
        EXACT_RECONSTRUCTION

    GATE_F = (
        FIBER_EQUIVARIANT_C3
        and FIBER_EQUIVARIANT_C5
        and c_pres
        and f_pres
    )

    print(
        "GATE A — native T-L incidence "
        "enumerated:",
        GATE_A
    )

    print(
        "GATE B — incidence is "
        "W(H4)-equivariant:",
        GATE_B
    )

    print(
        "GATE C — containment gives "
        "triality fibers:",
        GATE_C
    )

    print(
        "GATE D — all cross-fiber "
        "matrices classified:",
        GATE_D
    )

    print(
        "GATE E — T-fiber data exactly "
        "reconstructs L-pair types:",
        GATE_E
    )

    print(
        "GATE F — C3/C5 preserve "
        "fiber geometry:",
        GATE_F
    )

    # --------------------------------------------------------
    # W. MACHINE TRUTH PACKET
    # --------------------------------------------------------

    print(
        "\n" + "=" * 84
    )
    print(
        "W) MACHINE TRUTH PACKET"
    )
    print("=" * 84)

    truth = {
        "H4_order_14400":
            len(W) == 14400,

        "successful_T_count":
            nT,

        "compatible_L_count":
            nL,

        "total_TL_incidences":
            total_inc,

        "T_degree_hist":
            dict(
                sorted(
                    Counter(
                        row_degrees
                    ).items()
                )
            ),

        "L_degree_hist":
            dict(
                sorted(
                    Counter(
                        col_degrees
                    ).items()
                )
            ),

        "incidence_equivariant":
            equiv,

        "Levi_component_sizes":
            [
                len(C)
                for C in comps
            ],

        "T_unique_parent_L":
            unique_T_parent,

        "all_L_fibers_size_3":
            THREE_FIBERS,

        "C3_rotates_L0_fiber":
            C3_ROTATES_F0,

        "triality_fiber_picture":
            (
                nT == 75
                and nL == 25
                and unique_T_parent
                and THREE_FIBERS
                and C3_ROTATES_F0
            ),

        "cross_fiber_matrix_types":
            len(matrix_type_counts),

        "matrix_determines_L_type":
            MATRIX_DETERMINES_L,

        "L_type_determines_matrix":
            L_DETERMINES_MATRIX,

        "exact_T_to_L_rank3_reconstruction":
            EXACT_RECONSTRUCTION,

        "reconstructs_srg_25_8_3_2":
            TARGET_SRG_25_8_3_2,

        "L_pair_orbit_sizes":
            [
                len(O)
                for O in pair_orbits_L
            ],

        "C5_fiber_equivariant":
            FIBER_EQUIVARIANT_C5,

        "C3_fiber_equivariant":
            FIBER_EQUIVARIANT_C3,

        "C3_preserves_matrix_types":
            c_pres,

        "C5_preserves_matrix_types":
            f_pres,

        "z_in_all_T":
            z_in_all_T,

        "z_in_all_L":
            z_in_all_L,

        "all_T_intersection_order":
            len(common_T),

        "all_L_intersection_order":
            len(common_L),
    }

    for k, v in truth.items():
        print(
            f"{k:42s}: {v}"
        )

    # --------------------------------------------------------
    # X. INTERPRETATION GUARDRAILS
    # --------------------------------------------------------

    print(
        "\n" + "=" * 84
    )
    print(
        "X) INTERPRETATION GUARDRAILS"
    )
    print("=" * 84)

    print(r"""
1. The cardinalities of the T and L families are outputs, not assumptions.

2. A (25_3) incidence configuration is NOT assumed.

3. If the successful T-family has 75 elements and every T belongs to
   exactly one L while every L contains exactly three T's, interpret
   containment first as a 75 -> 25 fiber map, not as a symmetric design.

4. The three T's over one L are called a triality fiber only if the
   independently recovered local C3 complement acts as a 3-cycle on them.

5. The Levi graph may be disconnected.  If it is 25 disjoint K_1,3
   components, that is a structural result, not a failed incidence test.

6. BB^T and B^TB record shared containment only.  If they do not recover
   the SIM15.2 rank-3 relation, move to subgroup-intersection data rather
   than inventing a new incidence relation.

7. Cross-fiber matrices use only native subgroup intersections:

       M_ij[a,b] = |T_ia ∩ T_jb|.

   Their canonicalization removes arbitrary labels within each triality
   fiber.

8. The SIM15.2 rank-3 L relation is considered reconstructed from T-data
   only if canonical cross-fiber matrix type and L-intersection type
   determine one another exactly.

9. Reproducing the srg(25,8,3,2) signature from T-fiber data does NOT by
   itself identify the graph with a 5x5 lattice/rook graph.

10. No F5^2, affine plane, row/column system, pentagon decomposition, or
    finite geometry is inserted in advance.

11. C3 and C5 are not interpreted as physical layers.  Their role here is
    purely their computed action on the native subgroup hierarchy.

12. No RCFT physical interpretation follows from this run alone.

END SIM15.3
""")


if __name__ == "__main__":
    main()







