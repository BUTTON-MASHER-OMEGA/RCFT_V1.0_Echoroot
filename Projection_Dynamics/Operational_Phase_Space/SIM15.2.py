
SIM15.2 — PRIME-STRATIFIED H4 ACTION MAP
========================================

PURPOSE
-------
SIM15.2 studies how the 2-, 3-, and 5-primary structures of W(H4)
act relative to the hierarchy

    T ~= C2^3  triangleleft  L  triangleleft  N  <  W(H4),

with the SIM15.0/15.1 target orders

    |T| = 8,
    |L| = 192,
    |N| = 576,
    |W(H4)| = 14400.

The run is intentionally split into four hard gates:

A. Constructively identify L with W(D4) and determine whether the
   external order-3 automorphism is the standard D4 triality.

B. Verify that order-5 elements cannot normalize L and determine
   their action on the 25 conjugates of L.

C. Reconstruct the full degree-25 action

       W(H4) -> Sym(L_25)

   and determine its kernel, rank, blocks, subdegrees, orbitals,
   and invariant orbital graphs.

D. Ask whether the 25-set carries additional intrinsic combinatorial
   structure WITHOUT assuming F5^2, a 5x5 grid, affine geometry,
   five pentagons, etc.

The same representative C3 and C5 elements are also compared on:
    - successful T kernels,
    - L_25,
    - 600-cell vertices,
    - 120-cell vertices.

NO RCFT DYNAMICS.
NO LCO / ISP.
NO HILBERT SPACE.
NO J SEARCH.
NO ASSUMED F5^2.
NO ASSUMED FIVE-PENTAGON DECOMPOSITION.
NO ASSUMED TRIALITY UNTIL CONSTRUCTIVELY TESTED.

DEPENDENCIES
------------
Python 3
numpy

No GAP or Sage required.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict, deque
import numpy as np


# ============================================================
# 0. BASIC PERMUTATION UTILITIES
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
    n = len(a)
    r = pid(n)
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
    n = len(p)
    seen = [False] * n
    ans = 1

    for i in range(n):
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
    n = len(p)
    seen = [False] * n
    out = []

    for i in range(n):
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
    return dict(sorted(Counter(len(c) for c in pcycles(p, True)).items()))


def pstr(p):
    cyc = pcycles(p)
    if not cyc:
        return "()"
    return "".join("(" + " ".join(map(str, c)) + ")" for c in cyc)


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
        if all(pmul(z, g) == pmul(g, z) for g in G)
    }


def centralizer_in(W, H):
    H = set(H)

    return {
        g for g in W
        if all(pmul(g, h) == pmul(h, g) for h in H)
    }


def normalizer_in(W, H):
    H = set(H)

    return {
        g for g in W
        if {pconj(g, h) for h in H} == H
    }


def is_normal(H, G):
    H = set(H)

    return all(
        {pconj(g, h) for h in H} == H
        for g in G
    )


def order_hist(G):
    return dict(sorted(Counter(porder(g) for g in G).items()))


def orbit_of_point(G, x):
    return {g[x] for g in G}


def point_orbits(G, n):
    unseen = set(range(n))
    out = []

    while unseen:
        x = min(unseen)
        O = orbit_of_point(G, x)
        out.append(O)
        unseen -= O

    return sorted(out, key=lambda O: (len(O), sorted(O)))


def conjugate_subgroup(g, H):
    return frozenset(pconj(g, h) for h in H)


def subgroup_equal(H, K):
    return set(H) == set(K)


# ============================================================
# 1. H4 ROOT SYSTEM / 600-CELL
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

    return np.array(list(seen.values()), dtype=float)


def build_h4_roots():
    """
    Standard 120-root H4 realization / 600-cell vertices.

    Before final division by sqrt(2):
      8  : permutations of (±2,0,0,0)
      16 : (±1,±1,±1,±1)
      96 : even permutations of (0,±1,±phi,±1/phi)

    Final roots have squared norm 2.
    """
    roots = []

    # 8 coordinate roots
    for i in range(4):
        for s in (-1.0, 1.0):
            v = np.zeros(4)
            v[i] = 2.0 * s
            roots.append(v / SQRT2)

    # 16 hypercube roots
    for signs in itertools.product((-1.0, 1.0), repeat=4):
        roots.append(np.array(signs, dtype=float) / SQRT2)

    # 96 golden roots
    base = (0.0, 1.0, PHI, IPHI)

    even_perms = [
        p for p in itertools.permutations(range(4))
        if permutation_parity(p) == 0
    ]

    for p in even_perms:
        vals = np.array([base[p[i]] for i in range(4)], dtype=float)
        nz = [i for i, x in enumerate(vals) if abs(x) > TOL]

        for signs in itertools.product((-1.0, 1.0), repeat=3):
            v = vals.copy()

            for j, s in zip(nz, signs):
                v[j] *= s

            roots.append(v / SQRT2)

    R = unique_vectors(roots)

    if len(R) != 120:
        raise RuntimeError(
            f"H4 construction produced {len(R)} roots; expected 120."
        )

    norms = np.sum(R * R, axis=1)

    if not np.allclose(norms, 2.0, atol=1e-8):
        raise RuntimeError("H4 roots do not all have norm^2 = 2.")

    return R


def vec_key(v, decimals=9):
    return tuple(np.round(v, decimals))


def build_root_lookup(R):
    return {vec_key(v): i for i, v in enumerate(R)}


def reflection_matrix(alpha):
    alpha = np.asarray(alpha, dtype=float)
    return np.eye(4) - np.outer(alpha, alpha)


def linear_map_to_perm(M, R, lookup):
    p = []

    for v in R:
        w = M @ v
        key = vec_key(w)

        if key in lookup:
            p.append(lookup[key])
            continue

        d = np.linalg.norm(R - w[None, :], axis=1)
        j = int(np.argmin(d))

        if d[j] > 1e-6:
            raise RuntimeError("Linear map failed to permute H4 roots.")

        p.append(j)

    return tuple(p)


def find_simple_h4_roots(R):
    """
    H4 Coxeter chain:
        5 -- 3 -- 3

    With roots of norm^2 2:
        <a1,a2> = -phi
        <a2,a3> = -1
        <a3,a4> = -1
        non-neighbors orthogonal.
    """
    dots = R @ R.T

    def inds(i, target):
        return [
            j for j in range(len(R))
            if j != i and abs(dots[i, j] - target) < 1e-7
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
                return [a1, a2, a3, cand4[0]]

    raise RuntimeError("Could not find H4 simple roots.")


def build_W_H4(R):
    lookup = build_root_lookup(R)

    simple_idx = find_simple_h4_roots(R)
    simple = [R[i] for i in simple_idx]

    refl_mats = [reflection_matrix(a) for a in simple]
    refl_perms = [
        linear_map_to_perm(M, R, lookup)
        for M in refl_mats
    ]

    W = generated_group(refl_perms)

    if len(W) != 14400:
        raise RuntimeError(
            f"|W(H4)| = {len(W)}; expected 14400."
        )

    return W, refl_perms, simple_idx


# ============================================================
# 2. 600-CELL / 120-CELL
# ============================================================

def pairwise_dist2(P):
    diff = P[:, None, :] - P[None, :, :]
    return np.sum(diff * diff, axis=2)


def build_600_graph(R):
    D2 = pairwise_dist2(R)

    vals = sorted({
        round(float(D2[i, j]), 9)
        for i in range(len(R))
        for j in range(i + 1, len(R))
        if D2[i, j] > 1e-9
    })

    edge_d2 = vals[0]

    adj = [set() for _ in range(len(R))]
    edges = set()

    for i in range(len(R)):
        for j in range(i + 1, len(R)):

            if abs(D2[i, j] - edge_d2) < 1e-7:
                adj[i].add(j)
                adj[j].add(i)
                edges.add((i, j))

    if len(edges) != 720:
        raise RuntimeError(
            f"600-cell has {len(edges)} edges; expected 720."
        )

    if {len(A) for A in adj} != {12}:
        raise RuntimeError("600-cell degree is not uniformly 12.")

    return adj, edges


def tetrahedral_cells(adj):
    cells = set()
    n = len(adj)

    for a in range(n):
        for b in adj[a]:

            if b <= a:
                continue

            ab = adj[a] & adj[b]

            for c in ab:

                if c <= b:
                    continue

                abc = ab & adj[c]

                for d in abc:

                    if d <= c:
                        continue

                    cells.add(tuple(sorted((a, b, c, d))))

    if len(cells) != 600:
        raise RuntimeError(
            f"Found {len(cells)} tetrahedral cells; expected 600."
        )

    return sorted(cells)


def build_120_graph(cells):
    face_to_cells = defaultdict(list)

    for ci, cell in enumerate(cells):
        for face in itertools.combinations(cell, 3):
            face_to_cells[tuple(sorted(face))].append(ci)

    edges = set()
    adj = [set() for _ in cells]

    for face, owners in face_to_cells.items():

        if len(owners) != 2:
            raise RuntimeError(
                f"Face {face} belongs to {len(owners)} cells."
            )

        a, b = owners

        if a > b:
            a, b = b, a

        edges.add((a, b))
        adj[a].add(b)
        adj[b].add(a)

    if len(edges) != 1200:
        raise RuntimeError(
            f"120-cell has {len(edges)} edges; expected 1200."
        )

    if {len(A) for A in adj} != {4}:
        raise RuntimeError("120-cell degree is not uniformly 4.")

    return adj, edges


def induced_perm_on_subsets(p, subsets, subset_index):
    return tuple(
        subset_index[tuple(sorted(p[i] for i in S))]
        for S in subsets
    )


def build_action_on_cells(W, cells):
    idx = {c: i for i, c in enumerate(cells)}

    return {
        g: induced_perm_on_subsets(g, cells, idx)
        for g in W
    }


# ============================================================
# 3. C2^3 ENUMERATION
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

    print("involutions in W(H4):", len(invol))

    groups = set()

    for ia, a in enumerate(invol):

        for ib in range(ia + 1, len(invol)):
            b = invol[ib]

            if not commute(a, b):
                continue

            H4 = subgroup_generated([a, b], len(a))

            if len(H4) != 4:
                continue

            for c in invol:

                if c in H4:
                    continue

                if not commute(a, c) or not commute(b, c):
                    continue

                H8 = subgroup_generated([a, b, c], len(a))

                if len(H8) == 8:
                    if all(porder(x) in (1, 2) for x in H8):
                        groups.add(frozenset(H8))

    return [set(H) for H in groups]


def subgroup_conjugacy_classes(W, subs):
    unseen = {frozenset(H) for H in subs}
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
# 4. F2^3 ACTION / SUCCESSFUL CLASS
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


def choose_T_basis(T):
    e = pid(len(next(iter(T))))
    nonzero = [x for x in T if x != e]

    for a, b, c in itertools.combinations(nonzero, 3):

        H = subgroup_generated([a, b, c], len(a))

        if H == set(T):
            return a, b, c

    raise RuntimeError("Could not choose basis of T.")


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


I3 = (
    (1,0,0),
    (0,1,0),
    (0,0,1),
)


def mat_apply(M, v):
    return tuple(
        sum(M[i][j] * v[j] for j in range(3)) % 2
        for i in range(3)
    )


def mat_mul(A, B):
    return tuple(
        tuple(
            sum(A[i][k] * B[k][j] for k in range(3)) % 2
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

    raise RuntimeError("Matrix order failure.")


def mat_from_action_on_T(g, elem_to_coord, coord_to_elem):
    basis = [
        (1,0,0),
        (0,1,0),
        (0,0,1),
    ]

    cols = []

    for v in basis:

        x = coord_to_elem[v]
        y = pconj(g, x)

        cols.append(elem_to_coord[y])

    return tuple(
        tuple(cols[j][i] for j in range(3))
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

    elem_to_coord, coord_to_elem = T_coordinate_map(T)

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
        if all(mat_apply(M, v) == v for M in image)
    ]

    gl_hist = dict(
        sorted(Counter(mat_order(M) for M in image).items())
    )

    zN = center(N)

    e = pid(len(next(iter(T))))

    z = None

    if len(zN) == 2:
        z = next(x for x in zN if x != e)

    nonzero_fixed = [
        v for v in fixed
        if v != (0,0,0)
    ]

    center_fixed = False

    if z is not None and len(nonzero_fixed) == 1:
        center_fixed = (
            z == coord_to_elem[nonzero_fixed[0]]
        )

    checks = {
        "T8": len(T) == 8,
        "N192": len(N) == 192,
        "centralizer_T": C == set(T),
        "GL24": len(image) == 24,
        "GL_S4_hist": gl_hist == TARGET_GL_HIST,
        "one_nonzero_fixed": len(nonzero_fixed) == 1,
        "center2": len(zN) == 2,
        "center_fixed": center_fixed,
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
# 5. FIND SUCCESSFUL T, L, N
# ============================================================

def find_successful_structure(W):
    Tsubs = enumerate_C2_3_subgroups(W)

    print("C2^3 subgroups:", len(Tsubs))

    classes = subgroup_conjugacy_classes(W, Tsubs)

    print("C2^3 conjugacy classes:", len(classes))
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
            "Expected exactly one successful C2^3 class; "
            f"found {len(successful)}."
        )

    ci, successful_class, T, fp = successful[0]

    L = fp["N"]
    z = fp["z"]
    N = normalizer_in(W, L)

    return {
        "class_id": ci,
        "successful_class": successful_class,
        "T": T,
        "L": L,
        "N": N,
        "z": z,
    }


# ============================================================
# 6. COMPLEMENT C3
# ============================================================

def find_C3_complement(N, L):
    e = pid(len(next(iter(N))))

    candidates = [
        g for g in N
        if g not in L and porder(g) == 3
    ]

    witnesses = []

    for c in candidates:

        C3 = subgroup_generated([c], len(c))

        if C3 & set(L) != {e}:
            continue

        H = subgroup_generated(
            greedy_generators(L) + [c],
            len(c)
        )

        if H == set(N):
            witnesses.append(c)

    return witnesses


def is_inner_on_group(c, G):
    gens = greedy_generators(G)

    target = {
        x: pconj(c, x)
        for x in gens
    }

    for h in G:

        if all(
            pconj(h, x) == target[x]
            for x in gens
        ):
            return True, h

    return False, None


# ============================================================
# 7. STANDARD D4 ROOT SYSTEM
# ============================================================

def build_D4_roots():
    """
    D4 roots:
        ±e_i ± e_j, i < j

    24 roots in R^4, norm^2 2.
    """
    roots = []

    for i in range(4):
        for j in range(i + 1, 4):

            for si in (-1.0, 1.0):
                for sj in (-1.0, 1.0):

                    v = np.zeros(4)
                    v[i] = si
                    v[j] = sj
                    roots.append(v)

    return unique_vectors(roots)


def standard_D4_simple_roots():
    """
    Standard D4 simple roots.

        a1 = e1 - e2
        a2 = e2 - e3       central
        a3 = e3 - e4
        a4 = e3 + e4

    Diagram:
             a1
              |
        a3 -- a2 -- a4

    Up to diagram drawing / naming conventions, a2 is the
    trivalent central node and {a1,a3,a4} are the three outer nodes.
    """
    e1 = np.array([1.,0.,0.,0.])
    e2 = np.array([0.,1.,0.,0.])
    e3 = np.array([0.,0.,1.,0.])
    e4 = np.array([0.,0.,0.,1.])

    a1 = e1 - e2
    a2 = e2 - e3
    a3 = e3 - e4
    a4 = e3 + e4

    return [a1, a2, a3, a4]


def build_W_D4():
    R = build_D4_roots()
    lookup = build_root_lookup(R)

    simple = standard_D4_simple_roots()

    refl = [
        linear_map_to_perm(
            reflection_matrix(a),
            R,
            lookup
        )
        for a in simple
    ]

    WD4 = generated_group(refl)

    if len(WD4) != 192:
        raise RuntimeError(
            f"|W(D4)| = {len(WD4)}; expected 192."
        )

    return R, WD4, refl


# ============================================================
# 8. ABSTRACT GROUP ISOMORPHISM BY GENERATOR BACKTRACKING
# ============================================================

def multiplication_signature(g, gens):
    """
    Lightweight signature relative to chosen generators.
    """
    return (
        porder(g),
        tuple(porder(pmul(g, h)) for h in gens),
        tuple(porder(pmul(h, g)) for h in gens),
    )


def subgroup_closure_with_words(gens):
    """
    Return group and one word for each element.
    Word is tuple of generator indices.
    """
    if not gens:
        raise ValueError

    n = len(gens[0])
    e = pid(n)

    word = {e: ()}
    q = deque([e])

    while q:

        x = q.popleft()

        for i, g in enumerate(gens):

            y = pmul(g, x)

            if y not in word:
                word[y] = (i,) + word[x]
                q.append(y)

    return set(word), word


def eval_word(word, gens):
    if not gens:
        raise ValueError

    x = pid(len(gens[0]))

    for i in reversed(word):
        x = pmul(gens[i], x)

    return x


def find_isomorphism_via_generator_images(G, H):
    """
    Construct an explicit isomorphism G -> H by choosing a compact
    generating set of G and searching compatible images in H.

    Groups here both have order 192, so a generating tuple whose
    images satisfy all multiplication relations and generate H is
    enough.

    We use word consistency as the final proof.
    """
    G = set(G)
    H = set(H)

    if len(G) != len(H):
        return None

    ggens = greedy_generators(G)

    # Usually small.
    print("  L greedy generator count =", len(ggens))

    # Candidate images filtered by element order.
    candidates = []

    for g in ggens:

        og = porder(g)

        cand = [
            h for h in H
            if porder(h) == og
        ]

        candidates.append(cand)

    # Build source words.
    Ggen, words = subgroup_closure_with_words(ggens)

    if Ggen != G:
        raise RuntimeError("Greedy generators failed.")

    # Backtracking with partial relation checks.
    chosen = []

    def partial_ok():
        k = len(chosen)

        # Compare orders of all short pair products.
        for i in range(k):
            for j in range(k):

                if porder(pmul(ggens[i], ggens[j])) != \
                   porder(pmul(chosen[i], chosen[j])):
                    return False

        # Triple checks help heavily.
        for i in range(k):
            for j in range(k):
                for m in range(k):

                    a = pmul(
                        ggens[i],
                        pmul(ggens[j], ggens[m])
                    )

                    b = pmul(
                        chosen[i],
                        pmul(chosen[j], chosen[m])
                    )

                    if porder(a) != porder(b):
                        return False

        return True

    def verify_full(images):
        Hgen = subgroup_generated(images, len(images[0]))

        if Hgen != H:
            return None

        phi = {}

        for x, w in words.items():
            y = eval_word(w, images)
            phi[x] = y

        if len(set(phi.values())) != len(G):
            return None

        # Exact homomorphism check on generators suffices once
        # word evaluation is consistent, but do a robust full check
        # against generator multiplication.
        for x in G:
            for i, g in enumerate(ggens):

                lhs = phi[pmul(g, x)]
                rhs = pmul(images[i], phi[x])

                if lhs != rhs:
                    return None

        return phi

    def rec(depth):
        if depth == len(ggens):
            return verify_full(chosen)

        for h in candidates[depth]:

            chosen.append(h)

            if partial_ok():

                # Partial generated subgroup cannot exceed H;
                # if generators collapse too much at final stages,
                # verification will reject.
                ans = rec(depth + 1)

                if ans is not None:
                    return ans

            chosen.pop()

        return None

    phi = rec(0)

    if phi is None:
        return None

    return {
        "phi": phi,
        "source_generators": ggens,
        "target_generators": [
            phi[g] for g in ggens
        ],
    }


# ============================================================
# 9. D4 REFLECTION / COXETER SYSTEM SEARCH INSIDE L
# ============================================================

def involution_product_order(a, b):
    return porder(pmul(a, b))


def find_D4_coxeter_system(G):
    """
    Search for four involutions r1,r2,r3,r4 such that:

        r2 is central node,
        m(r2,ri)=3 for i in {1,3,4},
        outer nodes commute pairwise,
        generated group has order 192.

    This identifies a D4 Coxeter generating system abstractly.
    """
    inv = [
        g for g in G
        if porder(g) == 2
    ]

    for r2 in inv:

        neighbors = [
            x for x in inv
            if x != r2
            and involution_product_order(r2, x) == 3
        ]

        for r1, r3, r4 in itertools.combinations(neighbors, 3):

            outers = [r1, r3, r4]

            if not all(
                involution_product_order(a, b) == 2
                for a, b in itertools.combinations(outers, 2)
            ):
                continue

            H = subgroup_generated(
                [r1, r2, r3, r4],
                len(r1)
            )

            if H == set(G):
                return {
                    "outer": (r1, r3, r4),
                    "central": r2,
                    "generators": (r1, r2, r3, r4),
                }

    return None


# ============================================================
# 10. TRIALITY TEST MODULO INNER CONJUGACY
# ============================================================

def automorphism_matches_triality_mod_inner(c, L, cox):
    """
    We have a D4 Coxeter system inside L:

        central r2
        outer r1,r3,r4.

    Standard triality fixes central and cycles outer nodes.

    Because the chosen Coxeter system is not canonical, allow an
    inner conjugation h in L after applying c.

    Test whether there exists h in L and one of the two 3-cycles
    of outer nodes such that

        h (c r2 c^-1) h^-1 = r2

    and the three outer generators are cyclically permuted.
    """
    r1, r3, r4 = cox["outer"]
    r2 = cox["central"]

    outer = [r1, r3, r4]

    ccentral = pconj(c, r2)
    couter = [pconj(c, r) for r in outer]

    cycles = [
        (1,2,0),
        (2,0,1),
    ]

    for h in L:

        if pconj(h, ccentral) != r2:
            continue

        transported = [
            pconj(h, x)
            for x in couter
        ]

        for cyc in cycles:

            if all(
                transported[i] == outer[cyc[i]]
                for i in range(3)
            ):
                return {
                    "pass": True,
                    "inner_adjustment": h,
                    "outer_cycle": cyc,
                }

    return {
        "pass": False,
        "inner_adjustment": None,
        "outer_cycle": None,
    }


# ============================================================
# 11. 25 CONJUGATE L STRUCTURES
# ============================================================

def conjugacy_orbit_subgroup(W, H):
    return {
        conjugate_subgroup(w, H)
        for w in W
    }


def action_on_subgroup_family(g, family, index):
    p = []

    for H in family:

        Hg = conjugate_subgroup(g, H)

        if Hg not in index:
            raise RuntimeError(
                "Subgroup family not invariant."
            )

        p.append(index[Hg])

    return tuple(p)


def build_family_action(W, family):
    family = list(family)
    index = {
        H: i for i, H in enumerate(family)
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


# ============================================================
# 12. ACTION KERNEL
# ============================================================

def permutation_action_kernel(action):
    if not action:
        return set()

    n = len(next(iter(action.values())))
    e = pid(n)

    return {
        g for g, p in action.items()
        if p == e
    }


# ============================================================
# 13. STABILIZER SUBORBITS / ORBITALS
# ============================================================

def stabilizer_of_point_from_action(W, action, point):
    return {
        g for g in W
        if action[g][point] == point
    }


def suborbits(stab, action, n):
    perms = {
        action[g]
        for g in stab
    }

    return point_orbits(perms, n)


def orbital_from_suborbit(action, W, base, suborbit):
    """
    Orbit of ordered pairs (base, y), y in suborbit.
    """
    pairs = set()

    for y in suborbit:

        for g in W:

            p = action[g]
            pairs.add(
                (p[base], p[y])
            )

    return pairs


def reverse_orbital(O):
    return {
        (b, a)
        for a, b in O
    }


# ============================================================
# 14. ORBITAL GRAPHS
# ============================================================

def graph_from_orbital(n, O, symmetrize=True):
    adj = [set() for _ in range(n)]

    for a, b in O:

        if a == b:
            continue

        adj[a].add(b)

        if symmetrize:
            adj[b].add(a)

    return adj


def graph_edges(adj):
    E = set()

    for i, A in enumerate(adj):
        for j in A:

            if i < j:
                E.add((i, j))

    return E


def graph_components(adj):
    unseen = set(range(len(adj)))
    comps = []

    while unseen:

        s = min(unseen)
        seen = {s}
        q = deque([s])

        while q:

            x = q.popleft()

            for y in adj[x]:

                if y not in seen:
                    seen.add(y)
                    q.append(y)

        comps.append(seen)
        unseen -= seen

    return sorted(
        comps,
        key=lambda C: (len(C), sorted(C))
    )


def adjacency_matrix(adj):
    n = len(adj)
    A = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in adj[i]:
            A[i, j] = 1.0

    return A


def spectrum_signature(adj):
    A = adjacency_matrix(adj)
    vals = np.linalg.eigvalsh(A)
    vals = np.round(vals, 9)

    C = Counter(vals)

    return [
        (float(k), int(v))
        for k, v in sorted(C.items())
    ]


def triangle_count(adj):
    n = len(adj)
    total = 0

    for i in range(n):
        for j in adj[i]:

            if j <= i:
                continue

            for k in adj[i] & adj[j]:

                if k > j:
                    total += 1

    return total


def common_neighbor_hist(adj):
    C = Counter()
    n = len(adj)

    for i in range(n):
        for j in range(i + 1, n):

            key = (
                "adj" if j in adj[i] else "nonadj",
                len(adj[i] & adj[j])
            )

            C[key] += 1

    return dict(sorted(C.items(), key=repr))


def graph_signature(adj):
    degs = [len(A) for A in adj]
    E = graph_edges(adj)
    comps = graph_components(adj)

    return {
        "vertices": len(adj),
        "edges": len(E),
        "degree_hist": dict(sorted(Counter(degs).items())),
        "components": [len(C) for C in comps],
        "triangles": triangle_count(adj),
        "spectrum": spectrum_signature(adj),
        "common_neighbor_hist": common_neighbor_hist(adj),
    }


# ============================================================
# 15. BLOCK SYSTEM SEARCH ON 25 POINTS
# ============================================================

def orbit_of_subset(perms, B):
    B = frozenset(B)

    return {
        frozenset(p[x] for x in B)
        for p in perms
    }


def is_block(perms, B, n):
    B = frozenset(B)

    if not B or len(B) == n:
        return False

    for p in perms:

        C = frozenset(p[x] for x in B)
        I = B & C

        if I and C != B:
            return False

    return True


def find_nontrivial_blocks(perms, n, base=0):
    """
    For a transitive action, every block containing base is a union
    of suborbits of the point stabilizer.  Since n=25, brute force
    candidate subsets from stabilizer suborbits is tiny.
    """
    stab_perms = {
        p for p in perms
        if p[base] == base
    }

    sorbs = point_orbits(stab_perms, n)

    containing_base = [
        O for O in sorbs
        if base in O
    ]

    if len(containing_base) != 1:
        raise RuntimeError(
            "Unexpected stabilizer orbit containing base."
        )

    fixed_orbit = containing_base[0]

    others = [
        O for O in sorbs
        if O is not fixed_orbit
    ]

    blocks = []

    for mask in range(1 << len(others)):

        B = set(fixed_orbit)

        for i, O in enumerate(others):

            if (mask >> i) & 1:
                B |= set(O)

        if len(B) in (1, n):
            continue

        if is_block(perms, B, n):
            blocks.append(frozenset(B))

    # Deduplicate.
    blocks = sorted(
        set(blocks),
        key=lambda B: (len(B), sorted(B))
    )

    return blocks, sorbs


# ============================================================
# 16. C5 PENTAD SIGNATURES
# ============================================================

def cyclic_orbits_of_perm(p):
    return [
        set(c)
        for c in pcycles(p, include_fixed=True)
    ]


def relation_color_matrix(n, orbitals):
    """
    Assign each ordered pair (i,j) its orbital index.
    """
    color = {}

    for oi, O in enumerate(orbitals):

        for pair in O:
            color[pair] = oi

    if len(color) != n * n:
        raise RuntimeError(
            "Orbitals do not partition ordered pairs."
        )

    return color


def subset_orbital_signature(S, color):
    """
    Count orbital colors among ordered distinct pairs in S.
    """
    S = sorted(S)
    C = Counter()

    for a in S:
        for b in S:

            if a == b:
                continue

            C[color[(a, b)]] += 1

    return tuple(sorted(C.items()))


# ============================================================
# 17. CYCLE SIGNATURES ON POLYTOPES
# ============================================================

def cycle_signature_on_600(g):
    return cycle_signature(g)


def cycle_signature_on_120(g, cell_action):
    return cycle_signature(cell_action[g])


# ============================================================
# 18. SUCCESSFUL T SUBGROUPS INSIDE L
# ============================================================

def successful_T_inside_L(successful_class, L):
    return [
        set(H)
        for H in successful_class
        if set(H) <= set(L)
    ]


def transport_subgroups(g, subs):
    idx = {
        frozenset(H): i
        for i, H in enumerate(subs)
    }

    out = []

    for H in subs:

        H2 = conjugate_subgroup(g, H)
        out.append(idx.get(H2, None))

    return out


# ============================================================
# 19. ORDER-5 CONJUGACY CLASSES
# ============================================================

def element_conjugacy_class(W, x):
    return {
        pconj(g, x)
        for g in W
    }


def conjugacy_classes_of_selected_elements(W, elements):
    unseen = set(elements)
    classes = []

    while unseen:

        x = next(iter(unseen))
        C = element_conjugacy_class(W, x)
        C &= unseen

        classes.append(C)
        unseen -= C

    return classes


# ============================================================
# 20. MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SIM15.2 — PRIME-STRATIFIED H4 ACTION MAP")
    print("=" * 80)

    # --------------------------------------------------------
    # A. H4 REGRESSION
    # --------------------------------------------------------

    print("\nA) INDEPENDENT H4 / POLYTOPE REGRESSION")
    print("-" * 80)

    R = build_h4_roots()

    print("H4 roots =", len(R))
    print(
        "root norm^2 set =",
        sorted(
            set(
                np.round(
                    np.sum(R * R, axis=1),
                    10
                )
            )
        )
    )

    W, h4_simple, h4_simple_idx = build_W_H4(R)

    print("|W(H4)| =", len(W))
    print("H4 simple-root indices =", h4_simple_idx)

    adj600, E600 = build_600_graph(R)
    cells600 = tetrahedral_cells(adj600)
    adj120, E120 = build_120_graph(cells600)

    print(
        "600-cell =",
        len(R),
        "vertices,",
        len(E600),
        "edges,",
        len(cells600),
        "tetrahedral cells"
    )

    print(
        "120-cell =",
        len(cells600),
        "vertices,",
        len(E120),
        "edges"
    )

    print("building 120-cell action...")
    cell_action = build_action_on_cells(W, cells600)

    # --------------------------------------------------------
    # B. RECONSTRUCT T < L < N
    # --------------------------------------------------------

    print("\nB) RECONSTRUCT T < L < N")
    print("-" * 80)

    S = find_successful_structure(W)

    T = S["T"]
    L = S["L"]
    N = S["N"]
    z = S["z"]
    successful_class = S["successful_class"]

    print("\nSUCCESSFUL STRUCTURE")
    print("|T| =", len(T))
    print("|L| =", len(L))
    print("|N| =", len(N))
    print("|W| =", len(W))

    print(
        "prime decompositions target:",
        "8 = 2^3;",
        "192 = 2^6*3;",
        "576 = 2^6*3^2;",
        "14400 = 2^6*3^2*5^2"
    )

    print("T normal L =", is_normal(T, L))
    print("L normal N =", is_normal(L, N))
    print("[L:T] =", len(L) // len(T))
    print("[N:L] =", len(N) // len(L))
    print("[W:N] =", len(W) // len(N))

    # --------------------------------------------------------
    # C. C3 COMPLEMENT
    # --------------------------------------------------------

    print("\nC) EXTERNAL C3 COMPLEMENT")
    print("-" * 80)

    c_witnesses = find_C3_complement(N, L)

    print(
        "order-3 complement witnesses =",
        len(c_witnesses)
    )

    if not c_witnesses:
        raise RuntimeError(
            "No C3 complement found; SIM15.1 regression failed."
        )

    c = c_witnesses[0]

    print("chosen c order =", porder(c))
    print("c in L =", c in L)
    print("c in N =", c in N)

    inner, inner_witness = is_inner_on_group(c, L)

    print("conjugation by c inner on L =", inner)
    print("external action outer =", not inner)

    # --------------------------------------------------------
    # D. STANDARD D4 CONSTRUCTION
    # --------------------------------------------------------

    print("\nD) INDEPENDENT STANDARD W(D4)")
    print("-" * 80)

    RD4, WD4, d4_simple = build_W_D4()

    print("D4 roots =", len(RD4))
    print("|W(D4)| =", len(WD4))
    print(
        "W(D4) element-order histogram =",
        order_hist(WD4)
    )

    print(
        "L element-order histogram =",
        order_hist(L)
    )

    # --------------------------------------------------------
    # E. D4 COXETER SYSTEM INSIDE L
    # --------------------------------------------------------

    print("\nE) CONSTRUCT D4 COXETER SYSTEM INSIDE L")
    print("-" * 80)

    coxL = find_D4_coxeter_system(L)

    if coxL is None:
        print("D4 Coxeter system found = False")
    else:
        print("D4 Coxeter system found = True")

        r1, r3, r4 = coxL["outer"]
        r2 = coxL["central"]

        print("central generator order =", porder(r2))
        print(
            "outer generator orders =",
            [porder(x) for x in (r1,r3,r4)]
        )

        print(
            "central-outer product orders =",
            [
                porder(pmul(r2, x))
                for x in (r1,r3,r4)
            ]
        )

        print(
            "outer-outer product orders =",
            [
                porder(pmul(a,b))
                for a,b in itertools.combinations(
                    (r1,r3,r4), 2
                )
            ]
        )

        print(
            "generated order =",
            len(
                subgroup_generated(
                    coxL["generators"],
                    len(r2)
                )
            )
        )

    # --------------------------------------------------------
    # F. EXPLICIT ABSTRACT ISOMORPHISM L -> W(D4)
    # --------------------------------------------------------

    print("\nF) EXPLICIT L -> W(D4) ISOMORPHISM")
    print("-" * 80)

    iso = find_isomorphism_via_generator_images(L, WD4)

    ISO_PASS = iso is not None

    print("explicit group isomorphism found =", ISO_PASS)

    if ISO_PASS:
        phi = iso["phi"]

        print(
            "mapped elements =",
            len(phi),
            "distinct images =",
            len(set(phi.values()))
        )

        print(
            "source generator orders =",
            [
                porder(g)
                for g in iso["source_generators"]
            ]
        )

        print(
            "target generator orders =",
            [
                porder(g)
                for g in iso["target_generators"]
            ]
        )

    # --------------------------------------------------------
    # G. CONSTRUCTIVE TRIALITY TEST
    # --------------------------------------------------------

    print("\nG) CONSTRUCTIVE D4 TRIALITY TEST")
    print("-" * 80)

    if coxL is not None:

        trial = automorphism_matches_triality_mod_inner(
            c,
            L,
            coxL
        )

        print(
            "triality diagram action modulo inner =",
            trial["pass"]
        )

        if trial["pass"]:
            print(
                "outer-node cycle =",
                trial["outer_cycle"]
            )

            print(
                "inner adjustment order =",
                porder(
                    trial["inner_adjustment"]
                )
            )

    else:
        trial = {"pass": False}

    TRIALITY_IDENTIFIED = (
        ISO_PASS
        and coxL is not None
        and not inner
        and trial["pass"]
    )

    print(
        "D4 TRIALITY IDENTIFIED =",
        TRIALITY_IDENTIFIED
    )

    # --------------------------------------------------------
    # H. THREE SUCCESSFUL T KERNELS
    # --------------------------------------------------------

    print("\nH) THREE-KERNEL REGRESSION")
    print("-" * 80)

    Tinside = successful_T_inside_L(
        successful_class,
        L
    )

    print(
        "successful-class T <= L =",
        len(Tinside)
    )

    c_T_transport = transport_subgroups(
        c,
        Tinside
    )

    print(
        "c transport on successful T kernels =",
        c_T_transport
    )

    # Determine common central involutions.
    e120 = pid(120)

    nontrivial_intersection = (
        set.intersection(
            *[set(H) for H in Tinside]
        )
        if Tinside
        else set()
    )

    print(
        "intersection of successful T kernels order =",
        len(nontrivial_intersection)
    )

    print(
        "intersection elements orders =",
        sorted(
            Counter(
                porder(x)
                for x in nontrivial_intersection
            ).items()
        )
    )

    print(
        "z belongs to every successful T =",
        all(z in H for H in Tinside)
    )

    print(
        "c fixes z by conjugation =",
        pconj(c, z) == z
    )

    # --------------------------------------------------------
    # I. 25 COMPATIBLE L STRUCTURES
    # --------------------------------------------------------

    print("\nI) CONSTRUCT 25-ELEMENT L FAMILY")
    print("-" * 80)

    Lfamily_set = conjugacy_orbit_subgroup(W, L)

    print(
        "|conjugacy orbit of L| =",
        len(Lfamily_set)
    )

    if len(Lfamily_set) != 25:
        raise RuntimeError(
            "Expected 25 conjugates of L."
        )

    # Put chosen L at index 0.
    Lfrozen = frozenset(L)

    rest = sorted(
        [
            H for H in Lfamily_set
            if H != Lfrozen
        ],
        key=lambda H: tuple(sorted(H))
    )

    Lfamily = [Lfrozen] + rest

    family_index = {
        H: i
        for i, H in enumerate(Lfamily)
    }

    print("chosen L index =", family_index[Lfrozen])

    print("building W action on L_25...")

    action25 = {}

    for g in W:

        p = []

        for H in Lfamily:

            Hg = conjugate_subgroup(g, H)
            p.append(family_index[Hg])

        action25[g] = tuple(p)

    image25 = set(action25.values())

    print("|25-action image| =", len(image25))

    # --------------------------------------------------------
    # J. KERNEL / FAITHFULNESS
    # --------------------------------------------------------

    print("\nJ) KERNEL OF W(H4) -> S25")
    print("-" * 80)

    kernel25 = permutation_action_kernel(action25)

    print("|kernel| =", len(kernel25))
    print(
        "kernel order histogram =",
        order_hist(kernel25)
    )

    print("z in kernel =", z in kernel25)

    FAITHFUL25 = len(kernel25) == 1

    print(
        "25-action faithful =",
        FAITHFUL25
    )

    # --------------------------------------------------------
    # K. STABILIZER / SUBDEGREES / RANK
    # --------------------------------------------------------

    print("\nK) SUBDEGREES AND ACTION RANK")
    print("-" * 80)

    stab0 = stabilizer_of_point_from_action(
        W,
        action25,
        0
    )

    print("|Stab(L0)| =", len(stab0))
    print(
        "Stab(L0) == N =",
        stab0 == set(N)
    )

    sorbs = suborbits(
        stab0,
        action25,
        25
    )

    subdegrees = [
        len(O)
        for O in sorbs
    ]

    print("subdegrees =", subdegrees)
    print("rank =", len(sorbs))
    print("sum =", sum(subdegrees))

    for i, O in enumerate(sorbs):
        print(
            f"  suborbit {i}: "
            f"size={len(O)} "
            f"members={sorted(O)}"
        )

    # --------------------------------------------------------
    # L. BLOCK SYSTEM TEST
    # --------------------------------------------------------

    print("\nL) BLOCK SYSTEM / PRIMITIVITY TEST")
    print("-" * 80)

    blocks, block_sorbs = find_nontrivial_blocks(
        image25,
        25,
        base=0
    )

    print(
        "nontrivial blocks containing L0 =",
        len(blocks)
    )

    for i, B in enumerate(blocks):

        print(
            f"  block {i}: "
            f"size={len(B)} "
            f"members={sorted(B)}"
        )

        orbitB = orbit_of_subset(
            image25,
            B
        )

        print(
            "    block orbit size =",
            len(orbitB)
        )

    PRIMITIVE25 = len(blocks) == 0

    print(
        "25-action primitive =",
        PRIMITIVE25
    )

    # --------------------------------------------------------
    # M. ORBITALS
    # --------------------------------------------------------

    print("\nM) ORDERED-PAIR ORBITALS")
    print("-" * 80)

    orbitals = []

    for i, O in enumerate(sorbs):

        Orb = orbital_from_suborbit(
            action25,
            W,
            0,
            O
        )

        orbitals.append(Orb)

        rev = reverse_orbital(Orb)

        print(
            f"orbital {i}: "
            f"subdegree={len(O)} "
            f"ordered_pairs={len(Orb)} "
            f"self_paired={rev == Orb}"
        )

    color = relation_color_matrix(
        25,
        orbitals
    )

    # --------------------------------------------------------
    # N. INVARIANT ORBITAL GRAPHS
    # --------------------------------------------------------

    print("\nN) INVARIANT ORBITAL GRAPH SIGNATURES")
    print("-" * 80)

    graph_sigs = []

    for i, Orb in enumerate(orbitals):

        # Skip diagonal orbital.
        if all(a == b for a, b in Orb):
            print(
                f"orbital {i}: diagonal relation"
            )
            graph_sigs.append(None)
            continue

        adj = graph_from_orbital(
            25,
            Orb,
            symmetrize=True
        )

        sig = graph_signature(adj)
        graph_sigs.append(sig)

        print(f"\norbital graph {i}")
        print(
            "  vertices =",
            sig["vertices"]
        )
        print(
            "  edges =",
            sig["edges"]
        )
        print(
            "  degree_hist =",
            sig["degree_hist"]
        )
        print(
            "  components =",
            sig["components"]
        )
        print(
            "  triangles =",
            sig["triangles"]
        )
        print(
            "  spectrum =",
            sig["spectrum"]
        )
        print(
            "  common_neighbor_hist =",
            sig["common_neighbor_hist"]
        )

    # --------------------------------------------------------
    # O. ORDER-5 ELEMENTS
    # --------------------------------------------------------

    print("\nO) ORDER-5 ELEMENT CENSUS")
    print("-" * 80)

    order5 = [
        g for g in W
        if porder(g) == 5
    ]

    print(
        "order-5 elements in W(H4) =",
        len(order5)
    )

    inN5 = [
        g for g in order5
        if g in N
    ]

    print(
        "order-5 elements in N =",
        len(inN5)
    )

    normalizeL5 = [
        g for g in order5
        if conjugate_subgroup(g, L) == Lfrozen
    ]

    print(
        "order-5 elements normalizing L =",
        len(normalizeL5)
    )

    classes5 = conjugacy_classes_of_selected_elements(
        W,
        order5
    )

    print(
        "order-5 conjugacy classes =",
        len(classes5)
    )

    print(
        "order-5 class sizes =",
        sorted(len(C) for C in classes5)
    )

    # --------------------------------------------------------
    # P. C5 ACTION ON L_25
    # --------------------------------------------------------

    print("\nP) C5 ACTION ON L_25")
    print("-" * 80)

    sig5 = Counter(
        tuple(sorted(cycle_signature(action25[g]).items()))
        for g in order5
    )

    print(
        "distinct order-5 cycle signatures on L_25 =",
        len(sig5)
    )

    for sig, count in sig5.items():
        print(
            "  signature",
            dict(sig),
            "count",
            count
        )

    ALL_5_5 = (
        len(sig5) == 1
        and next(iter(sig5.keys())) == ((5,5),)
    )

    print(
        "all order-5 elements act as 5^5 =",
        ALL_5_5
    )

    # Representative f.
    f = order5[0]

    print(
        "representative f cycle signature L_25 =",
        cycle_signature(action25[f])
    )

    # --------------------------------------------------------
    # Q. C5 PENTADS VS INTRINSIC ORBITALS
    # --------------------------------------------------------

    print("\nQ) C5 PENTAD ORBITAL SIGNATURES")
    print("-" * 80)

    pentad_signatures = Counter()
    pentads = set()

    for g in order5:

        p = action25[g]

        for C in pcycles(p, include_fixed=True):

            if len(C) != 5:
                continue

            P = frozenset(C)
            pentads.add(P)

            sig = subset_orbital_signature(
                P,
                color
            )

            pentad_signatures[sig] += 1

    print(
        "distinct C5 pentads =",
        len(pentads)
    )

    print(
        "distinct intrinsic orbital signatures of C5 pentads =",
        len(pentad_signatures)
    )

    for i, (sig, count) in enumerate(
        sorted(
            pentad_signatures.items(),
            key=lambda kv: (repr(kv[0]), kv[1])
        )
    ):
        print(
            f"  pentad signature {i}: "
            f"occurrences={count}"
        )
        print(
            "    orbital counts =",
            sig
        )

    # Check whether any C5 pentad is a block.
    pentad_blocks = [
        P for P in pentads
        if is_block(image25, P, 25)
    ]

    print(
        "C5 pentads that are W-blocks =",
        len(pentad_blocks)
    )

    # --------------------------------------------------------
    # R. C3 ACTION ON L_25
    # --------------------------------------------------------

    print("\nR) C3 ACTION ON L_25")
    print("-" * 80)

    pc = action25[c]

    print(
        "c cycle signature on L_25 =",
        cycle_signature(pc)
    )

    print(
        "c fixes chosen L0 =",
        pc[0] == 0
    )

    # All complement witnesses.
    c3_sigs = Counter(
        tuple(
            sorted(
                cycle_signature(
                    action25[x]
                ).items()
            )
        )
        for x in c_witnesses
    )

    print(
        "cycle signatures of C3 complement witnesses on L_25:"
    )

    for sig, count in c3_sigs.items():
        print(
            "  ",
            dict(sig),
            "count",
            count
        )

    # --------------------------------------------------------
    # S. C3 / C5 ON SAME OBJECTS
    # --------------------------------------------------------

    print("\nS) PRIME-STRATIFIED C3 / C5 COMPARISON")
    print("-" * 80)

    print("C3 representative c:")
    print("  order =", porder(c))
    print("  in L =", c in L)
    print("  in N =", c in N)
    print(
        "  L_25 cycles =",
        cycle_signature(action25[c])
    )
    print(
        "  600-cell cycles =",
        cycle_signature_on_600(c)
    )
    print(
        "  120-cell cycles =",
        cycle_signature_on_120(
            c,
            cell_action
        )
    )

    print("\nC5 representative f:")
    print("  order =", porder(f))
    print("  in L =", f in L)
    print("  in N =", f in N)
    print(
        "  normalizes L =",
        conjugate_subgroup(f, L) == Lfrozen
    )
    print(
        "  L_25 cycles =",
        cycle_signature(action25[f])
    )
    print(
        "  600-cell cycles =",
        cycle_signature_on_600(f)
    )
    print(
        "  120-cell cycles =",
        cycle_signature_on_120(
            f,
            cell_action
        )
    )

    # --------------------------------------------------------
    # T. C3 / C5 ACTION ON SUCCESSFUL T KERNELS
    # --------------------------------------------------------

    print("\nT) ACTION ON SUCCESSFUL T KERNELS")
    print("-" * 80)

    print(
        "successful T kernels inside L =",
        len(Tinside)
    )

    print(
        "c transport within L =",
        transport_subgroups(c, Tinside)
    )

    # f moves L, so compare kernels to those inside fLf^-1.
    Lf = set(conjugate_subgroup(f, L))

    Tinside_fL = successful_T_inside_L(
        successful_class,
        Lf
    )

    print(
        "successful T kernels inside fLf^-1 =",
        len(Tinside_fL)
    )

    f_images = [
        set(conjugate_subgroup(f, H))
        for H in Tinside
    ]

    f_target_index = {
        frozenset(H): i
        for i, H in enumerate(Tinside_fL)
    }

    print(
        "f transport from kernels in L to kernels in fLf^-1 =",
        [
            f_target_index.get(
                frozenset(H),
                None
            )
            for H in f_images
        ]
    )

    # --------------------------------------------------------
    # U. INTERSECTION GEOMETRY OF THE 25 L'S
    # --------------------------------------------------------

    print("\nU) PAIRWISE INTERSECTION ORDERS ON L_25")
    print("-" * 80)

    intersection_hist = Counter()
    relation_by_intersection = defaultdict(Counter)

    for i in range(25):
        for j in range(i + 1, 25):

            s = len(
                set(Lfamily[i]) &
                set(Lfamily[j])
            )

            intersection_hist[s] += 1

            oi = color[(i, j)]
            oj = color[(j, i)]

            relation_by_intersection[s][
                (oi, oj)
            ] += 1

    print(
        "pairwise intersection-order histogram =",
        dict(sorted(intersection_hist.items()))
    )

    print(
        "intersection order -> orbital-pair counts:"
    )

    for s in sorted(relation_by_intersection):

        print(
            f"  |Li ∩ Lj|={s}:",
            dict(
                sorted(
                    relation_by_intersection[s].items()
                )
            )
        )

    # --------------------------------------------------------
    # V. COMMON T / CENTER INCIDENCE AMONG L'S
    # --------------------------------------------------------

    print("\nV) CENTER / SUCCESSFUL-KERNEL INCIDENCE ON L_25")
    print("-" * 80)

    centers_L = []

    for H in Lfamily:

        ZH = center(set(H))
        centers_L.append(ZH)

    center_sizes = Counter(
        len(ZH)
        for ZH in centers_L
    )

    print(
        "center-size histogram =",
        dict(sorted(center_sizes.items()))
    )

    central_involutions = []

    for ZH in centers_L:

        e = pid(120)

        nz = [
            x for x in ZH
            if x != e
        ]

        central_involutions.append(
            nz[0] if len(nz) == 1 else None
        )

    print(
        "distinct nontrivial central involutions among 25 L's =",
        len(
            set(
                x for x in central_involutions
                if x is not None
            )
        )
    )

    print(
        "all 25 share same central involution =",
        len(
            set(
                x for x in central_involutions
                if x is not None
            )
        ) == 1
    )

    print(
        "shared central involution equals z =",
        all(
            x == z
            for x in central_involutions
            if x is not None
        )
    )

    # --------------------------------------------------------
    # W. 25-SET STRUCTURE — NO F5^2 ASSUMPTION
    # --------------------------------------------------------

    print("\nW) BLIND 25-SET STRUCTURE SUMMARY")
    print("-" * 80)

    print("degree =", 25)
    print("image order =", len(image25))
    print("kernel order =", len(kernel25))
    print("faithful =", FAITHFUL25)
    print("rank =", len(sorbs))
    print("subdegrees =", subdegrees)
    print("primitive =", PRIMITIVE25)
    print(
        "nontrivial block sizes =",
        sorted(
            set(len(B) for B in blocks)
        )
    )

    print(
        "pairwise L-intersection orders =",
        sorted(intersection_hist)
    )

    print(
        "number of invariant non-diagonal orbital relations =",
        sum(
            1
            for O in orbitals
            if not all(a == b for a,b in O)
        )
    )

    print(
        "distinct C5 pentad orbital signatures =",
        len(pentad_signatures)
    )

    print(
        "C5 pentads forming W-blocks =",
        len(pentad_blocks)
    )

    # --------------------------------------------------------
    # X. HARD GATES
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("X) SIM15.2 HARD GATES")
    print("=" * 80)

    gateA = TRIALITY_IDENTIFIED

    gateB = (
        len(inN5) == 0
        and len(normalizeL5) == 0
        and ALL_5_5
    )

    gateC = (
        len(Lfamily) == 25
        and len(stab0) == 576
    )

    # Gate D is descriptive, not forced true/false:
    # the point is to determine structure without inserting F5^2.

    print(
        "GATE A — explicit D4 triality identification:",
        gateA
    )

    print(
        "GATE B — C5 excluded from N and acts 5^5:",
        gateB
    )

    print(
        "GATE C — 25-action reconstructed:",
        gateC
    )

    print(
        "GATE D — blind 25-set combinatorics mapped:",
        True
    )

    # --------------------------------------------------------
    # Y. MACHINE TRUTH PACKET
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("Y) MACHINE TRUTH PACKET")
    print("=" * 80)

    truth = {
        "H4_order_14400": len(W) == 14400,

        "T_order_8": len(T) == 8,

        "L_order_192": len(L) == 192,

        "N_order_576": len(N) == 576,

        "L_over_T_order_24":
            len(L) // len(T) == 24,

        "N_over_L_order_3":
            len(N) // len(L) == 3,

        "W_over_N_order_25":
            len(W) // len(N) == 25,

        "explicit_L_iso_WD4":
            ISO_PASS,

        "external_C3_outer":
            not inner,

        "D4_triality_identified":
            TRIALITY_IDENTIFIED,

        "three_successful_T_in_L":
            len(Tinside) == 3,

        "C3_cycles_successful_T":
            sorted(c_T_transport) == [0,1,2]
            if all(x is not None for x in c_T_transport)
            else False,

        "z_common_to_three_T":
            all(z in H for H in Tinside),

        "C3_fixes_z":
            pconj(c, z) == z,

        "L_family_size_25":
            len(Lfamily) == 25,

        "action25_kernel_order":
            len(kernel25),

        "action25_faithful":
            FAITHFUL25,

        "action25_rank":
            len(sorbs),

        "action25_subdegrees":
            subdegrees,

        "action25_primitive":
            PRIMITIVE25,

        "order5_elements":
            len(order5),

        "order5_in_N":
            len(inN5),

        "order5_normalizing_L":
            len(normalizeL5),

        "all_order5_cycle_5_5":
            ALL_5_5,

        "distinct_C5_pentads":
            len(pentads),

        "C5_pentad_signature_types":
            len(pentad_signatures),

        "C5_pentad_blocks":
            len(pentad_blocks),

        "distinct_L_intersection_orders":
            sorted(intersection_hist),

        "all_L_share_same_center":
            len(
                set(
                    x for x in central_involutions
                    if x is not None
                )
            ) == 1,

        "shared_center_is_z":
            all(
                x == z
                for x in central_involutions
                if x is not None
            ),
    }

    for k, v in truth.items():
        print(
            f"{k:38s}: {v}"
        )

    # --------------------------------------------------------
    # Z. INTERPRETATION GUARDRAILS
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("Z) INTERPRETATION GUARDRAILS")
    print("=" * 80)

    print("""
1. C2, C3, and C5 are NOT assumed to be three physical layers.

2. T ~= C2^3 is the distinguished operational 2-primary kernel.
   This does not imply all twofold symmetry in W(H4) belongs to T.

3. C3 is promoted to D4 triality ONLY if the explicit Coxeter-system
   test passes modulo inner conjugacy.

4. C5 cannot normalize L if N_W(L) has order 576.  Its appearance
   on the 25-set is therefore first interpreted as coset/conjugacy
   transport, not as an independent local operational layer.

5. 25 = 5^2 does NOT imply F5^2.

6. Five 5-cycles do NOT imply five intrinsic pentagons or a block
   system.  The block and orbital tests decide this.

7. If the 25-action is primitive, any proposed 5+5+5+5+5 global
   decomposition is rejected unless another independently defined
   structure supplies it.

8. If the 25-action has nontrivial kernel, report exactly which
   ambient H4 symmetry becomes invisible at architecture level.

9. Pairwise intersection orders and orbitals are treated as native
   relational data of the 25 compatible L structures.

10. No RCFT physical interpretation follows from this run alone.

END SIM15.2
""")


if __name__ == "__main__":
    main()







~~~~~~~~~~~~~~~~~~~





RESULTS:







================================================================================
SIM15.2 — PRIME-STRATIFIED H4 ACTION MAP
================================================================================

A) INDEPENDENT H4 / POLYTOPE REGRESSION--------------------------------------------------------------------------------
H4 roots = 120
root norm^2 set = [2.0]
|W(H4)| = 14400
H4 simple-root indices = [0, 76, 5, 41]
600-cell = 120 vertices, 720 edges, 600 tetrahedral cells
120-cell = 600 vertices, 1200 edges
building 120-cell action...

B) RECONSTRUCT T < L < N--------------------------------------------------------------------------------
involutions in W(H4): 571
C2^3 subgroups: 1200
C2^3 conjugacy classes: 5
class sizes: [75, 75, 300, 300, 450]
class 0: orbit=450 normalizer=32
class 1: orbit=300 normalizer=48
class 2: orbit=75 normalizer=192
  fingerprint: {'T8': True, 'N192': True, 'centralizer_T': False, 'GL24': False, 'GL_S4_hist': False, 'one_nonzero_fixed': True, 'center2': True, 'center_fixed': True} FULL= False
class 3: orbit=300 normalizer=48
class 4: orbit=75 normalizer=192
  fingerprint: {'T8': True, 'N192': True, 'centralizer_T': True, 'GL24': True, 'GL_S4_hist': True, 'one_nonzero_fixed': True, 'center2': True, 'center_fixed': True} FULL= True

SUCCESSFUL STRUCTURE|T| = 8
|L| = 192
|N| = 576
|W| = 14400
prime decompositions target: 8 = 2^3; 192 = 2^6*3; 576 = 2^6*3^2; 14400 = 2^6*3^2*5^2
T normal L = True
L normal N = True
[L:T] = 24
[N:L] = 3
[W:N] = 25

C) EXTERNAL C3 COMPLEMENT--------------------------------------------------------------------------------
order-3 complement witnesses = 48
chosen c order = 3
c in L = False
c in N = True
conjugation by c inner on L = False
external action outer = True

D) INDEPENDENT STANDARD W(D4)--------------------------------------------------------------------------------
D4 roots = 24
|W(D4)| = 192
W(D4) element-order histogram = {1: 1, 2: 43, 3: 32, 4: 84, 6: 32}
L element-order histogram = {1: 1, 2: 43, 3: 32, 4: 84, 6: 32}

E) CONSTRUCT D4 COXETER SYSTEM INSIDE L--------------------------------------------------------------------------------
D4 Coxeter system found = True
central generator order = 2
outer generator orders = [2, 2, 2]
central-outer product orders = [3, 3, 3]
outer-outer product orders = [2, 2, 2]
generated order = 192

F) EXPLICIT L -> W(D4) ISOMORPHISM--------------------------------------------------------------------------------
  L greedy generator count = 3
explicit group isomorphism found = True
mapped elements = 192 distinct images = 192
source generator orders = [4, 4, 4]
target generator orders = [4, 4, 4]

G) CONSTRUCTIVE D4 TRIALITY TEST--------------------------------------------------------------------------------
triality diagram action modulo inner = True
outer-node cycle = (2, 0, 1)
inner adjustment order = 2
D4 TRIALITY IDENTIFIED = True

H) THREE-KERNEL REGRESSION--------------------------------------------------------------------------------
successful-class T <= L = 3
c transport on successful T kernels = [2, 0, 1]
intersection of successful T kernels order = 2
intersection elements orders = [(1, 1), (2, 1)]
z belongs to every successful T = True
c fixes z by conjugation = True

I) CONSTRUCT 25-ELEMENT L FAMILY--------------------------------------------------------------------------------
|conjugacy orbit of L| = 25
chosen L index = 0
building W action on L_25...
|25-action image| = 7200

J) KERNEL OF W(H4) -> S25--------------------------------------------------------------------------------
|kernel| = 2
kernel order histogram = {1: 1, 2: 1}
z in kernel = True
25-action faithful = False

K) SUBDEGREES AND ACTION RANK--------------------------------------------------------------------------------
|Stab(L0)| = 576
Stab(L0) == N = True
subdegrees = [1, 8, 16]
rank = 3
sum = 25
  suborbit 0: size=1 members=[0]
  suborbit 1: size=8 members=[1, 3, 6, 10, 15, 18, 20, 23]
  suborbit 2: size=16 members=[2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 16, 17, 19, 21, 22, 24]

L) BLOCK SYSTEM / PRIMITIVITY TEST--------------------------------------------------------------------------------
nontrivial blocks containing L0 = 0
25-action primitive = True

M) ORDERED-PAIR ORBITALS--------------------------------------------------------------------------------
orbital 0: subdegree=1 ordered_pairs=25 self_paired=True
orbital 1: subdegree=8 ordered_pairs=200 self_paired=True
orbital 2: subdegree=16 ordered_pairs=400 self_paired=True

N) INVARIANT ORBITAL GRAPH SIGNATURES--------------------------------------------------------------------------------
orbital 0: diagonal relation

orbital graph 1  vertices = 25
  edges = 100
  degree_hist = {8: 25}
  components = [25]
  triangles = 100
  spectrum = [(-2.0, 16), (3.0, 8), (8.0, 1)]
  common_neighbor_hist = {('adj', 3): 100, ('nonadj', 2): 200}

orbital graph 2  vertices = 25
  edges = 200
  degree_hist = {16: 25}
  components = [25]
  triangles = 600
  spectrum = [(-4.0, 8), (1.0, 16), (16.0, 1)]
  common_neighbor_hist = {('adj', 9): 200, ('nonadj', 12): 100}

O) ORDER-5 ELEMENT CENSUS--------------------------------------------------------------------------------
order-5 elements in W(H4) = 624
order-5 elements in N = 0
order-5 elements normalizing L = 0
order-5 conjugacy classes = 5
order-5 class sizes = [24, 24, 144, 144, 288]

P) C5 ACTION ON L_25--------------------------------------------------------------------------------
distinct order-5 cycle signatures on L_25 = 1
  signature {5: 5} count 624
all order-5 elements act as 5^5 = True
representative f cycle signature L_25 = {5: 5}

Q) C5 PENTAD ORBITAL SIGNATURES--------------------------------------------------------------------------------
distinct C5 pentads = 130
distinct intrinsic orbital signatures of C5 pentads = 2
  pentad signature 0: occurrences=240
    orbital counts = ((1, 20),)
  pentad signature 1: occurrences=2880
    orbital counts = ((2, 20),)
C5 pentads that are W-blocks = 0

R) C3 ACTION ON L_25--------------------------------------------------------------------------------
c cycle signature on L_25 = {1: 4, 3: 7}
c fixes chosen L0 = True
cycle signatures of C3 complement witnesses on L_25:
   {1: 4, 3: 7} count 32
   {1: 10, 3: 5} count 16

S) PRIME-STRATIFIED C3 / C5 COMPARISON--------------------------------------------------------------------------------
C3 representative c:
  order = 3
  in L = False
  in N = True
  L_25 cycles = {1: 4, 3: 7}
  600-cell cycles = {1: 6, 3: 38}
  120-cell cycles = {1: 12, 3: 196}

C5 representative f:  order = 5
  in L = False
  in N = False
  normalizes L = False
  L_25 cycles = {5: 5}
  600-cell cycles = {5: 24}
  120-cell cycles = {5: 120}

T) ACTION ON SUCCESSFUL T KERNELS--------------------------------------------------------------------------------
successful T kernels inside L = 3
c transport within L = [2, 0, 1]
successful T kernels inside fLf^-1 = 3
f transport from kernels in L to kernels in fLf^-1 = [0, 1, 2]

U) PAIRWISE INTERSECTION ORDERS ON L_25--------------------------------------------------------------------------------
pairwise intersection-order histogram = {8: 100, 12: 200}
intersection order -> orbital-pair counts:
  |Li ∩ Lj|=8: {(1, 1): 100}
  |Li ∩ Lj|=12: {(2, 2): 200}

V) CENTER / SUCCESSFUL-KERNEL INCIDENCE ON L_25--------------------------------------------------------------------------------
center-size histogram = {2: 25}
distinct nontrivial central involutions among 25 L's = 1
all 25 share same central involution = True
shared central involution equals z = True

W) BLIND 25-SET STRUCTURE SUMMARY--------------------------------------------------------------------------------
degree = 25
image order = 7200
kernel order = 2
faithful = False
rank = 3
subdegrees = [1, 8, 16]
primitive = True
nontrivial block sizes = []
pairwise L-intersection orders = [8, 12]
number of invariant non-diagonal orbital relations = 2
distinct C5 pentad orbital signatures = 2
C5 pentads forming W-blocks = 0

================================================================================X) SIM15.2 HARD GATES
================================================================================
GATE A — explicit D4 triality identification: True
GATE B — C5 excluded from N and acts 5^5: True
GATE C — 25-action reconstructed: True
GATE D — blind 25-set combinatorics mapped: True

================================================================================Y) MACHINE TRUTH PACKET
================================================================================
H4_order_14400                        : True
T_order_8                             : True
L_order_192                           : True
N_order_576                           : True
L_over_T_order_24                     : True
N_over_L_order_3                      : True
W_over_N_order_25                     : True
explicit_L_iso_WD4                    : True
external_C3_outer                     : True
D4_triality_identified                : True
three_successful_T_in_L               : True
C3_cycles_successful_T                : True
z_common_to_three_T                   : True
C3_fixes_z                            : True
L_family_size_25                      : True
action25_kernel_order                 : 2
action25_faithful                     : False
action25_rank                         : 3
action25_subdegrees                   : [1, 8, 16]
action25_primitive                    : True
order5_elements                       : 624
order5_in_N                           : 0
order5_normalizing_L                  : 0
all_order5_cycle_5_5                  : True
distinct_C5_pentads                   : 130
C5_pentad_signature_types             : 2
C5_pentad_blocks                      : 0
distinct_L_intersection_orders        : [8, 12]
all_L_share_same_center               : True
shared_center_is_z                    : True

================================================================================Z) INTERPRETATION GUARDRAILS
================================================================================

1. C2, C3, and C5 are NOT assumed to be three physical layers.

2. T ~= C2^3 is the distinguished operational 2-primary kernel.
   This does not imply all twofold symmetry in W(H4) belongs to T.

3. C3 is promoted to D4 triality ONLY if the explicit Coxeter-system
   test passes modulo inner conjugacy.

4. C5 cannot normalize L if N_W(L) has order 576.  Its appearance
   on the 25-set is therefore first interpreted as coset/conjugacy
   transport, not as an independent local operational layer.

5. 25 = 5^2 does NOT imply F5^2.

6. Five 5-cycles do NOT imply five intrinsic pentagons or a block
   system.  The block and orbital tests decide this.

7. If the 25-action is primitive, any proposed 5+5+5+5+5 global
   decomposition is rejected unless another independently defined
   structure supplies it.

8. If the 25-action has nontrivial kernel, report exactly which
   ambient H4 symmetry becomes invisible at architecture level.

9. Pairwise intersection orders and orbitals are treated as native
   relational data of the 25 compatible L structures.

10. No RCFT physical interpretation follows from this run alone.

END SIM15.2
