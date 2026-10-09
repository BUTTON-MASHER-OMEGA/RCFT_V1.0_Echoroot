SIM15.1 — H4 NORMALIZER–PRIMITIVE CLOSURE MAP 
==============================================

NO RCFT DYNAMICS.
NO LCO / ISP.
NO J SEARCH.
NO METRIC INTERPRETATION BEYOND THE NATIVE H4 ROOT REALIZATION.
NO ASSUMPTION THAT THE 576-GROUP IS A TRIALITY EXTENSION.
NO ASSUMPTION THAT ANY H4 8-SET IS THE OPS X8.

PURPOSE
-------
SIM15.0 established that one conjugacy class of T ~= C2^3 inside W(H4)
has normalizer

    L = N_W(T),  |L| = 192,

with the full frozen SIM14.8 finite-group fingerprint.

SIM15.1 asks two questions, in this order:

PHASE I — AMBIENT CLOSURE
-------------------------
1. Reconstruct the successful SIM15.0 T and L blindly.
2. Compute N = N_W(L).
3. Verify |N| = 576 and N/L ~= C3.
4. Determine whether the extension splits.
5. Constructively compare L with W(D4) = C2^3 : S4.
6. Determine whether the external C3 acts nontrivially/outerly on L.
7. Enumerate successful-class T subgroups contained in L.
8. Test whether one C3 coset representative cycles them.
9. Test whether the SAME element cycles the repeated 600-cell and
   120-cell L-orbit triples.
10. Test whether the distinguished central involution z acts as -I
    (antipodal map) on the 600-cell and dual 120-cell.

PHASE II — PRIMITIVE DESCENT
----------------------------
11. Search natural H4 carriers for regular T-orbits of size 8.
12. On each Y8, inspect the z-pairing geometrically.
13. Inspect induced H4 adjacency on Y8 and classify any C8 survival.
14. Enumerate V4 subgroups K' <= T and compare their geometric signatures.
15. Do NOT manufacture an alternating form on Y8.  Only report native
    unsigned/signed structures actually supplied by H4 geometry.

IMPORTANT
---------
"triality" is NOT printed as confirmed merely because |N/L| = 3.
This script distinguishes:

    quotient C3,
    split C3 complement,
    non-inner action on L,
    cycling of successful T kernels,
    cycling of geometric orbit triples.

A genuine D4-triality identification requires the combined evidence,
and should still be compared explicitly with the standard D4 diagram
automorphism after this run.

DEPENDENCIES
------------
Python 3
numpy

The computation is finite/exhaustive.  No GAP/Sage required.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
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
        if not seen[i]:
            j = i
            L = 0
            while not seen[j]:
                seen[j] = True
                j = p[j]
                L += 1
            if L:
                ans = math.lcm(ans, L)
    return ans


def pcycles(p):
    n = len(p)
    seen = [False] * n
    cyc = []
    for i in range(n):
        if not seen[i]:
            c = []
            j = i
            while not seen[j]:
                seen[j] = True
                c.append(j)
                j = p[j]
            if len(c) > 1:
                cyc.append(tuple(c))
    return tuple(cyc)


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
    queue = [e]

    while queue:
        x = queue.pop()
        for g in gens:
            y = pmul(g, x)
            if y not in G:
                G.add(y)
                queue.append(y)
    return G


def subgroup_generated(elements, n):
    return generated_group(list(elements), n=n)


def center(G):
    Glist = list(G)
    return {
        z for z in Glist
        if all(pmul(z, g) == pmul(g, z) for g in Glist)
    }


def centralizer_in(W, H):
    Hlist = list(H)
    return {
        g for g in W
        if all(pmul(g, h) == pmul(h, g) for h in Hlist)
    }


def normalizer_in(W, H):
    Hset = set(H)
    return {
        g for g in W
        if {pconj(g, h) for h in Hset} == Hset
    }


def is_normal(H, G):
    Hset = set(H)
    return all({pconj(g, h) for h in Hset} == Hset for g in G)


def orbit_of_point(G, x):
    return {g[x] for g in G}


def orbits_on_points(G, n):
    unseen = set(range(n))
    out = []
    while unseen:
        x = min(unseen)
        O = orbit_of_point(G, x)
        out.append(O)
        unseen -= O
    return sorted(out, key=lambda z: (len(z), sorted(z)))


def orbit_of_object(G, obj, action):
    return {action(g, obj) for g in G}


def order_hist(G):
    return dict(sorted(Counter(porder(g) for g in G).items()))


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
    Standard 120-vertex 600-cell / H4 root system.

    Coordinates before final scaling:
      8:   permutations of (±2,0,0,0)
      16:  (±1,±1,±1,±1)
      96:  even permutations of (0, ±1, ±phi, ±1/phi)

    Divide by sqrt(2), giving squared norm 2.
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
        raise RuntimeError(f"H4 root construction produced {len(R)} roots, expected 120.")

    norms = np.sum(R * R, axis=1)
    if not np.allclose(norms, 2.0, atol=1e-8):
        raise RuntimeError("H4 roots do not all have norm^2 = 2.")

    return R


def vec_key(v, decimals=9):
    return tuple(np.round(v, decimals))


def build_root_lookup(R):
    return {vec_key(v): i for i, v in enumerate(R)}


def linear_map_to_perm(M, R, lookup):
    p = []
    for v in R:
        w = M @ v
        key = vec_key(w)
        if key not in lookup:
            # nearest fallback
            d = np.linalg.norm(R - w[None, :], axis=1)
            j = int(np.argmin(d))
            if d[j] > 1e-6:
                raise RuntimeError("Linear map failed to permute H4 roots.")
            p.append(j)
        else:
            p.append(lookup[key])
    return tuple(p)


def reflection_matrix(alpha):
    """
    For alpha.alpha = 2:
        s_alpha(v) = v - (v.alpha) alpha
    """
    alpha = np.asarray(alpha, dtype=float)
    return np.eye(4) - np.outer(alpha, alpha)


def find_simple_h4_roots(R):
    """
    Find alpha1,...,alpha4 with Coxeter diagram 5-3-3:

      alpha1 --5-- alpha2 --3-- alpha3 --3-- alpha4

    With norm^2 = 2:
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

    # Fix alpha1 to first root; transitivity makes this harmless.
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

    raise RuntimeError("Could not locate H4 simple-root system.")


def build_W_H4(R):
    lookup = build_root_lookup(R)
    simple_idx = find_simple_h4_roots(R)
    simple = [R[i] for i in simple_idx]

    refl_mats = [reflection_matrix(a) for a in simple]
    refl_perms = [linear_map_to_perm(M, R, lookup) for M in refl_mats]

    W = generated_group(refl_perms)

    if len(W) != 14400:
        raise RuntimeError(f"|W(H4)| = {len(W)}, expected 14400.")

    return W, refl_perms, simple_idx


# ============================================================
# 2. 600-CELL GRAPH / TETRAHEDRAL CELLS
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
        raise RuntimeError(f"600-cell edge count {len(edges)}, expected 720.")

    deg = {len(a) for a in adj}
    if deg != {12}:
        raise RuntimeError(f"600-cell degree set {deg}, expected {{12}}.")

    return adj, edges, edge_d2


def tetrahedral_cells(adj):
    """
    Tetrahedral cells are 4-cliques in the 600-cell graph.
    """
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
        raise RuntimeError(f"Found {len(cells)} tetrahedral cells, expected 600.")

    return sorted(cells)


# ============================================================
# 3. DUAL 120-CELL
# ============================================================

def build_dual_vertices(R, cells):
    """
    A dual vertex is represented by the normalized cell centroid direction.

    For permutation/combinatorial actions we primarily use tetrahedral-cell
    indices, so exact dual scaling is irrelevant.
    """
    dual = []
    for cell in cells:
        c = np.mean(R[list(cell)], axis=0)
        dual.append(c)
    return np.array(dual)


def induced_perm_on_subsets(p, subsets, subset_index):
    out = []
    for S in subsets:
        image = tuple(sorted(p[i] for i in S))
        out.append(subset_index[image])
    return tuple(out)


def build_action_on_cells(W, cells):
    cell_index = {c: i for i, c in enumerate(cells)}

    cache = {}
    for g in W:
        cache[g] = induced_perm_on_subsets(g, cells, cell_index)

    return cache


def build_120_graph(cells):
    """
    Dual cells are adjacent when the corresponding 600-cell tetrahedra
    share a triangular face.
    """
    face_to_cells = defaultdict(list)

    for ci, cell in enumerate(cells):
        for face in itertools.combinations(cell, 3):
            face_to_cells[tuple(sorted(face))].append(ci)

    edges = set()
    adj = [set() for _ in cells]

    for face, owners in face_to_cells.items():
        if len(owners) != 2:
            raise RuntimeError(
                f"Triangular face {face} belongs to {len(owners)} cells, expected 2."
            )

        a, b = owners
        if a > b:
            a, b = b, a

        edges.add((a, b))
        adj[a].add(b)
        adj[b].add(a)

    if len(edges) != 1200:
        raise RuntimeError(f"120-cell edge count {len(edges)}, expected 1200.")

    if {len(a) for a in adj} != {4}:
        raise RuntimeError("Dual 120-cell is not 4-regular.")

    return adj, edges


# ============================================================
# 4. FIND C2^3 SUBGROUPS
# ============================================================

def commute(a, b):
    return pmul(a, b) == pmul(b, a)


def enumerate_C2_3_subgroups(W):
    """
    Enumerate elementary abelian order-8 subgroups from commuting involutions.
    """
    Wlist = list(W)
    e = pid(len(Wlist[0]))

    invol = [g for g in Wlist if g != e and porder(g) == 2]

    print(f"involutions in W(H4): {len(invol)}")

    comm = {}
    for i, a in enumerate(invol):
        comm[a] = [b for b in invol if b != a and commute(a, b)]

    groups = set()

    for i, a in enumerate(invol):
        for b in comm[a]:
            if b <= a:
                continue

            H4 = subgroup_generated([a, b], len(a))
            if len(H4) != 4:
                continue

            for c in comm[a]:
                if c in H4:
                    continue
                if not commute(b, c):
                    continue

                H8 = subgroup_generated([a, b, c], len(a))

                if len(H8) == 8 and all(porder(x) in (1, 2) for x in H8):
                    groups.add(frozenset(H8))

    return [set(H) for H in groups]


# ============================================================
# 5. CONJUGACY CLASSES OF SUBGROUPS
# ============================================================

def conjugate_subgroup(g, H):
    return frozenset(pconj(g, h) for h in H)


def subgroup_conjugacy_classes(W, subs):
    unseen = {frozenset(H) for H in subs}
    classes = []

    while unseen:
        H = next(iter(unseen))
        orb = {conjugate_subgroup(g, H) for g in W}
        cls = orb & unseen
        classes.append(sorted(cls, key=lambda x: tuple(sorted(x))))
        unseen -= orb

    return classes


# ============================================================
# 6. F2^3 ACTION OF NORMALIZER ON T
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


def vxor(a, b):
    return tuple(x ^ y for x, y in zip(a, b))


def choose_T_basis(T):
    e = pid(len(next(iter(T))))
    nonzero = [x for x in T if x != e]

    for a, b, c in itertools.combinations(nonzero, 3):
        H = subgroup_generated([a, b, c], len(a))
        if H == set(T):
            return a, b, c

    raise RuntimeError("Could not choose F2^3 basis for T.")


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


def mat_from_action_on_T(g, elem_to_coord, coord_to_elem):
    basis = [(1,0,0), (0,1,0), (0,0,1)]
    cols = []

    for v in basis:
        x = coord_to_elem[v]
        y = pconj(g, x)
        cols.append(elem_to_coord[y])

    # matrix columns over F2
    M = tuple(
        tuple(cols[j][i] for j in range(3))
        for i in range(3)
    )
    return M


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


I3 = ((1,0,0),(0,1,0),(0,0,1))


def mat_order(M):
    x = I3
    for k in range(1, 100):
        x = mat_mul(M, x)
        if x == I3:
            return k
    raise RuntimeError("Matrix order failure.")


def normalizer_GL_image(N, T):
    elem_to_coord, coord_to_elem = T_coordinate_map(T)

    image = {
        mat_from_action_on_T(g, elem_to_coord, coord_to_elem)
        for g in N
    }

    fixed = []

    for v in F2V:
        if all(mat_apply(M, v) == v for M in image):
            fixed.append(v)

    return image, fixed, elem_to_coord, coord_to_elem


# ============================================================
# 7. SIM15.0 FULL-FINGERPRINT TEST
# ============================================================

TARGET_GL_HIST = {1:1, 2:9, 3:8, 4:6}


def full_fingerprint(W, T, verbose=False):
    N = normalizer_in(W, T)
    C = centralizer_in(W, T)

    image, fixed, elem_to_coord, coord_to_elem = normalizer_GL_image(N, T)

    zN = center(N)

    gl_hist = dict(sorted(Counter(mat_order(M) for M in image).items()))

    nonzero_fixed = [v for v in fixed if v != (0,0,0)]

    checks = {
        "T_order_8": len(T) == 8,
        "N_order_192": len(N) == 192,
        "centralizer_equals_T": C == set(T),
        "GL_image_order_24": len(image) == 24,
        "unique_nonzero_fixed": len(nonzero_fixed) == 1,
        "GL_hist_S4": gl_hist == TARGET_GL_HIST,
        "center_order_2": len(zN) == 2,
    }

    z = None
    if len(zN) == 2:
        e = pid(len(next(iter(T))))
        z = next(x for x in zN if x != e)

    if z is not None and len(nonzero_fixed) == 1:
        fixed_elem = coord_to_elem[nonzero_fixed[0]]
        checks["center_equals_fixed_line"] = (z == fixed_elem)
    else:
        checks["center_equals_fixed_line"] = False

    full = all(checks.values())

    if verbose:
        print("  |T| =", len(T))
        print("  |N_W(T)| =", len(N))
        print("  |C_W(T)| =", len(C))
        print("  |GL image| =", len(image))
        print("  fixed vectors =", fixed)
        print("  GL order histogram =", gl_hist)
        print("  |Z(N)| =", len(zN))
        print("  checks =", checks)
        print("  FULL MATCH =", full)

    return {
        "full": full,
        "N": N,
        "C": C,
        "image": image,
        "fixed": fixed,
        "elem_to_coord": elem_to_coord,
        "coord_to_elem": coord_to_elem,
        "center": zN,
        "z": z,
        "checks": checks,
    }


# ============================================================
# 8. SUBGROUP ORBITS / SET ACTIONS
# ============================================================

def orbit_partition_on_set(G, objects, action):
    objects = set(objects)
    unseen = set(objects)
    out = []

    while unseen:
        x = next(iter(unseen))
        O = orbit_of_object(G, x, action)
        O &= objects
        out.append(O)
        unseen -= O

    return sorted(out, key=lambda O: (len(O), repr(sorted(O, key=repr)[:1])))


def point_orbits(G, n):
    return orbits_on_points(G, n)


def subset_action(g, S):
    return frozenset(g[x] for x in S)


def permute_orbit_set(g, O):
    return frozenset(g[x] for x in O)


# ============================================================
# 9. NORMALIZER QUOTIENT N/L
# ============================================================

def cosets_right(N, L):
    unseen = set(N)
    cosets = []

    while unseen:
        g = next(iter(unseen))
        C = {pmul(l, g) for l in L}
        cosets.append(C)
        unseen -= C

    return cosets


def find_split_C3(N, L):
    outside = [g for g in N if g not in L]

    order3 = [g for g in outside if porder(g) == 3]

    witnesses = []

    for c in order3:
        C3 = subgroup_generated([c], len(c))
        if C3 & set(L) == {pid(len(c))}:
            P = {pmul(l, q) for l in L for q in C3}
            if P == set(N):
                witnesses.append(c)

    return witnesses


# ============================================================
# 10. INNER / OUTER ACTION TEST
# ============================================================

def automorphism_on_L_by_conjugation(c, L):
    return {x: pconj(c, x) for x in L}


def is_inner_automorphism_on_L(c, L):
    """
    Test whether conjugation by c restricted to L equals conjugation
    by some l in L.
    """
    Llist = list(L)

    # Use a small generating set if possible.
    gens = greedy_generators(L)

    target = {g: pconj(c, g) for g in gens}

    for l in Llist:
        if all(pconj(l, g) == target[g] for g in gens):
            return True, l

    return False, None


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


# ============================================================
# 11. STANDARD W(D4) ABSTRACT MODEL
# ============================================================

def perm4_mul(a, b):
    return tuple(a[b[i]] for i in range(4))


def perm4_inv(a):
    out = [0] * 4
    for i, j in enumerate(a):
        out[j] = i
    return tuple(out)


S4 = list(itertools.permutations(range(4)))

EVEN_SIGNS = [
    s for s in itertools.product((0,1), repeat=4)
    if sum(s) % 2 == 0
]


def d4_mul(x, y):
    """
    W(D4) as even sign changes C2^3 semidirect S4.

    x = (sign_bits, sigma)
    Action of sigma permutes sign coordinates.
    """
    sx, px = x
    sy, py = y

    # px acts on sy
    acted = tuple(sy[px[i]] for i in range(4))
    s = tuple(sx[i] ^ acted[i] for i in range(4))
    p = perm4_mul(px, py)

    return (s, p)


D4 = {
    (s, p)
    for s in EVEN_SIGNS
    for p in S4
}

D4_E = {
    (s, (0,1,2,3))
    for s in EVEN_SIGNS
}


# ============================================================
# 12. CONSTRUCTIVE SEMIDIRECT COMPARISON L ~ W(D4)
# ============================================================

def quotient_action_on_T(L, T):
    image, fixed, elem_to_coord, coord_to_elem = normalizer_GL_image(L, T)
    return image, fixed, elem_to_coord, coord_to_elem


def d4_fingerprint():
    """
    Structural fingerprint of standard W(D4) model.
    """
    # orders computed abstractly
    e = ((0,0,0,0), (0,1,2,3))

    def inv_d4(x):
        # brute-force tiny group
        for y in D4:
            if d4_mul(x, y) == e and d4_mul(y, x) == e:
                return y
        raise RuntimeError

    def ord_d4(x):
        y = e
        for k in range(1, 25):
            y = d4_mul(x, y)
            if y == e:
                return k
        raise RuntimeError

    hist = Counter(ord_d4(x) for x in D4)

    return {
        "order": len(D4),
        "E_order": len(D4_E),
        "order_hist": dict(sorted(hist.items())),
    }


# ============================================================
# 13. GEOMETRIC ACTION HELPERS
# ============================================================

def action_on_edges(g, edge):
    a, b = edge
    x, y = g[a], g[b]
    return tuple(sorted((x, y)))


def action_on_cell_perm(gcell, ci):
    return gcell[ci]


def induced_subgraph_signature(Y, adj):
    Y = sorted(Y)
    Yset = set(Y)

    degs = []
    edge_count = 0

    for x in Y:
        d = len(adj[x] & Yset)
        degs.append(d)
        edge_count += d

    edge_count //= 2

    return {
        "n": len(Y),
        "edges": edge_count,
        "degree_hist": dict(sorted(Counter(degs).items())),
        "is_literal_C8": (
            len(Y) == 8
            and edge_count == 8
            and set(degs) == {2}
            and is_connected_induced(Yset, adj)
        ),
    }


def is_connected_induced(Y, adj):
    if not Y:
        return True

    start = next(iter(Y))
    seen = {start}
    stack = [start]

    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y in Y and y not in seen:
                seen.add(y)
                stack.append(y)

    return seen == set(Y)


def graph_distance(adj, a, b):
    if a == b:
        return 0

    seen = {a}
    frontier = {a}
    d = 0

    while frontier:
        d += 1
        nxt = set()

        for x in frontier:
            for y in adj[x]:
                if y == b:
                    return d
                if y not in seen:
                    seen.add(y)
                    nxt.add(y)

        frontier = nxt

    return None


# ============================================================
# 14. T-TORSOR SEARCH
# ============================================================

def regular_T_orbits_on_points(T, n):
    out = []

    for O in point_orbits(T, n):
        if len(O) == 8:
            x = next(iter(O))
            stab = {g for g in T if g[x] == x}
            if len(stab) == 1:
                out.append(O)

    return out


def regular_T_orbits_on_objects(T, objects, action):
    objects = set(objects)
    parts = orbit_partition_on_set(T, objects, action)
    out = []

    for O in parts:
        if len(O) != 8:
            continue

        x = next(iter(O))
        stab = {g for g in T if action(g, x) == x}

        if len(stab) == 1:
            out.append(O)

    return out


# ============================================================
# 15. V4 SUBGROUPS OF T
# ============================================================

def all_V4_subgroups(T):
    n = len(next(iter(T)))
    e = pid(n)
    nonzero = [g for g in T if g != e]

    out = set()

    for a, b in itertools.combinations(nonzero, 2):
        H = subgroup_generated([a, b], n)
        if len(H) == 4:
            out.add(frozenset(H))

    return [set(H) for H in out]


# ============================================================
# 16. ANTIPODAL TEST
# ============================================================

def antipodal_perm_roots(R):
    lookup = build_root_lookup(R)
    return tuple(lookup[vec_key(-v)] for v in R)


def antipodal_perm_cells(root_antipode, cells):
    idx = {c: i for i, c in enumerate(cells)}
    out = []

    for cell in cells:
        c2 = tuple(sorted(root_antipode[v] for v in cell))
        out.append(idx[c2])

    return tuple(out)


# ============================================================
# 17. MAIN
# ============================================================

def main():

    print("=" * 78)
    print("SIM15.1 — H4 NORMALIZER–PRIMITIVE CLOSURE MAP")
    print("=" * 78)

    # --------------------------------------------------------
    # A. H4 REGRESSION
    # --------------------------------------------------------

    print("\nA) INDEPENDENT H4 / POLYTOPE REGRESSION")
    print("-" * 78)

    R = build_h4_roots()

    print("H4 roots:", len(R))
    print("root norm^2 set:",
          sorted(set(np.round(np.sum(R*R, axis=1), 10))))

    W, simple_reflections, simple_idx = build_W_H4(R)

    print("|W(H4)| =", len(W))
    print("simple root indices =", simple_idx)

    adj600, E600, edge_d2 = build_600_graph(R)
    cells600 = tetrahedral_cells(adj600)

    print("600-cell:")
    print("  |V| =", len(R))
    print("  |E| =", len(E600))
    print("  tetrahedral cells =", len(cells600))
    print("  degree =", sorted({len(a) for a in adj600}))

    dual = build_dual_vertices(R, cells600)
    adj120, E120 = build_120_graph(cells600)

    print("120-cell:")
    print("  |V| =", len(cells600))
    print("  |E| =", len(E120))
    print("  degree =", sorted({len(a) for a in adj120}))

    print("\nBuilding W(H4) action on 120-cell vertices...")
    cell_action = build_action_on_cells(W, cells600)

    # --------------------------------------------------------
    # B. REDISCOVER SIM15.0 SUCCESSFUL CLASS
    # --------------------------------------------------------

    print("\nB) REDISCOVER SUCCESSFUL C2^3 CLASS")
    print("-" * 78)

    Tsubs = enumerate_C2_3_subgroups(W)

    print("C2^3 subgroups found:", len(Tsubs))

    classes = subgroup_conjugacy_classes(W, Tsubs)

    print("conjugacy classes:", len(classes))
    print("class orbit sizes:", sorted(len(c) for c in classes))

    successful = []

    for ci, cls in enumerate(classes):
        T0 = set(next(iter(cls)))

        N0 = normalizer_in(W, T0)

        print(f"\nclass {ci}:")
        print("  conjugacy orbit =", len(cls))
        print("  |N_W(T)| =", len(N0))

        if len(N0) != 192:
            continue

        result = full_fingerprint(W, T0, verbose=True)

        if result["full"]:
            successful.append((ci, T0, result))

    print("\nfull-fingerprint successful classes:", len(successful))

    if len(successful) != 1:
        raise RuntimeError(
            f"Expected exactly one successful class, got {len(successful)}."
        )

    class_id, T, fp = successful[0]

    L = fp["N"]
    z = fp["z"]

    print("\nSUCCESSFUL CLASS =", class_id)
    print("|T| =", len(T))
    print("|L| =", len(L))
    print("|C_W(T)| =", len(fp["C"]))
    print("|Z(L)| =", len(fp["center"]))
    print("z =", pstr(z))

    # --------------------------------------------------------
    # C. NORMALIZER OF L
    # --------------------------------------------------------

    print("\nC) AMBIENT NORMALIZER OF L")
    print("-" * 78)

    N = normalizer_in(W, L)

    print("|N_W(L)| =", len(N))
    print("|L| =", len(L))
    print("[N:L] =", len(N) // len(L))
    print("L normal in N =", is_normal(L, N))

    cosets = cosets_right(N, L)

    print("number of right cosets =", len(cosets))
    print("coset sizes =", sorted(len(c) for c in cosets))

    if len(N) != 576:
        print("WARNING: expected |N| = 576 from SIM15.0.")

    if len(cosets) == 3:
        print("QUOTIENT ORDER = 3 => N/L ~= C3")
    else:
        print("QUOTIENT ORDER != 3")

    # --------------------------------------------------------
    # D. SPLITTING TEST
    # --------------------------------------------------------

    print("\nD) C3 SPLITTING TEST")
    print("-" * 78)

    split_witnesses = find_split_C3(N, L)

    print("order-3 complement witnesses outside L:", len(split_witnesses))

    if split_witnesses:
        c = split_witnesses[0]

        print("chosen c =", pstr(c))
        print("order(c) =", porder(c))
        print("<c> intersection L =",
              len(subgroup_generated([c], len(c)) & set(L)))
        print("<L,c> order =",
              len(subgroup_generated(greedy_generators(L) + [c], len(c))))
        print("SPLIT EXTENSION = True")
    else:
        c = None
        print("SPLIT EXTENSION = False / no order-3 complement found")

    # --------------------------------------------------------
    # E. D4 STRUCTURAL COMPARISON
    # --------------------------------------------------------

    print("\nE) W(D4) STRUCTURAL COMPARISON")
    print("-" * 78)

    d4fp = d4_fingerprint()

    print("standard W(D4):")
    print("  order =", d4fp["order"])
    print("  normal even-sign subgroup order =", d4fp["E_order"])
    print("  element-order histogram =", d4fp["order_hist"])

    print("\nL:")
    print("  order =", len(L))
    print("  T order =", len(T))
    print("  element-order histogram =", order_hist(L))

    d4_hist_match = (order_hist(L) == d4fp["order_hist"])

    image, fixed, _, _ = quotient_action_on_T(L, T)

    print("|conjugation image on T| =", len(image))
    print("GL image histogram =",
          dict(sorted(Counter(mat_order(M) for M in image).items())))
    print("common fixed vectors =", fixed)

    print("D4 order match =", len(L) == 192)
    print("D4 normal 2^3 match =", len(T) == 8 and is_normal(T, L))
    print("D4 element-order histogram match =", d4_hist_match)

    # This is intentionally labelled a structural identification target,
    # not a formal proof of equality with a chosen root representation.
    D4_STRUCTURAL_PASS = (
        len(L) == 192
        and len(T) == 8
        and is_normal(T, L)
        and len(image) == 24
        and d4_hist_match
    )

    print("D4 STRUCTURAL IDENTIFICATION PASS =", D4_STRUCTURAL_PASS)

    # --------------------------------------------------------
    # F. OUTER ACTION TEST
    # --------------------------------------------------------

    print("\nF) EXTERNAL C3 ACTION ON L")
    print("-" * 78)

    if c is not None:
        inner, witness = is_inner_automorphism_on_L(c, L)

        print("conjugation by c preserves L =",
              all(pconj(c, x) in L for x in L))
        print("conjugation action inner on L =", inner)

        if witness is not None:
            print("inner witness =", pstr(witness))

        print("OUTER C3 ACTION =", not inner)

        if D4_STRUCTURAL_PASS and not inner:
            print("TRIALITY CANDIDATE = True")
            print("NOTE: candidate only; explicit D4 diagram-action comparison")
            print("      remains required for theorem-level 'triality'.")
        else:
            print("TRIALITY CANDIDATE = False")
    else:
        inner = None
        print("No split C3 witness; outer-action test skipped.")

    # --------------------------------------------------------
    # G. SUCCESSFUL-CLASS T SUBGROUPS INSIDE L
    # --------------------------------------------------------

    print("\nG) SUCCESSFUL T-KERNEL INCIDENCE INSIDE L")
    print("-" * 78)

    successful_class = {
        frozenset(H)
        for H in classes[class_id]
    }

    T_inside_L = []

    for Hf in successful_class:
        H = set(Hf)
        if H <= set(L):
            T_inside_L.append(H)

    print("successful-class T <= L:", len(T_inside_L))

    for i, H in enumerate(T_inside_L):
        print(f"  T[{i}] order =", len(H))

    if c is not None and T_inside_L:
        index = {frozenset(H): i for i, H in enumerate(T_inside_L)}

        transport = []

        for H in T_inside_L:
            H2 = frozenset(pconj(c, x) for x in H)
            transport.append(index.get(H2, None))

        print("c transport on successful T kernels =", transport)

        if len(T_inside_L) == 3 and sorted(transport) == [0,1,2]:
            # determine whether one 3-cycle
            pT = tuple(transport)
            print("kernel transport permutation =", pT)
            print("kernel transport order =", porder(pT))
            print("C3 CYCLES THREE SUCCESSFUL KERNELS =",
                  porder(pT) == 3)
        else:
            print("C3 CYCLES THREE SUCCESSFUL KERNELS = False")

    # --------------------------------------------------------
    # H. L ORBITS ON 600-CELL
    # --------------------------------------------------------

    print("\nH) L ACTION ON 600-CELL VERTICES")
    print("-" * 78)

    L_orb600 = point_orbits(L, 120)

    print("orbit sizes =", [len(O) for O in L_orb600])

    for i, O in enumerate(L_orb600):
        x = next(iter(O))
        stab = {g for g in L if g[x] == x}

        print(
            f"  orbit {i}: size={len(O):3d}, "
            f"stabilizer={len(stab):3d}"
        )

    if c is not None:
        idx600 = {frozenset(O): i for i, O in enumerate(L_orb600)}
        tr600 = []

        for O in L_orb600:
            O2 = frozenset(c[x] for x in O)
            tr600.append(idx600.get(O2, None))

        print("c transport on 600-cell L-orbits =", tr600)

    # --------------------------------------------------------
    # I. L ORBITS ON 120-CELL
    # --------------------------------------------------------

    print("\nI) L ACTION ON 120-CELL VERTICES")
    print("-" * 78)

    Lcell_perms = {cell_action[g] for g in L}
    Ncell_perms = {cell_action[g] for g in N}

    if len(Lcell_perms) != len(L):
        print("WARNING: L action on 120-cell vertices has kernel.")

    L_orb120 = point_orbits(Lcell_perms, 600)

    print("orbit sizes =", [len(O) for O in L_orb120])

    for i, O in enumerate(L_orb120):
        x = next(iter(O))
        stab = {g for g in Lcell_perms if g[x] == x}

        print(
            f"  orbit {i}: size={len(O):3d}, "
            f"stabilizer={len(stab):3d}"
        )

    if c is not None:
        ccell = cell_action[c]
        idx120 = {frozenset(O): i for i, O in enumerate(L_orb120)}
        tr120 = []

        for O in L_orb120:
            O2 = frozenset(ccell[x] for x in O)
            tr120.append(idx120.get(O2, None))

        print("c transport on 120-cell L-orbits =", tr120)

    # --------------------------------------------------------
    # J. SAME-C3 TRIPLE TEST
    # --------------------------------------------------------

    print("\nJ) SAME-C3 GEOMETRIC TRIPLE TEST")
    print("-" * 78)

    def summarize_size_transport(orbits, transport):
        groups = defaultdict(list)

        for i, O in enumerate(orbits):
            groups[len(O)].append(i)

        for size in sorted(groups):
            inds = groups[size]
            print(f"size {size}: orbit indices {inds}")

            if transport is not None:
                print("  c images:",
                      {i: transport[i] for i in inds})

    if c is not None:
        summarize_size_transport(L_orb600, tr600)
        summarize_size_transport(L_orb120, tr120)
    else:
        print("No c witness; skipped.")

    # --------------------------------------------------------
    # K. CENTRAL INVOLUTION / ANTIPODAL TEST
    # --------------------------------------------------------

    print("\nK) CENTRAL INVOLUTION z / ANTIPODAL TEST")
    print("-" * 78)

    antip600 = antipodal_perm_roots(R)

    print("z == root antipodal map =", z == antip600)
    print("z fixed 600-cell vertices =",
          sum(z[i] == i for i in range(120)))

    antip120 = antipodal_perm_cells(antip600, cells600)
    z120 = cell_action[z]

    print("z == dual antipodal map =", z120 == antip120)
    print("z fixed 120-cell vertices =",
          sum(z120[i] == i for i in range(600)))

    # --------------------------------------------------------
    # L. T ORBITS / REGULAR TORSORS — 600 VERTICES
    # --------------------------------------------------------

    print("\nL) PRIMITIVE DESCENT — REGULAR T-TORSORS")
    print("-" * 78)

    tors600V = regular_T_orbits_on_points(T, 120)

    print("regular T-torsors on 600-cell vertices:", len(tors600V))
    print("sizes:", [len(O) for O in tors600V])

    # 600 edges
    tors600E = regular_T_orbits_on_objects(
        T,
        E600,
        action_on_edges
    )

    print("regular T-torsors on 600-cell edges:", len(tors600E))

    # 600 tetrahedral cells = 120-cell vertices
    Tcell_perms = {cell_action[g] for g in T}
    tors120V = regular_T_orbits_on_points(Tcell_perms, 600)

    print("regular T-torsors on 120-cell vertices:", len(tors120V))

    # 120-cell edges
    def action120edge(groot, edge):
        pg = cell_action[groot]
        a, b = edge
        x, y = pg[a], pg[b]
        return tuple(sorted((x, y)))

    tors120E = regular_T_orbits_on_objects(
        T,
        E120,
        action120edge
    )

    print("regular T-torsors on 120-cell edges:", len(tors120E))

    # --------------------------------------------------------
    # M. Y8 GEOMETRIC SIGNATURES
    # --------------------------------------------------------

    print("\nM) Y8 GEOMETRIC SIGNATURES")
    print("-" * 78)

    def report_vertex_torsors(name, torsors, adj, coords, zperm):
        print(f"\n{name}")

        for i, Y in enumerate(torsors):
            sig = induced_subgraph_signature(Y, adj)

            pair_dist = []
            pair_ip = []
            seen_pairs = set()

            for y in Y:
                zy = zperm[y]
                pair = tuple(sorted((y, zy)))

                if pair in seen_pairs:
                    continue

                seen_pairs.add(pair)

                pair_dist.append(graph_distance(adj, y, zy))

                if coords is not None:
                    pair_ip.append(
                        round(float(np.dot(coords[y], coords[zy])), 10)
                    )

            print(f"  Y8[{i}] = {sorted(Y)}")
            print("    induced graph =", sig)
            print("    z-pair graph distances =", sorted(pair_dist))

            if pair_ip:
                print("    z-pair inner products =", sorted(pair_ip))

    report_vertex_torsors(
        "600-cell vertex torsors",
        tors600V,
        adj600,
        R,
        z
    )

    report_vertex_torsors(
        "120-cell vertex torsors",
        tors120V,
        adj120,
        dual,
        z120
    )

    # --------------------------------------------------------
    # N. V4 SUBGROUPS K' <= T
    # --------------------------------------------------------

    print("\nN) V4 SUBGROUPS INSIDE SUCCESSFUL T")
    print("-" * 78)

    V4s = all_V4_subgroups(T)

    print("number of V4 subgroups of T =", len(V4s))

    for i, Kp in enumerate(V4s):
        NK_L = normalizer_in(L, Kp)
        CK_L = centralizer_in(L, Kp)

        orb600 = point_orbits(Kp, 120)

        Kpcell = {cell_action[g] for g in Kp}
        orb120 = point_orbits(Kpcell, 600)

        print(f"\nK'[{i}]")
        print("  |N_L(K')| =", len(NK_L))
        print("  |C_L(K')| =", len(CK_L))
        print("  600 vertex orbit histogram =",
              dict(sorted(Counter(len(O) for O in orb600).items())))
        print("  120 vertex orbit histogram =",
              dict(sorted(Counter(len(O) for O in orb120).items())))

    # --------------------------------------------------------
    # O. C8 SURVIVAL GRADING
    # --------------------------------------------------------

    print("\nO) C8 PRIMITIVE SURVIVAL")
    print("-" * 78)

    literal_C8_600 = 0
    literal_C8_120 = 0

    for Y in tors600V:
        sig = induced_subgraph_signature(Y, adj600)
        if sig["is_literal_C8"]:
            literal_C8_600 += 1

    for Y in tors120V:
        sig = induced_subgraph_signature(Y, adj120)
        if sig["is_literal_C8"]:
            literal_C8_120 += 1

    print("literal induced C8 among 600-vertex Y8 =", literal_C8_600)
    print("literal induced C8 among 120-vertex Y8 =", literal_C8_120)

    if literal_C8_600 or literal_C8_120:
        C8_grade = "A: literal induced C8 exists"
    else:
        C8_grade = (
            "NO A-GRADE C8: no literal induced C8 on vertex torsors; "
            "Hamiltonian/subrelation tests deferred unless structurally motivated"
        )

    print("C8 grade =", C8_grade)

    # --------------------------------------------------------
    # P. TRUTH PACKETS
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("P) AMBIENT CLOSURE TRUTH PACKET")
    print("=" * 78)

    ambient = {
        "successful_SIM15_class_unique": len(successful) == 1,
        "L_order_192": len(L) == 192,
        "N_order_576": len(N) == 576,
        "N_over_L_order_3": len(N) == 3 * len(L),
        "split_C3_found": bool(split_witnesses),
        "D4_structural_pass": D4_STRUCTURAL_PASS,
        "external_C3_outer_on_L": (None if c is None else not inner),
        "successful_T_inside_L": len(T_inside_L),
        "z_antipodal_600": z == antip600,
        "z_antipodal_120": z120 == antip120,
    }

    for k, v in ambient.items():
        print(f"{k:36s}: {v}")

    print("\n" + "=" * 78)
    print("Q) PRIMITIVE SURVIVAL TRUTH PACKET")
    print("=" * 78)

    primitive = {
        "regular_T_torsors_600_vertices": len(tors600V),
        "regular_T_torsors_600_edges": len(tors600E),
        "regular_T_torsors_120_vertices": len(tors120V),
        "regular_T_torsors_120_edges": len(tors120E),
        "z_pairing_available": z is not None,
        "literal_C8_600_vertex_torsor": literal_C8_600,
        "literal_C8_120_vertex_torsor": literal_C8_120,
        "V4_subgroups_inside_T": len(V4s),
        "signed_Omega_recovered": False,
    }

    for k, v in primitive.items():
        print(f"{k:36s}: {v}")

    print("\nSIGNED OMEGA STATUS:")
    print("  NOT FORCED.")
    print("  SIM15.1 does not manufacture Omega_H4 from Omega_OPS.")
    print("  Only native H4 pairing/incidence information is reported.")

    print("\n" + "=" * 78)
    print("SIM15.1 MACHINE SUMMARY")
    print("=" * 78)

    print("""
Interpretation gates:

1. |N/L| = 3 alone DOES NOT establish triality.

2. A split extension plus an outer order-3 action on a constructively
   identified W(D4) is a TRIALITY CANDIDATE.

3. The strongest threefold result occurs only if the SAME c:
      - cycles successful T kernels,
      - cycles the 600-cell equal-size orbit triples,
      - cycles the 120-cell equal-size orbit triples.

4. A regular T-orbit of size 8 is only a weak X8 analogue.
   Primitive survival requires additional relational structure.

5. Literal induced C8 survival is strong.
   Merely being able to draw some Hamiltonian 8-cycle is not counted
   as primitive survival in this run.

6. z reproduces unsigned partner structure algebraically.
   z = -I would additionally give it an intrinsic H4 geometric meaning.

7. No signed symplectic form is inferred from H4 Euclidean geometry.

END SIM15.1
""")


if __name__ == "__main__":
    main()





~~~~~~~~~~~~~~~~~~~~~~~~





RESULTS:




==============================================================================
SIM15.1 — H4 NORMALIZER–PRIMITIVE CLOSURE MAP
==============================================================================

A) INDEPENDENT H4 / POLYTOPE REGRESSION------------------------------------------------------------------------------
H4 roots: 120
root norm^2 set: [2.0]
|W(H4)| = 14400
simple root indices = [0, 76, 5, 41]
600-cell:
  |V| = 120
  |E| = 720
  tetrahedral cells = 600
  degree = [12]
120-cell:
  |V| = 600
  |E| = 1200
  degree = [4]

Building W(H4) action on 120-cell vertices...
B) REDISCOVER SUCCESSFUL C2^3 CLASS------------------------------------------------------------------------------
involutions in W(H4): 571
C2^3 subgroups found: 1200
conjugacy classes: 5
class orbit sizes: [75, 75, 300, 300, 450]

class 0:  conjugacy orbit = 450
  |N_W(T)| = 32

class 1:  conjugacy orbit = 300
  |N_W(T)| = 48

class 2:  conjugacy orbit = 75
  |N_W(T)| = 192
  |T| = 8
  |N_W(T)| = 192
  |C_W(T)| = 16
  |GL image| = 12
  fixed vectors = [(0, 0, 0), (1, 1, 1)]
  GL order histogram = {1: 1, 2: 3, 3: 8}
  |Z(N)| = 2
  checks = {'T_order_8': True, 'N_order_192': True, 'centralizer_equals_T': False, 'GL_image_order_24': False, 'unique_nonzero_fixed': True, 'GL_hist_S4': False, 'center_order_2': True, 'center_equals_fixed_line': True}
  FULL MATCH = False

class 3:  conjugacy orbit = 300
  |N_W(T)| = 48

class 4:  conjugacy orbit = 75
  |N_W(T)| = 192
  |T| = 8
  |N_W(T)| = 192
  |C_W(T)| = 8
  |GL image| = 24
  fixed vectors = [(0, 0, 0), (0, 1, 1)]
  GL order histogram = {1: 1, 2: 9, 3: 8, 4: 6}
  |Z(N)| = 2
  checks = {'T_order_8': True, 'N_order_192': True, 'centralizer_equals_T': True, 'GL_image_order_24': True, 'unique_nonzero_fixed': True, 'GL_hist_S4': True, 'center_order_2': True, 'center_equals_fixed_line': True}
  FULL MATCH = True

full-fingerprint successful classes: 1

SUCCESSFUL CLASS = 4
|T| = 8
|L| = 192
|C_W(T)| = 8
|Z(L)| = 2
z = (0 1)(2 3)(4 5)(6 7)(8 23)(9 22)(10 21)(11 20)(12 19)(13 18)(14 17)(15 16)(24 31)(25 30)(26 29)(27 28)(32 39)(33 38)(34 37)(35 36)(40 47)(41 46)(42 45)(43 44)(48 55)(49 54)(50 53)(51 52)(56 63)(57 62)(58 61)(59 60)(64 71)(65 70)(66 69)(67 68)(72 79)(73 78)(74 77)(75 76)(80 87)(81 86)(82 85)(83 84)(88 95)(89 94)(90 93)(91 92)(96 103)(97 102)(98 101)(99 100)(104 111)(105 110)(106 109)(107 108)(112 119)(113 118)(114 117)(115 116)

C) AMBIENT NORMALIZER OF L------------------------------------------------------------------------------
|N_W(L)| = 576
|L| = 192
[N:L] = 3
L normal in N = True
number of right cosets = 3
coset sizes = [192, 192, 192]
QUOTIENT ORDER = 3 => N/L ~= C3

D) C3 SPLITTING TEST------------------------------------------------------------------------------
order-3 complement witnesses outside L: 48
chosen c = (0 3 4)(1 2 5)(8 14 20)(9 15 21)(10 22 16)(11 23 17)(24 74 62)(25 75 63)(26 78 60)(27 79 61)(28 72 58)(29 73 59)(30 76 56)(31 77 57)(32 98 94)(33 99 95)(34 102 92)(35 103 93)(36 96 90)(37 97 91)(38 100 88)(39 101 89)(40 50 110)(41 51 111)(42 54 108)(43 55 109)(44 48 106)(45 49 107)(46 52 104)(47 53 105)(64 83 118)(65 87 116)(66 82 114)(67 86 112)(68 81 119)(69 85 117)(70 80 115)(71 84 113)
order(c) = 3
<c> intersection L = 1
<L,c> order = 576
SPLIT EXTENSION = True

E) W(D4) STRUCTURAL COMPARISON------------------------------------------------------------------------------
standard W(D4):
  order = 192
  normal even-sign subgroup order = 8
  element-order histogram = {1: 1, 2: 43, 3: 32, 4: 84, 6: 32}

L:  order = 192
  T order = 8
  element-order histogram = {1: 1, 2: 43, 3: 32, 4: 84, 6: 32}
|conjugation image on T| = 24
GL image histogram = {1: 1, 2: 9, 3: 8, 4: 6}
common fixed vectors = [(0, 0, 0), (0, 1, 1)]
D4 order match = True
D4 normal 2^3 match = True
D4 element-order histogram match = True
D4 STRUCTURAL IDENTIFICATION PASS = True

F) EXTERNAL C3 ACTION ON L------------------------------------------------------------------------------
conjugation by c preserves L = True
conjugation action inner on L = False
OUTER C3 ACTION = True
TRIALITY CANDIDATE = True
NOTE: candidate only; explicit D4 diagram-action comparison
      remains required for theorem-level 'triality'.

G) SUCCESSFUL T-KERNEL INCIDENCE INSIDE L------------------------------------------------------------------------------
successful-class T <= L: 3
  T[0] order = 8
  T[1] order = 8
  T[2] order = 8
c transport on successful T kernels = [1, 2, 0]
kernel transport permutation = (1, 2, 0)
kernel transport order = 3
C3 CYCLES THREE SUCCESSFUL KERNELS = True

H) L ACTION ON 600-CELL VERTICES------------------------------------------------------------------------------
orbit sizes = [24, 32, 32, 32]
  orbit 0: size= 24, stabilizer=  8
  orbit 1: size= 32, stabilizer=  6
  orbit 2: size= 32, stabilizer=  6
  orbit 3: size= 32, stabilizer=  6
c transport on 600-cell L-orbits = [0, 2, 3, 1]

I) L ACTION ON 120-CELL VERTICES------------------------------------------------------------------------------
orbit sizes = [8, 8, 8, 32, 32, 32, 96, 96, 96, 192]
  orbit 0: size=  8, stabilizer= 24
  orbit 1: size=  8, stabilizer= 24
  orbit 2: size=  8, stabilizer= 24
  orbit 3: size= 32, stabilizer=  6
  orbit 4: size= 32, stabilizer=  6
  orbit 5: size= 32, stabilizer=  6
  orbit 6: size= 96, stabilizer=  2
  orbit 7: size= 96, stabilizer=  2
  orbit 8: size= 96, stabilizer=  2
  orbit 9: size=192, stabilizer=  1
c transport on 120-cell L-orbits = [1, 2, 0, 5, 3, 4, 8, 6, 7, 9]

J) SAME-C3 GEOMETRIC TRIPLE TEST------------------------------------------------------------------------------
size 24: orbit indices [0]
  c images: {0: 0}
size 32: orbit indices [1, 2, 3]
  c images: {1: 2, 2: 3, 3: 1}
size 8: orbit indices [0, 1, 2]
  c images: {0: 1, 1: 2, 2: 0}
size 32: orbit indices [3, 4, 5]
  c images: {3: 5, 4: 3, 5: 4}
size 96: orbit indices [6, 7, 8]
  c images: {6: 8, 7: 6, 8: 7}
size 192: orbit indices [9]
  c images: {9: 9}

K) CENTRAL INVOLUTION z / ANTIPODAL TEST------------------------------------------------------------------------------
z == root antipodal map = True
z fixed 600-cell vertices = 0
z == dual antipodal map = True
z fixed 120-cell vertices = 0

L) PRIMITIVE DESCENT — REGULAR T-TORSORS------------------------------------------------------------------------------
regular T-torsors on 600-cell vertices: 12
sizes: [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
regular T-torsors on 600-cell edges: 84
regular T-torsors on 120-cell vertices: 74
regular T-torsors on 120-cell edges: 144

M) Y8 GEOMETRIC SIGNATURES------------------------------------------------------------------------------

600-cell vertex torsors  Y8[0] = [24, 27, 28, 31, 48, 51, 52, 55]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [5, 5, 5, 5]
    z-pair inner products = [-2.0, -2.0, -2.0, -2.0]
  Y8[1] = [25, 26, 29, 30, 49, 50, 53, 54]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [5, 5, 5, 5]
    z-pair inner products = [-2.0, -2.0, -2.0, -2.0]
  Y8[2] = [32, 35, 36, 39, 72, 75, 76, 79]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [5, 5, 5, 5]
    z-pair inner products = [-2.0, -2.0, -2.0, -2.0]
  Y8[3] = [33, 34, 37, 38, 73, 74, 77, 78]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [5, 5, 5, 5]
    z-pair inner products = [-2.0, -2.0, -2.0, -2.0]
  Y8[4] = [40, 43, 44, 47, 96, 99, 100, 103]
    induced graph = {'n': 8, 'edges': 12, 'degree_hist': {3: 8}, 'is_literal_C8': False}
    z-pair graph distances = [5, 5, 5, 5]
    z-pair inner products = [-2.0, -2.0, -2.0, -2.0]
  Y8[5] = [41, 42, 45, 46, 97, 98, 101, 102]
    induced graph = {'n': 8, 'edges': 12, 'degree_hist': {3: 8}, 'is_literal_C8': False}
    z-pair graph distances = [5, 5, 5, 5]
    z-pair inner products = [-2.0, -2.0, -2.0, -2.0]
  Y8[6] = [56, 57, 62, 63, 80, 81, 86, 87]
    induced graph = {'n': 8, 'edges': 12, 'degree_hist': {3: 8}, 'is_literal_C8': False}
    z-pair graph distances = [5, 5, 5, 5]
    z-pair inner products = [-2.0, -2.0, -2.0, -2.0]
  Y8[7] = [58, 59, 60, 61, 82, 83, 84, 85]
    induced graph = {'n': 8, 'edges': 12, 'degree_hist': {3: 8}, 'is_literal_C8': False}
    z-pair graph distances = [5, 5, 5, 5]
    z-pair inner products = [-2.0, -2.0, -2.0, -2.0]
  Y8[8] = [64, 65, 70, 71, 104, 105, 110, 111]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [5, 5, 5, 5]
    z-pair inner products = [-2.0, -2.0, -2.0, -2.0]
  Y8[9] = [66, 67, 68, 69, 106, 107, 108, 109]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [5, 5, 5, 5]
    z-pair inner products = [-2.0, -2.0, -2.0, -2.0]
  Y8[10] = [88, 89, 94, 95, 112, 113, 118, 119]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [5, 5, 5, 5]
    z-pair inner products = [-2.0, -2.0, -2.0, -2.0]
  Y8[11] = [90, 91, 92, 93, 114, 115, 116, 117]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [5, 5, 5, 5]
    z-pair inner products = [-2.0, -2.0, -2.0, -2.0]

120-cell vertex torsors  Y8[0] = [0, 8, 21, 29, 40, 45, 61, 66]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[1] = [1, 9, 20, 28, 41, 46, 60, 65]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[2] = [2, 13, 23, 34, 42, 53, 63, 74]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[3] = [3, 14, 22, 33, 43, 54, 62, 73]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[4] = [4, 15, 24, 35, 44, 55, 64, 75]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[5] = [5, 10, 26, 31, 47, 50, 68, 71]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[6] = [6, 11, 25, 30, 48, 51, 67, 70]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[7] = [7, 12, 27, 32, 49, 52, 69, 72]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[8] = [16, 17, 38, 39, 56, 57, 78, 79]
    induced graph = {'n': 8, 'edges': 4, 'degree_hist': {1: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[9] = [18, 19, 36, 37, 58, 59, 76, 77]
    induced graph = {'n': 8, 'edges': 4, 'degree_hist': {1: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[10] = [80, 89, 100, 109, 136, 139, 156, 159]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[11] = [81, 88, 101, 108, 137, 138, 157, 158]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[12] = [82, 91, 105, 114, 122, 131, 147, 154]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[13] = [83, 90, 106, 113, 123, 130, 148, 153]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[14] = [84, 92, 107, 115, 120, 121, 145, 146]
    induced graph = {'n': 8, 'edges': 4, 'degree_hist': {1: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[15] = [85, 94, 102, 111, 127, 134, 142, 151]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[16] = [86, 93, 103, 110, 128, 133, 143, 150]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[17] = [87, 95, 104, 112, 125, 126, 140, 141]
    induced graph = {'n': 8, 'edges': 4, 'degree_hist': {1: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[18] = [96, 98, 117, 119, 124, 132, 149, 155]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[19] = [97, 99, 116, 118, 129, 135, 144, 152]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[20] = [160, 171, 220, 231, 400, 411, 460, 471]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[21] = [161, 170, 221, 230, 401, 410, 461, 470]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[22] = [162, 168, 222, 228, 402, 408, 462, 468]
    induced graph = {'n': 8, 'edges': 4, 'degree_hist': {1: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[23] = [163, 169, 223, 229, 403, 409, 463, 469]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[24] = [164, 172, 224, 232, 404, 412, 464, 472]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[25] = [165, 177, 225, 237, 405, 417, 465, 477]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[26] = [166, 176, 226, 236, 406, 416, 466, 476]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[27] = [167, 179, 227, 239, 407, 419, 467, 479]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[28] = [173, 174, 233, 234, 413, 414, 473, 474]
    induced graph = {'n': 8, 'edges': 4, 'degree_hist': {1: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[29] = [175, 178, 235, 238, 415, 418, 475, 478]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[30] = [180, 191, 200, 211, 420, 431, 440, 451]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[31] = [181, 190, 201, 210, 421, 430, 441, 450]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[32] = [182, 188, 202, 208, 422, 428, 442, 448]
    induced graph = {'n': 8, 'edges': 4, 'degree_hist': {1: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[33] = [183, 189, 203, 209, 423, 429, 443, 449]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[34] = [184, 192, 204, 212, 424, 432, 444, 452]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[35] = [185, 197, 205, 217, 425, 437, 445, 457]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[36] = [186, 196, 206, 216, 426, 436, 446, 456]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[37] = [187, 199, 207, 219, 427, 439, 447, 459]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[38] = [193, 194, 213, 214, 433, 434, 453, 454]
    induced graph = {'n': 8, 'edges': 4, 'degree_hist': {1: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[39] = [195, 198, 215, 218, 435, 438, 455, 458]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[40] = [240, 251, 300, 311, 320, 331, 380, 391]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[41] = [241, 250, 301, 310, 321, 330, 381, 390]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[42] = [242, 248, 302, 308, 322, 328, 382, 388]
    induced graph = {'n': 8, 'edges': 4, 'degree_hist': {1: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[43] = [243, 249, 303, 309, 323, 329, 383, 389]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[44] = [244, 252, 304, 312, 324, 332, 384, 392]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[45] = [245, 257, 305, 317, 325, 337, 385, 397]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[46] = [246, 256, 306, 316, 326, 336, 386, 396]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[47] = [247, 259, 307, 319, 327, 339, 387, 399]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[48] = [253, 254, 313, 314, 333, 334, 393, 394]
    induced graph = {'n': 8, 'edges': 4, 'degree_hist': {1: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[49] = [255, 258, 315, 318, 335, 338, 395, 398]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[50] = [260, 271, 280, 291, 340, 351, 360, 371]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[51] = [261, 270, 281, 290, 341, 350, 361, 370]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[52] = [262, 268, 282, 288, 342, 348, 362, 368]
    induced graph = {'n': 8, 'edges': 4, 'degree_hist': {1: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[53] = [263, 269, 283, 289, 343, 349, 363, 369]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[54] = [264, 272, 284, 292, 344, 352, 364, 372]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[55] = [265, 277, 285, 297, 345, 357, 365, 377]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[56] = [266, 276, 286, 296, 346, 356, 366, 376]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[57] = [267, 279, 287, 299, 347, 359, 367, 379]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[58] = [273, 274, 293, 294, 353, 354, 373, 374]
    induced graph = {'n': 8, 'edges': 4, 'degree_hist': {1: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[59] = [275, 278, 295, 298, 355, 358, 375, 378]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[60] = [480, 487, 495, 502, 541, 545, 552, 556]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[61] = [481, 488, 494, 501, 542, 546, 551, 555]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[62] = [482, 489, 496, 503, 540, 544, 550, 554]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[63] = [483, 492, 497, 506, 543, 549, 553, 559]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[64] = [484, 493, 498, 507, 528, 529, 537, 538]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[65] = [485, 490, 499, 504, 547, 548, 557, 558]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[66] = [486, 491, 500, 505, 531, 532, 534, 535]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[67] = [508, 512, 519, 523, 583, 587, 598, 599]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[68] = [509, 513, 518, 522, 588, 589, 593, 597]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[69] = [510, 514, 520, 524, 580, 584, 590, 594]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[70] = [511, 517, 521, 527, 581, 586, 591, 596]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[71] = [515, 516, 525, 526, 582, 585, 592, 595]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[72] = [561, 562, 563, 564, 576, 577, 578, 579]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]
  Y8[73] = [566, 567, 568, 569, 571, 572, 573, 574]
    induced graph = {'n': 8, 'edges': 0, 'degree_hist': {0: 8}, 'is_literal_C8': False}
    z-pair graph distances = [15, 15, 15, 15]
    z-pair inner products = [-1.7135254916, -1.7135254916, -1.7135254916, -1.7135254916]

N) V4 SUBGROUPS INSIDE SUCCESSFUL T------------------------------------------------------------------------------
number of V4 subgroups of T = 7

K'[0]  |N_L(K')| = 48
  |C_L(K')| = 8
  600 vertex orbit histogram = {2: 6, 4: 27}
  120 vertex orbit histogram = {1: 2, 2: 3, 4: 148}

K'[1]  |N_L(K')| = 64
  |C_L(K')| = 32
  600 vertex orbit histogram = {2: 4, 4: 28}
  120 vertex orbit histogram = {2: 4, 4: 148}

K'[2]  |N_L(K')| = 48
  |C_L(K')| = 8
  600 vertex orbit histogram = {2: 6, 4: 27}
  120 vertex orbit histogram = {1: 2, 2: 3, 4: 148}

K'[3]  |N_L(K')| = 48
  |C_L(K')| = 8
  600 vertex orbit histogram = {2: 6, 4: 27}
  120 vertex orbit histogram = {1: 2, 2: 3, 4: 148}

K'[4]  |N_L(K')| = 64
  |C_L(K')| = 32
  600 vertex orbit histogram = {2: 4, 4: 28}
  120 vertex orbit histogram = {2: 4, 4: 148}

K'[5]  |N_L(K')| = 48
  |C_L(K')| = 8
  600 vertex orbit histogram = {2: 6, 4: 27}
  120 vertex orbit histogram = {1: 2, 2: 3, 4: 148}

K'[6]  |N_L(K')| = 64
  |C_L(K')| = 32
  600 vertex orbit histogram = {2: 4, 4: 28}
  120 vertex orbit histogram = {2: 4, 4: 148}

O) C8 PRIMITIVE SURVIVAL------------------------------------------------------------------------------
literal induced C8 among 600-vertex Y8 = 0
literal induced C8 among 120-vertex Y8 = 0
C8 grade = NO A-GRADE C8: no literal induced C8 on vertex torsors; Hamiltonian/subrelation tests deferred unless structurally motivated

==============================================================================P) AMBIENT CLOSURE TRUTH PACKET
==============================================================================
successful_SIM15_class_unique       : True
L_order_192                         : True
N_order_576                         : True
N_over_L_order_3                    : True
split_C3_found                      : True
D4_structural_pass                  : True
external_C3_outer_on_L              : True
successful_T_inside_L               : 3
z_antipodal_600                     : True
z_antipodal_120                     : True

==============================================================================Q) PRIMITIVE SURVIVAL TRUTH PACKET
==============================================================================
regular_T_torsors_600_vertices      : 12
regular_T_torsors_600_edges         : 84
regular_T_torsors_120_vertices      : 74
regular_T_torsors_120_edges         : 144
z_pairing_available                 : True
literal_C8_600_vertex_torsor        : 0
literal_C8_120_vertex_torsor        : 0
V4_subgroups_inside_T               : 7
signed_Omega_recovered              : False

SIGNED OMEGA STATUS:  NOT FORCED.
  SIM15.1 does not manufacture Omega_H4 from Omega_OPS.
  Only native H4 pairing/incidence information is reported.

==============================================================================SIM15.1 MACHINE SUMMARY
==============================================================================

Interpretation gates:

1. |N/L| = 3 alone DOES NOT establish triality.

2. A split extension plus an outer order-3 action on a constructively
   identified W(D4) is a TRIALITY CANDIDATE.

3. The strongest threefold result occurs only if the SAME c:
      - cycles successful T kernels,
      - cycles the 600-cell equal-size orbit triples,
      - cycles the 120-cell equal-size orbit triples.

4. A regular T-orbit of size 8 is only a weak X8 analogue.
   Primitive survival requires additional relational structure.

5. Literal induced C8 survival is strong.
   Merely being able to draw some Hamiltonian 8-cycle is not counted
   as primitive survival in this run.

6. z reproduces unsigned partner structure algebraically.
   z = -I would additionally give it an intrinsic H4 geometric meaning.

7. No signed symplectic form is inferred from H4 Euclidean geometry.

END SIM15.1
