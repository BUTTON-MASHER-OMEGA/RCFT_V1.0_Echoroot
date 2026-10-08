import itertools
import math
from collections import Counter, defaultdict, deque
import numpy as np

# =============================================================================
# SIM15.5 — 24-CELL STABILIZER / TRIALITY-NORMALIZER IDENTIFICATION
# =============================================================================
#
# FROZEN QUESTION
# ---------------
# Are the 25 native 24-cell stabilizers exactly the 25 N_W(L) triality
# normalizers, and does this identify the native 0/6 geometry with the
# previously discovered 8/12 architecture relations?
#
# HARD GATES
# ----------
# A  Frozen SIM15.4 regression
# B  Independently construct N25 and S25
# C  KILLER: S25 == N25 literally as subgroup families
# D  Unique W(H4)-equivariant C <-> N <-> L correspondence
# E  L normal N, [N:L]=3, N/L=C3, triality cycles T-fiber
# F  Native 0/6 geometry versus N-intersection algebra
# G  Classify N-intersection groups
# H  Native 0/6 geometry versus frozen L-intersection 8/12
# I  Classify I8/I12 and compare I <= J
# J  Same rank-3 relation transported through C25, N25, L25
#
# EXPLORATORY
# -----------
# K  What does N576 -> L192 forget geometrically?
# L  C3 / C5 controls
#
# No downstream result rescues a failed upstream gate.
# =============================================================================


# =============================================================================
# 0. BASIC PERMUTATION / GROUP MACHINERY
# =============================================================================

def pid(n):
    return tuple(range(n))


def pmul(a, b):
    # composition a o b
    return tuple(a[b[i]] for i in range(len(a)))


def pinv(a):
    q = [0] * len(a)
    for i, j in enumerate(a):
        q[j] = i
    return tuple(q)


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
    return dict(sorted(Counter(len(c) for c in pcycles(p, True)).items()))


def generated_group(gens, n=None):
    if not gens:
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
    return generated_group(list(gens), n)


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


def commute(a, b):
    return pmul(a, b) == pmul(b, a)


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


def order_hist(G):
    return dict(sorted(Counter(porder(g) for g in G).items()))


def conjugate_subgroup(g, H):
    return frozenset(pconj(g, h) for h in H)


def conjugacy_orbit_subgroup(W, H):
    return {conjugate_subgroup(g, H) for g in W}


def commutator(a, b):
    return pmul(pmul(pmul(a, b), pinv(a)), pinv(b))


def derived_subgroup(G):
    G = set(G)
    n = len(next(iter(G)))

    comms = {
        commutator(a, b)
        for a in G
        for b in G
    }

    return frozenset(subgroup_generated(list(comms), n))


def is_normal(H, G):
    H = set(H)
    return all(
        {pconj(g, h) for h in H} == H
        for g in G
    )


# =============================================================================
# 1. H4 CONSTRUCTION
# =============================================================================

PHI = (1 + math.sqrt(5)) / 2
IPHI = 1 / PHI
SQRT2 = math.sqrt(2)
TOL = 1e-7


def permutation_parity(p):
    inv = 0
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] > p[j]:
                inv += 1
    return inv & 1


def unique_vectors(vectors, decimals=10):
    d = {}
    for v in vectors:
        d[tuple(np.round(v, decimals))] = np.asarray(v, float)
    return np.array(list(d.values()), float)


def build_h4_roots():
    roots = []

    # 8 coordinate roots
    for i in range(4):
        for s in (-1.0, 1.0):
            v = np.zeros(4)
            v[i] = 2 * s
            roots.append(v / SQRT2)

    # 16 half-hypercube roots
    for signs in itertools.product((-1.0, 1.0), repeat=4):
        roots.append(np.array(signs, float) / SQRT2)

    # 96 golden roots
    base = (0.0, 1.0, PHI, IPHI)

    even = [
        p for p in itertools.permutations(range(4))
        if permutation_parity(p) == 0
    ]

    for p in even:
        vals = np.array([base[p[i]] for i in range(4)], float)
        nz = [i for i, x in enumerate(vals) if abs(x) > 1e-8]

        for signs in itertools.product((-1.0, 1.0), repeat=3):
            v = vals.copy()
            for j, s in zip(nz, signs):
                v[j] *= s
            roots.append(v / SQRT2)

    R = unique_vectors(roots)

    if len(R) != 120:
        raise RuntimeError(f"H4 roots={len(R)}, expected 120")

    if not np.allclose(np.sum(R * R, axis=1), 2, atol=1e-8):
        raise RuntimeError("H4 root norms failed")

    return R


def vec_key(v):
    return tuple(np.round(v, 9))


def build_root_lookup(R):
    return {vec_key(v): i for i, v in enumerate(R)}


def reflection_matrix(alpha):
    # roots have alpha.alpha = 2
    return np.eye(4) - np.outer(alpha, alpha)


def linear_map_to_perm(M, R, lookup):
    p = []

    for v in R:
        w = M @ v
        k = vec_key(w)

        if k in lookup:
            p.append(lookup[k])
        else:
            d = np.linalg.norm(R - w[None, :], axis=1)
            j = int(np.argmin(d))
            if d[j] > 1e-6:
                raise RuntimeError("linear map does not permute roots")
            p.append(j)

    return tuple(p)


def find_simple_h4_roots(R):
    dots = R @ R.T

    def inds(i, target):
        return [
            j for j in range(len(R))
            if j != i and abs(dots[i, j] - target) < 1e-7
        ]

    a1 = 0

    for a2 in inds(a1, -PHI):
        for a3 in [
            x for x in inds(a2, -1)
            if abs(dots[a1, x]) < 1e-7
        ]:
            cand = [
                a4 for a4 in inds(a3, -1)
                if abs(dots[a2, a4]) < 1e-7
                and abs(dots[a1, a4]) < 1e-7
            ]

            if cand:
                return [a1, a2, a3, cand[0]]

    raise RuntimeError("H4 simple roots not found")


def build_W_H4(R):
    lookup = build_root_lookup(R)
    idx = find_simple_h4_roots(R)

    gens = [
        linear_map_to_perm(reflection_matrix(R[i]), R, lookup)
        for i in idx
    ]

    W = generated_group(gens)

    if len(W) != 14400:
        raise RuntimeError(f"|W(H4)|={len(W)}, expected 14400")

    return W, gens, idx


# =============================================================================
# 2. FROZEN SUCCESSFUL C2^3 STRUCTURE
# =============================================================================

F2V = [
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    (0, 1, 1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
]

I3 = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
)

TARGET_GL_HIST = {1: 1, 2: 9, 3: 8, 4: 6}


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

                if (
                    len(H8) == 8
                    and all(porder(x) in (1, 2) for x in H8)
                ):
                    groups.add(frozenset(H8))

    return [set(H) for H in groups]


def subgroup_conjugacy_classes(W, subs):
    unseen = {frozenset(H) for H in subs}
    classes = []

    while unseen:
        H = next(iter(unseen))
        orb = {conjugate_subgroup(g, H) for g in W}
        cls = orb & unseen
        classes.append(cls)
        unseen -= orb

    return classes


def choose_T_basis(T):
    e = pid(len(next(iter(T))))
    nz = [x for x in T if x != e]

    for a, b, c in itertools.combinations(nz, 3):
        if subgroup_generated([a, b, c], len(a)) == set(T):
            return a, b, c

    raise RuntimeError("T basis failed")


def T_coordinate_map(T):
    e = pid(len(next(iter(T))))
    a, b, c = choose_T_basis(T)

    c2e = {}
    e2c = {}

    for v in F2V:
        x = e

        if v[0]:
            x = pmul(a, x)
        if v[1]:
            x = pmul(b, x)
        if v[2]:
            x = pmul(c, x)

        c2e[v] = x
        e2c[x] = v

    return e2c, c2e


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

    raise RuntimeError("matrix order failure")


def mat_from_action_on_T(g, e2c, c2e):
    basis = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    ]

    cols = []

    for v in basis:
        x = c2e[v]
        y = pconj(g, x)
        cols.append(e2c[y])

    return tuple(
        tuple(cols[j][i] for j in range(3))
        for i in range(3)
    )


def full_fingerprint(W, T):
    N = normalizer_in(W, T)
    C = centralizer_in(W, T)

    e2c, c2e = T_coordinate_map(T)

    image = {
        mat_from_action_on_T(g, e2c, c2e)
        for g in N
    }

    fixed = [
        v for v in F2V
        if all(mat_apply(M, v) == v for M in image)
    ]

    gh = dict(sorted(Counter(mat_order(M) for M in image).items()))

    zN = center(N)
    e = pid(len(next(iter(T))))

    z = None
    if len(zN) == 2:
        z = next(x for x in zN if x != e)

    nf = [v for v in fixed if v != (0, 0, 0)]

    center_fixed = (
        z is not None
        and len(nf) == 1
        and z == c2e[nf[0]]
    )

    checks = {
        "T8": len(T) == 8,
        "N192": len(N) == 192,
        "centralizer_T": C == set(T),
        "GL24": len(image) == 24,
        "GL_S4_hist": gh == TARGET_GL_HIST,
        "one_nonzero_fixed": len(nf) == 1,
        "center2": len(zN) == 2,
        "center_fixed": center_fixed,
    }

    return {
        "full": all(checks.values()),
        "N": N,
        "C": C,
        "z": z,
        "checks": checks,
    }


def find_successful_structure(W):
    Tsubs = enumerate_C2_3_subgroups(W)

    print("C2^3 subgroups:", len(Tsubs))

    classes = subgroup_conjugacy_classes(W, Tsubs)

    print("C2^3 conjugacy classes:", len(classes))
    print("class sizes:", sorted(len(C) for C in classes))

    good = []

    for ci, C in enumerate(classes):
        T = set(next(iter(C)))
        L = normalizer_in(W, T)

        print(
            f"class {ci}: orbit={len(C)} normalizer={len(L)}"
        )

        if len(L) != 192:
            continue

        fp = full_fingerprint(W, T)

        print(
            "  fingerprint:",
            fp["checks"],
            "FULL=",
            fp["full"],
        )

        if fp["full"]:
            good.append((ci, C, T, fp))

    if len(good) != 1:
        raise RuntimeError(
            f"Expected one successful class, found {len(good)}"
        )

    ci, C, T, fp = good[0]

    L = fp["N"]
    N = normalizer_in(W, L)

    return {
        "class_id": ci,
        "successful_class": C,
        "T": T,
        "L": L,
        "N": N,
        "z": fp["z"],
    }


def find_C3_complement(N, L):
    e = pid(len(next(iter(N))))
    Lgens = greedy_generators(L)
    out = []

    for c in N:
        if c in L or porder(c) != 3:
            continue

        C3 = subgroup_generated([c], len(c))

        if C3 & set(L) != {e}:
            continue

        if subgroup_generated(Lgens + [c], len(c)) == set(N):
            out.append(c)

    return out


# =============================================================================
# 3. NATIVE 24-CELL CONSTRUCTION — FROZEN FROM SIM15.4
# =============================================================================

def root_reflection_perm(i, R, lookup):
    a = R[i]
    p = []

    for v in R:
        w = v - np.dot(v, a) * a
        k = vec_key(w)

        if k in lookup:
            p.append(lookup[k])
        else:
            d = np.linalg.norm(R - w[None, :], axis=1)
            j = int(np.argmin(d))

            if d[j] > 1e-6:
                raise RuntimeError("root reflection failure")

            p.append(j)

    return tuple(p)


def d4_frames(R):
    D = R @ R.T
    n = len(R)

    minus1 = [
        [
            j for j in range(n)
            if j != i and abs(D[i, j] + 1) < TOL
        ]
        for i in range(n)
    ]

    for c in range(n):
        for a, b, d in itertools.combinations(minus1[c], 3):
            if (
                abs(D[a, b]) < TOL
                and abs(D[a, d]) < TOL
                and abs(D[b, d]) < TOL
            ):
                yield c, a, b, d


def close_under_reflections(seed, refl):
    C = set(seed)
    q = deque(seed)

    while q:
        x = q.popleft()

        for r in refl:
            y = r[x]

            if y not in C:
                C.add(y)
                q.append(y)

    return frozenset(C)


def enumerate_native_24cells(R):
    lookup = build_root_lookup(R)
    cache = {}
    cells = set()
    hist = Counter()
    nf = 0

    for frame in d4_frames(R):
        nf += 1
        rr = []

        for i in frame:
            if i not in cache:
                cache[i] = root_reflection_perm(i, R, lookup)

            rr.append(cache[i])

        C = close_under_reflections(frame, rr)

        hist[len(C)] += 1

        if len(C) == 24:
            cells.add(C)

    return (
        tuple(sorted(cells, key=lambda C: tuple(sorted(C)))),
        nf,
        dict(sorted(hist.items())),
    )


def induced_edges(C, R, target=1.0):
    V = sorted(C)
    E = []
    A = {v: set() for v in V}

    for ia, a in enumerate(V):
        for b in V[ia + 1:]:
            if abs(float(np.dot(R[a], R[b])) - target) < TOL:
                E.append((a, b))
                A[a].add(b)
                A[b].add(a)

    return tuple(E), A


def triangles(A):
    T = set()

    for a in sorted(A):
        for b in [x for x in A[a] if x > a]:
            for c in A[a] & A[b]:
                if c > b:
                    T.add((a, b, c))

    return tuple(sorted(T))


def octahedral_facets(C, R):
    V = sorted(C)
    X = R[V]
    facets = set()

    for ids in itertools.combinations(range(24), 4):
        P = X[list(ids)]
        D = P[1:] - P[0]

        if np.linalg.matrix_rank(D, tol=1e-8) < 3:
            continue

        _, _, vh = np.linalg.svd(D)
        n = vh[-1]

        if np.linalg.norm(n) < 1e-10:
            continue

        n = n / np.linalg.norm(n)
        vals = X @ n
        h = float(P[0] @ n)

        for s in (1.0, -1.0):
            vv = s * vals
            hh = s * h
            mx = float(np.max(vv))

            if abs(mx - hh) > 1e-7:
                continue

            inds = [
                k for k, x in enumerate(vv)
                if abs(x - mx) < 1e-7
            ]

            if len(inds) == 6:
                facets.add(
                    frozenset(V[k] for k in inds)
                )

    good = set()

    for F in facets:
        E, A = induced_edges(F, R)

        if (
            len(E) == 12
            and Counter(len(A[v]) for v in F) == {4: 6}
        ):
            good.add(F)

    return tuple(
        sorted(good, key=lambda F: tuple(sorted(F)))
    )


def ip_spectrum(C, R):
    V = sorted(C)
    H = Counter()

    for ia, a in enumerate(V):
        for b in V[ia + 1:]:
            x = float(np.dot(R[a], R[b]))

            for t in (-2.0, -1.0, 0.0, 1.0, 2.0):
                if abs(x - t) < 1e-7:
                    x = t
                    break

            H[round(x, 8)] += 1

    return dict(sorted(H.items()))


def certify_24cell(C, R):
    V = sorted(C)
    X = R[V]

    E, A = induced_edges(C, R)
    T = triangles(A)
    F = octahedral_facets(C, R)

    lookup = {vec_key(R[v]) for v in V}

    antipodal = all(
        vec_key(-R[v]) in lookup
        for v in V
    )

    cert = {
        "vertices": len(V),
        "rank": int(np.linalg.matrix_rank(X, tol=1e-8)),
        "antipodal": antipodal,
        "degree_hist": dict(
            sorted(Counter(len(A[v]) for v in V).items())
        ),
        "edges": len(E),
        "triangles": len(T),
        "octahedral_facets": len(F),
        "ip_spectrum": ip_spectrum(C, R),
    }

    cert["passes"] = (
        cert["vertices"] == 24
        and cert["rank"] == 4
        and cert["antipodal"]
        and cert["degree_hist"] == {8: 24}
        and cert["edges"] == 96
        and cert["triangles"] == 96
        and cert["octahedral_facets"] == 24
    )

    return (
        cert,
        frozenset(tuple(sorted(e)) for e in E),
        frozenset(frozenset(t) for t in T),
        frozenset(F),
    )


# =============================================================================
# 4. FAMILY / ACTION UTILITIES
# =============================================================================

def header(label, title):
    print("\n" + "=" * 100)
    print(f"{label}) {title}")
    print("-" * 100)


def frozen_family_with_first(S, first):
    first = frozenset(first)

    rest = sorted(
        [H for H in S if H != first],
        key=lambda H: tuple(sorted(H)),
    )

    return [first] + rest


def subset_image(g, C):
    return frozenset(g[v] for v in C)


def setwise_stabilizer(W, C):
    return frozenset(
        g for g in W
        if subset_image(g, C) == C
    )


def build_cell_action(W, F):
    index = {C: i for i, C in enumerate(F)}
    action = {}

    for g in W:
        p = []

        for C in F:
            D = subset_image(g, C)

            if D not in index:
                return index, None, False, (g, C, D)

            p.append(index[D])

        action[g] = tuple(p)

    return index, action, True, None


def action_kernel(action):
    n = len(next(iter(action.values())))
    e = pid(n)

    return {
        g for g, p in action.items()
        if p == e
    }


def subgroup_family_action(W, family):
    idx = {frozenset(H): i for i, H in enumerate(family)}
    action = {}

    for g in W:
        p = []

        for H in family:
            K = conjugate_subgroup(g, H)

            if K not in idx:
                return idx, None, False

            p.append(idx[K])

        action[g] = tuple(p)

    return idx, action, True


def pair_orbits(action, n):
    unseen = {
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
    }

    image = set(action.values())
    out = []

    while unseen:
        a, b = min(unseen)

        O = {
            tuple(sorted((p[a], p[b])))
            for p in image
        }

        out.append(frozenset(O))
        unseen -= O

    return tuple(
        sorted(
            out,
            key=lambda O: (len(O), tuple(sorted(O))),
        )
    )


def intersection_size(A, B):
    return len(set(A) & set(B))


def point_orbits_group(G, C):
    C = set(C)
    unseen = set(C)
    out = []

    while unseen:
        x = min(unseen)

        O = {g[x] for g in G} & C

        out.append(tuple(sorted(O)))
        unseen -= O

    return tuple(
        sorted(out, key=lambda x: (len(x), x))
    )


# =============================================================================
# 5. 600-CELL GEOMETRY / NATIVE PAIR RELATION
# =============================================================================

def build_600_graph(R):
    A = [set() for _ in R]

    for i in range(len(R)):
        for j in range(i + 1, len(R)):
            if abs(float(R[i] @ R[j]) - PHI) < 1e-7:
                A[i].add(j)
                A[j].add(i)

    return A


def native_pair_type(C, D):
    k = len(C & D)

    if k == 0:
        return "R0_disjoint"

    if k == 6:
        return "R6_shared_vertices"

    return f"R_other_{k}"


# =============================================================================
# 6. SUBGROUP INVARIANTS
# =============================================================================

def subgroup_invariants(I, W, include_ambient=True):
    I = frozenset(I)

    Z = frozenset(center(I))
    D = derived_subgroup(I)

    out = {
        "order": len(I),
        "order_hist": order_hist(I),
        "center_order": len(Z),
        "center_hist": order_hist(Z),
        "derived_order": len(D),
        "derived_hist": order_hist(D),
        "abelianization_order": len(I) // len(D),
    }

    if include_ambient:
        out["normalizer_order"] = len(normalizer_in(W, I))
        out["centralizer_order"] = len(centralizer_in(W, I))

    return out


def invariant_key(inv):
    return (
        inv["order"],
        tuple(inv["order_hist"].items()),
        inv["center_order"],
        tuple(inv["center_hist"].items()),
        inv["derived_order"],
        tuple(inv["derived_hist"].items()),
        inv["abelianization_order"],
        inv.get("normalizer_order"),
        inv.get("centralizer_order"),
    )


def print_invariant(inv, indent="  "):
    for k, v in inv.items():
        print(indent + f"{k:24s}= {v}")


def group_label(inv):
    n = inv["order"]
    h = inv["order_hist"]
    z = inv["center_order"]
    d = inv["derived_order"]

    if n == 8:
        if h == {1: 1, 2: 7}:
            return "C2^3"

        if h == {1: 1, 2: 3, 4: 4} and z == 2 and d == 2:
            return "D8"

        if h == {1: 1, 2: 1, 4: 6} and z == 2 and d == 2:
            return "Q8"

        if h == {1: 1, 2: 1, 4: 2, 8: 4}:
            return "C8"

    if n == 12:
        if h == {1: 1, 2: 3, 3: 8}:
            return "A4"

        if h == {1: 1, 2: 7, 3: 2, 6: 2}:
            return "D12"

    return "unresolved"


# =============================================================================
# 7. GRAPH SIGNATURE
# =============================================================================

def graph_from_pairs(n, pairs):
    A = np.zeros((n, n), int)

    for i, j in pairs:
        A[i, j] = 1
        A[j, i] = 1

    return A


def graph_signature(A):
    n = len(A)

    deg = Counter(int(x) for x in A.sum(axis=1))

    adj = Counter()
    non = Counter()

    for i in range(n):
        for j in range(i + 1, n):
            c = int(A[i] @ A[j])

            if A[i, j]:
                adj[c] += 1
            else:
                non[c] += 1

    vals = np.round(
        np.linalg.eigvalsh(A.astype(float)),
        9,
    )

    seen = {0}
    q = deque([0])

    while q:
        x = q.popleft()

        for y in np.flatnonzero(A[x]):
            y = int(y)

            if y not in seen:
                seen.add(y)
                q.append(y)

    return {
        "v": n,
        "edges": int(A.sum() // 2),
        "degree_hist": dict(sorted(deg.items())),
        "connected": len(seen) == n,
        "adjacent_common_neighbor_hist": dict(sorted(adj.items())),
        "nonadjacent_common_neighbor_hist": dict(sorted(non.items())),
        "spectrum": [
            (float(x), int(c))
            for x, c in sorted(Counter(vals).items())
        ],
    }


# =============================================================================
# 8. FACE ACTION UTILITIES FOR EXPLORATORY GATE K
# =============================================================================

def induced_faces(C, R):
    E, A = induced_edges(C, R)
    T = triangles(A)
    F = octahedral_facets(C, R)

    return {
        "vertices": tuple(frozenset([v]) for v in sorted(C)),
        "edges": tuple(frozenset(e) for e in E),
        "triangles": tuple(frozenset(t) for t in T),
        "octahedral_cells": tuple(F),
    }


def orbits_on_objects(G, objects):
    objects = tuple(objects)
    objset = set(objects)
    unseen = set(objects)
    out = []

    while unseen:
        X = next(iter(unseen))

        O = set()

        for g in G:
            Y = frozenset(g[x] for x in X)

            if Y in objset:
                O.add(Y)

        out.append(frozenset(O))
        unseen -= O

    return tuple(
        sorted(
            out,
            key=lambda O: (
                len(O),
                tuple(sorted(tuple(sorted(x)) for x in O)),
            ),
        )
    )


def orbit_size_signature(G, objects):
    O = orbits_on_objects(G, objects)
    return tuple(sorted(len(x) for x in O))


# =============================================================================
# 9. STATUS REPORTING
# =============================================================================

NOT_TESTED = "NOT TESTED"


def status(x):
    if x is NOT_TESTED or x == NOT_TESTED:
        return "NOT TESTED"
    return str(bool(x))


# =============================================================================
# 10. MAIN
# =============================================================================

def main():

    print("=" * 100)
    print("SIM15.5 — 24-CELL STABILIZER / TRIALITY-NORMALIZER IDENTIFICATION")
    print("=" * 100)

    print("""
FROZEN QUESTION:

Are the 25 native 24-cell stabilizers exactly the 25 N_W(L)
triality normalizers, and does this identify the native 0/6
geometry with the previously discovered 8/12 architecture relations?

No isomorphism or histogram match may substitute for literal subgroup equality.
No downstream success rescues a failed upstream gate.
""")

    # =========================================================================
    # A — FROZEN REGRESSION
    # =========================================================================

    header("A", "FROZEN SIM15.4 REGRESSION")

    R = build_h4_roots()
    W, Wgens, simple_idx = build_W_H4(R)

    print("H4 roots =", len(R))
    print("|W(H4)| =", len(W))
    print("simple-root indices =", simple_idx)

    S = find_successful_structure(W)

    T0 = frozenset(S["T"])
    L0 = frozenset(S["L"])
    N0 = frozenset(S["N"])
    z = S["z"]

    Tfamily = frozen_family_with_first(
        set(S["successful_class"]),
        T0,
    )

    Lfamily = frozen_family_with_first(
        conjugacy_orbit_subgroup(W, L0),
        L0,
    )

    print("|T0| =", len(T0))
    print("|L0| =", len(L0))
    print("|N0| =", len(N0))
    print("|T75| =", len(Tfamily))
    print("|L25| =", len(Lfamily))

    # Blind native 24-cell construction
    CRAW, nframes, closure_hist = enumerate_native_24cells(R)

    print("D4 frame count =", nframes)
    print("closure-size histogram =", closure_hist)
    print("distinct 24-root closures =", len(CRAW))

    Cfamily = []
    face_data = {}
    cert_hist = Counter()

    for C in CRAW:
        cert, E, T, F = certify_24cell(C, R)

        key = (
            cert["vertices"],
            cert["rank"],
            cert["antipodal"],
            tuple(cert["degree_hist"].items()),
            cert["edges"],
            cert["triangles"],
            cert["octahedral_facets"],
            tuple(cert["ip_spectrum"].items()),
            cert["passes"],
        )

        cert_hist[key] += 1

        if cert["passes"]:
            Cfamily.append(C)
            face_data[C] = {
                "edges": E,
                "triangles": T,
                "octahedral_cells": F,
            }

    Cfamily = tuple(
        sorted(Cfamily, key=lambda C: tuple(sorted(C)))
    )

    print("certified native 24-cells =", len(Cfamily))

    Sfamily = tuple(
        setwise_stabilizer(W, C)
        for C in Cfamily
    )

    print(
        "native stabilizer-order histogram =",
        dict(sorted(Counter(len(H) for H in Sfamily).items())),
    )

    # Native pair relation
    native_pair_hist = Counter()

    for i in range(len(Cfamily)):
        for j in range(i + 1, len(Cfamily)):
            native_pair_hist[
                native_pair_type(Cfamily[i], Cfamily[j])
            ] += 1

    print(
        "native pair-relation histogram =",
        dict(native_pair_hist),
    )

    Cindex, actionC, closedC, failC = build_cell_action(
        W,
        Cfamily,
    )

    if not closedC:
        print("WARNING: native C24 family not W-closed:", failC)

    Cpair_orbits = (
        pair_orbits(actionC, len(Cfamily))
        if closedC else ()
    )

    print(
        "native C24 pair orbital sizes =",
        [len(O) for O in Cpair_orbits],
    )

    Ckernel = action_kernel(actionC) if closedC else set()

    print(
        "|C24 action image| =",
        len(set(actionC.values())) if closedC else None,
    )

    print(
        "|C24 action kernel| =",
        len(Ckernel) if closedC else None,
    )

    print(
        "z in C24 kernel =",
        z in Ckernel if closedC else None,
    )

    GATE_A = (
        len(R) == 120
        and len(W) == 14400
        and len(T0) == 8
        and len(L0) == 192
        and len(N0) == 576
        and len(Tfamily) == 75
        and len(Lfamily) == 25
        and nframes == 4800
        and closure_hist == {24: 4800}
        and len(CRAW) == 25
        and len(Cfamily) == 25
        and Counter(len(H) for H in Sfamily) == {576: 25}
        and native_pair_hist == {
            "R0_disjoint": 100,
            "R6_shared_vertices": 200,
        }
        and closedC
        and sorted(len(O) for O in Cpair_orbits) == [100, 200]
        and len(set(actionC.values())) == 7200
        and len(Ckernel) == 2
        and z in Ckernel
    )

    print("GATE A =", GATE_A)

    # =========================================================================
    # B — INDEPENDENT N25 AND S25
    # =========================================================================

    header("B", "INDEPENDENT ORDER-576 FAMILIES N25 AND S25")

    # Operational branch only:
    # N_i = N_W(L_i)
    Nfamily_raw = [
        frozenset(normalizer_in(W, L))
        for L in Lfamily
    ]

    Nset = set(Nfamily_raw)
    Sset = set(Sfamily)

    print("L entries =", len(Lfamily))
    print("distinct N_W(L_i) =", len(Nset))
    print("distinct native stabilizers =", len(Sset))

    print(
        "N-family subgroup-order histogram =",
        dict(sorted(Counter(len(N) for N in Nset).items())),
    )

    print(
        "S-family subgroup-order histogram =",
        dict(sorted(Counter(len(Sg) for Sg in Sset).items())),
    )

    N_hist_types = Counter(
        tuple(order_hist(N).items())
        for N in Nset
    )

    S_hist_types = Counter(
        tuple(order_hist(Sg).items())
        for Sg in Sset
    )

    print("N-family element-order histogram types:")

    for h, n in N_hist_types.items():
        print(" count", n, "hist =", dict(h))

    print("S-family element-order histogram types:")

    for h, n in S_hist_types.items():
        print(" count", n, "hist =", dict(h))

    # How many L's map to each normalizer?
    L_per_N = Counter(Nfamily_raw)

    print(
        "number of L_i per distinct N_i histogram =",
        dict(sorted(Counter(L_per_N.values()).items())),
    )

    GATE_B = (
        GATE_A
        and len(Nset) == 25
        and len(Sset) == 25
        and all(len(N) == 576 for N in Nset)
        and all(len(Sg) == 576 for Sg in Sset)
        and set(L_per_N.values()) == {1}
    )

    print("GATE B =", GATE_B)

    # =========================================================================
    # C — KILLER GATE
    # =========================================================================

    header("C", "KILLER GATE — LITERAL S25 == N25")

    print("|N25| =", len(Nset))
    print("|S25| =", len(Sset))
    print("|N25 cap S25| =", len(Nset & Sset))
    print("|N25 \\ S25| =", len(Nset - Sset))
    print("|S25 \\ N25| =", len(Sset - Nset))

    GATE_C = (
        GATE_B
        and Nset == Sset
    )

    print("GATE C — LITERAL S25 == N25 =", GATE_C)

    # =========================================================================
    # D — CANONICAL C <-> N <-> L
    # =========================================================================

    header("D", "CANONICAL C <-> N <-> L CORRESPONDENCE")

    GATE_D = NOT_TESTED

    C_to_N = {}
    C_to_L = {}
    N_to_C = {}
    N_to_L = {}

    if not GATE_C:
        print("NOT TESTED: Gate C failed.")

    else:
        N_to_L_candidates = defaultdict(list)

        for li, N in enumerate(Nfamily_raw):
            N_to_L_candidates[N].append(li)

        S_to_C_candidates = defaultdict(list)

        for ci, Sg in enumerate(Sfamily):
            S_to_C_candidates[Sg].append(ci)

        unique_N_to_L = all(
            len(v) == 1
            for v in N_to_L_candidates.values()
        )

        unique_N_to_C = all(
            len(v) == 1
            for v in S_to_C_candidates.values()
        )

        print("unique L per N =", unique_N_to_L)
        print("unique C per N =", unique_N_to_C)

        if unique_N_to_L and unique_N_to_C:

            for N in Nset:
                li = N_to_L_candidates[N][0]
                ci = S_to_C_candidates[N][0]

                C_to_N[ci] = N
                C_to_L[ci] = li
                N_to_C[N] = ci
                N_to_L[N] = li

            equiv = True

            for g in W:
                for ci, C in enumerate(Cfamily):
                    Cg = subset_image(g, C)

                    cj = Cindex[Cg]

                    Ni = C_to_N[ci]
                    Nj_expected = conjugate_subgroup(g, Ni)

                    if C_to_N[cj] != Nj_expected:
                        equiv = False
                        print(
                            "equivariance failure:",
                            "ci =", ci,
                            "cj =", cj,
                        )
                        break

                if not equiv:
                    break

            print("C <-> N equivariant =", equiv)

            # Also check L transport through the earned correspondence.
            Lidx = {
                frozenset(L): i
                for i, L in enumerate(Lfamily)
            }

            equiv_L = True

            for g in W:
                for ci in range(25):
                    li = C_to_L[ci]

                    Cg = subset_image(g, Cfamily[ci])
                    cj = Cindex[Cg]

                    Lg = conjugate_subgroup(g, Lfamily[li])

                    if Lg not in Lidx:
                        equiv_L = False
                        break

                    if C_to_L[cj] != Lidx[Lg]:
                        equiv_L = False
                        break

                if not equiv_L:
                    break

            print("C <-> L transported equivariantly =", equiv_L)

            GATE_D = (
                len(C_to_N) == 25
                and len(C_to_L) == 25
                and equiv
                and equiv_L
            )

        else:
            GATE_D = False

    print("GATE D =", status(GATE_D))

    # =========================================================================
    # E — INTERNAL TRIALITY EXTENSION
    # =========================================================================

    header("E", "INTERNAL TRIALITY EXTENSION N576 > L192")

    GATE_E = NOT_TESTED

    triality_records = []

    if GATE_D is not True:
        print("NOT TESTED: Gate D did not pass.")

    else:
        all_ok = True

        for ci in range(25):
            N = C_to_N[ci]
            li = C_to_L[ci]
            L = frozenset(Lfamily[li])

            normal = is_normal(L, N)
            index = len(N) // len(L)

            cands = find_C3_complement(N, L)

            contained_T = [
                ti for ti, T in enumerate(Tfamily)
                if set(T) <= set(L)
            ]

            cycle_ok = False
            witness = None
            transport = None

            for c in cands:
                if len(contained_T) != 3:
                    continue

                Tidx = {
                    frozenset(Tfamily[ti]): ti
                    for ti in contained_T
                }

                tr = []

                valid = True

                for ti in contained_T:
                    U = conjugate_subgroup(c, Tfamily[ti])

                    if U not in Tidx:
                        valid = False
                        break

                    tr.append(Tidx[U])

                if valid:
                    local_map = {
                        contained_T[k]: tr[k]
                        for k in range(3)
                    }

                    # require one 3-cycle, not three fixed points
                    start = contained_T[0]
                    x1 = local_map[start]
                    x2 = local_map[x1]
                    x3 = local_map[x2]

                    if (
                        x1 != start
                        and x2 != start
                        and x3 == start
                    ):
                        cycle_ok = True
                        witness = c
                        transport = local_map
                        break

            rec = {
                "cell": ci,
                "normal": normal,
                "index": index,
                "C3_complement_count": len(cands),
                "successful_T_inside_L": len(contained_T),
                "triality_3cycle": cycle_ok,
            }

            triality_records.append(rec)

            if not (
                normal
                and index == 3
                and len(cands) > 0
                and len(contained_T) == 3
                and cycle_ok
            ):
                all_ok = False

        summary = Counter(
            (
                r["normal"],
                r["index"],
                r["successful_T_inside_L"],
                r["triality_3cycle"],
            )
            for r in triality_records
        )

        print("triality-record types:")

        for k, n in summary.items():
            print(" count", n, "record =", k)

        GATE_E = all_ok

    print("GATE E =", status(GATE_E))

    # =========================================================================
    # F — NATIVE 0/6 GEOMETRY VS N-INTERSECTION ALGEBRA
    # =========================================================================

    header("F", "NATIVE 0/6 GEOMETRY VS N-INTERSECTION ALGEBRA")

    GATE_F = NOT_TESTED
    Npair_records = []
    N_intersection_by_relation = defaultdict(Counter)

    if GATE_D is not True:
        print("NOT TESTED: canonical C <-> N correspondence not earned.")

    else:
        for i in range(25):
            for j in range(i + 1, 25):

                rel = native_pair_type(
                    Cfamily[i],
                    Cfamily[j],
                )

                Ni = C_to_N[i]
                Nj = C_to_N[j]

                J = frozenset(set(Ni) & set(Nj))
                jo = len(J)

                N_intersection_by_relation[rel][jo] += 1

                Npair_records.append(
                    (i, j, rel, J)
                )

        for rel, H in sorted(N_intersection_by_relation.items()):
            print(rel, "-> |N_i cap N_j| histogram =", dict(sorted(H.items())))

        expected_relations = {
            "R0_disjoint",
            "R6_shared_vertices",
        }

        uniform = (
            set(N_intersection_by_relation) == expected_relations
            and all(
                len(H) == 1
                for H in N_intersection_by_relation.values()
            )
        )

        distinct = False

        if uniform:
            vals = [
                next(iter(H))
                for H in N_intersection_by_relation.values()
            ]

            distinct = len(set(vals)) == 2

        print("uniform within native relations =", uniform)
        print("distinct N-intersection orders =", distinct)

        GATE_F = uniform and distinct

    print("GATE F =", status(GATE_F))

    # =========================================================================
    # G — CLASSIFY N-INTERSECTION GROUPS
    # =========================================================================

    header("G", "CLASSIFY N-INTERSECTION GROUPS")

    GATE_G = NOT_TESTED
    J_class_data = {}

    if GATE_F is not True:
        print("NOT TESTED: Gate F did not pass.")

    else:
        # Cache by actual subgroup so ambient normalizer/centralizer scans
        # are not repeated unnecessarily.
        inv_cache = {}

        relation_invtypes = defaultdict(Counter)
        relation_subgroups = defaultdict(set)

        for i, j, rel, J in Npair_records:

            relation_subgroups[rel].add(J)

            if J not in inv_cache:
                inv_cache[J] = subgroup_invariants(J, W)

            inv = inv_cache[J]
            relation_invtypes[rel][invariant_key(inv)] += 1

        for rel in sorted(relation_invtypes):

            print("\nRELATION:", rel)
            print(
                "distinct actual intersection subgroups =",
                len(relation_subgroups[rel]),
            )
            print(
                "invariant types =",
                len(relation_invtypes[rel]),
            )

            for key, count in relation_invtypes[rel].items():
                representative = next(
                    J for J in relation_subgroups[rel]
                    if invariant_key(inv_cache[J]) == key
                )

                inv = inv_cache[representative]

                print(" count =", count)
                print_invariant(inv)

            # Check conjugacy of all J's in this relation
            reps = list(relation_subgroups[rel])
            J0 = reps[0]
            orbJ0 = conjugacy_orbit_subgroup(W, J0)

            all_conjugate = all(
                J in orbJ0
                for J in reps
            )

            print(
                "all J in relation W(H4)-conjugate =",
                all_conjugate,
            )

            J_class_data[rel] = {
                "subgroups": relation_subgroups[rel],
                "invariant_types": relation_invtypes[rel],
                "all_conjugate": all_conjugate,
            }

        # Shared-geometry action for R6
        r6_orbit_sigs = Counter()

        for i, j, rel, J in Npair_records:
            if rel != "R6_shared_vertices":
                continue

            shared = Cfamily[i] & Cfamily[j]

            sig = tuple(
                sorted(
                    len(O)
                    for O in point_orbits_group(J, shared)
                )
            )

            r6_orbit_sigs[sig] += 1

        print(
            "\nR6 action on six shared vertices:",
            dict(r6_orbit_sigs),
        )

        uniform_types = all(
            len(d["invariant_types"]) == 1
            and d["all_conjugate"]
            for d in J_class_data.values()
        )

        GATE_G = (
            set(J_class_data) == {
                "R0_disjoint",
                "R6_shared_vertices",
            }
            and uniform_types
        )

    print("GATE G =", status(GATE_G))

    # =========================================================================
    # H — DESCEND TO FROZEN L-INTERSECTION 8/12
    # =========================================================================

    header("H", "NATIVE 0/6 GEOMETRY VS L-INTERSECTION 8/12")

    GATE_H = NOT_TESTED
    Lpair_records = []
    L_intersection_by_relation = defaultdict(Counter)

    if GATE_D is not True:
        print("NOT TESTED: canonical C <-> L correspondence not earned.")

    else:
        for i in range(25):
            for j in range(i + 1, 25):

                rel = native_pair_type(
                    Cfamily[i],
                    Cfamily[j],
                )

                Li = frozenset(Lfamily[C_to_L[i]])
                Lj = frozenset(Lfamily[C_to_L[j]])

                I = frozenset(set(Li) & set(Lj))
                io = len(I)

                L_intersection_by_relation[rel][io] += 1

                Lpair_records.append(
                    (i, j, rel, I)
                )

        for rel, H in sorted(L_intersection_by_relation.items()):
            print(rel, "-> |L_i cap L_j| histogram =", dict(sorted(H.items())))

        r0 = L_intersection_by_relation.get(
            "R0_disjoint",
            Counter(),
        )

        r6 = L_intersection_by_relation.get(
            "R6_shared_vertices",
            Counter(),
        )

        direct = (
            r0 == {8: 100}
            and r6 == {12: 200}
        )

        reverse = (
            r0 == {12: 100}
            and r6 == {8: 200}
        )

        print("0 -> 8 and 6 -> 12 =", direct)
        print("0 -> 12 and 6 -> 8 =", reverse)

        GATE_H = direct

        if reverse:
            print(
                "NOTE: exact relation descent exists but with labels reversed."
            )

    print("GATE H =", status(GATE_H))

    # =========================================================================
    # I — CLASSIFY I8 / I12 AND COMPARE I <= J
    # =========================================================================

    header("I", "CLASSIFY I8 / I12 AND COMPARE I <= J")

    GATE_I = NOT_TESTED

    if GATE_H is not True:
        print("NOT TESTED: frozen 0/6 -> 8/12 descent not earned.")

    else:
        I_cache = {}
        I_by_order = defaultdict(set)

        # Pair lookup for N intersections
        J_lookup = {
            (i, j): J
            for i, j, rel, J in Npair_records
        }

        nested_index_hist = defaultdict(Counter)
        containment_failures = 0

        for i, j, rel, I in Lpair_records:

            I_by_order[len(I)].add(I)

            if I not in I_cache:
                I_cache[I] = subgroup_invariants(I, W)

            J = J_lookup[i, j]

            contained = set(I) <= set(J)

            if not contained:
                containment_failures += 1
            else:
                nested_index_hist[rel][
                    len(J) // len(I)
                ] += 1

        labels = defaultdict(Counter)
        uniform = True

        for k in sorted(I_by_order):

            print(f"\n|I| = {k}")
            print(
                "distinct actual I subgroups =",
                len(I_by_order[k]),
            )

            type_counter = Counter()

            for I in I_by_order[k]:
                inv = I_cache[I]
                type_counter[invariant_key(inv)] += 1
                labels[k][group_label(inv)] += 1

            print("invariant types =", len(type_counter))

            for key, count in type_counter.items():
                representative = next(
                    I for I in I_by_order[k]
                    if invariant_key(I_cache[I]) == key
                )

                print(" count =", count)
                print_invariant(I_cache[representative])

            print("labels =", dict(labels[k]))

            orb = conjugacy_orbit_subgroup(
                W,
                next(iter(I_by_order[k])),
            )

            all_conjugate = all(
                I in orb
                for I in I_by_order[k]
            )

            print(
                "all same-order I W(H4)-conjugate =",
                all_conjugate,
            )

            if len(type_counter) != 1 or not all_conjugate:
                uniform = False

        print(
            "\nI <= J containment failures =",
            containment_failures,
        )

        for rel, H in sorted(nested_index_hist.items()):
            print(
                rel,
                "-> [J:I] histogram =",
                dict(sorted(H.items())),
            )

        GATE_I = (
            set(I_by_order) == {8, 12}
            and uniform
            and containment_failures == 0
        )

    print("GATE I =", status(GATE_I))

    # =========================================================================
    # J — SAME RANK-3 RELATION AT C, N, L LEVELS
    # =========================================================================

    header("J", "EQUIVARIANT RANK-3 RELATION AT C25 / N25 / L25")

    GATE_J = NOT_TESTED
    graph_sig = None

    if GATE_D is not True:
        print("NOT TESTED: canonical correspondence not earned.")

    else:
        # Native disjointness relation
        native_R0 = frozenset(
            (i, j)
            for i in range(25)
            for j in range(i + 1, 25)
            if native_pair_type(Cfamily[i], Cfamily[j])
            == "R0_disjoint"
        )

        # N family action
        Nordered = [
            C_to_N[i]
            for i in range(25)
        ]

        Nidx, actionN, closedN = subgroup_family_action(
            W,
            Nordered,
        )

        # L family ordered according to cells
        Lordered = [
            frozenset(Lfamily[C_to_L[i]])
            for i in range(25)
        ]

        Lidx2, actionL, closedL = subgroup_family_action(
            W,
            Lordered,
        )

        print("N-family action closed =", closedN)
        print("L-family action closed =", closedL)

        if closedN:
            print(
                "|N action image| =",
                len(set(actionN.values())),
            )
            print(
                "|N action kernel| =",
                len(action_kernel(actionN)),
            )
            print(
                "N pair orbital sizes =",
                [
                    len(O)
                    for O in pair_orbits(actionN, 25)
                ],
            )

        if closedL:
            print(
                "|L action image| =",
                len(set(actionL.values())),
            )
            print(
                "|L action kernel| =",
                len(action_kernel(actionL)),
            )
            print(
                "L pair orbital sizes =",
                [
                    len(O)
                    for O in pair_orbits(actionL, 25)
                ],
            )

        # Because all three families are ordered through the earned C mapping,
        # equivariance means the actual 25-point permutations must coincide.
        same_actions_CN = (
            closedN
            and all(
                actionC[g] == actionN[g]
                for g in W
            )
        )

        same_actions_CL = (
            closedL
            and all(
                actionC[g] == actionL[g]
                for g in W
            )
        )

        print(
            "actual C25 and N25 permutation actions identical =",
            same_actions_CN,
        )

        print(
            "actual C25 and L25 permutation actions identical =",
            same_actions_CL,
        )

        G = graph_from_pairs(25, native_R0)
        graph_sig = graph_signature(G)

        print("\nnative R0 graph signature:")

        for k, v in graph_sig.items():
            print(f"{k:38s}= {v}")

        srg_ok = (
            graph_sig["v"] == 25
            and graph_sig["edges"] == 100
            and graph_sig["degree_hist"] == {8: 25}
            and graph_sig["connected"]
            and graph_sig["adjacent_common_neighbor_hist"] == {3: 100}
            and graph_sig["nonadjacent_common_neighbor_hist"] == {2: 200}
            and graph_sig["spectrum"] == [
                (-2.0, 16),
                (3.0, 8),
                (8.0, 1),
            ]
        )

        print(
            "native graph = srg(25,8,3,2) fingerprint:",
            srg_ok,
        )

        GATE_J = (
            same_actions_CN
            and same_actions_CL
            and srg_ok
        )

    print("GATE J =", status(GATE_J))

    # =========================================================================
    # K — EXPLORATORY: WHAT DOES N -> L FORGET?
    # =========================================================================

    header("K", "EXPLORATORY — WHAT DOES N576 -> L192 FORGET?")

    exploratory_K = NOT_TESTED

    if GATE_E is not True:
        print("NOT TESTED: triality refinement not earned.")

    else:
        # One canonical representative is enough for the first blind probe.
        ci = 0
        C = Cfamily[ci]
        N = C_to_N[ci]
        L = frozenset(Lfamily[C_to_L[ci]])

        objects = induced_faces(C, R)

        print("representative cell =", ci)
        print("|N| =", len(N))
        print("|L| =", len(L))

        print("\nN versus L orbit partitions on native incidence sets:")

        for name, objs in objects.items():
            sigN = orbit_size_signature(N, objs)
            sigL = orbit_size_signature(L, objs)

            print(f"{name:20s}")
            print("  N576 =", sigN)
            print("  L192 =", sigL)

        contained_T = [
            Tfamily[ti]
            for ti in range(len(Tfamily))
            if set(Tfamily[ti]) <= set(L)
        ]

        print(
            "\nsuccessful T kernels inside representative L =",
            len(contained_T),
        )

        for name, objs in objects.items():
            sigs = [
                orbit_size_signature(T, objs)
                for T in contained_T
            ]

            print(
                f"{name:20s} T-fiber signatures =",
                sigs,
            )

        cands = find_C3_complement(N, L)

        if cands:
            c = cands[0]

            print(
                "\nexplicit triality witness order =",
                porder(c),
            )

            # Report how the witness transports the three T's.
            Tindex_local = {
                frozenset(T): i
                for i, T in enumerate(contained_T)
            }

            transport = []

            for i, T in enumerate(contained_T):
                U = conjugate_subgroup(c, T)
                transport.append(
                    Tindex_local.get(U, None)
                )

            print(
                "triality transport on local T triple =",
                transport,
            )

        exploratory_K = True

    print("EXPLORATORY K COMPLETED =", status(exploratory_K))

    # =========================================================================
    # L — C3 / C5 CONTROLS
    # =========================================================================

    header("L", "C3 / C5 CONTROLS")

    exploratory_L = NOT_TESTED
    C3_cell_sig = None
    C5_cell_sig = None

    if GATE_D is not True:
        print("NOT TESTED: native correspondence not earned.")

    else:
        # C3 from N0/L0
        cands = find_C3_complement(N0, L0)

        print("C3 complement witnesses for N0/L0 =", len(cands))

        if cands:
            c = cands[0]
            pc = actionC[c]

            C3_cell_sig = cycle_signature(pc)

            print(
                "representative C3 action on native C25 =",
                C3_cell_sig,
            )

            # Cell matched to N0
            ci0 = N_to_C.get(N0, None)

            if ci0 is not None:
                print(
                    "C3 fixes native cell over N0 =",
                    pc[ci0] == ci0,
                )

        order5 = [
            g for g in W
            if porder(g) == 5
        ]

        print("order-5 elements in W(H4) =", len(order5))

        if order5:
            f = order5[0]
            pf = actionC[f]

            C5_cell_sig = cycle_signature(pf)

            print(
                "representative C5 action on native C25 =",
                C5_cell_sig,
            )

        exploratory_L = True

    print("EXPLORATORY L COMPLETED =", status(exploratory_L))

    # =========================================================================
    # HARD GATE SUMMARY
    # =========================================================================

    print("\n" + "=" * 100)
    print("SIM15.5 HARD GATES")
    print("=" * 100)

    gates = {
        "A frozen SIM15.4 regression": GATE_A,
        "B independent N25 and S25 families": GATE_B,
        "C LITERAL S25 == N25": GATE_C,
        "D unique equivariant C <-> N <-> L": GATE_D,
        "E triality extension N/L = C3": GATE_E,
        "F native relation -> uniform N intersections": GATE_F,
        "G classify native N-intersection classes": GATE_G,
        "H native 0/6 -> frozen L 8/12": GATE_H,
        "I classify I8/I12 and I <= J": GATE_I,
        "J same equivariant rank-3 relation": GATE_J,
    }

    for k, v in gates.items():
        print(f"{k:55s}: {status(v)}")

    # =========================================================================
    # MACHINE TRUTH PACKET
    # =========================================================================

    print("\n" + "=" * 100)
    print("MACHINE TRUTH PACKET")
    print("=" * 100)

    native_normalizer_identity = (
        GATE_C is True
    )

    triality_refinement = (
        GATE_E is True
    )

    relation_descent = (
        GATE_H is True
    )

    all_hard = all(
        v is True
        for v in gates.values()
    )

    truth = {
        "H4_roots": len(R),
        "H4_order": len(W),

        "successful_T_count": len(Tfamily),
        "compatible_L_count": len(Lfamily),

        "native_D4_frame_count": nframes,
        "native_24cell_count": len(Cfamily),

        "distinct_N_normalizers": len(Nset),
        "distinct_native_stabilizers": len(Sset),

        "N_intersect_S_count": len(Nset & Sset),
        "N_minus_S_count": len(Nset - Sset),
        "S_minus_N_count": len(Sset - Nset),

        "NATIVE_NORMALIZER_IDENTITY":
            native_normalizer_identity,

        "TRIALITY_REFINEMENT":
            triality_refinement,

        "RELATION_DESCENT":
            relation_descent,

        "native_pair_relation_hist":
            dict(native_pair_hist),

        "N_intersection_by_native_relation":
            {
                rel: dict(sorted(H.items()))
                for rel, H
                in N_intersection_by_relation.items()
            },

        "L_intersection_by_native_relation":
            {
                rel: dict(sorted(H.items()))
                for rel, H
                in L_intersection_by_relation.items()
            },

        "native_rank3_graph":
            graph_sig,

        "C3_native_cell_cycle_signature":
            C3_cell_sig,

        "C5_native_cell_cycle_signature":
            C5_cell_sig,

        "ALL_HARD_GATES_A_TO_J":
            all_hard,
    }

    for k, v in truth.items():
        print(f"{k:48s}: {v}")

    # =========================================================================
    # INTERPRETATION GUARDRAILS
    # =========================================================================

    print("\n" + "=" * 100)
    print("INTERPRETATION GUARDRAILS")
    print("=" * 100)

    print("""
1. SIM15.4 already falsified:

       L_192 = Stab_W(C_24).

   SIM15.5 does not revise or erase that result.

2. The new killer hypothesis is:

       {N_W(L_i)} == {Stab_W(C_i)}

   as literal subgroup families inside the same W(H4).

3. Equal order 576 is NOT sufficient.

4. Matching element-order histograms are NOT sufficient.

5. Conjugacy or abstract isomorphism is NOT sufficient.

6. Gate C requires actual subgroup equality.

7. If Gate C fails, the phrase "one triality layer too low" is NOT earned.

8. If Gate C passes, the C <-> N <-> L correspondence must still be unique
   and W(H4)-equivariant.

9. N/L ~= C3 is rechecked inside the earned native correspondence.

10. The three successful C2^3 kernels are required to be cyclically
    transported by an explicit order-3 complement witness.

11. N_i cap N_j intersection orders are NOT predicted in advance.
    Native geometry is allowed to tell us what they are.

12. The native geometric relations remain frozen from SIM15.4:

       R0 : |C_i cap C_j| = 0     (100 pairs)
       R6 : |C_i cap C_j| = 6     (200 pairs)

13. Only after C <-> N is earned do we descend to L and test:

       R0 <-> |L_i cap L_j| = 8
       R6 <-> |L_i cap L_j| = 12.

14. I8 and I12 are classified from their actual elements, not their orders.

15. Gate J requires the actual transported W(H4) actions to coincide,
    not merely three abstractly isomorphic SRGs.

16. Gate K is exploratory. No geometric structure explaining 576 -> 192
    is named in advance.

17. Failure to find such a structure does not falsify N/L ~= C3.

18. C3 and C5 are controls, not physical interpretations.

19. No E8, F4 finite geometry, rook-graph identification, or RCFT physics
    is used to construct or rescue this experiment.

20. FALSE means tested and falsified.
    NOT TESTED means an upstream logical prerequisite failed.

21. No downstream success rescues an upstream failed gate.

END SIM15.5
""")


main()






~~~~~~~~~~~~~~~







RESULTS:



====================================================================================================
SIM15.5 — 24-CELL STABILIZER / TRIALITY-NORMALIZER IDENTIFICATION
====================================================================================================

FROZEN QUESTION:

Are the 25 native 24-cell stabilizers exactly the 25 N_W(L)
triality normalizers, and does this identify the native 0/6
geometry with the previously discovered 8/12 architecture relations?

No isomorphism or histogram match may substitute for literal subgroup equality.
No downstream success rescues a failed upstream gate.

====================================================================================================A) FROZEN SIM15.4 REGRESSION
----------------------------------------------------------------------------------------------------
H4 roots = 120
|W(H4)| = 14400
simple-root indices = [0, 76, 5, 41]
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
|T0| = 8
|L0| = 192
|N0| = 576
|T75| = 75
|L25| = 25
D4 frame count = 4800
closure-size histogram = {24: 4800}
distinct 24-root closures = 25
certified native 24-cells = 25
native stabilizer-order histogram = {576: 25}
native pair-relation histogram = {'R6_shared_vertices': 200, 'R0_disjoint': 100}
native C24 pair orbital sizes = [100, 200]
|C24 action image| = 7200
|C24 action kernel| = 2
z in C24 kernel = True
GATE A = True

====================================================================================================B) INDEPENDENT ORDER-576 FAMILIES N25 AND S25
----------------------------------------------------------------------------------------------------
L entries = 25
distinct N_W(L_i) = 25
distinct native stabilizers = 25
N-family subgroup-order histogram = {576: 25}
S-family subgroup-order histogram = {576: 25}
N-family element-order histogram types:
 count 25 hist = {1: 1, 2: 43, 3: 80, 4: 84, 6: 272, 12: 96}
S-family element-order histogram types:
 count 25 hist = {1: 1, 2: 43, 3: 80, 4: 84, 6: 272, 12: 96}
number of L_i per distinct N_i histogram = {1: 25}
GATE B = True

====================================================================================================C) KILLER GATE — LITERAL S25 == N25
----------------------------------------------------------------------------------------------------
|N25| = 25
|S25| = 25
|N25 cap S25| = 25
|N25 \ S25| = 0
|S25 \ N25| = 0
GATE C — LITERAL S25 == N25 = True

====================================================================================================D) CANONICAL C <-> N <-> L CORRESPONDENCE
----------------------------------------------------------------------------------------------------
unique L per N = True
unique C per N = True
C <-> N equivariant = True
C <-> L transported equivariantly = True
GATE D = True

====================================================================================================E) INTERNAL TRIALITY EXTENSION N576 > L192
----------------------------------------------------------------------------------------------------
triality-record types:
 count 25 record = (True, 3, 3, True)
GATE E = True

====================================================================================================F) NATIVE 0/6 GEOMETRY VS N-INTERSECTION ALGEBRA
----------------------------------------------------------------------------------------------------
R0_disjoint -> |N_i cap N_j| histogram = {72: 100}
R6_shared_vertices -> |N_i cap N_j| histogram = {36: 200}
uniform within native relations = True
distinct N-intersection orders = True
GATE F = True

====================================================================================================G) CLASSIFY N-INTERSECTION GROUPS
----------------------------------------------------------------------------------------------------

RELATION: R0_disjoint
distinct actual intersection subgroups = 100
invariant types = 1
 count = 100
  order                   = 72
  order_hist              = {1: 1, 2: 1, 3: 26, 4: 6, 6: 26, 12: 12}
  center_order            = 6
  center_hist             = {1: 1, 2: 1, 3: 2, 6: 2}
  derived_order           = 8
  derived_hist            = {1: 1, 2: 1, 4: 6}
  abelianization_order    = 9
  normalizer_order        = 144
  centralizer_order       = 6
all J in relation W(H4)-conjugate = True

RELATION: R6_shared_vertices
distinct actual intersection subgroups = 200
invariant types = 1
 count = 200
  order                   = 36
  order_hist              = {1: 1, 2: 7, 3: 8, 6: 20}
  center_order            = 6
  center_hist             = {1: 1, 2: 1, 3: 2, 6: 2}
  derived_order           = 3
  derived_hist            = {1: 1, 3: 2}
  abelianization_order    = 12
  normalizer_order        = 72
  centralizer_order       = 6
all J in relation W(H4)-conjugate = True

R6 action on six shared vertices: {(6,): 200}
GATE G = True

====================================================================================================H) NATIVE 0/6 GEOMETRY VS L-INTERSECTION 8/12
----------------------------------------------------------------------------------------------------
R0_disjoint -> |L_i cap L_j| histogram = {8: 100}
R6_shared_vertices -> |L_i cap L_j| histogram = {12: 200}
0 -> 8 and 6 -> 12 = True
0 -> 12 and 6 -> 8 = False
GATE H = True

====================================================================================================I) CLASSIFY I8 / I12 AND COMPARE I <= J
----------------------------------------------------------------------------------------------------
