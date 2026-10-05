#!/usr/bin/env python3
"""
SIM15.0 — H4 SYMMETRY SURVIVABILITY MAP
========================================

FIRST EXTERNAL SYMMETRY STRESS TEST AFTER SIM14.8.

PURPOSE
-------
Test whether the frozen SIM14.8 operational architecture can survive
inside an independently constructed H4 Coxeter symmetry environment,
and—only if it survives—measure its natural action on the two primary
H4 geometric witnesses:

    600-cell : 120 vertices, 720 edges, 600 tetrahedral cells
    120-cell : 600 vertices, 1200 edges, dual to the 600-cell

FROZEN SIM14.8 FINGERPRINT
--------------------------
The H4 construction is completed BEFORE this fingerprint is used.

Target architecture:

    G ~= C2^3
    H192 = G ⋊_rho S4

with

    rho(S4) = Stab_GL(3,2)(v),  v != 0
    Fix_G(S4) = <v>
    Z(H192)   = <v> ~= C2

The test is stronger than:
    "Does H4 contain a subgroup of order 192?"

It asks whether H4 contains the realized extension architecture.

NO EIGHT-STATE ORBIT IS REQUIRED OR PREFERENTIALLY SEARCHED.

EXPERIMENTAL ORDER
------------------
A. Construct H4 independently over Q(phi).
B. Verify Coxeter relations and |W(H4)| = 14400.
C. Construct the 600-cell from the H4 root system.
D. Construct the dual 120-cell from the 600 tetrahedral cells.
E. Only now search elementary-abelian C2^3 subgroups.
F. Classify their H4 conjugacy classes.
G. Search normalizers for the SIM14.8 extension fingerprint.
H. For full matches, map actions on 600-cell and 120-cell.
I. Measure conjugacy multiplicity.
J. Compare order-3 and order-5 transport only after survival.
K. Print machine truth packet.

INTERPRETIVE CEILING
--------------------
A positive result does NOT establish:
- RCFT derives H4,
- H4 derives the SIM14 operational phase space,
- the 600-cell or 120-cell is physical spacetime,
- three generations,
- projection friction,
- LCO/ISP dynamics,
- an E8 -> H4 mechanism,
- a D7 bridge,
- F4 compatibility.

A negative result rejects only the tested direct
SIM14.8 -> H4 survivability architecture.

No stochastic sampling.
No fitting.
No optimization.
No RCFT dynamics.
"""

from collections import Counter, defaultdict, deque
from itertools import combinations
import math
import time


# =============================================================================
# 0. BASIC EXACT ARITHMETIC: Q(phi)
# =============================================================================

# Represent a + b*phi as the integer pair (a,b),
# with phi^2 = phi + 1.

ZERO = (0, 0)
ONE  = (1, 0)
PHI  = (0, 1)


def qadd(x, y):
    return (x[0] + y[0], x[1] + y[1])


def qneg(x):
    return (-x[0], -x[1])


def qsub(x, y):
    return (x[0] - y[0], x[1] - y[1])


def qmul(x, y):
    # (a+b phi)(c+d phi)
    # = ac + bd + (ad+bc+bd) phi
    a, b = x
    c, d = y
    return (a*c + b*d,
            a*d + b*c + b*d)


def qstr(x):
    a, b = x
    if b == 0:
        return str(a)
    if a == 0:
        if b == 1:
            return "phi"
        if b == -1:
            return "-phi"
        return f"{b}*phi"
    return f"({a}{'+' if b >= 0 else ''}{b}*phi)"


# =============================================================================
# 1. EXACT H4 ROOT SYSTEM
# =============================================================================

# Simple-root Gram matrix for Coxeter chain 5-3-3.
#
# All simple roots have norm^2 = 2.
#
# For adjacent roots:
#   <a0,a1> = -2 cos(pi/5) = -phi
#   <a1,a2> = -2 cos(pi/3) = -1
#   <a2,a3> = -1
#
# Nonadjacent roots are orthogonal.

GRAM = [[ZERO for _ in range(4)] for _ in range(4)]

for i in range(4):
    GRAM[i][i] = (2, 0)

GRAM[0][1] = GRAM[1][0] = qneg(PHI)
GRAM[1][2] = GRAM[2][1] = (-1, 0)
GRAM[2][3] = GRAM[3][2] = (-1, 0)


def basis_vector(i):
    return tuple(ONE if j == i else ZERO for j in range(4))


SIMPLE_ROOTS = tuple(basis_vector(i) for i in range(4))


def inner(v, w):
    out = ZERO
    for i in range(4):
        for j in range(4):
            out = qadd(
                out,
                qmul(qmul(v[i], GRAM[i][j]), w[j])
            )
    return out


def reflect_root(v, i):
    """
    Reflection in simple root alpha_i.

    Since <alpha_i,alpha_i> = 2,

        s_i(v) = v - <v,alpha_i> alpha_i.

    In simple-root coordinates alpha_i is the i-th basis vector.
    """
    c = inner(v, SIMPLE_ROOTS[i])
    out = list(v)
    out[i] = qsub(out[i], c)
    return tuple(out)


def generate_h4_roots():
    roots = set(SIMPLE_ROOTS)
    q = deque(SIMPLE_ROOTS)

    while q:
        v = q.popleft()
        for i in range(4):
            w = reflect_root(v, i)
            if w not in roots:
                roots.add(w)
                q.append(w)

    return tuple(sorted(roots))


# =============================================================================
# 2. PERMUTATION UTILITIES
# =============================================================================

def identity_perm(n):
    return tuple(range(n))


def compose(p, q):
    """
    p o q
    """
    return tuple(p[q[i]] for i in range(len(q)))


def inverse_perm(p):
    out = [0] * len(p)
    for i, j in enumerate(p):
        out[j] = i
    return tuple(out)


def perm_power(p, n):
    out = identity_perm(len(p))
    base = p

    while n:
        if n & 1:
            out = compose(base, out)
        base = compose(base, base)
        n >>= 1

    return out


def perm_order(p):
    seen = [False] * len(p)
    ans = 1

    for i in range(len(p)):
        if seen[i]:
            continue

        j = i
        length = 0

        while not seen[j]:
            seen[j] = True
            j = p[j]
            length += 1

        if length:
            ans = math.lcm(ans, length)

    return ans


def perm_cycles(p):
    seen = [False] * len(p)
    cycles = []

    for i in range(len(p)):
        if seen[i]:
            continue

        cyc = []
        j = i

        while not seen[j]:
            seen[j] = True
            cyc.append(j)
            j = p[j]

        if len(cyc) > 1:
            cycles.append(tuple(cyc))

    return cycles


def perm_string(p):
    cs = perm_cycles(p)
    if not cs:
        return "()"
    return "".join(
        "(" + " ".join(map(str, c)) + ")"
        for c in cs
    )


def group_closure(generators, degree):
    ID = identity_perm(degree)

    group = {ID}
    q = deque([ID])

    while q:
        g = q.popleft()

        for s in generators:
            h = compose(s, g)

            if h not in group:
                group.add(h)
                q.append(h)

    return tuple(group)


def conjugate(g, x, ginv=None):
    if ginv is None:
        ginv = inverse_perm(g)
    return compose(compose(g, x), ginv)


def orbit_of_points(group, n):
    unseen = set(range(n))
    orbits = []

    while unseen:
        x = min(unseen)
        orb = {g[x] for g in group}
        orbits.append(tuple(sorted(orb)))
        unseen -= orb

    return tuple(sorted(orbits, key=lambda z: (len(z), z)))


def orbit_sizes(group, n):
    return tuple(sorted(len(o) for o in orbit_of_points(group, n)))


def group_order_histogram(group):
    return dict(sorted(Counter(perm_order(g) for g in group).items()))


def center_of_group(group):
    G = set(group)
    out = []

    for z in group:
        if all(compose(z, g) == compose(g, z) for g in group):
            out.append(z)

    return tuple(out)


# =============================================================================
# 3. INDEPENDENT H4 CONSTRUCTION
# =============================================================================

def build_h4():
    roots = generate_h4_roots()
    root_index = {r: i for i, r in enumerate(roots)}

    reflection_perms = []

    for k in range(4):
        p = tuple(
            root_index[reflect_root(r, k)]
            for r in roots
        )
        reflection_perms.append(p)

    W = group_closure(reflection_perms, len(roots))

    return roots, tuple(reflection_perms), W


# =============================================================================
# 4. COXETER SANITY
# =============================================================================

def coxeter_checks(reflections):
    r1, r2, r3, r4 = reflections
    ID = identity_perm(len(r1))

    checks = {
        "r1^2=e": compose(r1, r1) == ID,
        "r2^2=e": compose(r2, r2) == ID,
        "r3^2=e": compose(r3, r3) == ID,
        "r4^2=e": compose(r4, r4) == ID,

        "(r1r2)^5=e":
            perm_power(compose(r1, r2), 5) == ID,

        "(r2r3)^3=e":
            perm_power(compose(r2, r3), 3) == ID,

        "(r3r4)^3=e":
            perm_power(compose(r3, r4), 3) == ID,

        "(r1r3)^2=e":
            perm_power(compose(r1, r3), 2) == ID,

        "(r1r4)^2=e":
            perm_power(compose(r1, r4), 2) == ID,

        "(r2r4)^2=e":
            perm_power(compose(r2, r4), 2) == ID,
    }

    return checks


# =============================================================================
# 5. 600-CELL FROM THE H4 ROOT SYSTEM
# =============================================================================

def build_600_cell_graph(roots):
    """
    With this normalization, adjacent 600-cell vertices have
    inner product phi.

    The construction is checked afterward by:
        120 vertices
        degree 12
        720 edges
        600 tetrahedral K4 cells
    """
    n = len(roots)
    adj = [set() for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if inner(roots[i], roots[j]) == PHI:
                adj[i].add(j)
                adj[j].add(i)

    return adj


def edge_count(adj):
    return sum(len(x) for x in adj) // 2


def degree_histogram(adj):
    return dict(sorted(Counter(len(x) for x in adj).items()))


def enumerate_tetrahedral_cells(adj):
    """
    Enumerate all K4 cliques in the 600-cell graph.

    Each K4 is a tetrahedral cell.
    """
    cells = []

    n = len(adj)

    for a in range(n):
        for b in sorted(x for x in adj[a] if x > a):

            common_ab = adj[a] & adj[b]

            for c in sorted(
                x for x in common_ab
                if x > b
            ):
                common_abc = common_ab & adj[c]

                for d in sorted(
                    x for x in common_abc
                    if x > c
                ):
                    cells.append((a, b, c, d))

    return tuple(sorted(set(cells)))


# =============================================================================
# 6. DUAL 120-CELL
# =============================================================================

def build_dual_120_cell(cells600):
    """
    Vertices of the dual 120-cell correspond to tetrahedral cells
    of the 600-cell.

    Two dual vertices are adjacent iff the corresponding tetrahedra
    share a triangular face.
    """
    face_to_cells = defaultdict(list)

    for ci, cell in enumerate(cells600):
        for face in combinations(cell, 3):
            face_to_cells[tuple(sorted(face))].append(ci)

    adj = [set() for _ in range(len(cells600))]

    for face, owners in face_to_cells.items():
        if len(owners) != 2:
            raise RuntimeError(
                f"600-cell triangular face {face} "
                f"belongs to {len(owners)} cells, expected 2."
            )

        a, b = owners
        adj[a].add(b)
        adj[b].add(a)

    return adj, face_to_cells


# =============================================================================
# 7. INDUCED ACTION ON THE DUAL 120-CELL
# =============================================================================

def induced_cell_perm(g, cells, cell_index):
    return tuple(
        cell_index[tuple(sorted(g[v] for v in cell))]
        for cell in cells
    )


def induced_group_on_cells(group, cells):
    cell_index = {c: i for i, c in enumerate(cells)}

    return tuple(
        induced_cell_perm(g, cells, cell_index)
        for g in group
    )


# =============================================================================
# 8. ENUMERATE ELEMENTARY-ABELIAN C2^3 SUBGROUPS
# =============================================================================

def enumerate_c2cubed_subgroups(W):
    """
    Search all elementary-abelian order-8 subgroups.

    Strategy:
      1. collect involutions;
      2. build commuting relation;
      3. construct each V4 once;
      4. extend V4 by a commuting involution.

    Groups are stored as frozensets of permutation tuples.
    """
    ID = identity_perm(len(W[0]))

    involutions = [
        g for g in W
        if g != ID and compose(g, g) == ID
    ]

    invol_index = {
        g: i for i, g in enumerate(involutions)
    }

    commute = [set() for _ in involutions]

    for i in range(len(involutions)):
        a = involutions[i]

        for j in range(i + 1, len(involutions)):
            b = involutions[j]

            if compose(a, b) == compose(b, a):
                commute[i].add(j)
                commute[j].add(i)

    v4s = set()

    for i, a in enumerate(involutions):
        for j in commute[i]:
            if j <= i:
                continue

            b = involutions[j]
            ab = compose(a, b)

            if ab == ID:
                continue

            k = invol_index.get(ab)

            if k is None:
                continue

            V = frozenset((ID, a, b, ab))

            if len(V) == 4:
                v4s.add(V)

    Ts = set()

    for V in v4s:
        nontriv = [x for x in V if x != ID]
        inds = [invol_index[x] for x in nontriv]

        common = (
            commute[inds[0]]
            & commute[inds[1]]
            & commute[inds[2]]
        )

        for ci in common:
            c = involutions[ci]

            if c in V:
                continue

            generated = set(V)

            for v in V:
                generated.add(compose(c, v))

            if len(generated) != 8:
                continue

            # Every nonidentity element must be an involution.
            if not all(
                x == ID or compose(x, x) == ID
                for x in generated
            ):
                continue

            Ts.add(frozenset(generated))

    return tuple(Ts), tuple(involutions)


# =============================================================================
# 9. H4 CONJUGACY CLASSES OF C2^3 SUBGROUPS
# =============================================================================

def conjugate_subgroup_by_generator(T, g):
    # H4 simple reflections are involutions, so g^-1 = g.
    return frozenset(
        compose(compose(g, x), g)
        for x in T
    )


def subgroup_conjugacy_orbit(T, reflections):
    orb = {T}
    q = deque([T])

    while q:
        U = q.popleft()

        for r in reflections:
            V = conjugate_subgroup_by_generator(U, r)

            if V not in orb:
                orb.add(V)
                q.append(V)

    return frozenset(orb)


def classify_c2cubed_conjugacy(Ts, reflections):
    unseen = set(Ts)
    classes = []

    while unseen:
        T = next(iter(unseen))
        orb = subgroup_conjugacy_orbit(T, reflections)

        classes.append(orb)
        unseen -= set(orb)

    classes.sort(key=lambda C: (len(C), repr(next(iter(C)))))

    return tuple(classes)


# =============================================================================
# 10. NORMALIZER / CENTRALIZER
# =============================================================================

def normalizer_and_centralizer(W, T):
    Tset = set(T)

    normalizer = []
    centralizer = []

    nontriv = [x for x in T if x != identity_perm(len(x))]

    for g in W:
        gi = inverse_perm(g)

        images = [
            conjugate(g, x, gi)
            for x in nontriv
        ]

        if all(y in Tset for y in images):
            normalizer.append(g)

            if all(y == x for x, y in zip(nontriv, images)):
                centralizer.append(g)

    return tuple(normalizer), tuple(centralizer)


# =============================================================================
# 11. F2^3 MODEL OF A C2^3 SUBGROUP
# =============================================================================

BITS = (
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    (0, 1, 1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
)


def bxor(a, b):
    return tuple(x ^ y for x, y in zip(a, b))


def bits_str(v):
    return "".join(map(str, v))


def independent_pair_in_T(T):
    ID = identity_perm(len(next(iter(T))))
    elems = [x for x in T if x != ID]

    a = elems[0]

    for b in elems[1:]:
        if b != a:
            ab = compose(a, b)

            if ab != ID and ab != a and ab != b:
                # Need third generator outside <a,b>.
                V = {ID, a, b, ab}

                for c in elems:
                    if c not in V:
                        return a, b, c

    raise RuntimeError("Could not choose F2^3 basis.")


def coordinate_model_T(T):
    """
    Choose a basis a,b,c for T and build:
        element -> binary coordinate
        binary coordinate -> element
    """
    ID = identity_perm(len(next(iter(T))))
    a, b, c = independent_pair_in_T(T)

    coord_to_elem = {}

    for bits in BITS:
        x = ID

        if bits[0]:
            x = compose(a, x)
        if bits[1]:
            x = compose(b, x)
        if bits[2]:
            x = compose(c, x)

        coord_to_elem[bits] = x

    if len(set(coord_to_elem.values())) != 8:
        raise RuntimeError("F2^3 coordinate construction failed.")

    elem_to_coord = {
        g: bits
        for bits, g in coord_to_elem.items()
    }

    return elem_to_coord, coord_to_elem


# =============================================================================
# 12. GL(3,2) ACTION INDUCED BY NORMALIZER
# =============================================================================

def mat_vec_f2(M, v):
    return tuple(
        sum(M[i][j] * v[j] for j in range(3)) % 2
        for i in range(3)
    )


def matrix_from_normalizer_element(g, T, elem_to_coord, coord_to_elem):
    gi = inverse_perm(g)

    basis_bits = (
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    )

    columns = []

    for b in basis_bits:
        x = coord_to_elem[b]
        y = conjugate(g, x, gi)
        columns.append(elem_to_coord[y])

    # columns -> row-major matrix
    M = tuple(
        tuple(columns[j][i] for j in range(3))
        for i in range(3)
    )

    return M


def matrix_string(M):
    return "[" + ";".join(
        "".join(map(str, row))
        for row in M
    ) + "]"


def induced_GL_image(N, T):
    elem_to_coord, coord_to_elem = coordinate_model_T(T)

    image = {
        matrix_from_normalizer_element(
            g, T, elem_to_coord, coord_to_elem
        )
        for g in N
    }

    return image, elem_to_coord, coord_to_elem


def fixed_vectors_of_matrix_group(Ms):
    fixed = []

    for v in BITS:
        if all(mat_vec_f2(M, v) == v for M in Ms):
            fixed.append(v)

    return tuple(fixed)


# =============================================================================
# 13. CONSTRUCTIVE S4 TEST FOR THE GL(3,2) IMAGE
# =============================================================================

def span2(a, b):
    return frozenset((
        (0, 0, 0),
        a,
        b,
        bxor(a, b),
    ))


def all_2d_subspaces():
    nonzero = BITS[1:]
    planes = set()

    for a, b in combinations(nonzero, 2):
        if a != b:
            P = span2(a, b)
            if len(P) == 4:
                planes.add(P)

    return tuple(sorted(
        planes,
        key=lambda P: tuple(sorted(P))
    ))


def image_plane(M, P):
    return frozenset(mat_vec_f2(M, v) for v in P)


def constructive_s4_test(Ms, fixed_nonzero):
    """
    Stab_GL(3,2)(v) acts faithfully on the four 2D subspaces
    not containing v.

    If image order is 24 and the four-object action is faithful,
    this gives a constructive S4 realization.
    """
    planes = all_2d_subspaces()
    four = tuple(P for P in planes if fixed_nonzero not in P)

    if len(four) != 4:
        return False, None

    pindex = {P: i for i, P in enumerate(four)}

    action = set()

    for M in Ms:
        p = tuple(
            pindex[image_plane(M, P)]
            for P in four
        )
        action.add(p)

    kernel = [
        p for p in action
        if p == identity_perm(4)
    ]

    ok = (
        len(Ms) == 24
        and len(action) == 24
        and len(kernel) == 1
    )

    return ok, {
        "four_planes": four,
        "action_order": len(action),
        "kernel_order": len(kernel),
    }


# =============================================================================
# 14. FULL SIM14.8 FINGERPRINT TEST
# =============================================================================

def fingerprint_test(N, C, T):
    """
    Test the realized SIM14.8 architecture.

    Strong target:
        |T| = 8
        T ~= C2^3
        |N| = 192
        C_N(T) = T
        induced image order = 24
        image ~= S4 constructively
        unique nonzero common fixed vector
        center(N) = <fixed vector>
    """
    result = {}

    result["T_order_8"] = (len(T) == 8)
    result["N_order_192"] = (len(N) == 192)
    result["centralizer_equals_T"] = (set(C) == set(T))

    Ms, elem_to_coord, coord_to_elem = induced_GL_image(N, T)

    result["GL_image_order_24"] = (len(Ms) == 24)

    fixed = fixed_vectors_of_matrix_group(Ms)
    nonzero_fixed = [
        v for v in fixed
        if v != (0, 0, 0)
    ]

    result["unique_nonzero_fixed_vector"] = (
        len(nonzero_fixed) == 1
    )

    s4_ok = False
    s4_data = None

    if len(nonzero_fixed) == 1:
        s4_ok, s4_data = constructive_s4_test(
            Ms, nonzero_fixed[0]
        )

    result["constructive_S4"] = s4_ok

    Z = center_of_group(N)

    center_ok = False
    fixed_elem = None

    if len(nonzero_fixed) == 1:
        fixed_elem = coord_to_elem[nonzero_fixed[0]]

        center_ok = (
            len(Z) == 2
            and fixed_elem in Z
        )

    result["center_order_2"] = (len(Z) == 2)
    result["center_is_fixed_line"] = center_ok

    full = all(result.values())

    return {
        "checks": result,
        "full_match": full,
        "GL_image": Ms,
        "fixed_vectors": fixed,
        "fixed_nonzero": tuple(nonzero_fixed),
        "fixed_element": fixed_elem,
        "center": Z,
        "s4_data": s4_data,
        "elem_to_coord": elem_to_coord,
        "coord_to_elem": coord_to_elem,
    }


# =============================================================================
# 15. RESTRICT A SUBGROUP TO 600-CELL / 120-CELL ACTIONS
# =============================================================================

def subgroup_cell_action(L, cells600):
    return induced_group_on_cells(L, cells600)


def print_orbit_signature(label, group, degree):
    orbits = orbit_of_points(group, degree)

    print(label)
    print("  orbit sizes", [len(o) for o in orbits])

    for i, orb in enumerate(orbits):
        stab_order = len(group) // len(orb)
        print(
            f"  orbit {i}: size={len(orb)} "
            f"stabilizer={stab_order} "
            f"representative={orb[0]}"
        )

    return orbits


# =============================================================================
# 16. EDGE-ORBIT DIAGNOSTICS
# =============================================================================

def edge_set_from_adj(adj):
    return frozenset(
        (i, j)
        for i in range(len(adj))
        for j in adj[i]
        if i < j
    )


def edge_orbits(group, edges):
    unseen = set(edges)
    orbits = []

    while unseen:
        e = min(unseen)
        a, b = e

        orb = set()

        for g in group:
            x, y = g[a], g[b]
            if x > y:
                x, y = y, x
            orb.add((x, y))

        orbits.append(frozenset(orb))
        unseen -= orb

    return tuple(sorted(orbits, key=lambda O: len(O)))


# =============================================================================
# 17. CONJUGACY MULTIPLICITY OF A FULL MATCH
# =============================================================================

def conjugacy_orbit_of_subgroup(L, reflections):
    orb = {frozenset(L)}
    q = deque([frozenset(L)])

    while q:
        U = q.popleft()

        for r in reflections:
            V = frozenset(
                compose(compose(r, x), r)
                for x in U
            )

            if V not in orb:
                orb.add(V)
                q.append(V)

    return frozenset(orb)


# =============================================================================
# 18. ORDER-3 / ORDER-5 TRANSPORT ON MATCH FAMILY
# =============================================================================

def transport_cycle_histogram(match_orbit, W):
    """
    For each order-3 and order-5 element of W, measure its permutation
    cycle structure on the conjugacy orbit of a surviving subgroup.

    This is diagnostic only.

    We do NOT search for triples or fivefolds.
    """
    family = tuple(match_orbit)
    findex = {L: i for i, L in enumerate(family)}

    results = {}

    for target_order in (3, 5):
        hist = Counter()
        normalizing = 0
        count = 0

        for g in W:
            if perm_order(g) != target_order:
                continue

            count += 1
            gi = inverse_perm(g)

            p = []

            for L in family:
                L2 = frozenset(
                    conjugate(g, x, gi)
                    for x in L
                )
                p.append(findex[L2])

            p = tuple(p)

            if all(p[i] == i for i in range(len(p))):
                normalizing += 1

            cyc_lengths = tuple(sorted(
                len(c) for c in perm_cycles(p)
            ))

            fixed = sum(
                1 for i in range(len(p))
                if p[i] == i
            )

            hist[(fixed, cyc_lengths)] += 1

        results[target_order] = {
            "element_count": count,
            "pointwise_family_normalizers": normalizing,
            "histogram": hist,
        }

    return results


# =============================================================================
# 19. MAIN
# =============================================================================

def main():
    t0 = time.time()

    print("=" * 94)
    print("SIM15.0 — H4 SYMMETRY SURVIVABILITY MAP")
    print("=" * 94)

    # -------------------------------------------------------------------------
    # A. INDEPENDENT H4 CONSTRUCTION
    # -------------------------------------------------------------------------

    print("\nA. INDEPENDENT H4 CONSTRUCTION")

    roots, reflections, W = build_h4()

    print("|H4 root system|", len(roots))
    print("root norm histogram",
          dict(sorted(Counter(inner(r, r) for r in roots).items())))

    checks = coxeter_checks(reflections)

    for k, v in checks.items():
        print(k, v)

    print("|W(H4)|", len(W))
    print("expected |W(H4)|=14400", len(W) == 14400)

    h4_sanity = (
        len(roots) == 120
        and all(checks.values())
        and len(W) == 14400
    )

    print("H4 independent sanity PASS", h4_sanity)

    if not h4_sanity:
        raise RuntimeError(
            "STOP — independent H4 construction failed."
        )

    # -------------------------------------------------------------------------
    # B. 600-CELL
    # -------------------------------------------------------------------------

    print("\nB. 600-CELL PRIMARY GEOMETRIC WITNESS")

    adj600 = build_600_cell_graph(roots)
    edges600 = edge_set_from_adj(adj600)
    cells600 = enumerate_tetrahedral_cells(adj600)

    print("|V_600|", len(adj600))
    print("|E_600|", len(edges600))
    print("degree histogram", degree_histogram(adj600))
    print("|tetrahedral cells|", len(cells600))

    sanity600 = (
        len(adj600) == 120
        and len(edges600) == 720
        and degree_histogram(adj600) == {12: 120}
        and len(cells600) == 600
    )

    print("600-cell sanity PASS", sanity600)

    if not sanity600:
        raise RuntimeError(
            "STOP — 600-cell construction failed."
        )

    # -------------------------------------------------------------------------
    # C. DUAL 120-CELL
    # -------------------------------------------------------------------------

    print("\nC. 120-CELL DUAL GEOMETRIC WITNESS")

    adj120, face_to_cells = build_dual_120_cell(cells600)
    edges120 = edge_set_from_adj(adj120)

    print("|V_120cell|", len(adj120))
    print("|E_120cell|", len(edges120))
    print("degree histogram", degree_histogram(adj120))
    print("|600-cell triangular faces|", len(face_to_cells))

    sanity120 = (
        len(adj120) == 600
        and len(edges120) == 1200
        and degree_histogram(adj120) == {4: 600}
        and len(face_to_cells) == 1200
    )

    print("120-cell dual sanity PASS", sanity120)

    if not sanity120:
        raise RuntimeError(
            "STOP — dual 120-cell construction failed."
        )

    # Verify generators induce permutations of all 600 dual vertices.
    W120_gens = induced_group_on_cells(
        reflections,
        cells600
    )

    dual_action_sanity = all(
        sorted(p) == list(range(600))
        for p in W120_gens
    )

    print("H4 generators act on dual 120-cell", dual_action_sanity)

    if not dual_action_sanity:
        raise RuntimeError(
            "STOP — H4 dual action failed."
        )

    # -------------------------------------------------------------------------
    # D. UNSEAL SIM14.8 FINGERPRINT
    # -------------------------------------------------------------------------

    print("\nD. SIM14.8 FINGERPRINT UNSEALED")
    print("Target:")
    print("  T ~= C2^3")
    print("  L ~= C2^3 ⋊_rho S4")
    print("  |L| = 192")
    print("  C_L(T) = T")
    print("  |rho(S4)| = 24")
    print("  rho(S4) ~= Stab_GL(3,2)(v)")
    print("  Fix_T(rho) = <v>, v != 0")
    print("  Z(L) = <v> ~= C2")
    print("No X8 orbit is required.")

    # -------------------------------------------------------------------------
    # E. C2^3 SEARCH
    # -------------------------------------------------------------------------

    print("\nE. ELEMENTARY-ABELIAN C2^3 SEARCH")

    Ts, involutions = enumerate_c2cubed_subgroups(W)

    print("|involutions in W(H4)|", len(involutions))
    print("|C2^3 subgroups|", len(Ts))

    classes = classify_c2cubed_conjugacy(
        Ts,
        reflections
    )

    print("|C2^3 conjugacy classes|", len(classes))
    print(
        "C2^3 conjugacy orbit sizes",
        [len(C) for C in classes]
    )

    # -------------------------------------------------------------------------
    # F. CLASS-BY-CLASS NORMALIZER SCREEN
    # -------------------------------------------------------------------------

    print("\nF. C2^3 NORMALIZER SCREEN")

    class_records = []
    full_matches = []

    for ci, Cclass in enumerate(classes):
        T = next(iter(Cclass))

        # Orbit-stabilizer already predicts normalizer order.
        predicted_N = len(W) // len(Cclass)

        print(f"\nC2^3 class {ci}")
        print("  conjugacy orbit size", len(Cclass))
        print("  predicted normalizer order", predicted_N)

        # Full SIM fingerprint requires |N|=192.
        # We still record every class, but only expensive-normalize
        # classes whose stabilizer order can possibly match.
        if predicted_N != 192:
            print("  full H192 fingerprint possible", False)

            class_records.append({
                "class": ci,
                "orbit_size": len(Cclass),
                "predicted_N": predicted_N,
                "screened": False,
                "full_match": False,
            })

            continue

        N, C = normalizer_and_centralizer(W, T)

        print("  |N_W(T)|", len(N))
        print("  |C_W(T)|", len(C))
        print("  C_W(T) == T", set(C) == set(T))

        fp = fingerprint_test(N, C, T)

        print("  fingerprint checks:")
        for k, v in fp["checks"].items():
            print("   ", k, v)

        print(
            "  fixed vectors",
            [bits_str(v) for v in fp["fixed_vectors"]]
        )

        print(
            "  nonzero fixed vectors",
            [bits_str(v) for v in fp["fixed_nonzero"]]
        )

        print(
            "  |induced GL image|",
            len(fp["GL_image"])
        )

        print(
            "  induced GL order histogram",
            dict(sorted(Counter(
                # compute matrix order by repeated action on 8 vectors
                matrix_order_f2(M)
                for M in fp["GL_image"]
            ).items()))
        )

        print("  FULL SIM14.8 FINGERPRINT MATCH",
              fp["full_match"])

        rec = {
            "class": ci,
            "orbit_size": len(Cclass),
            "predicted_N": predicted_N,
            "screened": True,
            "N": N,
            "C": C,
            "T": T,
            "fp": fp,
            "full_match": fp["full_match"],
        }

        class_records.append(rec)

        if fp["full_match"]:
            full_matches.append(rec)

    # -------------------------------------------------------------------------
    # G. FULL-MATCH GEOMETRY
    # -------------------------------------------------------------------------

    print("\nG. FULL-MATCH GEOMETRIC SURVIVAL")
    print("|full fingerprint conjugacy-class matches|",
          len(full_matches))

    for mi, rec in enumerate(full_matches):
        print(f"\nFULL MATCH {mi}")
        print("C2^3 class", rec["class"])

        L = rec["N"]
        T = tuple(rec["T"])

        print("|L|", len(L))
        print("|T|", len(T))
        print("L order histogram", group_order_histogram(L))

        # 600-cell action is already the root permutation action.
        print_orbit_signature(
            "L action on 600-cell vertices",
            L,
            120
        )

        print_orbit_signature(
            "T action on 600-cell vertices",
            T,
            120
        )

        # Dual 120-cell action.
        L120 = subgroup_cell_action(L, cells600)
        T120 = subgroup_cell_action(T, cells600)

        print_orbit_signature(
            "L action on 120-cell vertices",
            L120,
            600
        )

        print_orbit_signature(
            "T action on 120-cell vertices",
            T120,
            600
        )

        # Edge orbit signatures.
        eo600 = edge_orbits(L, edges600)
        eo120 = edge_orbits(L120, edges120)

        print(
            "L edge-orbit sizes on 600-cell",
            [len(O) for O in eo600]
        )

        print(
            "L edge-orbit sizes on 120-cell",
            [len(O) for O in eo120]
        )

        # Distinguished fixed central involution.
        z = rec["fp"]["fixed_element"]

        if z is not None:
            z600_fixed = sum(
                1 for i in range(120)
                if z[i] == i
            )

            z120 = induced_group_on_cells(
                (z,),
                cells600
            )[0]

            z120_fixed = sum(
                1 for i in range(600)
                if z120[i] == i
            )

            print(
                "distinguished central involution "
                "fixed 600-cell vertices",
                z600_fixed
            )

            print(
                "distinguished central involution "
                "fixed 120-cell vertices",
                z120_fixed
            )

    # -------------------------------------------------------------------------
    # H. CONJUGACY MULTIPLICITY / THREEFOLD TEST
    # -------------------------------------------------------------------------

    print("\nH. MATCH-FAMILY CONJUGACY GEOMETRY")

    match_families = []

    for mi, rec in enumerate(full_matches):
        L = rec["N"]

        fam = conjugacy_orbit_of_subgroup(
            L,
            reflections
        )

        normalizer_order = len(W) // len(fam)

        print(f"match {mi}")
        print("  conjugacy orbit size", len(fam))
        print("  normalizer order", normalizer_order)

        match_families.append(fam)

    print(
        "observed full-match orbit sizes",
        [len(F) for F in match_families]
    )

    print(
        "intrinsic orbit size 3 observed",
        any(len(F) == 3 for F in match_families)
    )

    # -------------------------------------------------------------------------
    # I. ORDER-3 / ORDER-5 TRANSPORT CONTROL
    # -------------------------------------------------------------------------

    print("\nI. ORDER-3 / ORDER-5 TRANSPORT DIAGNOSTIC")

    # Avoid duplicate analysis if several class representatives
    # generate identical match-family orbits.
    unique_families = []

    seen_family_keys = set()

    for fam in match_families:
        key = frozenset(fam)

        if key not in seen_family_keys:
            seen_family_keys.add(key)
            unique_families.append(fam)

    for fi, fam in enumerate(unique_families):
        print(f"\nmatch family {fi}")
        print("family size", len(fam))

        transport = transport_cycle_histogram(
            fam,
            W
        )

        for order in (3, 5):
            data = transport[order]

            print(f"order-{order} elements",
                  data["element_count"])

            print(
                f"order-{order} elements fixing "
                "every match-family member",
                data["pointwise_family_normalizers"]
            )

            print(f"order-{order} transport histogram")

            for key, count in sorted(
                data["histogram"].items(),
                key=lambda kv: repr(kv[0])
            ):
                fixed, cycles = key
                print(
                    "   fixed=",
                    fixed,
                    "nontrivial_cycles=",
                    cycles,
                    "count=",
                    count
                )

    # -------------------------------------------------------------------------
    # J. FAILURE / SURVIVAL LADDER
    # -------------------------------------------------------------------------

    print("\nJ. SURVIVAL LADDER")

    any_c2cubed = len(Ts) > 0

    any_192_normalizer = any(
        rec["predicted_N"] == 192
        for rec in class_records
    )

    any_extension = any(
        rec.get("screened", False)
        and rec["fp"]["checks"]["centralizer_equals_T"]
        and rec["fp"]["checks"]["GL_image_order_24"]
        for rec in class_records
    )

    any_rho = any(
        rec.get("screened", False)
        and rec["fp"]["checks"]["constructive_S4"]
        for rec in class_records
    )

    any_fixed = any(
        rec.get("screened", False)
        and rec["fp"]["checks"]["unique_nonzero_fixed_vector"]
        for rec in class_records
    )

    any_center = any(
        rec.get("screened", False)
        and rec["fp"]["checks"]["center_is_fixed_line"]
        for rec in class_records
    )

    any_full = len(full_matches) > 0

    print("H4 constructed", h4_sanity)
    print("600-cell constructed", sanity600)
    print("120-cell constructed", sanity120)
    print("C2^3 occurs", any_c2cubed)
    print("order-192 C2^3 normalizer occurs", any_192_normalizer)
    print("C2^3 ⋊ order-24 extension architecture occurs",
          any_extension)
    print("rho/S4 action compatible", any_rho)
    print("unique fixed-line architecture compatible", any_fixed)
    print("center/fixed-line architecture compatible", any_center)
    print("FULL SIM14.8 fingerprint survives", any_full)

    # -------------------------------------------------------------------------
    # K. MACHINE TRUTH PACKET
    # -------------------------------------------------------------------------

    print("\nK. MACHINE TRUTH PACKET")

    frozen_h4 = {
        "|roots|=120": len(roots) == 120,
        "Coxeter relations": all(checks.values()),
        "|W(H4)|=14400": len(W) == 14400,

        "600-cell |V|=120": len(adj600) == 120,
        "600-cell |E|=720": len(edges600) == 720,
        "600-cell degree=12": degree_histogram(adj600) == {12: 120},
        "600 tetrahedral cells": len(cells600) == 600,

        "120-cell |V|=600": len(adj120) == 600,
        "120-cell |E|=1200": len(edges120) == 1200,
        "120-cell degree=4": degree_histogram(adj120) == {4: 600},
        "dual H4 action valid": dual_action_sanity,
    }

    print("INDEPENDENT H4 / POLYTOPE SANITY")

    for k, v in frozen_h4.items():
        print(k, v)

    print(
        "ALL INDEPENDENT H4/POLYTOPE CHECKS PASS",
        all(frozen_h4.values())
    )

    print("\nSIM14.8 -> H4 SURVIVABILITY")
    print("C2^3 candidates found", len(Ts))
    print("C2^3 conjugacy classes", len(classes))
    print(
        "192-normalizer candidate classes",
        sum(
            rec["predicted_N"] == 192
            for rec in class_records
        )
    )
    print(
        "full fingerprint class matches",
        len(full_matches)
    )

    print(
        "DIRECT SIM14.8 -> H4 SURVIVABILITY",
        "YES" if any_full else "NO"
    )

    print("\nINTERPRETIVE CEILING")
    print("No RCFT dynamics added.")
    print("No LCO/ISP added.")
    print("No X8 orbit required.")
    print("No threefold multiplicity assumed.")
    print("No fivefold multiplicity assumed.")
    print("No E8 -> H4 mechanism claimed.")
    print("No D7 bridge claimed.")
    print("No F4/B4/A4 compatibility claimed.")
    print("No projection-friction interpretation added.")

    print("\nElapsed seconds", round(time.time() - t0, 3))

    print("\n" + "=" * 94)
    print("SIM15.0 COMPLETE")
    print("=" * 94)


# =============================================================================
# MATRIX ORDER HELPER — defined late because it only needs F2 routines above
# =============================================================================

def matrix_order_f2(M):
    """
    Order of a 3x3 F2 matrix by its permutation action on F2^3.
    """
    p = tuple(
        BITS.index(mat_vec_f2(M, v))
        for v in BITS
    )

    return perm_order(p)


if __name__ == "__main__":
    main()





~~~~~~~~~~~~~~~~~~~~~







Results:





==============================================================================================
SIM15.0 — H4 SYMMETRY SURVIVABILITY MAP
==============================================================================================

A. INDEPENDENT H4 CONSTRUCTION|H4 root system| 120
root norm histogram {(2, 0): 120}
r1^2=e True
r2^2=e True
r3^2=e True
r4^2=e True
(r1r2)^5=e True
(r2r3)^3=e True
(r3r4)^3=e True
(r1r3)^2=e True
(r1r4)^2=e True
(r2r4)^2=e True
|W(H4)| 14400
expected |W(H4)|=14400 True
H4 independent sanity PASS True

B. 600-CELL PRIMARY GEOMETRIC WITNESS|V_600| 120
|E_600| 720
degree histogram {12: 120}
|tetrahedral cells| 600
600-cell sanity PASS True

C. 120-CELL DUAL GEOMETRIC WITNESS|V_120cell| 600
|E_120cell| 1200
degree histogram {4: 600}
|600-cell triangular faces| 1200
120-cell dual sanity PASS True
H4 generators act on dual 120-cell True

D. SIM14.8 FINGERPRINT UNSEALEDTarget:
  T ~= C2^3
  L ~= C2^3 ⋊_rho S4
  |L| = 192
  C_L(T) = T
  |rho(S4)| = 24
  rho(S4) ~= Stab_GL(3,2)(v)
  Fix_T(rho) = <v>, v != 0
  Z(L) = <v> ~= C2
No X8 orbit is required.

E. ELEMENTARY-ABELIAN C2^3 SEARCH|involutions in W(H4)| 571
|C2^3 subgroups| 1200
|C2^3 conjugacy classes| 5
C2^3 conjugacy orbit sizes [75, 75, 300, 300, 450]

F. C2^3 NORMALIZER SCREEN
C2^3 class 0  conjugacy orbit size 75
  predicted normalizer order 192
  |N_W(T)| 192
  |C_W(T)| 16
  C_W(T) == T False
  fingerprint checks:
    T_order_8 True
    N_order_192 True
    centralizer_equals_T False
    GL_image_order_24 False
    unique_nonzero_fixed_vector True
    constructive_S4 False
    center_order_2 True
    center_is_fixed_line True
  fixed vectors ['000', '010']
  nonzero fixed vectors ['010']
  |induced GL image| 12
  induced GL order histogram {1: 1, 2: 3, 3: 8}
  FULL SIM14.8 FINGERPRINT MATCH False

C2^3 class 1  conjugacy orbit size 75
  predicted normalizer order 192
  |N_W(T)| 192
  |C_W(T)| 8
  C_W(T) == T True
  fingerprint checks:
    T_order_8 True
    N_order_192 True
    centralizer_equals_T True
    GL_image_order_24 True
    unique_nonzero_fixed_vector True
    constructive_S4 True
    center_order_2 True
    center_is_fixed_line True
  fixed vectors ['000', '001']
  nonzero fixed vectors ['001']
  |induced GL image| 24
  induced GL order histogram {1: 1, 2: 9, 3: 8, 4: 6}
  FULL SIM14.8 FINGERPRINT MATCH True

C2^3 class 2  conjugacy orbit size 300
  predicted normalizer order 48
  full H192 fingerprint possible False

C2^3 class 3  conjugacy orbit size 300
  predicted normalizer order 48
  full H192 fingerprint possible False

C2^3 class 4  conjugacy orbit size 450
  predicted normalizer order 32
  full H192 fingerprint possible False

G. FULL-MATCH GEOMETRIC SURVIVAL|full fingerprint conjugacy-class matches| 1

FULL MATCH 0C2^3 class 1
|L| 192
|T| 8
L order histogram {1: 1, 2: 43, 3: 32, 4: 84, 6: 32}
L action on 600-cell vertices
  orbit sizes [24, 32, 32, 32]
  orbit 0: size=24 stabilizer=8 representative=0
  orbit 1: size=32 stabilizer=6 representative=1
  orbit 2: size=32 stabilizer=6 representative=3
  orbit 3: size=32 stabilizer=6 representative=8
T action on 600-cell vertices
  orbit sizes [4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
  orbit 0: size=4 stabilizer=2 representative=0
  orbit 1: size=4 stabilizer=2 representative=4
  orbit 2: size=4 stabilizer=2 representative=5
  orbit 3: size=4 stabilizer=2 representative=30
  orbit 4: size=4 stabilizer=2 representative=33
  orbit 5: size=4 stabilizer=2 representative=38
  orbit 6: size=8 stabilizer=1 representative=1
  orbit 7: size=8 stabilizer=1 representative=3
  orbit 8: size=8 stabilizer=1 representative=7
  orbit 9: size=8 stabilizer=1 representative=8
  orbit 10: size=8 stabilizer=1 representative=9
  orbit 11: size=8 stabilizer=1 representative=14
  orbit 12: size=8 stabilizer=1 representative=16
  orbit 13: size=8 stabilizer=1 representative=17
  orbit 14: size=8 stabilizer=1 representative=24
  orbit 15: size=8 stabilizer=1 representative=29
  orbit 16: size=8 stabilizer=1 representative=35
  orbit 17: size=8 stabilizer=1 representative=36
L action on 120-cell vertices
  orbit sizes [8, 8, 8, 32, 32, 32, 96, 96, 96, 192]
  orbit 0: size=8 stabilizer=24 representative=21
  orbit 1: size=8 stabilizer=24 representative=55
  orbit 2: size=8 stabilizer=24 representative=105
  orbit 3: size=32 stabilizer=6 representative=20
  orbit 4: size=32 stabilizer=6 representative=31
  orbit 5: size=32 stabilizer=6 representative=53
  orbit 6: size=96 stabilizer=2 representative=0
  orbit 7: size=96 stabilizer=2 representative=3
  orbit 8: size=96 stabilizer=2 representative=8
  orbit 9: size=192 stabilizer=1 representative=2
T action on 120-cell vertices
  orbit sizes [2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
  orbit 0: size=2 stabilizer=4 representative=21
  orbit 1: size=2 stabilizer=4 representative=210
  orbit 2: size=2 stabilizer=4 representative=284
  orbit 3: size=2 stabilizer=4 representative=289
  orbit 4: size=8 stabilizer=1 representative=0
  orbit 5: size=8 stabilizer=1 representative=2
  orbit 6: size=8 stabilizer=1 representative=3
  orbit 7: size=8 stabilizer=1 representative=4
  orbit 8: size=8 stabilizer=1 representative=8
  orbit 9: size=8 stabilizer=1 representative=9
  orbit 10: size=8 stabilizer=1 representative=10
  orbit 11: size=8 stabilizer=1 representative=11
  orbit 12: size=8 stabilizer=1 representative=12
  orbit 13: size=8 stabilizer=1 representative=13
  orbit 14: size=8 stabilizer=1 representative=20
  orbit 15: size=8 stabilizer=1 representative=23
  orbit 16: size=8 stabilizer=1 representative=24
  orbit 17: size=8 stabilizer=1 representative=26
  orbit 18: size=8 stabilizer=1 representative=27
  orbit 19: size=8 stabilizer=1 representative=29
  orbit 20: size=8 stabilizer=1 representative=31
  orbit 21: size=8 stabilizer=1 representative=32
  orbit 22: size=8 stabilizer=1 representative=33
  orbit 23: size=8 stabilizer=1 representative=34
  orbit 24: size=8 stabilizer=1 representative=48
  orbit 25: size=8 stabilizer=1 representative=49
  orbit 26: size=8 stabilizer=1 representative=51
  orbit 27: size=8 stabilizer=1 representative=52
  orbit 28: size=8 stabilizer=1 representative=53
  orbit 29: size=8 stabilizer=1 representative=54
  orbit 30: size=8 stabilizer=1 representative=55
  orbit 31: size=8 stabilizer=1 representative=56
  orbit 32: size=8 stabilizer=1 representative=59
  orbit 33: size=8 stabilizer=1 representative=60
  orbit 34: size=8 stabilizer=1 representative=61
  orbit 35: size=8 stabilizer=1 representative=62
  orbit 36: size=8 stabilizer=1 representative=71
  orbit 37: size=8 stabilizer=1 representative=72
  orbit 38: size=8 stabilizer=1 representative=73
  orbit 39: size=8 stabilizer=1 representative=74
  orbit 40: size=8 stabilizer=1 representative=93
  orbit 41: size=8 stabilizer=1 representative=94
  orbit 42: size=8 stabilizer=1 representative=95
  orbit 43: size=8 stabilizer=1 representative=96
  orbit 44: size=8 stabilizer=1 representative=97
  orbit 45: size=8 stabilizer=1 representative=98
  orbit 46: size=8 stabilizer=1 representative=99
  orbit 47: size=8 stabilizer=1 representative=100
  orbit 48: size=8 stabilizer=1 representative=101
  orbit 49: size=8 stabilizer=1 representative=102
  orbit 50: size=8 stabilizer=1 representative=103
  orbit 51: size=8 stabilizer=1 representative=104
  orbit 52: size=8 stabilizer=1 representative=105
  orbit 53: size=8 stabilizer=1 representative=106
  orbit 54: size=8 stabilizer=1 representative=111
  orbit 55: size=8 stabilizer=1 representative=114
  orbit 56: size=8 stabilizer=1 representative=115
  orbit 57: size=8 stabilizer=1 representative=116
  orbit 58: size=8 stabilizer=1 representative=123
  orbit 59: size=8 stabilizer=1 representative=126
  orbit 60: size=8 stabilizer=1 representative=127
  orbit 61: size=8 stabilizer=1 representative=128
  orbit 62: size=8 stabilizer=1 representative=145
  orbit 63: size=8 stabilizer=1 representative=146
  orbit 64: size=8 stabilizer=1 representative=147
  orbit 65: size=8 stabilizer=1 representative=148
  orbit 66: size=8 stabilizer=1 representative=149
  orbit 67: size=8 stabilizer=1 representative=150
  orbit 68: size=8 stabilizer=1 representative=151
  orbit 69: size=8 stabilizer=1 representative=152
  orbit 70: size=8 stabilizer=1 representative=161
  orbit 71: size=8 stabilizer=1 representative=162
  orbit 72: size=8 stabilizer=1 representative=163
  orbit 73: size=8 stabilizer=1 representative=164
  orbit 74: size=8 stabilizer=1 representative=165
  orbit 75: size=8 stabilizer=1 representative=166
  orbit 76: size=8 stabilizer=1 representative=168
  orbit 77: size=8 stabilizer=1 representative=169
L edge-orbit sizes on 600-cell [48, 48, 48, 96, 96, 96, 96, 96, 96]
L edge-orbit sizes on 120-cell [32, 32, 32, 48, 48, 48, 96, 96, 96, 96, 192, 192, 192]
distinguished central involution fixed 600-cell vertices 0
distinguished central involution fixed 120-cell vertices 0

H. MATCH-FAMILY CONJUGACY GEOMETRYmatch 0
  conjugacy orbit size 25
  normalizer order 576
observed full-match orbit sizes [25]
intrinsic orbit size 3 observed False

I. ORDER-3 / ORDER-5 TRANSPORT DIAGNOSTIC
match family 0family size 25
order-3 elements 440
order-3 elements fixing every match-family member 0
order-3 transport histogram
   fixed= 10 nontrivial_cycles= (3, 3, 3, 3, 3) count= 40
   fixed= 4 nontrivial_cycles= (3, 3, 3, 3, 3, 3, 3) count= 400
order-5 elements 624
order-5 elements fixing every match-family member 0
order-5 transport histogram
   fixed= 0 nontrivial_cycles= (5, 5, 5, 5, 5) count= 624

J. SURVIVAL LADDERH4 constructed True
600-cell constructed True
120-cell constructed True
C2^3 occurs True
order-192 C2^3 normalizer occurs True
C2^3 ⋊ order-24 extension architecture occurs True
rho/S4 action compatible True
unique fixed-line architecture compatible True
center/fixed-line architecture compatible True
FULL SIM14.8 fingerprint survives True

K. MACHINE TRUTH PACKETINDEPENDENT H4 / POLYTOPE SANITY
|roots|=120 True
Coxeter relations True
|W(H4)|=14400 True
600-cell |V|=120 True
600-cell |E|=720 True
600-cell degree=12 True
600 tetrahedral cells True
120-cell |V|=600 True
120-cell |E|=1200 True
120-cell degree=4 True
dual H4 action valid True
ALL INDEPENDENT H4/POLYTOPE CHECKS PASS True

SIM14.8 -> H4 SURVIVABILITYC2^3 candidates found 1200
C2^3 conjugacy classes 5
192-normalizer candidate classes 2
full fingerprint class matches 1
DIRECT SIM14.8 -> H4 SURVIVABILITY YES

INTERPRETIVE CEILINGNo RCFT dynamics added.
No LCO/ISP added.
No X8 orbit required.
No threefold multiplicity assumed.
No fivefold multiplicity assumed.
No E8 -> H4 mechanism claimed.
No D7 bridge claimed.
No F4/B4/A4 compatibility claimed.
No projection-friction interpretation added.

Elapsed seconds
