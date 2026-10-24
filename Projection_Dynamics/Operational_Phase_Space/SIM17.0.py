
#!/usr/bin/env python3
"""
SIM17.0 — Frozen Fano–Octonion Associator Audit.

Hard order:
A. Frozen OPS reconstruction
B. Independent seven-sector checks WITHOUT octonions
C. Signed Cayley–Dickson algebra and J0 lift
D. Associator-projector classification
E. Quadratic and Jt compatibility
F. Local symplectic negative controls
G. Graph and complexified idempotents
H. Signs, origin, and invalid-algebra controls

Exact symbolic algebra. Dependency: sympy.
No physical dynamics, gravity, ISP, or sector selection.
"""

import itertools
import json
import sympy as sp

N = 8
I = sp.eye(N)
Z = sp.zeros(N)

PAIRS = ((0, 2), (1, 3), (4, 6), (5, 7))
EDGES = {
    (0, 1), (0, 2), (1, 3), (2, 4),
    (3, 5), (4, 6), (5, 7), (6, 7)
}
COORD = {
    0: 0, 1: 4, 2: 1, 3: 5,
    4: 7, 5: 3, 6: 6, 7: 2
}
VERTEX = {v: k for k, v in COORD.items()}

p = (1, 0, 3, 2, 5, 4, 7, 6)
h = (7, 6, 5, 4, 3, 2, 1, 0)
s = (2, 3, 0, 1, 6, 7, 4, 5)

OMEGA = sp.zeros(N)
A = sp.zeros(N)

for a, b in PAIRS:
    OMEGA[a, b] = 1
    OMEGA[b, a] = -1

for a, b in EDGES:
    A[a, b] = A[b, a] = 1

K = OMEGA.applyfunc(abs)
J0 = -OMEGA

B = sp.zeros(4, N)
for i in range(4):
    B[i, 2*i] = B[i, 2*i+1] = 1


def pm(g):
    M = sp.zeros(N)
    for i, j in enumerate(g):
        M[j, i] = 1
    return M


P = pm(p)


def check(title, condition):
    if not condition:
        raise AssertionError(title)
    print("PASS | " + title)


def exact(M):
    return M.applyfunc(sp.simplify)


def span_rank(matrices):
    if not matrices:
        return 0
    return sp.Matrix.hstack(
        *(sp.Matrix(list(M)) for M in matrices)
    ).rank()


def projector(vertices):
    Q = sp.zeros(N)
    for v in vertices:
        Q[v, v] = 1
    return Q


def subspaces():
    out = {}
    for a, b in itertools.combinations(range(1, 8), 2):
        U = frozenset((0, a, b, a ^ b))
        if len(U) == 4:
            out[tuple(sorted(U))] = U
    return list(out.values())


print("SIM17.0 — FROZEN FANO / OCTONION ASSOCIATOR AUDIT")

# =========================================================
# GATE A — FROZEN RECONSTRUCTION
# NO OCTONION MULTIPLICATION
# =========================================================

check(
    "A frozen SIM14/16 carrier and coordinates",
    len(set(COORD.values())) == 8
    and len(EDGES) == 8
    and OMEGA.T == -OMEGA
    and OMEGA*OMEGA == -I
    and K*K == I
    and K*OMEGA == -OMEGA*K
    and P*K == K*P
    and P*OMEGA == OMEGA*P
    and all(COORD[p[i]] ^ COORD[i] == 4 for i in range(8))
    and all(COORD[h[i]] ^ COORD[i] == 2 for i in range(8))
    and all(COORD[s[i]] ^ COORD[i] == 1 for i in range(8))
)

# =========================================================
# GATE B — INDEPENDENT FANO SECTOR CLASSIFICATION
# STILL NO OCTONION MULTIPLICATION
# =========================================================

US = subspaces()
all_q = {}
classes = {}
survivors = []

for U in US:
    vertices = frozenset(VERTEX[g] for g in U)
    Q = projector(vertices)

    basis_restriction = sp.Matrix.hstack(
        *[I[:, j] for j in sorted(vertices)]
    )

    rankB = (B*basis_restriction).rank()
    rankOm = OMEGA.extract(
        sorted(vertices), sorted(vertices)
    ).rank()

    contains_s = (1 in U)
    contains_p = (4 in U)

    inside = {
        edge for edge in EDGES
        if edge[0] in vertices and edge[1] in vertices
    }
    cross = {
        edge for edge in EDGES
        if (edge[0] in vertices) != (edge[1] in vertices)
    }

    unseen = set(vertices)
    components = []

    while unseen:
        root = min(unseen)
        reached = {root}
        pending = [root]

        while pending:
            current = pending.pop()

            for a, b in inside:
                neighbor = (
                    b if a == current else
                    a if b == current else None
                )
                if neighbor is not None and neighbor not in reached:
                    reached.add(neighbor)
                    pending.append(neighbor)

        unseen -= reached
        components.append(len(reached))

    signature = (
        len(inside),
        tuple(sorted(components)),
        len(cross)
    )

    tag = (contains_s, contains_p)
    classes[tag] = classes.get(tag, 0) + 1
    all_q[U] = (Q, vertices, rankB, rankOm, signature)

    assert rankB == (2 if contains_p else 4)
    assert rankOm == (4 if contains_s else 0)
    assert (Q*J0 == J0*Q) == contains_s
    assert (Q*K == K*Q) == contains_s

    if contains_s and not contains_p:
        survivors.append(U)

check("B exactly seven Fano coordinate 4-planes", len(US) == 7)

counts = [
    classes[(True, False)],
    classes[(True, True)],
    classes[(False, False)],
    classes[(False, True)]
]

check("B predeclared class counts 2/1/2/2", counts == [2, 1, 2, 2])

expected = {
    frozenset((0, 1, 2, 3)),
    frozenset((0, 1, 6, 7))
}

check(
    "B exactly two symplectic/full-basin sectors",
    set(survivors) == expected
)

expected_graph = {
    frozenset((0, 1, 2, 3)): (2, (2, 2), 4),
    frozenset((0, 1, 6, 7)): (3, (4,), 2)
}

check(
    "B two distinct frozen induced graph signatures",
    all(
        all_q[U][4] == expected_graph[frozenset(U)]
        for U in survivors
    )
)

check(
    "B basin exchanges survivor and affine complement",
    all(
        P*all_q[U][0]*P == I-all_q[U][0]
        for U in survivors
    )
)

print("B | class counts:", counts)
print("B | survivors:", [sorted(all_q[U][1]) for U in survivors])
print(
    "B | graph signatures:",
    {str(sorted(all_q[U][1])): list(all_q[U][4])
     for U in survivors}
)
print("B | PRE-OCTONION INDEPENDENT CHECKS COMPLETE")

# =========================================================
# GATE C — OCTONION MULTIPLICATION
# This is introduced only AFTER Gates A and B pass.
# =========================================================

# Recursive Cayley-Dickson convention:
# (a,b)(c,d) = (ac-conj(d)b, da+b conj(c))

def cd(i, j, n=8):
    if n == 1:
        return (1, 0)

    half = n // 2

    if i < half and j < half:
        return cd(i, j, half)

    if i < half and j >= half:
        sign, k = cd(j-half, i, half)
        return sign, k+half

    if i >= half and j < half:
        sign, k = cd(i-half, j, half)
        return sign*(1 if j == 0 else -1), k+half

    sign, k = cd(j-half, i-half, half)
    return -sign*(1 if j-half == 0 else -1), k


ETA = (1, 1, 1, 1, 1, 1, 1, -1)


def make_mult(eta):
    table = {}
    for a in range(N):
        for b in range(N):
            sign, k = cd(COORD[a], COORD[b])
            vertex = VERTEX[k]
            table[a, b] = (
                sign*eta[a]*eta[b]*eta[vertex],
                vertex
            )
    return table


MUL = make_mult(ETA)


def product(u, v, table=MUL):
    out = sp.zeros(N, 1)
    for i in range(N):
        for j in range(N):
            sign, k = table[i, j]
            out[k] += sign*u[i]*v[j]
    return out


basis = [I[:, i] for i in range(N)]


def left(a, table=MUL):
    M = sp.zeros(N)
    for j in range(N):
        sign, k = table[a, j]
        M[k, j] = sign
    return M


LEFT = [left(i) for i in range(N)]

check(
    "C octonion identity and imaginary squares",
    all(
        MUL[0, i] == (1, i) and MUL[i, 0] == (1, i)
        for i in range(N)
    )
    and all(
        product(basis[i], basis[i]) == -basis[0]
        for i in range(1, N)
    )
)

check(
    "C imaginary anticommutation and XOR support",
    all(
        product(basis[i], basis[j])
        == -product(basis[j], basis[i])
        for i, j in itertools.combinations(range(1, N), 2)
    )
    and all(
        MUL[a, b][1] == VERTEX[COORD[a] ^ COORD[b]]
        for a in range(N) for b in range(N)
    )
)


def assoc(a, b, c):
    return product(product(a, b), c) - product(
        a, product(b, c)
    )


check(
    "C alternativity via trilinear basis identities",
    all(
        assoc(basis[i], basis[i], basis[j]) == sp.zeros(N, 1)
        and assoc(basis[j], basis[i], basis[i]) == sp.zeros(N, 1)
        for i in range(N) for j in range(N)
    )
    and all(
        assoc(basis[i], basis[j], basis[k])
        == -assoc(basis[j], basis[i], basis[k])
        and assoc(basis[i], basis[j], basis[k])
        == -assoc(basis[i], basis[k], basis[j])
        for i, j, k in itertools.product(range(N), repeat=3)
    )
)

check(
    "C norm composition via polarized full-basis identities",
    all(LEFT[i].T*LEFT[i] == I for i in range(N))
    and all(
        LEFT[i].T*LEFT[j] + LEFT[j].T*LEFT[i] == Z
        for i, j in itertools.combinations(range(N), 2)
    )
)

nonassoc = next(
    (
        (i, j, k)
        for i, j, k in itertools.product(range(1, N), repeat=3)
        if assoc(basis[i], basis[j], basis[k]) != sp.zeros(N, 1)
    ),
    None
)

check("C genuinely nonassociative triple", nonassoc is not None)

check(
    "C signed left-multiplication realization of J0",
    LEFT[VERTEX[1]] == J0
)

print("C | nonassociative witness:", nonassoc)
print("C | signed J0 witness eta:", ETA)

# =========================================================
# GATE D — ASSOCIATOR DEFECT / PROJECTORS
# =========================================================

assoc_ranks = []
projector_ids = set()
all_assoc_identity = True

for a, b in itertools.combinations(range(1, N), 2):
    sign, c = MUL[a, b]
    D = sign*LEFT[c] - LEFT[a]*LEFT[b]

    U = frozenset((
        0, COORD[a], COORD[b],
        COORD[a] ^ COORD[b]
    ))
    Q = all_q[U][0]

    good = (
        D.T*D == 4*(I-Q)
        and D.rank() == 4
        and D*Q == Z
        and (I-Q)*D == D
    )

    if not good:
        all_assoc_identity = False

    assoc_ranks.append(D.rank())
    projector_ids.add(U)

check(
    "D all 21 associator projector identities",
    all_assoc_identity
)

check(
    "D all seven Fano projectors reconstructed",
    len(projector_ids) == 7
    and len(assoc_ranks) == 21
    and set(assoc_ranks) == {4}
)

check(
    "D quaternionic closure for all seven lines",
    all(
        MUL[a, b][1] in all_q[U][1]
        for U in US
        for a, b in itertools.product(all_q[U][1], repeat=2)
    )
)

print("D | associator ranks:", assoc_ranks)

# =========================================================
# GATE E — Jt AND QUADRATIC COMPATIBILITY
# =========================================================

# r = exp(t) > 0, exact Laurent arithmetic.
r = sp.symbols("r", positive=True)
G = ((r+1/r)/2)*I + ((r-1/r)/2)*K
J = J0*G

Hq = {}

for U in survivors:
    Q = all_q[U][0]
    W = I-Q
    Hq[U] = exact(W*G*W)

    assert Hq[U].T == Hq[U]
    assert exact(J.T*Hq[U] + Hq[U]*J) == Z
    assert W*G == G*W
    assert Hq[U].rank() == 4

check(
    "E metric quadratic forms conserve under fixed Jt",
    all(
        exact(J.T*Hq[U] + Hq[U]*J) == Z
        for U in survivors
    )
)

check(
    "E positive semidefinite with exact sector kernels",
    all(
        Hq[U].rank() == 4
        and Hq[U]*all_q[U][0] == Z
        and all_q[U][0]*J == J*all_q[U][0]
        for U in survivors
    )
)

lam = sp.symbols("lam")

check(
    "E positive complement spectrum for all positive exp(t)",
    all(
        sp.factor(
            G.extract(
                [i for i in range(N) if i not in all_q[U][1]],
                [i for i in range(N) if i not in all_q[U][1]]
            ).charpoly(lam).as_expr()
            - (lam-r)**2*(lam-1/r)**2
        ) == 0
        for U in survivors
    )
)

a = sp.symbols("a", positive=True)

S = (
    ((sp.sqrt(a)+1/sp.sqrt(a))/2)*I
    - ((sp.sqrt(a)-1/sp.sqrt(a))/2)*K
)

Gshift = G.subs(r, r*a)

check(
    "E full-family transport covariance of quadratic forms",
    all(
        exact(
            S.T*(I-all_q[U][0])*Gshift*(I-all_q[U][0])*S
            - Hq[U]
        ) == Z
        for U in survivors
    )
)

check(
    "E generic Jt is not ordinary Euclidean unit-octonion multiplication",
    exact(J.T*J-G*G) == Z
    and exact((J.T*J-I).subs(r, sp.Integer(2))) != Z
)

# =========================================================
# GATE F — LOCAL SYMPLECTIC NEGATIVE CONTROL
# =========================================================

local_sp = []

for a, b in PAIRS:
    local_sp.extend((
        I[:, a]*I[:, b].T,
        I[:, b]*I[:, a].T,
        I[:, a]*I[:, a].T - I[:, b]*I[:, b].T
    ))

check(
    "F twelve independent local symplectic generators",
    span_rank(local_sp) == 12
    and all(
        L.T*OMEGA + OMEGA*L == Z
        for L in local_sp
    )
)

check(
    "F every local generator preserves compatible Fano sectors",
    all(
        all_q[U][0]*L == L*all_q[U][0]
        for U in survivors
        for L in local_sp
    )
)

print("F | no automatic sector attraction or selection derived")

# =========================================================
# GATE G — GRAPH AND COMPLEXIFIED IDEMPOTENTS
# =========================================================

check(
    "G neither survivor projector commutes with full C8 adjacency",
    all(
        all_q[U][0]*A != A*all_q[U][0]
        for U in survivors
    )
)

imaginary = sp.I
s_vertex = VERTEX[1]

rho_plus = (
    basis[0] + imaginary*basis[s_vertex]
)/2

rho_minus = (
    basis[0] - imaginary*basis[s_vertex]
)/2

check(
    "G complexified octonion idempotents and zero divisors",
    product(rho_plus, rho_plus) == rho_plus
    and product(rho_minus, rho_minus) == rho_minus
    and product(rho_plus, rho_minus) == sp.zeros(N, 1)
    and rho_plus+rho_minus == basis[0]
)

Pi_plus = (I+imaginary*J0)/2
Pi_minus = (I-imaginary*J0)/2

check(
    "G rank-four complex projectors distinct from frozen real projectors",
    Pi_plus*Pi_plus == Pi_plus
    and Pi_minus*Pi_minus == Pi_minus
    and Pi_plus.rank() == 4
    and Pi_minus.rank() == 4
    and all(Pi_plus != all_q[U][0] for U in US)
    and Pi_plus != (I+P)/2
)

# =========================================================
# GATE H — SIGN, ORIGIN, AND INVALID-LAW CONTROLS
# =========================================================

number_lifts = 0

for mask in range(1 << 7):
    eta = (1,) + tuple(
        -1 if mask & (1 << (j-1)) else 1
        for j in range(1, N)
    )

    if left(VERTEX[1], make_mult(eta)) == J0:
        number_lifts += 1

check(
    "H independent sign-rephasing census finds valid J0 lift",
    number_lifts > 0
)

unsigned = {
    pair: (1, k)
    for pair, (_, k) in MUL.items()
}

unsigned_squares = all(
    product(basis[i], basis[i], unsigned) == -basis[0]
    for i in range(1, N)
)

check(
    "H unsigned Fano law fails imaginary-square test",
    not unsigned_squares
)

invalid = dict(MUL)
sign, k = invalid[1, 2]
invalid[1, 2] = (-sign, k)

check(
    "H deliberately broken sign law fails norm composition",
    left(1, invalid).T*left(1, invalid) != I
    or left(1, invalid).T*left(2, invalid)
       + left(2, invalid).T*left(1, invalid) != Z
)

origin_rows = []

for shift in range(8):
    g = tuple(
        VERTEX[COORD[i] ^ shift]
        for i in range(N)
    )

    R = pm(g)

    graph_ok = all(
        tuple(sorted((g[a], g[b]))) in EDGES
        for a, b in EDGES
    )

    omega_sign = (
        1 if R.T*OMEGA*R == OMEGA else
        -1 if R.T*OMEGA*R == -OMEGA else 0
    )

    origin_rows.append({
        "shift": shift,
        "graph": graph_ok,
        "omega_sign": omega_sign,
        "basin_kernel": R*P == P*R
    })

check(
    "H all eight affine origins audited separately",
    len(origin_rows) == 8
)

print("H | J0-compatible identity-preserving rephasings:", number_lifts)
print("H | affine-origin signatures:", origin_rows)

# =========================================================
# MACHINE-READABLE TRUTH PACKET
# =========================================================

packet = {
    "sim": "17.0",
    "status": "EXACT_ALGEBRAIC_AUDIT_PASSED",
    "scope": (
        "Frozen OPS plus introduced octonionic multiplication; "
        "no physical dynamics"
    ),
    "pre_octonion": {
        "fano_sectors": len(US),
        "class_counts": counts,
        "survivors": [
            sorted(all_q[U][1])
            for U in survivors
        ]
    },
    "octonion": {
        "signed_j0_lift": True,
        "eta": list(ETA),
        "alternative": True,
        "norm_composition": True,
        "nonassoc_witness": list(nonassoc),
        "associator_ranks": assoc_ranks,
        "projectors_from_defect": len(projector_ids)
    },
    "quadratic": {
        "positive_semidefinite": True,
        "rank_each": 4,
        "fixed_J_conservation": True,
        "family_transport_covariance": True
    },
    "local_generators_preserve_survivors": True,
    "graph_signatures": {
        str(sorted(all_q[U][1])): list(all_q[U][4])
        for U in survivors
    },
    "sign_rephasings_with_frozen_J0": number_lifts,
    "affine_origin_controls": origin_rows,
    "dynamical_sector_selection_derived": False,
    "h4_bridge_derived": False,
    "octonion_physical_necessity": "UNRESOLVED"
}

print(
    "TRUTH_PACKET_JSON="
    + json.dumps(packet, sort_keys=True)
)

print("END SIM17.0")







##############Results:








SIM17.0 — FROZEN FANO / OCTONION ASSOCIATOR AUDIT
PASS | A frozen SIM14/16 carrier and coordinates
PASS | B exactly seven Fano coordinate 4-planes
PASS | B predeclared class counts 2/1/2/2
PASS | B exactly two symplectic/full-basin sectors
PASS | B two distinct frozen induced graph signatures
PASS | B basin exchanges survivor and affine complement
B | class counts: [2, 1, 2, 2]
B | survivors: [[0, 2, 5, 7], [0, 2, 4, 6]]
B | graph signatures: {'[0, 2, 5, 7]': [2, (2, 2), 4], '[0, 2, 4, 6]': [3, (4,), 2]}
B | PRE-OCTONION INDEPENDENT CHECKS COMPLETE
PASS | C octonion identity and imaginary squares
PASS | C imaginary anticommutation and XOR support
PASS | C alternativity via trilinear basis identities
PASS | C norm composition via polarized full-basis identities
PASS | C genuinely nonassociative triple
PASS | C signed left-multiplication realization of J0
C | nonassociative witness: (1, 2, 4)
C | signed J0 witness eta: (1, 1, 1, 1, 1, 1, 1, -1)
PASS | D all 21 associator projector identities
PASS | D all seven Fano projectors reconstructed
PASS | D quaternionic closure for all seven lines
D | associator ranks: [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
PASS | E metric quadratic forms conserve under fixed Jt
PASS | E positive semidefinite with exact sector kernels
PASS | E positive complement spectrum for all positive exp(t)
PASS | E full-family transport covariance of quadratic forms
PASS | E generic Jt is not ordinary Euclidean unit-octonion multiplication
PASS | F twelve independent local symplectic generators
PASS | F every local generator preserves compatible Fano sectors
F | no automatic sector attraction or selection derived
PASS | G neither survivor projector commutes with full C8 adjacency
PASS | G complexified octonion idempotents and zero divisors
PASS | G rank-four complex projectors distinct from frozen real projectors
PASS | H independent sign-rephasing census finds valid J0 lift
PASS | H unsigned Fano law fails imaginary-square test
PASS | H deliberately broken sign law fails norm composition
PASS | H all eight affine origins audited separately
H | J0-compatible identity-preserving rephasings: 16
H | affine-origin signatures: [{'shift': 0, 'graph': True, 'omega_sign': 1, 'basin_kernel': True}, {'shift': 1, 'graph': False, 'omega_sign': -1, 'basin_kernel': True}, {'shift': 2, 'graph': True, 'omega_sign': -1, 'basin_kernel': True}, {'shift': 3, 'graph': False, 'omega_sign': 1, 'basin_kernel': True}, {'shift': 4, 'graph': True, 'omega_sign': 1, 'basin_kernel': True}, {'shift': 5, 'graph': False, 'omega_sign': -1, 'basin_kernel': True}, {'shift': 6, 'graph': True, 'omega_sign': -1, 'basin_kernel': True}, {'shift': 7, 'graph': False, 'omega_sign': 1, 'basin_kernel': True}]
TRUTH_PACKET_JSON={"affine_origin_controls": [{"basin_kernel": true, "graph": true, "omega_sign": 1, "shift": 0}, {"basin_kernel": true, "graph": false, "omega_sign": -1, "shift": 1}, {"basin_kernel": true, "graph": true, "omega_sign": -1, "shift": 2}, {"basin_kernel": true, "graph": false, "omega_sign": 1, "shift": 3}, {"basin_kernel": true, "graph": true, "omega_sign": 1, "shift": 4}, {"basin_kernel": true, "graph": false, "omega_sign": -1, "shift": 5}, {"basin_kernel": true, "graph": true, "omega_sign": -1, "shift": 6}, {"basin_kernel": true, "graph": false, "omega_sign": 1, "shift": 7}], "dynamical_sector_selection_derived": false, "graph_signatures": {"[0, 2, 4, 6]": [3, [4], 2], "[0, 2, 5, 7]": [2, [2, 2], 4]}, "h4_bridge_derived": false, "local_generators_preserve_survivors": true, "octonion": {"alternative": true, "associator_ranks": [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], "eta": [1, 1, 1, 1, 1, 1, 1, -1], "nonassoc_witness": [1, 2, 4], "norm_composition": true, "projectors_from_defect": 7, "signed_j0_lift": true}, "octonion_physical_necessity": "UNRESOLVED", "pre_octonion": {"class_counts": [2, 1, 2, 2], "fano_sectors": 7, "survivors": [[0, 2, 5, 7], [0, 2, 4, 6]]}, "quadratic": {"family_transport_covariance": true, "fixed_J_conservation": true, "positive_semidefinite": true, "rank_each": 4}, "scope": "Frozen OPS plus introduced octonionic multiplication; no physical dynamics", "sign_rephasings_with_frozen_J0": 16, "sim": "17.0", "status": "EXACT_ALGEBRAIC_AUDIT_PASSED"}
END SIM17.0
