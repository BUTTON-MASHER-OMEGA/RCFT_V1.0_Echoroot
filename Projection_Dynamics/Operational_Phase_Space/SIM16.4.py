
#!/usr/bin/env python3
"""
SIM16.4 — Frozen Local Symplectic Generator
and Family-Transport Audit.

Exact finite/linear algebra; SymPy only.
No ISP, physical dynamics, or new gauge assumptions.
"""

import json
import sympy as sp

N = 8
I = sp.eye(N)
ZERO = sp.zeros(N)

PAIRS = ((0, 2), (1, 3), (4, 6), (5, 7))
EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (3, 5), (4, 6), (5, 7), (6, 7)
)
p = (1, 0, 3, 2, 5, 4, 7, 6)
h = (7, 6, 5, 4, 3, 2, 1, 0)
s = (2, 3, 0, 1, 6, 7, 4, 5)

Om = sp.zeros(N)
for a, b in PAIRS:
    Om[a, b] = 1
    Om[b, a] = -1

K = Om.applyfunc(abs)
J0 = -Om
A = sp.zeros(N)

for a, b in EDGES:
    A[a, b] = A[b, a] = 1


def perm_matrix(g):
    m = sp.zeros(N)
    for i, j in enumerate(g):
        m[j, i] = 1
    return m


P = perm_matrix(p)
E = sp.eye(N)

Up = sp.Matrix.hstack(*[
    (E[:, 2*i] + E[:, 2*i+1])/sp.sqrt(2)
    for i in range(4)
])

Um = sp.Matrix.hstack(*[
    (E[:, 2*i] - E[:, 2*i+1])/sp.sqrt(2)
    for i in range(4)
])

U = Up.row_join(Um)
O4 = Up.T*Om*Up
Ap = Up.T*A*Up
Am = Um.T*A*Um

B = sp.zeros(4, 8)
for i in range(4):
    B[i, 2*i] = 1
    B[i, 2*i+1] = 1


def flatten(m):
    return sp.Matrix(N*N, 1, list(m))


def nullspace_on(basis, maps):
    if not basis:
        return []

    cols = [
        sp.Matrix(sum((list(f(b)) for f in maps), []))
        for b in basis
    ]
    coeff = sp.Matrix.hstack(*cols)

    return [
        sum(
            (basis[i]*v[i] for i in range(len(basis))),
            sp.zeros(N)
        )
        for v in coeff.nullspace()
    ]


def span_rank(basis):
    if not basis:
        return 0
    return sp.Matrix.hstack(
        *(flatten(b) for b in basis)
    ).rank()


def image_rank(basis, f):
    return span_rank([f(b) for b in basis])


def comm(a, b):
    return a*b - b*a


def is_local(m):
    return all(
        m[i, j] == 0
        for i in range(N)
        for j in range(N)
        if i != j
        and tuple(sorted((i, j))) not in EDGES
    )


def is_sp(m):
    return m.T*Om + Om*m == ZERO


def check(label, condition):
    if not condition:
        raise AssertionError(label)
    print("PASS | " + label)


def serialize(m):
    return [
        [str(m[i, j]) for j in range(m.cols)]
        for i in range(m.rows)
    ]


def expK(x):
    return sp.cosh(x)*I + sp.sinh(x)*K


def Jq(q):
    return -Om*(
        (q + 1/q)/2*I
        + (q - 1/q)/2*K
    )


print(
    "SIM16.4 — FROZEN LOCAL SYMPLECTIC "
    "GENERATOR / FAMILY TRANSPORT AUDIT"
)

# ====================================================
# GATE A — FROZEN INPUTS
# ====================================================

check(
    "A frozen carrier, C8, sector decomposition",
    Om.T == -Om
    and Om*Om == -I
    and K*K == I
    and K*Om == -Om*K
    and P*P == I
    and P*Om == Om*P
    and P*K == K*P
    and P*A == A*P
    and Ap.trace() == 2
    and Am.trace() == -2
    and B*Um == sp.zeros(4)
    and U.T*U == I
)

# ====================================================
# GATE B — STRICT ADJACENCY CONTROLS
# ====================================================

F4 = []
for i in range(4):
    for j in range(4):
        m = sp.zeros(4)
        m[i, j] = 1
        F4.append(m)


def null4(maps):
    cols = [
        sp.Matrix(sum((list(f(b)) for f in maps), []))
        for b in F4
    ]
    mat = sp.Matrix.hstack(*cols)

    return [
        sum(
            (v[i]*F4[i] for i in range(16)),
            sp.zeros(4)
        )
        for v in mat.nullspace()
    ]


inter = null4([
    lambda x: Ap*x - x*Am
])

J4 = Up.T*J0*Up

inter_j = null4([
    lambda x: Ap*x - x*Am,
    lambda x: J4*x - x*J4
])

inter_sp = null4([
    lambda x: Ap*x - x*Am,
    lambda x: (
        Am*(O4*x.T*O4)
        - (O4*x.T*O4)*Ap
    )
])

check(
    "B strict adjacency intertwiner dimensions 3/0/0",
    len(inter) == 3
    and len(inter_j) == 0
    and len(inter_sp) == 0
)

# ====================================================
# GATE C — ALL GRAPH-LOCAL DIRECTIONS
# ====================================================

allowed = sorted(
    {(i, i) for i in range(N)}
    | {
        (i, j)
        for a, b in EDGES
        for i, j in ((a, b), (b, a))
    }
)

local = []
for i, j in allowed:
    m = sp.zeros(N)
    m[i, j] = 1
    local.append(m)

check(
    "C 24 independent C8-local directions",
    len(local) == 24
    and span_rank(local) == 24
    and all(is_local(x) for x in local)
)

# ====================================================
# GATE D — LOCAL SYMPLECTIC GENERATORS
# ====================================================

sps = nullspace_on(
    local,
    [lambda x: x.T*Om + Om*x]
)

check(
    "D graph-local symplectic dimension 12",
    len(sps) == 12
    and span_rank(sps) == 12
    and all(
        is_local(x) and is_sp(x)
        for x in sps
    )
)

partner_support = {
    (a, b)
    for x, y in PAIRS
    for a, b in (
        (x, y), (y, x), (x, x), (y, y)
    )
}

check(
    "D symplectic generators supported only on four partner blocks",
    all(
        m[i, j] == 0
        for m in sps
        for i in range(N)
        for j in range(N)
        if (i, j) not in partner_support
    )
)

# ====================================================
# GATE E — SECTOR-MIXING RANK
# ====================================================

mixrank = image_rank(
    sps,
    lambda x: comm(x, P)
)

nonmix = nullspace_on(
    sps,
    [lambda x: comm(x, P)]
)

check(
    "E six independent mixing directions",
    mixrank == 6
    and len(nonmix) == 6
)

# ====================================================
# GATE F — FIXED COMPLEX STRUCTURES
# ====================================================

def fixed(q):
    return nullspace_on(
        sps,
        [lambda x: comm(x, Jq(q))]
    )


f0 = fixed(sp.Integer(1))
f2 = fixed(sp.Integer(2))
f5 = fixed(sp.Integer(5))

f_all = nullspace_on(
    sps,
    [
        lambda x: comm(x, J0),
        lambda x: comm(x, J0*K)
    ]
)

fixed_data = {}

for qvalue, space in (
    (1, f0), (2, f2), (5, f5)
):
    rank = image_rank(
        space,
        lambda x: comm(x, P)
    )

    fixed_data[str(qvalue)] = {
        "dimension": len(space),
        "mixing_rank": rank
    }

    check(
        "F fixed J exp(t)=" + str(qvalue)
        + " dims 4, mixing rank 2",
        len(space) == 4
        and rank == 2
        and all(
            comm(x, Jq(sp.Integer(qvalue))) == ZERO
            for x in space
        )
    )

check(
    "F common stabilizer all J_t zero",
    len(f_all) == 0
)

# Explicit fixed-J0 rotation basis.
rot = []

for a, b in PAIRS:
    m = sp.zeros(N)
    m[b, a] = 1
    m[a, b] = -1
    rot.append(m)

check(
    "F J0 stabilizer exactly four partner-plane rotations",
    len(rot) == 4
    and span_rank(rot + f0) == 4
    and all(
        is_local(x)
        and is_sp(x)
        and comm(x, J0) == ZERO
        for x in rot
    )
    and all(
        comm(x, y) == ZERO
        for x in rot
        for y in rot
    )
)

# All-positive-q symbolic stabilizer witnesses.
q = sp.symbols("q", positive=True)
root = sp.sqrt(q)

Sq = (
    (root + 1/root)*I
    - (root - 1/root)*K
)/2

Sqi = (
    (root + 1/root)*I
    + (root - 1/root)*K
)/2

rot_q = [
    sp.simplify(Sq*R*Sqi)
    for R in rot
]

check(
    "F symbolic every-real-t fixed stabilizer witnesses",
    all(
        is_local(x)
        and sp.simplify(x.T*Om + Om*x) == ZERO
        and sp.simplify(comm(x, Jq(q))) == ZERO
        for x in rot_q
    )
)

# Each independent symplectic two-plane has
# a one-dimensional fixed-complex centralizer.
def pair_centralizer_rank(a, b):
    ea = E[:, a]
    eb = E[:, b]

    basis = (
        ea*eb.T,
        eb*ea.T,
        ea*ea.T - eb*eb.T
    )

    cols = [
        sp.Matrix(
            list(comm(x, Jq(sp.Integer(2))))
        ).reshape(64, 1)
        for x in basis
    ]
    return sp.Matrix.hstack(*cols).rank()


check(
    "F pairwise 2x2 fixed centralizer dimension one",
    all(
        pair_centralizer_rank(a, b) == 2
        for a, b in PAIRS
    )
)

# ====================================================
# GATE G — FAMILY-PRESERVING TRANSPORT
# ====================================================

# Infinitesimal translation of the J_t family:
# [L,J0]   = c J0 K
# [L,J0 K] = c J0

cols = [
    sp.Matrix(
        list(comm(x, J0))
        + list(comm(x, J0*K))
    )
    for x in sps
]

cols.append(
    sp.Matrix(list(-J0*K) + list(-J0))
)

transport_coeff = sp.Matrix.hstack(*cols)
transport_kernel = transport_coeff.nullspace()

check(
    "G unique family transport direction",
    len(transport_kernel) == 1
    and transport_kernel[0][-1] != 0
)

Lstar = -K/sp.Integer(2)

check(
    "G Lstar symplectic local nonmix and correct tangent",
    is_local(Lstar)
    and is_sp(Lstar)
    and comm(Lstar, P) == ZERO
    and comm(Lstar, J0) == J0*K
    and comm(Lstar, J0*K) == J0
)

# ====================================================
# GATE H — BRACKET AND FINITE-TIME AUDITS
# ====================================================

check(
    "H fixed J0 algebra abelian",
    all(
        comm(x, y) == ZERO
        for x in rot
        for y in rot
    )
)

z = sp.symbols("z", real=True)
finite_rot = []

for R in rot:
    e = (
        I
        + sp.sin(z)*R
        + (1-sp.cos(z))*(R*R)
    )
    finite_rot.append(e)

check(
    "H rotations preserve exact finite-time locality",
    all(
        is_local(e)
        and sp.trigsimp(e.T*e-I) == ZERO
        and sp.trigsimp(e.T*Om*e-Om) == ZERO
        for e in finite_rot
    )
)

S = expK(-z/2)

check(
    "H K transport finite-time locality and conjugacy",
    is_local(S)
    and sp.simplify(S.T*Om*S-Om) == ZERO
    and sp.simplify(
        S*J0*expK(z/2)-Jq(sp.exp(z))
    ) == ZERO
)

# This closure result is derived rather than assumed.
all_brackets_local = all(
    is_local(comm(x, y))
    and is_sp(comm(x, y))
    for x in sps
    for y in sps
)

check(
    "H full local symplectic bracket classification",
    all_brackets_local
)

bracket_at_q = {}

for qvalue, space in (
    (1, f0), (2, f2), (5, f5)
):
    failures = [
        (i, j)
        for i, x in enumerate(space)
        for j, y in enumerate(space)
        if i < j
        and not is_local(comm(x, y))
    ]

    bracket_at_q[str(qvalue)] = {
        "local_bracket_closed": not bool(failures),
        "counterexample_indices": (
            list(failures[0]) if failures else None
        )
    }

    print(
        "H | fixed exp(t)=",
        qvalue,
        "local bracket closed:",
        not bool(failures)
    )

# ====================================================
# HUMAN-READABLE SUMMARY
# ====================================================

print("D | local symplectic basis count:", len(sps))
print("E | mixing image rank:", mixrank)
print("F | fixed-J spaces:", fixed_data)
print("F | fixed-all-J dimension:", len(f_all))
print("G | family transport dimension:", 1)
print("G | representative transport generator: -K/2")
print(
    "H | full local symplectic bracket closed:",
    all_brackets_local
)
print(
    "H | Lie algebra: direct sum of four sp(2,R) blocks"
)

# ====================================================
# MACHINE-READABLE TRUTH PACKET
# ====================================================

packet = {
    "sim": "16.4",
    "status": "EXACT_ALGEBRAIC_AUDIT_PASSED",
    "scope": (
        "Frozen OPS exact algebra; "
        "no dynamics or physical interpretation"
    ),
    "carrier_dim": 8,
    "local_basis_dim": len(local),
    "local_symplectic_dim": len(sps),
    "local_symplectic_lie_algebra":
        "direct sum sp(2,R)^4",
    "local_symplectic_mixing_rank": mixrank,
    "local_symplectic_nonmix_dim": len(nonmix),
    "strict_adjacency_intertwiner_dim": len(inter),
    "adjacency_J0_intertwiner_dim": len(inter_j),
    "adjacency_sp_intertwiner_dim": len(inter_sp),
    "fixed_J_exp_t": fixed_data,
    "fixed_all_J_dim": len(f_all),
    "J0_basis_rotation_pairs": [
        list(pair) for pair in PAIRS
    ],
    "J0_abelian": True,
    "all_real_t_fixed_stabilizer":
        "4D abelian, transported partner rotations",
    "family_transport_dim": 1,
    "family_transport_generator": "-K/2",
    "family_transport_mixing_rank": 0,
    "family_transport_finite_local": True,
    "fixed_rotation_finite_local": True,
    "full_local_sp_bracket_closed":
        all_brackets_local,
    "fixed_J_bracket_tests": bracket_at_q,
    "witnesses": {
        "local_symplectic_basis": [
            serialize(x) for x in sps
        ],
        "fixed_J0_basis": [
            serialize(x) for x in rot
        ],
        "adjacency_intertwiner_basis": [
            serialize(x) for x in inter
        ]
    }
}

print(
    "TRUTH_PACKET_JSON="
    + json.dumps(packet, sort_keys=True)
)

print("END SIM16.4")






###############





RESULTS:





SIM16.4 — FROZEN LOCAL SYMPLECTIC GENERATOR / FAMILY TRANSPORT AUDIT
PASS | A frozen carrier, C8, sector decomposition
PASS | B strict adjacency intertwiner dimensions 3/0/0
PASS | C 24 independent C8-local directions
PASS | D graph-local symplectic dimension 12
PASS | D symplectic generators supported only on four partner blocks
PASS | E six independent mixing directions
PASS | F fixed J exp(t)=1 dims 4, mixing rank 2
PASS | F fixed J exp(t)=2 dims 4, mixing rank 2
PASS | F fixed J exp(t)=5 dims 4, mixing rank 2
PASS | F common stabilizer all J_t zero
PASS | F J0 stabilizer exactly four partner-plane rotations
PASS | F symbolic every-real-t fixed stabilizer witnesses
PASS | F pairwise 2x2 fixed centralizer dimension one
PASS | G unique family transport direction
PASS | G Lstar symplectic local nonmix and correct tangent
PASS | H fixed J0 algebra abelian
PASS | H rotations preserve exact finite-time locality
PASS | H K transport finite-time locality and conjugacy
PASS | H full local symplectic bracket classification
H | fixed exp(t)= 1 local bracket closed: True
H | fixed exp(t)= 2 local bracket closed: True
H | fixed exp(t)= 5 local bracket closed: True
D | local symplectic basis count: 12
E | mixing image rank: 6
F | fixed-J spaces: {'1': {'dimension': 4, 'mixing_rank': 2}, '2': {'dimension': 4, 'mixing_rank': 2}, '5': {'dimension': 4, 'mixing_rank': 2}}
F | fixed-all-J dimension: 0
G | family transport dimension: 1
G | representative transport generator: -K/2
H | full local symplectic bracket closed: True
H | Lie algebra: direct sum of four sp(2,R) blocks
TRUTH_PACKET_JSON={"J0_abelian": true, "J0_basis_rotation_pairs": [[0, 2], [1, 3], [4, 6], [5, 7]], "adjacency_J0_intertwiner_dim": 0, "adjacency_sp_intertwiner_dim": 0, "all_real_t_fixed_stabilizer": "4D abelian, transported partner rotations", "carrier_dim": 8, "family_transport_dim": 1, "family_transport_finite_local": true, "family_transport_generator": "-K/2", "family_transport_mixing_rank": 0, "fixed_J_bracket_tests": {"1": {"counterexample_indices": null, "local_bracket_closed": true}, "2": {"counterexample_indices": null, "local_bracket_closed": true}, "5": {"counterexample_indices": null, "local_bracket_closed": true}}, "fixed_J_exp_t": {"1": {"dimension": 4, "mixing_rank": 2}, "2": {"dimension": 4, "mixing_rank": 2}, "5": {"dimension": 4, "mixing_rank": 2}}, "fixed_all_J_dim": 0, "fixed_rotation_finite_local": true, "full_local_sp_bracket_closed": true, "local_basis_dim": 24, "local_symplectic_dim": 12, "local_symplectic_lie_algebra": "direct sum sp(2,R)^4", "local_symplectic_mixing_rank": 6, "local_symplectic_nonmix_dim": 6, "scope": "Frozen OPS exact algebra; no dynamics or physical interpretation", "sim": "16.4", "status": "EXACT_ALGEBRAIC_AUDIT_PASSED", "strict_adjacency_intertwiner_dim": 3, "witnesses": {"adjacency_intertwiner_basis": [[["0", "0", "-1", "-1"], ["0", "-1", "0", "1"], ["-1", "0", "1", "0"], ["1", "1", "0", "0"]], [["0", "-1", "0", "1"], ["-1", "1", "0", "-2"], ["2", "0", "-1", "1"], ["-1", "0", "1", "0"]], [["-1", "0", "0", "-1"], ["2", "-1", "-1", "2"], ["-2", "1", "1", "-2"], ["1", "0", "0", "1"]]], "fixed_J0_basis": [[["0", "0", "-1", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["1", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"]], [["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "-1", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "1", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"]], [["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "-1", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "1", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"]], [["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "-1"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "1", "0", "0"]]], "local_symplectic_basis": [[["0", "0", "1", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"]], [["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "1", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"]], [["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["1", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"]], [["-1", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "1", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"]], [["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "1", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"]], [["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "-1", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "1", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"]], [["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "1", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"]], [["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "1"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"]], [["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "1", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"]], [["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "-1", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "1", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"]], [["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "1", "0", "0"]], [["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "-1", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0", "0", "1"]]]}}
END SIM16.4
