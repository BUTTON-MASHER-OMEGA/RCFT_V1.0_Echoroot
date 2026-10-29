
#!/usr/bin/env python3
"""
SIM16.3 — Frozen Projection–Complex Structure Compatibility Audit

Exact finite and symbolic algebra only.
No ISP dynamics, new physical assumptions, or gauge identifications.
Dependency: sympy

Includes the final Gate H frozen C8 adjacency closure patch.
"""

import itertools
import json
import sympy as sp

n = 8
I = sp.eye(n)
Z = sp.zeros(n)

# ============================================================
# FROZEN SIM14–16 DEFINITIONS
# ============================================================

p = (1, 0, 3, 2, 5, 4, 7, 6)
h = (7, 6, 5, 4, 3, 2, 1, 0)
s = (2, 3, 0, 1, 6, 7, 4, 5)

PAIRS = ((0, 2), (1, 3), (4, 6), (5, 7))

EDGES = {
    (0, 1), (0, 2), (1, 3), (2, 4),
    (3, 5), (4, 6), (5, 7), (6, 7)
}

Om = sp.zeros(n)
for a, b in PAIRS:
    Om[a, b] = 1
    Om[b, a] = -1

K = Om.applyfunc(abs)
J0 = -Om
ident = tuple(range(n))


def compose(a, b):
    return tuple(a[b[i]] for i in range(n))


def closure(gens):
    group = {ident}
    frontier = [ident]

    while frontier:
        a = frontier.pop()
        for b in gens:
            c = compose(a, b)
            if c not in group:
                group.add(c)
                frontier.append(c)

    return group


def pm(g):
    M = sp.zeros(n)
    for i, j in enumerate(g):
        M[j, i] = 1
    return M


def sign(g):
    v = pm(g).T * Om * pm(g)
    if v == Om:
        return 1
    if v == -Om:
        return -1
    return 0


def expK(x):
    return sp.cosh(x) * I + sp.sinh(x) * K


def exact(M):
    return sp.simplify(sp.expand_trig(M))


def chk(label, condition):
    if not condition:
        raise AssertionError(label)
    print("PASS | " + label)


def rows(M):
    return [
        [str(M[i, j]) for j in range(M.cols)]
        for i in range(M.rows)
    ]


print("SIM16.3 — FROZEN PROJECTION–COMPLEX STRUCTURE AUDIT")


# ============================================================
# GATE A — FROZEN SOURCE RECONSTRUCTION
# ============================================================

T = closure((p, h, s))
coord = {}

for bits in range(8):
    g = ident
    for k, gen in enumerate((p, h, s)):
        if bits & (1 << k):
            g = compose(gen, g)
    coord[g[0]] = bits

vertex = {v: k for k, v in coord.items()}

aff = set()

for a, b, c in itertools.product(range(1, 8), repeat=3):
    directions = {
        0, a, b, c, a ^ b, a ^ c, b ^ c, a ^ b ^ c
    }
    if len(directions) != 8:
        continue

    for shift in range(8):
        q = []
        for i in range(8):
            v = coord[i]
            w = shift

            if v & 1:
                w ^= a
            if v & 2:
                w ^= b
            if v & 4:
                w ^= c

            q.append(vertex[w])

        aff.add(tuple(q))

Nplus = sorted(g for g in aff if sign(g) == 1)
Nminus = sorted(g for g in aff if sign(g) == -1)
Npm = sorted(Nplus + Nminus)

chk(
    "A frozen groups and matrix identities",
    (len(T), len(aff), len(Nplus), len(Nminus), len(Npm))
    == (8, 1344, 24, 24, 48)
    and Om.T == -Om
    and Om * Om == -I
    and K * K == I
    and K * Om == -Om * K
    and len(EDGES) == 8
    and all(edge in EDGES for edge in PAIRS)
)


# ============================================================
# GATE B — ORTHOGONAL PROJECTION SECTORS
# ============================================================

P, H, Q = map(pm, (p, h, s))

B = sp.zeros(4, 8)
for i in range(4):
    B[i, 2*i] = 1
    B[i, 2*i + 1] = 1

eye8 = sp.eye(8)

Up = sp.Matrix.hstack(*[
    (eye8[:, 2*i] + eye8[:, 2*i + 1]) / sp.sqrt(2)
    for i in range(4)
])

Um = sp.Matrix.hstack(*[
    (eye8[:, 2*i] - eye8[:, 2*i + 1]) / sp.sqrt(2)
    for i in range(4)
])

Ep = (I + P) / 2
Em = (I - P) / 2

chk(
    "B orthogonal 4+4 decomposition",
    P * P == I
    and Ep * Ep == Ep
    and Em * Em == Em
    and Ep * Em == Z
    and Ep + Em == I
    and Ep == Up * Up.T
    and Em == Um * Um.T
    and Up.T * Up == sp.eye(4)
    and Um.T * Um == sp.eye(4)
    and Up.T * Um == sp.zeros(4)
    and B * Um == sp.zeros(4, 4)
    and B * Up == sp.sqrt(2) * sp.eye(4)
)

chk(
    "B exact all-t commutation coefficients",
    all(
        P*M == M*P
        for M in (Om, K, J0, J0*K)
    )
)

t = sp.symbols("t", real=True)
G = expK(t)
J = J0 * G


# ============================================================
# GATE C — RESTRICTED COMPLEX-SYMPLECTIC GEOMETRY
# ============================================================

Op = Up.T * Om * Up
Ominus = Um.T * Om * Um

Kp = Up.T * K * Up
Km = Um.T * K * Um

Gp = Up.T * G * Up
Gm = Um.T * G * Um

Jp = Up.T * J * Up
Jm = Um.T * J * Um

chk(
    "C restricted symplectic and complex structures",
    Op == Ominus
    and Kp == Km
    and Op.rank() == Ominus.rank() == 4
    and exact(Jp*Jp + sp.eye(4)) == sp.zeros(4)
    and exact(Jm*Jm + sp.eye(4)) == sp.zeros(4)
    and exact(Jp - Jm) == sp.zeros(4)
    and exact(Gp - Gm) == sp.zeros(4)
    and exact(Jp.T*Op*Jp - Op) == sp.zeros(4)
    and exact(Gp*Op*Gp - Op) == sp.zeros(4)
)

print("C | Omega_plus = Omega_minus =", rows(Op))
print("C | K_plus = K_minus =", rows(Kp))
print("C | J_plus = J_minus =", rows(Jp))
print("C | J spectrum: +i (2), -i (2)")
print("C | G spectrum: exp(t) (2), exp(-t) (2)")


# ============================================================
# GATE D — BASIN QUOTIENT
# ============================================================

Jbar = sp.simplify(B * J * Up / sp.sqrt(2))

# The canonical Euclidean section maps y -> Up*y/sqrt(2).
Ob = Op / 2

chk(
    "D quotient intertwining and induced geometry",
    exact(B*J - Jbar*B) == sp.zeros(4, 8)
    and exact(Jbar*Jbar + sp.eye(4)) == sp.zeros(4)
    and B*Em == sp.zeros(4, 8)
    and Om*Um != sp.zeros(8, 4)
    and Ob.rank() == 4
)

print("D | Direct full-form descent: OBSTRUCTED")
print("D | Split-induced quotient form: Omega_plus / 2")


# ============================================================
# GATE E — PARAMETER IDENTIFIABILITY
# ============================================================

chk(
    "E signed-t algebraic recovery",
    exact(sp.trace(Kp*Gp) - 4*sp.sinh(t)) == 0
    and exact(sp.trace(Km*Gm) - 4*sp.sinh(t)) == 0
    and exact(Jbar - Jp) == sp.zeros(4)
)

print("E | t = asinh(tr(K_sector * G_sector)/4)")
print("E | Algebraic recoverability: YES")
print("E | Operational observability: UNRESOLVED")
print("E | Physical parameter selection: UNRESOLVED")


# ============================================================
# GATE F — FINITE SYMMETRIES AND NEGATIVE CONTROLS
# ============================================================

chk(
    "F p/h/s covariance and sector invariance",
    sign(p) == 1
    and sign(h) == -1
    and sign(s) == -1
    and all(M*P == P*M for M in (P, H, Q))
    and all(M*K == K*M for M in (P, H, Q))
    and all(
        exact(M.T*J*M - sign(g)*J) == Z
        for g, M in ((p, P), (h, H), (s, Q))
    )
)

central = [
    g for g in Npm
    if pm(g)*P == P*pm(g)
]

noncentral = [
    g for g in Npm
    if pm(g)*P != P*pm(g)
]

chk(
    "F signed-group sector classification",
    len(central) == 16
    and len(noncentral) == 32
)

print("F | Npm sector-preserving:", len(central))
print("F | Npm sector-nonpreserving:", len(noncentral))


# Alternative SIM14 perfect matching M1.
# Orientation is declared; it is not substituted into frozen OPS.

ALT_PAIRS = ((0, 1), (2, 4), (3, 5), (6, 7))
Om1 = sp.zeros(n)

for a, b in ALT_PAIRS:
    Om1[a, b] = 1
    Om1[b, a] = -1

K1 = Om1.applyfunc(abs)

chk(
    "F M1 valid distinct symplectic carrier",
    Om1.T == -Om1
    and Om1*Om1 == -I
    and K1*K1 == I
    and K1*Om1 == -Om1*K1
    and K1 != K
    and all(
        tuple(sorted(pair)) in EDGES
        for pair in ALT_PAIRS
    )
)

print("F | M1 changes the form; frozen Omega unchanged")


# Alternative four-basin observation partition.

Balt = sp.zeros(4, 8)

for i, (a, b) in enumerate(
    ((0, 2), (1, 3), (4, 5), (6, 7))
):
    Balt[i, a] = 1
    Balt[i, b] = 1

Nalt = Balt.nullspace()

chk(
    "F alternate partition obstructs J0 descent",
    Balt.rank() == 4
    and any(
        Balt*J0*v != sp.zeros(4, 1)
        for v in Nalt
    )
)


# Generic symplectic conjugacy control.
# A complex structure need not descend through B.

v = Up[:, 0] + Um[:, 0]
Tcontrol = I + v*(v.T*Om)
Jgeneric = Tcontrol*J0*Tcontrol.inv()

chk(
    "F generic compatible complex structure need not descend",
    Tcontrol.T*Om*Tcontrol == Om
    and Jgeneric*Jgeneric == -I
    and P*Jgeneric != Jgeneric*P
    and any(
        B*Jgeneric*Um[:, i] != sp.zeros(4, 1)
        for i in range(4)
    )
)


# ============================================================
# GATE G — SYMPLECTIC DESCENT AND CONJUGACY
# ============================================================

S = expK(-t/2)
Sinv = expK(t/2)

chk(
    "G symplectic conjugacy descends for all t",
    exact(S.T*Om*S - Om) == Z
    and exact(S*J0*Sinv - J) == Z
    and S*Em == Em*S
    and exact(
        (Up.T*S*Up)
        * (Up.T*J0*Up)
        * (Up.T*Sinv*Up)
        - Jp
    ) == sp.zeros(4)
)


# Explicit symplectic transvection that does not descend.

v = Up[:, 0] + Um[:, 0]
Trans = I + v*(v.T*Om)

chk(
    "G explicit non-descending symplectic transvection",
    Trans.T*Om*Trans == Om
    and B*Trans*Um != sp.zeros(4, 4)
)

chk(
    "G symplectic-orthogonal sector splitting",
    Up.T*Om*Um == sp.zeros(4)
)

print("G | Descent criterion: S(ker B) = ker B")
print("G | Symplectic descent implies block-diagonal action")
print("G | Counterexample transvection vector:", rows(v.T))


# ============================================================
# GATE H — INTRINSIC VS OBSERVATIONAL EQUIVALENCE
# ============================================================

Swap = Up*Um.T + Um*Up.T

chk(
    "H intrinsic complex-symplectic sector swap",
    Swap.T*Om*Swap == Om
    and Swap*Up == Um
    and Swap*Um == Up
    and exact(Swap*J - J*Swap) == Z
    and Swap*K == K*Swap
)

chk(
    "H swap fails fixed basin observation",
    B*Swap != B
)

chk(
    "H swap exchanges p eigenspaces",
    Swap*P*Swap == -P
)

print("H | Intrinsic sector equivalence: YES")
print("H | Swap preserves frozen P: NO")
print("H | Swap preserves B: NO")


# ============================================================
# GATE H — MINIMAL C8 ADJACENCY CLOSURE PATCH
# ============================================================

# Frozen graph adjacency operator.
A = sp.zeros(n)

for i, j in EDGES:
    A[i, j] = 1
    A[j, i] = 1

# Restricted adjacency operators in frozen normalized bases.
Ap = Up.T * A * Up
Am = Um.T * A * Um

chk(
    "H frozen C8 adjacency distinguishes intrinsically equivalent sectors",
    P*A == A*P
    and Up.T*A*Um == sp.zeros(4)
    and sp.trace(Ap) == 2
    and sp.trace(Am) == -2
    and Swap.T*A*Swap != A
)

print("H | A_plus =", rows(Ap))
print("H | A_minus =", rows(Am))
print(
    "H | adjacency traces (+,-):",
    sp.trace(Ap), sp.trace(Am)
)
print(
    "H | adjacency representations not similar "
    "(unequal traces)"
)


# ============================================================
# FINAL FINITE GRAPH-SYMMETRY COUNT
# ============================================================

def preserves_graph(g):
    image_edges = {
        tuple(sorted((g[i], g[j])))
        for i, j in EDGES
    }
    return image_edges == EDGES


graph_count = sum(
    preserves_graph(g)
    for g in Npm
)

chk(
    "F signed-group graph automorphism count",
    graph_count == 4
)

print("F | Npm graph-preserving:", graph_count)


# ============================================================
# MACHINE-READABLE TRUTH PACKET
# ============================================================

packet = {
    "sim": "16.3",
    "status": "EXACT_ALGEBRAIC_AUDIT_PASSED",
    "scope": "Frozen OPS, linear projection, no ISP or physical boundary",

    # Metadata: referenced source SHAs; not runtime-fetched.
    "source_shas": {
        "SIM16.0.py":
            "e01b90b751353273241e6c516e30b16571dcbfdf",
        "SIM16.1.py":
            "76970c358d128d136782310a2b6afa41efd793f7",
        "SIM16.2.py":
            "d8acde23c8ee0c45de29aa88c4005ab4efa26e26"
    },

    "orders": {
        "T": len(T),
        "AGL": len(aff),
        "Nplus": len(Nplus),
        "Nminus": len(Nminus),
        "Npm": len(Npm)
    },

    "ranks": {
        "Vplus": 4,
        "Vminus": 4,
        "Omega_plus": Op.rank(),
        "Omega_minus": Ominus.rank(),
        "B": B.rank()
    },

    "sector_equal_in_normalized_frozen_bases": True,

    "parameter_recovery":
        "t=asinh(tr(K_sector G_sector)/4)",

    "direct_symplectic_form_descent": False,
    "split_induced_symplectic_form": True,
    "complex_quotient_descent": True,
    "symplectic_conjugacy_descent": True,

    "Npm_sector_preserving": len(central),
    "Npm_sector_nonpreserving": len(noncentral),
    "Npm_graph_preserving": graph_count,

    "explicit_nondescending_symplectic_witness":
        rows(Trans),

    "alternative_M1_valid": True,
    "alternative_basin_obstructs_J0": True,
    "generic_complex_control_obstructs_descent": True,

    "intrinsic_sector_swap_symplectic": True,
    "sector_swap_preserves_p": False,
    "sector_swap_preserves_B": False,

    # New Gate H adjacency findings.
    "adjacency_sector_traces": [
        int(sp.trace(Ap)),
        int(sp.trace(Am))
    ],
    "adjacency_restrictions_similar": False,
    "sector_swap_preserves_adjacency": False,

    "operational_observability": "UNRESOLVED",
    "parameter_selection": "UNRESOLVED",
    "new_dynamics": False,
    "new_gauge_group": False
}

print(
    "TRUTH_PACKET_JSON="
    + json.dumps(packet, sort_keys=True)
)

print("END SIM16.3")








"""

RESULTS:


SIM16.3 — FROZEN PROJECTION–COMPLEX STRUCTURE AUDIT
PASS | A frozen groups and matrix identities
PASS | B orthogonal 4+4 decomposition
PASS | B exact all-t commutation coefficients
PASS | C restricted symplectic and complex structures
C | Omega_plus = Omega_minus = [['0', '1', '0', '0'], ['-1', '0', '0', '0'], ['0', '0', '0', '1'], ['0', '0', '-1', '0']]
C | K_plus = K_minus = [['0', '1', '0', '0'], ['1', '0', '0', '0'], ['0', '0', '0', '1'], ['0', '0', '1', '0']]
C | J_plus = J_minus = [['-sinh(t)', '-cosh(t)', '0', '0'], ['cosh(t)', 'sinh(t)', '0', '0'], ['0', '0', '-sinh(t)', '-cosh(t)'], ['0', '0', 'cosh(t)', 'sinh(t)']]
C | J spectrum: +i (2), -i (2)
C | G spectrum: exp(t) (2), exp(-t) (2)
PASS | D quotient intertwining and induced geometry
D | Direct full-form descent: OBSTRUCTED
D | Split-induced quotient form: Omega_plus / 2
PASS | E signed-t algebraic recovery
E | t = asinh(tr(K_sector * G_sector)/4)
E | Algebraic recoverability: YES
E | Operational observability: UNRESOLVED
E | Physical parameter selection: UNRESOLVED
PASS | F p/h/s covariance and sector invariance
PASS | F signed-group sector classification
F | Npm sector-preserving: 16
F | Npm sector-nonpreserving: 32
PASS | F M1 valid distinct symplectic carrier
F | M1 changes the form; frozen Omega unchanged
PASS | F alternate partition obstructs J0 descent
PASS | F generic compatible complex structure need not descend
PASS | G symplectic conjugacy descends for all t
PASS | G explicit non-descending symplectic transvection
PASS | G symplectic-orthogonal sector splitting
G | Descent criterion: S(ker B) = ker B
G | Symplectic descent implies block-diagonal action
G | Counterexample transvection vector: [['sqrt(2)', '0', '0', '0', '0', '0', '0', '0']]
PASS | H intrinsic complex-symplectic sector swap
PASS | H swap fails fixed basin observation
PASS | H swap exchanges p eigenspaces
H | Intrinsic sector equivalence: YES
H | Swap preserves frozen P: NO
H | Swap preserves B: NO
PASS | H frozen C8 adjacency distinguishes intrinsically equivalent sectors
H | A_plus = [['1', '1', '0', '0'], ['1', '0', '1', '0'], ['0', '1', '0', '1'], ['0', '0', '1', '1']]
H | A_minus = [['-1', '1', '0', '0'], ['1', '0', '1', '0'], ['0', '1', '0', '1'], ['0', '0', '1', '-1']]
H | adjacency traces (+,-): 2 -2
H | adjacency representations not similar (unequal traces)
PASS | F signed-group graph automorphism count
F | Npm graph-preserving: 4
TRUTH_PACKET_JSON={"Npm_graph_preserving": 4, "Npm_sector_nonpreserving": 32, "Npm_sector_preserving": 16, "adjacency_restrictions_similar": false, "adjacency_sector_traces": [2, -2], "alternative_M1_valid": true, "alternative_basin_obstructs_J0": true, "complex_quotient_descent": true, "direct_symplectic_form_descent": false, "explicit_nondescending_symplectic_witness": [["1", "0", "2", "0", "0", "0", "0", "0"], ["0", "1", "0", "0", "0", "0", "0", "0"], ["0", "0", "1", "0", "0", "0", "0", "0"], ["0", "0", "0", "1", "0", "0", "0", "0"], ["0", "0", "0", "0", "1", "0", "0", "0"], ["0", "0", "0", "0", "0", "1", "0", "0"], ["0", "0", "0", "0", "0", "0", "1", "0"], ["0", "0", "0", "0", "0", "0", "0", "1"]], "generic_complex_control_obstructs_descent": true, "intrinsic_sector_swap_symplectic": true, "new_dynamics": false, "new_gauge_group": false, "operational_observability": "UNRESOLVED", "orders": {"AGL": 1344, "Nminus": 24, "Nplus": 24, "Npm": 48, "T": 8}, "parameter_recovery": "t=asinh(tr(K_sector G_sector)/4)", "parameter_selection": "UNRESOLVED", "ranks": {"B": 4, "Omega_minus": 4, "Omega_plus": 4, "Vminus": 4, "Vplus": 4}, "scope": "Frozen OPS, linear projection, no ISP or physical boundary", "sector_equal_in_normalized_frozen_bases": true, "sector_swap_preserves_B": false, "sector_swap_preserves_adjacency": false, "sector_swap_preserves_p": false, "sim": "16.3", "source_shas": {"SIM16.0.py": "e01b90b751353273241e6c516e30b16571dcbfdf", "SIM16.1.py": "76970c358d128d136782310a2b6afa41efd793f7", "SIM16.2.py": "d8acde23c8ee0c45de29aa88c4005ab4efa26e26"}, "split_induced_symplectic_form": true, "status": "EXACT_ALGEBRAIC_AUDIT_PASSED", "symplectic_conjugacy_descent": true}
END SIM16.3
"""
