
#!/usr/bin/env python3
"""SIM16.1 — EXACT C8 METRIC LOCALITY / J_COMPLEX SURVIVABILITY 
Frozen SIM16.0 OPS. Exact global classification. No fitting.
Requires numpy and sympy.
"""
import itertools
import json
import numpy as np
import sympy as sp

N = 8
I = sp.eye(N)
PAIRS = ((0, 2), (1, 3), (4, 6), (5, 7))
C8_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (3, 5), (4, 6), (5, 7), (6, 7)
)
edges = {tuple(sorted(e)) for e in C8_EDGES}
forbidden = [
    (i, j) for i in range(N) for j in range(i + 1, N)
    if (i, j) not in edges
]

Omega = sp.zeros(N)
for a, b in PAIRS:
    Omega[a, b], Omega[b, a] = 1, -1
K = Omega.applyfunc(abs)

p = (1, 0, 3, 2, 5, 4, 7, 6)
h = (7, 6, 5, 4, 3, 2, 1, 0)
s = (2, 3, 0, 1, 6, 7, 4, 5)
identity = tuple(range(N))

def compose(a, b):
    return tuple(a[b[i]] for i in range(N))

def closure(gens):
    group, frontier = {identity}, [identity]
    while frontier:
        a = frontier.pop()
        for b in gens:
            c = compose(a, b)
            if c not in group:
                group.add(c)
                frontier.append(c)
    return group

T = closure((p, h, s))
assert len(T) == 8
assert len({g[0] for g in T}) == 8

# Reconstruct frozen F2^3 affine coordinates.
coord = {}
for bits in range(8):
    g = identity
    for k, gen in enumerate((p, h, s)):
        if (bits >> k) & 1:
            g = compose(gen, g)
    coord[g[0]] = bits
assert len(coord) == 8
vertex = {v: k for k, v in coord.items()}

def independent(a, b, c):
    return len({
        0, a, b, c, a ^ b, a ^ c,
        b ^ c, a ^ b ^ c
    }) == 8

affine = set()
for a, b, c in itertools.product(range(1, 8), repeat=3):
    if not independent(a, b, c):
        continue
    for shift in range(8):
        perm = []
        for i in range(N):
            v, w = coord[i], shift
            if v & 1: w ^= a
            if v & 2: w ^= b
            if v & 4: w ^= c
            perm.append(vertex[w])
        affine.add(tuple(perm))
assert len(affine) == 1344

def Pmat(g):
    P = sp.zeros(N)
    for i, j in enumerate(g):
        P[j, i] = 1
    return P

def omega_sign(g):
    A = Pmat(g).T * Omega * Pmat(g)
    return 1 if A == Omega else (-1 if A == -Omega else 0)

Nplus = sorted(g for g in affine if omega_sign(g) == 1)
Nminus = sorted(g for g in affine if omega_sign(g) == -1)
Npm = sorted(Nplus + Nminus)
assert (len(Nplus), len(Nminus), len(Npm)) == (24, 24, 48)

def orbit_matrices(group):
    remaining = {(i, j) for i in range(N) for j in range(i, N)}
    matrices = []
    while remaining:
        i, j = min(remaining)
        orbit = {
            tuple(sorted((g[i], g[j]))) for g in group
        }
        assert orbit <= remaining
        remaining -= orbit
        M = sp.zeros(N)
        for a, b in orbit:
            M[a, b] = M[b, a] = 1
        matrices.append(M)
    return matrices

def kernel(basis, equations):
    columns = []
    for B in basis:
        parts = []
        for f in equations:
            value = f(B)
            if isinstance(value, sp.MatrixBase):
                parts.extend(list(value))
            elif isinstance(value, (list, tuple)):
                parts.extend(value)
            else:
                parts.append(value)
        columns.append(sp.Matrix(parts))
    A = sp.Matrix.hstack(*columns)
    vectors = A.nullspace()
    assert all(A * v == sp.zeros(A.rows, 1) for v in vectors)
    return [
        sum((B * v[i] for i, B in enumerate(basis)), sp.zeros(N))
        for v in vectors
    ]

def local_equations(B):
    return [B[i, j] for i, j in forbidden]

def tangent_equations(B):
    return B * Omega + Omega * B

def in_span(M, basis):
    A = sp.Matrix.hstack(*[sp.Matrix(list(B)) for B in basis])
    return A.rank() == A.row_join(sp.Matrix(list(M))).rank()

def matlist(M):
    return [[int(v) if v.q == 1 else str(v) for v in M.row(i)]
            for i in range(N)]

def witness(X, t, group):
    lam, Q = np.linalg.eigh(np.array(X, dtype=float))
    G = (Q * np.exp(t * lam)) @ Q.T
    Om = np.array(Omega, dtype=float)
    J = -Om @ G
    residuals = {
        "symplectic": float(np.linalg.norm(G @ Om @ G - Om, ord=np.inf)),
        "complex_square": float(np.linalg.norm(J @ J + np.eye(N), ord=np.inf)),
        "metric_symmetry": float(max(
            np.linalg.norm(
                np.array(Pmat(g), dtype=float).T @ G
                @ np.array(Pmat(g), dtype=float) - G,
                ord=np.inf
            ) for g in group
        )),
        "signed_J_covariance": float(max(
            np.linalg.norm(
                np.array(Pmat(g), dtype=float).T @ J
                @ np.array(Pmat(g), dtype=float)
                - omega_sign(g) * J,
                ord=np.inf
            ) for g in group
        )),
        "forbidden_max": float(max(abs(G[i, j]) for i, j in forbidden)),
        "identity_distance": float(np.linalg.norm(G - np.eye(N))),
        "minimum_eigenvalue": float(np.min(np.linalg.eigvalsh(G)))
    }
    for key in ("symplectic", "complex_square",
                "metric_symmetry", "signed_J_covariance"):
        assert residuals[key] < 1e-9
    assert residuals["minimum_eigenvalue"] > 0
    return residuals

print("SIM16.1 — EXACT C8 METRIC LOCALITY")

# A — Frozen input regression
assert Omega.T == -Omega
assert Omega * Omega == -I
assert K * K == I
assert K * Omega + Omega * K == sp.zeros(N)
assert len(edges) == 8 and len(forbidden) == 20
assert all(tuple(sorted(e)) in edges for e in PAIRS)
print("A | PASS frozen orders:", len(T), len(affine),
      len(Nplus), len(Nminus), len(Npm))
print("A | C8 edges / forbidden pairs:", len(edges), len(forbidden))

# B — Exact symmetry and compatible tangent spaces
sym_plus = orbit_matrices(Nplus)
sym_pm = orbit_matrices(Npm)
tan_plus = kernel(sym_plus, [tangent_equations])
tan_pm = kernel(sym_pm, [tangent_equations])
assert tuple(map(len, (sym_plus, sym_pm, tan_plus, tan_pm))) == (6, 4, 4, 2)
assert in_span(K, tan_pm)
for group, tangent in ((Nplus, tan_plus), (Npm, tan_pm)):
    for X in tangent:
        assert X == X.T
        assert tangent_equations(X) == sp.zeros(N)
        assert all(Pmat(g).T * X * Pmat(g) == X for g in group)
print("B | PASS symmetric dims:", len(sym_plus), len(sym_pm))
print("B | PASS tangent dims:", len(tan_plus), len(tan_pm))
print("B | PASS |Omega| belongs to signed tangent space")

# C — Exact linear locality intersections
local_sym_plus = kernel(sym_plus, [local_equations])
local_sym_pm = kernel(sym_pm, [local_equations])
local_tan_plus = kernel(tan_plus, [local_equations])
local_tan_pm = kernel(tan_pm, [local_equations])

assert in_span(I, local_sym_plus) and in_span(K, local_sym_plus)
assert in_span(I, local_sym_pm) and in_span(K, local_sym_pm)
assert in_span(K, local_tan_pm)
assert (len(local_sym_plus), len(local_sym_pm)) == (3, 2)
assert (len(local_tan_plus), len(local_tan_pm)) == (2, 1)
print("C | PASS local symmetric dims:", len(local_sym_plus),
      len(local_sym_pm))
print("C | PASS local tangent dims:", len(local_tan_plus),
      len(local_tan_pm))

# D — Does each group preserve the full C8 graph?
def preserves_graph(g):
    return all(
        tuple(sorted((g[i], g[j]))) in edges for i, j in edges
    )
graph_plus = sum(preserves_graph(g) for g in Nplus)
graph_pm = sum(preserves_graph(g) for g in Npm)
print("D | graph-preserving elements N+/Npm:",
      graph_plus, graph_pm)

# E — Exact partner-family identity
t = sp.symbols("t", real=True)
Gt = sp.cosh(t) * I + sp.sinh(t) * K
assert sp.simplify(Gt * Omega * Gt - Omega) == sp.zeros(N)
assert all(Gt[i, j] == 0 for i, j in forbidden)
assert all(Pmat(g).T * Gt * Pmat(g) == Gt for g in Npm)
assert sp.simplify(
    Gt - sp.exp(t) * (I + K) / 2
       - sp.exp(-t) * (I - K) / 2
) == sp.zeros(N)
print("E | PASS exact partner family for every real t")

# F — Global classification directly in sparse metric space
for label, basis in (("Nplus", local_sym_plus),
                     ("Npm", local_sym_pm)):
    z = sp.symbols("z0:" + str(len(basis)), real=True)
    G = sum((z[i] * B for i, B in enumerate(basis)), sp.zeros(N))
    equations = sorted(
        {sp.expand(v) for v in G * Omega * G - Omega if v != 0},
        key=str
    )
    print(f"F | {label}: {len(basis)} metric coordinates")
    for eq in equations:
        print("F | exact compatibility equation:", sp.factor(eq))

    if label == "Nplus":
        assert len(basis) == 3
        assert basis[0] + basis[2] == I
        assert basis[1] == K
        assert basis[0] * basis[2] == sp.zeros(N)
        assert len(equations) == 2
        assert sp.expand(equations[0] + equations[1]) == 0
        assert sp.expand(
            equations[0] + z[0] * z[2] - z[1]**2 - 1
        ) == 0
        print("F | Nplus: G=u D0+v K+w D1")
        print("F | u*w-v^2=1, u>0, w>0")
        print("F | GLOBAL VERDICT: 2D positive family")

    if label == "Npm":
        assert len(basis) == 2
        assert in_span(I, basis) and in_span(K, basis)
        alpha, beta = sp.symbols("alpha beta", real=True)
        G2 = alpha * I + beta * K
        assert sp.expand(
            G2 * Omega * G2 - Omega
        ) == (alpha**2 - beta**2 - 1) * Omega
        print("F | Npm: G=alpha I+beta K")
        print("F | alpha^2-beta^2=1, alpha>|beta|")
        print("F | GLOBAL THEOREM: G(t)=cosh(t)I+sinh(t)K")
        print("F | GLOBAL VERDICT: PARTNER FAMILY ONLY")

# G — Numerical regression, including signed J covariance
for label, group, tangent in (
    ("Nplus", Nplus, tan_plus),
    ("Npm", Npm, tan_pm)
):
    for i, X in enumerate(tangent):
        m = witness(X, 0.23, group)
        print(f"G | {label} witness {i}: "
              f"forbidden={m['forbidden_max']:.3e}, "
              f"signed_J={m['signed_J_covariance']:.3e}")

for value in (0.25, -0.6, 1.0):
    m = witness(K, value, Npm)
    assert m["forbidden_max"] < 1e-9
    assert m["identity_distance"] > 1e-3
    print(f"G | partner t={value:+.2f}: "
          f"forbidden={m['forbidden_max']:.3e}, "
          f"signed_J={m['signed_J_covariance']:.3e}")

# H — Machine-readable truth packet
packet = {
    "sim": "16.1",
    "status": "EXACT_PROOF",
    "scope": "Frozen OPS, strong C8 metric sparsity",
    "group_orders": {
        "Nplus": len(Nplus),
        "Nminus": len(Nminus),
        "Npm": len(Npm)
    },
    "symmetric_dimensions": {
        "Nplus": len(sym_plus), "Npm": len(sym_pm)
    },
    "tangent_dimensions": {
        "Nplus": len(tan_plus), "Npm": len(tan_pm)
    },
    "local_symmetric_dimensions": {
        "Nplus": len(local_sym_plus),
        "Npm": len(local_sym_pm)
    },
    "local_tangent_dimensions": {
        "Nplus": len(local_tan_plus),
        "Npm": len(local_tan_pm)
    },
    "graph_preserving_elements": {
        "Nplus": graph_plus, "Npm": graph_pm
    },
    "Nplus_global_local_family_dimension": 2,
    "signed_global_verdict": "PARTNER_FAMILY_ONLY",
    "signed_local_family":
        "G(t)=cosh(t)*I+sinh(t)*abs(Omega), t in R",
    "partner_matrix": matlist(K),
    "proof_method":
        "Exact invariant sparse metric space, "
        "G*Omega*G=Omega, and SPD eigenvalue criterion"
}
print("H | EXACT GLOBAL CLASSIFICATION PASSED")
print("TRUTH_PACKET_JSON=" + json.dumps(packet, sort_keys=True))
print("END SIM16.1")






~~~~~~~~~~~~~~~~~~~~~~





RESULTS:




SIM16.1 — EXACT C8 METRIC LOCALITY
A | PASS frozen orders: 8 1344 24 24 48
A | C8 edges / forbidden pairs: 8 20
B | PASS symmetric dims: 6 4
B | PASS tangent dims: 4 2
B | PASS |Omega| belongs to signed tangent space
C | PASS local symmetric dims: 3 2
C | PASS local tangent dims: 2 1
D | graph-preserving elements N+/Npm: 2 4
E | PASS exact partner family for every real t
F | Nplus: 3 metric coordinates
F | exact compatibility equation: -z0*z2 + z1**2 + 1
F | exact compatibility equation: z0*z2 - z1**2 - 1
F | Nplus: G=u D0+v K+w D1
F | u*w-v^2=1, u>0, w>0
F | GLOBAL VERDICT: 2D positive family
F | Npm: 2 metric coordinates
F | exact compatibility equation: -z0**2 + z1**2 + 1
F | exact compatibility equation: z0**2 - z1**2 - 1
F | Npm: G=alpha I+beta K
F | alpha^2-beta^2=1, alpha>|beta|
F | GLOBAL THEOREM: G(t)=cosh(t)I+sinh(t)K
F | GLOBAL VERDICT: PARTNER FAMILY ONLY
G | Nplus witness 0: forbidden=0.000e+00, signed_J=0.000e+00
G | Nplus witness 1: forbidden=2.445e-01, signed_J=2.276e-15
G | Nplus witness 2: forbidden=0.000e+00, signed_J=0.000e+00
G | Nplus witness 3: forbidden=2.998e-01, signed_J=4.163e-16
G | Npm witness 0: forbidden=0.000e+00, signed_J=0.000e+00
G | Npm witness 1: forbidden=2.445e-01, signed_J=2.359e-15
G | partner t=+0.25: forbidden=0.000e+00, signed_J=0.000e+00
G | partner t=-0.60: forbidden=0.000e+00, signed_J=0.000e+00
G | partner t=+1.00: forbidden=0.000e+00, signed_J=0.000e+00
H | EXACT GLOBAL CLASSIFICATION PASSED
TRUTH_PACKET_JSON={"Nplus_global_local_family_dimension": 2, "graph_preserving_elements": {"Nplus": 2, "Npm": 4}, "group_orders": {"Nminus": 24, "Nplus": 24, "Npm": 48}, "local_symmetric_dimensions": {"Nplus": 3, "Npm": 2}, "local_tangent_dimensions": {"Nplus": 2, "Npm": 1}, "partner_matrix": [[0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0]], "proof_method": "Exact invariant sparse metric space, G*Omega*G=Omega, and SPD eigenvalue criterion", "scope": "Frozen OPS, strong C8 metric sparsity", "signed_global_verdict": "PARTNER_FAMILY_ONLY", "signed_local_family": "G(t)=cosh(t)*I+sinh(t)*abs(Omega), t in R", "sim": "16.1", "status": "EXACT_PROOF", "symmetric_dimensions": {"Nplus": 6, "Npm": 4}, "tangent_dimensions": {"Nplus": 4, "Npm": 2}}
END SIM16.1
