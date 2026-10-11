
#!/usr/bin/env python3
"""SIM16.0 — J_COMPLEX EXACT RIGIDITY
Frozen OPS inputs. Exact linear algebra. No dynamics or fitting.
Requires: numpy, sympy
"""
import itertools
import json
import numpy as np
import sympy as sp

N = 8
I = sp.eye(N)
PAIRS = [(0, 2), (1, 3), (4, 6), (5, 7)]
Omega = sp.zeros(N)
for a, b in PAIRS:
    Omega[a, b], Omega[b, a] = 1, -1

# Frozen SIM14 operational translations
p = (1, 0, 3, 2, 5, 4, 7, 6)
h = (7, 6, 5, 4, 3, 2, 1, 0)
s = (2, 3, 0, 1, 6, 7, 4, 5)
identity = tuple(range(N))

def compose(a, b):
    return tuple(a[b[i]] for i in range(N))

def closure(gens):
    group = {identity}
    frontier = [identity]
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

# Recover F2^3 coordinates from frozen translations.
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
        0, a, b, c, a ^ b, a ^ c, b ^ c, a ^ b ^ c
    }) == 8

# Exhaustively reconstruct AGL(3,2).
affine = set()
for a, b, c in itertools.product(range(1, 8), repeat=3):
    if not independent(a, b, c):
        continue
    for shift in range(8):
        perm = []
        for i in range(N):
            v = coord[i]
            w = shift
            if v & 1:
                w ^= a
            if v & 2:
                w ^= b
            if v & 4:
                w ^= c
            perm.append(vertex[w])
        affine.add(tuple(perm))
assert len(affine) == 1344

def pmat(g):
    P = sp.zeros(N)
    for i, j in enumerate(g):
        P[j, i] = 1
    return P

def omega_sign(g):
    A = pmat(g).T * Omega * pmat(g)
    if A == Omega:
        return 1
    if A == -Omega:
        return -1
    return 0

Nplus = sorted(g for g in affine if omega_sign(g) == 1)
Nminus = sorted(g for g in affine if omega_sign(g) == -1)
Npm = sorted(Nplus + Nminus)
assert (len(Nplus), len(Nminus), len(Npm)) == (24, 24, 48)

def generators(group):
    found = {identity}
    gens = []
    for g in group:
        if g not in found:
            gens.append(g)
            found = closure(gens)
            if len(found) == len(group):
                break
    assert found == set(group)
    return gens

gens_plus = generators(Nplus)
gens_pm = generators(Npm)

def exact_nullspace(basis, gens, anti=False):
    matrices = [pmat(g) for g in gens]
    images = []
    for B in basis:
        blocks = [B * P - P * B for P in matrices]
        if anti:
            blocks.append(B * Omega + Omega * B)
        images.append(
            sp.Matrix.vstack(*blocks).reshape(
                len(blocks) * N * N, 1
            )
        )
    A = sp.Matrix.hstack(*images)
    null = A.nullspace()
    witnesses = [
        sum(
            (basis[i] * v[i] for i in range(len(basis))),
            sp.zeros(N)
        )
        for v in null
    ]
    assert all(A * v == sp.zeros(A.rows, 1) for v in null)
    return witnesses

symmetric_basis = []
for i in range(N):
    for j in range(i, N):
        B = sp.zeros(N)
        B[i, j] = B[j, i] = 1
        symmetric_basis.append(B)

full_basis = []
for i in range(N):
    for j in range(N):
        B = sp.zeros(N)
        B[i, j] = 1
        full_basis.append(B)

print("SIM16.0 — J_COMPLEX EXACT RIGIDITY")
print("A | frozen group orders:",
      len(T), len(affine), len(Nplus), len(Nminus), len(Npm))

# GATE A: Baseline identities and covariance.
assert Omega.T == -Omega
assert Omega * Omega == -I
J0 = -Omega
assert J0 * J0 == -I
assert J0.T * Omega * J0 == Omega
assert Omega * J0 == I
for g in Npm:
    P = pmat(g)
    assert P.T * I * P == I
    assert P.T * J0 * P == omega_sign(g) * J0
print("A | PASS exact baseline and full signed covariance")

# GATE B: Reproduce the full commutant.
commutant = exact_nullspace(full_basis, gens_plus)
assert len(commutant) == 8
print("B | PASS commutant dimension:", len(commutant))

# GATE C: Symmetric invariant matrix spaces.
sym_plus = exact_nullspace(symmetric_basis, gens_plus)
sym_pm = exact_nullspace(symmetric_basis, gens_pm)
print("C | symmetric invariant dimensions:",
      len(sym_plus), len(sym_pm))

# GATE D: Exact compatible-metric tangent spaces.
tangent_plus = exact_nullspace(sym_plus, [], anti=True)
tangent_pm = exact_nullspace(sym_pm, [], anti=True)
dplus, dpm = len(tangent_plus), len(tangent_pm)
assert dpm <= dplus
print("D | EXACT tangent dimensions:", dplus, dpm)

for label, group, tangent in [
    ("Nplus", Nplus, tangent_plus),
    ("Npm", Npm, tangent_pm)
]:
    for X in tangent:
        assert X.T == X
        assert X * Omega + Omega * X == sp.zeros(N)
        assert all(
            X * pmat(g) == pmat(g) * X for g in group
        )
print("D | PASS exact verification against all group elements")

# GATE E: Unconstrained negative control.
Xcontrol = sp.zeros(N)
Xcontrol[0, 0], Xcontrol[2, 2] = 1, -1
assert Xcontrol * Omega + Omega * Xcontrol == sp.zeros(N)
assert any(
    Xcontrol * pmat(g) != pmat(g) * Xcontrol
    for g in Nplus
)
print("E | PASS nontrivial squeeze rejected by Nplus")

# GATE F: Integrate directions and check nonlinear equations.
for label, group, tangent in [
    ("Nplus", Nplus, tangent_plus),
    ("Npm", Npm, tangent_pm)
]:
    for k, X in enumerate(tangent):
        lam, Q = np.linalg.eigh(np.array(X, dtype=float))
        G = (Q * np.exp(0.2 * lam)) @ Q.T
        Om = np.array(Omega, dtype=float)
        J = -Om @ G

        residuals = [
            np.linalg.norm(
                G @ Om @ G - Om, ord=np.inf
            ),
            np.linalg.norm(
                J @ J + np.eye(N), ord=np.inf
            ),
            max(
                np.linalg.norm(
                    np.array(pmat(g), dtype=float).T
                    @ G @ np.array(pmat(g), dtype=float) - G,
                    ord=np.inf
                )
                for g in group
            )
        ]
        assert min(np.linalg.eigvalsh(G)) > 0
        assert max(residuals) < 1e-9
        assert np.linalg.norm(G - np.eye(N)) > 1e-6
        print(
            f"F | {label} witness {k}: "
            f"max_residual={max(residuals):.3g}, "
            f"||G-I||={np.linalg.norm(G-np.eye(N)):.6g}"
        )

# GATE G: Global classification by principal logarithm.
print("G | GLOBAL CLASSIFICATION")
print("G | Nplus:",
      "RIGID" if dplus == 0
      else f"FAMILY SURVIVES (dimension {dplus})")
print("G | Npm:",
      "RIGID" if dpm == 0
      else f"FAMILY SURVIVES (dimension {dpm})")

packet = {
    "sim": "16.0",
    "omega_pairs": PAIRS,
    "group_orders": {
        "Nplus": len(Nplus),
        "Nminus": len(Nminus),
        "Npm": len(Npm)
    },
    "commutant_dimension": len(commutant),
    "symmetric_dimensions": {
        "Nplus": len(sym_plus),
        "Npm": len(sym_pm)
    },
    "tangent_dimensions": {
        "Nplus": dplus,
        "Npm": dpm
    },
    "verdict": {
        "Nplus": "RIGID" if dplus == 0 else "FAMILY SURVIVES",
        "Npm": "RIGID" if dpm == 0 else "FAMILY SURVIVES"
    },
    "scope": "Frozen orthogonal permutation representation"
}
print("TRUTH_PACKET_JSON=" + json.dumps(packet, sort_keys=True))
print("END SIM16.0")







~~~~~~~~~~~~~~~~~






RESULTS:


       



SIM16.0 — J_COMPLEX EXACT RIGIDITY
A | frozen group orders: 8 1344 24 24 48
A | PASS exact baseline and full signed covariance
B | PASS commutant dimension: 8
C | symmetric invariant dimensions: 6 4
D | EXACT tangent dimensions: 4 2
D | PASS exact verification against all group elements
E | PASS nontrivial squeeze rejected by Nplus
F | Nplus witness 0: max_residual=2.33e-15, ||G-I||=1.0607
F | Nplus witness 1: max_residual=3.33e-16, ||G-I||=0.572286
F | Nplus witness 2: max_residual=7.63e-16, ||G-I||=1.0607
F | Nplus witness 3: max_residual=0, ||G-I||=0.572286
F | Npm witness 0: max_residual=2.41e-15, ||G-I||=1.0607
F | Npm witness 1: max_residual=3.33e-16, ||G-I||=0.572286
G | GLOBAL CLASSIFICATION
G | Nplus: FAMILY SURVIVES (dimension 4)
G | Npm: FAMILY SURVIVES (dimension 2)
TRUTH_PACKET_JSON={"commutant_dimension": 8, "group_orders": {"Nminus": 24, "Nplus": 24, "Npm": 48}, "omega_pairs": [[0, 2], [1, 3], [4, 6], [5, 7]], "scope": "Frozen orthogonal permutation representation", "sim": "16.0", "symmetric_dimensions": {"Nplus": 6, "Npm": 4}, "tangent_dimensions": {"Nplus": 4, "Npm": 2}, "verdict": {"Nplus": "FAMILY SURVIVES", "Npm": "FAMILY SURVIVES"}}
END SIM16.0
