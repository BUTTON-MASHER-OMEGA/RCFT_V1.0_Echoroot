
#!/usr/bin/env python3
"""SIM16.2 — OPERATIONAL METRIC IDENTIFIABILITY AND EQUIVALENCE CLOSURE
Exact frozen OPS audit. No dynamics, fitting, or new gauge assumptions.
Requires: sympy. Gate H is explicitly source-limited provenance metadata.
"""
import itertools
import json
import sympy as sp

N = 8
I = sp.eye(N)
Z = sp.zeros(N)
PAIRS = ((0, 2), (1, 3), (4, 6), (5, 7))
C8_EDGES = ((0, 1), (0, 2), (1, 3), (2, 4),
            (3, 5), (4, 6), (5, 7), (6, 7))
EDGES = {tuple(sorted(e)) for e in C8_EDGES}
FORBIDDEN = [(i, j) for i in range(N) for j in range(i + 1, N)
             if (i, j) not in EDGES]
Omega = sp.zeros(N)
for a, b in PAIRS:
    Omega[a, b], Omega[b, a] = 1, -1
K = Omega.applyfunc(abs)
J0 = -Omega
p = (1, 0, 3, 2, 5, 4, 7, 6)
h = (7, 6, 5, 4, 3, 2, 1, 0)
s = (2, 3, 0, 1, 6, 7, 4, 5)
IDENTITY = tuple(range(N))

def check(label, condition):
    if not condition:
        raise AssertionError(label)
    print('PASS | ' + label)

def compose(a, b):
    return tuple(a[b[i]] for i in range(N))

def closure(gens):
    group, frontier = {IDENTITY}, [IDENTITY]
    while frontier:
        a = frontier.pop()
        for b in gens:
            c = compose(a, b)
            if c not in group:
                group.add(c)
                frontier.append(c)
    return group

def pmat(g):
    M = sp.zeros(N)
    for i, j in enumerate(g):
        M[j, i] = 1
    return M

def independent(a, b, c):
    return len({0, a, b, c, a ^ b, a ^ c, b ^ c, a ^ b ^ c}) == 8

def omega_sign(g):
    x = pmat(g).T * Omega * pmat(g)
    return 1 if x == Omega else (-1 if x == -Omega else 0)

def invariant_pair_orbits(group):
    remaining = {(i, j) for i in range(N) for j in range(i, N)}
    orbits = []
    while remaining:
        i, j = min(remaining)
        orbit = {tuple(sorted((g[i], g[j]))) for g in group}
        assert orbit <= remaining
        remaining -= orbit
        orbits.append(sorted(orbit))
    return orbits

def orbit_matrix(orbit):
    M = sp.zeros(N)
    for i, j in orbit:
        M[i, j] = M[j, i] = 1
    return M

def nullspace_intersection(basis, equations):
    if not basis:
        return []
    cols = []
    for B in basis:
        col = []
        for equation in equations:
            col.extend(list(equation(B)))
        cols.append(sp.Matrix(col))
    M = sp.Matrix.hstack(*cols)
    return [sum((v[i] * basis[i] for i in range(len(basis))), Z)
            for v in M.nullspace()]

def local_eq(M):
    return sp.Matrix([M[i, j] for i, j in FORBIDDEN])

def tangent_eq(M):
    return M * Omega + Omega * M

def in_span(M, basis):
    if not basis:
        return M == Z
    B = sp.Matrix.hstack(*(sp.Matrix(b).reshape(N*N, 1) for b in basis))
    v = sp.Matrix(M).reshape(N*N, 1)
    return B.rank() == B.row_join(v).rank()

def compact_matrix(M):
    return [[int(M[i, j]) for j in range(N)] for i in range(N)]

print('SIM16.2 — OPERATIONAL METRIC IDENTIFIABILITY AND EQUIVALENCE CLOSURE')
print('CLASS | EXACT_ALGEBRAIC_AUDIT; NO ISP DYNAMICS')

# A — frozen regression, independent affine and invariant-space reconstruction
T = closure((p, h, s))
coord = {}
for bits in range(8):
    g = IDENTITY
    for k, gen in enumerate((p, h, s)):
        if bits & (1 << k):
            g = compose(gen, g)
    coord[g[0]] = bits
vertex = {v: k for k, v in coord.items()}
check('A1 translation torsor', len(T) == 8 and len(coord) == 8
      and len(vertex) == 8 and len({g[0] for g in T}) == 8)
affine = set()
for a, b, c in itertools.product(range(1, 8), repeat=3):
    if not independent(a, b, c):
        continue
    for shift in range(8):
        g = []
        for i in range(N):
            v, w = coord[i], shift
            if v & 1: w ^= a
            if v & 2: w ^= b
            if v & 4: w ^= c
            g.append(vertex[w])
        affine.add(tuple(g))
Nplus = sorted(g for g in affine if omega_sign(g) == 1)
Nminus = sorted(g for g in affine if omega_sign(g) == -1)
Npm = sorted(Nplus + Nminus)
check('A2 affine and signed group orders',
      (len(affine), len(Nplus), len(Nminus), len(Npm)) == (1344, 24, 24, 48))
check('A3 frozen matrix identities', Omega.T == -Omega and Omega*Omega == -I
      and K.T == K and K*K == I and K*Omega + Omega*K == Z
      and len(EDGES) == 8 and len(FORBIDDEN) == 20
      and all(tuple(sorted(e)) in EDGES for e in PAIRS))
orbits_plus = invariant_pair_orbits(Nplus)
orbits_pm = invariant_pair_orbits(Npm)
sym_plus = [orbit_matrix(o) for o in orbits_plus]
sym_pm = [orbit_matrix(o) for o in orbits_pm]
tan_plus = nullspace_intersection(sym_plus, [tangent_eq])
tan_pm = nullspace_intersection(sym_pm, [tangent_eq])
loc_sym_plus = nullspace_intersection(sym_plus, [local_eq])
loc_sym_pm = nullspace_intersection(sym_pm, [local_eq])
loc_tan_plus = nullspace_intersection(tan_plus, [local_eq])
loc_tan_pm = nullspace_intersection(tan_pm, [local_eq])
check('A4 SIM16.1 exact dimension regression',
      tuple(map(len, (sym_plus, sym_pm, tan_plus, tan_pm,
                      loc_sym_plus, loc_sym_pm, loc_tan_plus, loc_tan_pm)))
      == (6, 4, 4, 2, 3, 2, 2, 1))
check('A5 signed local symmetric span',
      in_span(I, loc_sym_pm) and in_span(K, loc_sym_pm)
      and len(loc_sym_pm) == 2 and in_span(K, loc_tan_pm))

t, u = sp.symbols('t u', real=True)
Gt = sp.cosh(t)*I + sp.sinh(t)*K
Jt = -Omega*Gt
Gu = sp.cosh(u)*I + sp.sinh(u)*K
check('A6 exact compatibility and locality',
      sp.simplify(Gt*Omega*Gt-Omega) == Z
      and sp.simplify(Jt*Jt+I) == Z
      and all(Gt[i,j] == 0 for i,j in FORBIDDEN))

# B — injectivity via strictly monotone signed pairing trace
check('B signed trace recovers t',
      sp.simplify(sp.trace(K*Gt)-8*sp.sinh(t)) == 0
      and sp.simplify(sp.trace(K*Gu)-8*sp.sinh(u)) == 0
      and Omega.inv() == -Omega)
print('PROOF | B: sinh is strictly increasing on R; J=-Omega G is injective in G.')

# C — full signed permutation action
check('C all 48 metrics fixed and J signed-covariant', all(
    pmat(g).T*Gt*pmat(g) == Gt and
    sp.simplify(pmat(g).T*Jt*pmat(g) - omega_sign(g)*Jt) == Z
    for g in Npm))

# D — exact invariants, spectral projector certificate
Eplus, Eminus = (I+K)/2, (I-K)/2
check('D projector and spectral decomposition',
      Eplus*Eplus == Eplus and Eminus*Eminus == Eminus
      and Eplus*Eminus == Z and Eplus+Eminus == I
      and (sp.trace(Eplus), sp.trace(Eminus)) == (4,4)
      and sp.simplify(Gt - sp.exp(t)*Eplus-sp.exp(-t)*Eminus) == Z
      and sp.simplify(sp.trace(Gt)-8*sp.cosh(t)) == 0
      and sp.simplify(Gt.det()-1) == 0)
print('PROOF | D: spectrum {exp(t):4, exp(-t):4}; ordinary trace gives |t|.')

# E — exact symplectic conjugacy, with wrong-orientation control
# Exponentials evaluated algebraically using K^2=I.
def expK(x):
    return sp.cosh(x)*I + sp.sinh(x)*K

S = expK(-t/2)
Sinv = expK(t/2)
check('E1 S is symplectic', sp.simplify(S.T*Omega*S-Omega) == Z)
check('E2 correct conjugation S J0 S^-1 = Jt',
      sp.simplify(S*J0*Sinv-Jt) == Z)
check('E3 reverse orientation produces J_-t',
      sp.simplify(Sinv*J0*S-(-Omega*expK(-t))) == Z)
Sdiff = expK(-(t-u)/2)
check('E4 two-parameter symplectic transport',
      sp.trigsimp(sp.expand_trig(Sdiff.T*Omega*Sdiff-Omega)) == Z
      and sp.trigsimp(sp.expand_trig(Sdiff*(-Omega*Gu)*expK((t-u)/2)-Jt)) == Z)
check('E5 reverse orientation is wrong at t=1',
      sp.simplify((Sinv*J0*S-Jt).subs(t,1)) != Z)

# F — external sign-flip: one minus sign in each partner pair
# The diagonal witness flips exactly one endpoint of each pair.
Q = sp.diag(-1,-1,1,1,-1,-1,1,1)
check('F external sign flip is anti-symplectic',
      Q.T*K*Q == -K and Q.T*Omega*Q == -Omega
      and sp.simplify(Q.T*Gt*Q - (sp.cosh(t)*I-sp.sinh(t)*K)) == Z)
check('F sign flip not a frozen unsigned permutation',
      Q not in [pmat(g) for g in Npm])

# G — universal theorems recorded with exact rational witnesses.
# Proof 1: a one-dimensional real fixed space cannot carry J^2=-I.
# Proof 2: G-isometry implies G^(1/2) P G^(-1/2) orthogonal;
# primitive stochastic P has seven eigenvalues strictly inside unit disk.
Pmix = sp.ones(N)/N
check('G1 rational primitive stochastic witness',
      Pmix*sp.ones(N,1) == sp.ones(N,1)
      and (Pmix-I).nullspace() == [sp.ones(N,1)]
      and sp.simplify(Pmix*J0-J0*Pmix) != Z)
check('G2 exact non-isometry witness',
      Pmix.T*I*Pmix != I and Pmix.rank() == 1)
print('PROOF | G1: [P,J]=0 preserves ker(P-I); dim=1 contradicts J^2=-I.')
print('PROOF | G2: P^T G P=G with G>0 forces all |eigenvalues(P)|=1;')
print('PROOF |     primitive stochastic P has 7 nonstationary |lambda|<1.')

# H — human source audit only; not a theorem proved by this executable.
provenance = {
    'reviewed': ['SIM14.6.py', 'SIM16.0.py', 'SIM16.1.py'],
    'sim1_13_status': 'EXPLORATORY_NOT_FROZEN',
    'bridge_equation_supplied_to_executable': False,
    'audit_scope': 'DECLARED_SOURCE_SCOPE_NOT_AUTOMATED_REPOSITORY_SEARCH',
    'bridge_status': 'NO_DOCUMENTED_BRIDGE_IDENTIFIED_IN_REVIEWED_SOURCES',
    'observability': 'UNRESOLVED',
    'no_bridge_impossibility_theorem': False
}
print('H | SOURCE-LIMITED provenance only; no operational observability proof.')

# I — machine-readable audit and explicit projector/orbit exports
D0 = sp.diag(1,1,0,0,1,1,0,0)
D1 = I-D0
check('I N+ projector regression',
      D0*D1 == Z and D0+D1 == I and D0*K == K*D1
      and all(pmat(g).T*D0*pmat(g) == D0 for g in Nplus)
      and in_span(D0, loc_sym_plus) and in_span(D1, loc_sym_plus))

def orbit_packet(orbits):
    return [{'pairs': [list(e) for e in orbit],
             'forbidden_pairs': [list(e) for e in orbit if e[0] != e[1] and e in FORBIDDEN]}
            for orbit in orbits]

packet = {
    'sim': '16.2',
    'experiment_class': 'EXACT_ALGEBRAIC_AUDIT',
    'frozen_regression': 'PASS',
    'group_orders': {'T':len(T),'AGL':len(affine),'Nplus':len(Nplus),
                     'Nminus':len(Nminus),'Npm':len(Npm)},
    'parameter_identifiability': 'EXACT_PROOF',
    'frozen_metric_equivalence': 'SINGLETON_ORBITS',
    'signed_complex_structure_covariance': 'EXACT_PROOF',
    'symplectic_complex_structure_equivalence': 'SAME_ORBIT',
    'symplectic_conjugacy_is_operational_gauge': 'NOT_ESTABLISHED',
    'external_sign_flip': 'NOT_FROZEN_OPERATIONAL_GAUGE',
    'stochastic_commutation_obstruction': 'GENERAL_ALGEBRAIC_THEOREM',
    'primitive_isometry_obstruction': 'GENERAL_ALGEBRAIC_THEOREM',
    'provenance': provenance,
    'operational_observability': 'UNRESOLVED',
    'parameter_selection': 'UNDERDETERMINED_BY_FROZEN_CONSTRAINTS',
    'new_dynamics_introduced': False,
    'new_gauge_group_introduced': False,
    'D0': compact_matrix(D0), 'D1': compact_matrix(D1),
    'orbits_Nplus': orbit_packet(orbits_plus),
    'orbits_Npm': orbit_packet(orbits_pm),
    'status': 'EXACT_ALGEBRAIC_AUDIT_PASSED'
}
print('I | PASS exact algebraic gates and source-limited provenance labeling')
print('TRUTH_PACKET_JSON=' + json.dumps(packet, sort_keys=True))
print('END SIM16.2')






~~~~~~~~~~~~~~~






RESULTS:





SIM16.2 — OPERATIONAL METRIC IDENTIFIABILITY AND EQUIVALENCE CLOSURE
CLASS | EXACT_ALGEBRAIC_AUDIT; NO ISP DYNAMICS
PASS | A1 translation torsor
PASS | A2 affine and signed group orders
PASS | A3 frozen matrix identities
PASS | A4 SIM16.1 exact dimension regression
PASS | A5 signed local symmetric span
PASS | A6 exact compatibility and locality
PASS | B signed trace recovers t
PROOF | B: sinh is strictly increasing on R; J=-Omega G is injective in G.
PASS | C all 48 metrics fixed and J signed-covariant
PASS | D projector and spectral decomposition
PROOF | D: spectrum {exp(t):4, exp(-t):4}; ordinary trace gives |t|.
PASS | E1 S is symplectic
PASS | E2 correct conjugation S J0 S^-1 = Jt
PASS | E3 reverse orientation produces J_-t
PASS | E4 two-parameter symplectic transport
PASS | E5 reverse orientation is wrong at t=1
PASS | F external sign flip is anti-symplectic
PASS | F sign flip not a frozen unsigned permutation
PASS | G1 rational primitive stochastic witness
PASS | G2 exact non-isometry witness
PROOF | G1: [P,J]=0 preserves ker(P-I); dim=1 contradicts J^2=-I.
PROOF | G2: P^T G P=G with G>0 forces all |eigenvalues(P)|=1;
PROOF |     primitive stochastic P has 7 nonstationary |lambda|<1.
H | SOURCE-LIMITED provenance only; no operational observability proof.
PASS | I N+ projector regression
I | PASS exact algebraic gates and source-limited provenance labeling
TRUTH_PACKET_JSON={"D0": [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], "D1": [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]], "experiment_class": "EXACT_ALGEBRAIC_AUDIT", "external_sign_flip": "NOT_FROZEN_OPERATIONAL_GAUGE", "frozen_metric_equivalence": "SINGLETON_ORBITS", "frozen_regression": "PASS", "group_orders": {"AGL": 1344, "Nminus": 24, "Nplus": 24, "Npm": 48, "T": 8}, "new_dynamics_introduced": false, "new_gauge_group_introduced": false, "operational_observability": "UNRESOLVED", "orbits_Nplus": [{"forbidden_pairs": [], "pairs": [[0, 0], [1, 1], [4, 4], [5, 5]]}, {"forbidden_pairs": [[0, 4], [0, 5], [1, 4], [1, 5], [4, 5]], "pairs": [[0, 1], [0, 4], [0, 5], [1, 4], [1, 5], [4, 5]]}, {"forbidden_pairs": [], "pairs": [[0, 2], [1, 3], [4, 6], [5, 7]]}, {"forbidden_pairs": [[0, 3], [0, 6], [0, 7], [1, 2], [1, 6], [1, 7], [2, 5], [3, 4], [4, 7], [5, 6]], "pairs": [[0, 3], [0, 6], [0, 7], [1, 2], [1, 6], [1, 7], [2, 4], [2, 5], [3, 4], [3, 5], [4, 7], [5, 6]]}, {"forbidden_pairs": [], "pairs": [[2, 2], [3, 3], [6, 6], [7, 7]]}, {"forbidden_pairs": [[2, 3], [2, 6], [2, 7], [3, 6], [3, 7]], "pairs": [[2, 3], [2, 6], [2, 7], [3, 6], [3, 7], [6, 7]]}], "orbits_Npm": [{"forbidden_pairs": [], "pairs": [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]]}, {"forbidden_pairs": [[0, 4], [0, 5], [1, 4], [1, 5], [2, 3], [2, 6], [2, 7], [3, 6], [3, 7], [4, 5]], "pairs": [[0, 1], [0, 4], [0, 5], [1, 4], [1, 5], [2, 3], [2, 6], [2, 7], [3, 6], [3, 7], [4, 5], [6, 7]]}, {"forbidden_pairs": [], "pairs": [[0, 2], [1, 3], [4, 6], [5, 7]]}, {"forbidden_pairs": [[0, 3], [0, 6], [0, 7], [1, 2], [1, 6], [1, 7], [2, 5], [3, 4], [4, 7], [5, 6]], "pairs": [[0, 3], [0, 6], [0, 7], [1, 2], [1, 6], [1, 7], [2, 4], [2, 5], [3, 4], [3, 5], [4, 7], [5, 6]]}], "parameter_identifiability": "EXACT_PROOF", "parameter_selection": "UNDERDETERMINED_BY_FROZEN_CONSTRAINTS", "primitive_isometry_obstruction": "GENERAL_ALGEBRAIC_THEOREM", "provenance": {"audit_scope": "DECLARED_SOURCE_SCOPE_NOT_AUTOMATED_REPOSITORY_SEARCH", "bridge_equation_supplied_to_executable": false, "bridge_status": "NO_DOCUMENTED_BRIDGE_IDENTIFIED_IN_REVIEWED_SOURCES", "no_bridge_impossibility_theorem": false, "observability": "UNRESOLVED", "reviewed": ["SIM14.6.py", "SIM16.0.py", "SIM16.1.py"], "sim1_13_status": "EXPLORATORY_NOT_FROZEN"}, "signed_complex_structure_covariance": "EXACT_PROOF", "sim": "16.2", "status": "EXACT_ALGEBRAIC_AUDIT_PASSED", "stochastic_commutation_obstruction": "GENERAL_ALGEBRAIC_THEOREM", "symplectic_complex_structure_equivalence": "SAME_ORBIT", "symplectic_conjugacy_is_operational_gauge": "NOT_ESTABLISHED"}
END SIM16.2
