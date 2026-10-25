#!/usr/bin/env python3
"""
OPS-003: COMPLETE OCTONIONIC TRIALITY AND SUBSTRATE REPRESENTATION AUDIT
========================================================================
Agent: Antigravity
Workspace: Projects/ops-antigravity/research/
Authority: Exploratory research findings only; no architectural promotion.

Exact symbolic and algebraic verification script testing:
1. Standard real octonion algebra O, Fano orientations on F_2^3, and Spin(8) triality.
2. Exact infinitesimal triality triples (D_v, D_s, D_c) in so(8)^3 and g_2 = Der(O).
3. Finite D_4 triality in H_192 (three normal C_2^3 subgroups T_0, T_1, T_2) vs. X_8.
4. OPS Correspondence checks across all 16 octonion orientations on F_2^3:
   - A. Carrier homogeneity vs. octonion 1+7 identity splitting.
   - B. Pairing involution p vs. octonion conjugation, L_a, R_a, and Clifford products.
   - C. Frozen C_8 adjacency vs. octonion multiplication locality and triality transport.
   - D. Symplectic form Omega preservation and triality transport across (8_v, 8_s, 8_c).
   - E. Basin projection B descent across so(8), spin(8), and octonion operators.
5. Nonassociativity vs. LCO composition, associator [L_a, R_b], and locality closure.
"""

import itertools
from collections import Counter
import sympy as sp
import numpy as np

# =============================================================================
# 0. FROZEN OPS SIM14-SIM16 BASELINE DEFINITIONS
# =============================================================================
N = 8
I8 = sp.eye(N)
Z8 = sp.zeros(N)

p = (1, 0, 3, 2, 5, 4, 7, 6)
h = (7, 6, 5, 4, 3, 2, 1, 0)
s = (2, 3, 0, 1, 6, 7, 4, 5)
PAIRS = ((0, 2), (1, 3), (4, 6), (5, 7))
EDGES = {
    (0, 1), (0, 2), (1, 3), (2, 4),
    (3, 5), (4, 6), (5, 7), (6, 7)
}

Omega = sp.zeros(N)
for a, b in PAIRS:
    Omega[a, b] = 1
    Omega[b, a] = -1
K_mat = Omega.applyfunc(abs)
J0 = -Omega

def compose(a, b):
    return tuple(a[b[i]] for i in range(N))

def perm_mat(g):
    M = sp.zeros(N)
    for i, j in enumerate(g):
        M[j, i] = 1
    return M

P_mat = perm_mat(p)
H_mat = perm_mat(h)
S_mat = perm_mat(s)

ident = tuple(range(N))
coord = {}
for bits in range(8):
    g = ident
    for k, gen in enumerate((p, h, s)):
        if (bits >> k) & 1:
            g = compose(gen, g)
    coord[g[0]] = bits
vertex_of_bits = {v: k for k, v in coord.items()}

print("=== GATE 0: FROZEN OPS BASELINE ===")
print("Vertex -> F2^3 bits (b0=p, b1=h, b2=s):", coord)
print("Bits -> Vertex:", vertex_of_bits)

# =============================================================================
# 1. OCTONION ALGEBRA ON F_2^3 AND ENUMERATION OF ALL VALID ORIENTATIONS
# =============================================================================
fano_lines = sorted({
    tuple(sorted((a, b, a ^ b)))
    for a in range(1, 8) for b in range(a + 1, 8)
})
assert len(fano_lines) == 7
print("\n=== GATE 1: FANO LINES & OCTONION MULTIPLICATION TABLES ===")
print("7 Fano lines in F_2^3 bits:", fano_lines)
print("7 Fano lines in vertex labels:", [
    tuple(vertex_of_bits[x] for x in line) for line in fano_lines
])

valid_tables = []
for signs in itertools.product([-1, 1], repeat=7):
    eps = [[0]*8 for _ in range(8)]
    for a in range(8):
        eps[0][a] = 1
        eps[a][0] = 1
    for a in range(1, 8):
        eps[a][a] = -1
    for idx, (a, b, c) in enumerate(fano_lines):
        sgn = signs[idx]
        eps[a][b], eps[b][a] = sgn, -sgn
        eps[b][c], eps[c][b] = sgn, -sgn
        eps[c][a], eps[a][c] = sgn, -sgn
    is_oct = True
    for a in range(1, 8):
        for b in range(1, 8):
            if b == a:
                continue
            ab = a ^ b
            for c in range(1, 8):
                if c in (a, b, ab):
                    continue
                if eps[a][b] * eps[ab][c] != -eps[b][c] * eps[a][b ^ c]:
                    is_oct = False
                    break
            if not is_oct:
                break
        if not is_oct:
            break
    if is_oct:
        valid_tables.append(eps)

print(f"Total valid octonion orientations on fixed F_2^3 Fano plane: {len(valid_tables)} (out of 128)")
assert len(valid_tables) == 16

def build_LR_vertex_basis(eps):
    L = []
    R = []
    for a_bits in range(8):
        La = np.zeros((8, 8), dtype=int)
        Ra = np.zeros((8, 8), dtype=int)
        for v in range(8):
            b_bits = coord[v]
            out_v = vertex_of_bits[a_bits ^ b_bits]
            La[out_v, v] = eps[a_bits][b_bits]
            Ra[out_v, v] = eps[b_bits][a_bits]
        L.append(La)
        R.append(Ra)
    return L, R

eps0 = valid_tables[0]
L0, R0 = build_LR_vertex_basis(eps0)

assoc_count = 0
nonassoc_count = 0
for a in range(1, 8):
    for b in range(1, 8):
        for c in range(1, 8):
            lhs = eps0[a][b] * eps0[a ^ b][c]
            rhs = eps0[b][c] * eps0[a][b ^ c]
            if lhs == rhs:
                assoc_count += 1
            else:
                nonassoc_count += 1
print(f"Imaginary triples (7^3 = 343): associative = {assoc_count}, non-associative = {nonassoc_count}")
assert assoc_count == 175 and nonassoc_count == 168

# =============================================================================
# 2. EXACT CONSTRUCTION OF SPIN(8) TRIALITY & G_2 DERIVATIONS
# =============================================================================
print("\n=== GATE 2: EXACT SPIN(8) TRIALITY AND G_2 DERIVATIONS ===")
conj_diag = np.array([1 if coord[v] == 0 else -1 for v in range(8)], dtype=int)
C_oct = np.diag(conj_diag)
print(f"Octonion conjugation trace = {np.trace(C_oct)}, det = {int(np.round(np.linalg.det(C_oct)))}")
print(f"OPS Pairing P_mat trace    = {int(sp.trace(P_mat))}, det = {int(P_mat.det())}")

T_tensor = np.zeros((8, 8, 8), dtype=int)
for i in range(8):
    Li = L0[coord[i]]
    for j in range(8):
        for k in range(8):
            T_tensor[i, j, k] = Li[k, j] * eps0[coord[k]][coord[k]]

assert np.array_equal(T_tensor, np.transpose(T_tensor, (1, 2, 0)))
assert np.array_equal(T_tensor, np.transpose(T_tensor, (2, 0, 1)))
print("Verified: T(x, y, z) = Re((xy)z) is strictly invariant under cyclic C_3 permutations (x, y, z) -> (y, z, x)!")

T_trans = np.einsum('ia,jb,kc,abc->ijk', C_oct, C_oct, C_oct, np.transpose(T_tensor, (1, 0, 2)))
assert np.array_equal(T_tensor, T_trans)
print("Verified: T(conj(y), conj(x), conj(z)) == T(x, y, z) gives full S_3 outer triality!")

so8_pairs = [(p_idx, q_idx) for p_idx in range(8) for q_idx in range(p_idx + 1, 8)]
so8_pair_to_idx = {pair: idx for idx, pair in enumerate(so8_pairs)}
so8_basis = []
for p_idx, q_idx in so8_pairs:
    M = np.zeros((8, 8), dtype=int)
    M[p_idx, q_idx] = 1
    M[q_idx, p_idx] = -1
    so8_basis.append(M)

T_v = np.zeros((28, 512), dtype=int)
T_s = np.zeros((28, 512), dtype=int)
T_c = np.zeros((28, 512), dtype=int)
for m, Bm in enumerate(so8_basis):
    T_v[m, :] = np.einsum('ai,ajk->ijk', Bm, T_tensor).reshape(-1)
    T_s[m, :] = np.einsum('bj,ibk->ijk', Bm, T_tensor).reshape(-1)
    T_c[m, :] = np.einsum('ck,ijc->ijk', Bm, T_tensor).reshape(-1)

# Exact solve via 56x56 normal equations (verified exact on all 512 rows):
M_sc_np = np.hstack([T_s.T, T_c.T])
G_56 = sp.Matrix(M_sc_np.T @ M_sc_np)
RHS_56 = sp.Matrix(-M_sc_np.T @ T_v.T)
sol_sc = G_56.LUsolve(RHS_56)
assert sp.Matrix(M_sc_np) * sol_sc == sp.Matrix(-T_v.T)

Pi_vs = sol_sc[:28, :]
Pi_vc = sol_sc[28:, :]
assert Pi_vs * Pi_vs == Pi_vc and Pi_vs ** 3 == sp.eye(28)
print(f"Pi_vs rank = {Pi_vs.rank()}, Pi_vc rank = {Pi_vc.rank()} (exact order-3 triality automorphisms!)")

g2_null = (Pi_vs - sp.eye(28)).nullspace()
print(f"Dimension of triality-fixed subalgebra g_2 = Der(O): {len(g2_null)} (expected 14)")
assert len(g2_null) == 14

for vec in g2_null:
    D_mat = sum((vec[m] * sp.Matrix(so8_basis[m]) for m in range(28)), sp.zeros(8))
    assert D_mat[:, 0] == sp.zeros(8, 1)
print("Verified: All 14 generators of g_2 = Der(O) annihilate vertex 0 (the octonion identity e_0)!")

def mat_to_so8_vec(M):
    assert M.T == -M
    return sp.Matrix(28, 1, [M[p_i, q_i] for p_i, q_i in so8_pairs])

def so8_vec_to_mat(v):
    M = sp.zeros(8, 8)
    for idx, (p_i, q_i) in enumerate(so8_pairs):
        M[p_i, q_i] = v[idx]
        M[q_i, p_i] = -v[idx]
    return M

# =============================================================================
# 3. GATE 3: FINITE D_4 TRIALITY IN H_192 VS. CARRIER X_8
# =============================================================================
print("\n=== GATE 3: FINITE H_192 TRIALITY KERNELS VS. X_8 CARRIER ===")
affine = set()
for a, b, c in itertools.product(range(1, 8), repeat=3):
    if len({0, a, b, c, a ^ b, a ^ c, b ^ c, a ^ b ^ c}) != 8:
        continue
    for shift in range(8):
        perm = []
        for i in range(N):
            v, w = coord[i], shift
            if v & 1: w ^= a
            if v & 2: w ^= b
            if v & 4: w ^= c
            perm.append(vertex_of_bits[w])
        affine.add(tuple(perm))

def preserves_abs_omega(g):
    Pm = perm_mat(g)
    return Pm.T * K_mat * Pm == K_mat

H192 = sorted(g for g in affine if preserves_abs_omega(g))
assert len(H192) == 192

def inv_perm(g):
    out = [0]*N
    for i, j in enumerate(g):
        out[j] = i
    return tuple(out)

inv_H192 = [g for g in H192 if g != ident and compose(g, g) == ident]
conj_classes = []
seen_inv = set()
for g in inv_H192:
    if g in seen_inv:
        continue
    cls = {compose(compose(h_el, g), inv_perm(h_el)) for h_el in H192}
    seen_inv |= cls
    conj_classes.append(sorted(cls))

normal_C2_3 = []
for r in range(1, len(conj_classes) + 1):
    for comb in itertools.combinations(conj_classes, r):
        cand = {ident}
        for c in comb:
            cand |= set(c)
        if len(cand) == 8:
            if all(compose(x, y) in cand for x in cand for y in cand):
                normal_C2_3.append(sorted(cand))

print(f"Number of normal C_2^3 subgroups inside H_192: {len(normal_C2_3)}")
for idx, T_sub in enumerate(normal_C2_3):
    orbits = []
    unseen = set(range(8))
    while unseen:
        v0 = min(unseen)
        orb = {g[v0] for g in T_sub}
        orbits.append(sorted(orb))
        unseen -= orb
    traces = Counter(sum(1 for i in range(8) if g[i] == i) for g in T_sub)
    print(f"  Normal C_2^3 #{idx}: orbits on X_8 = {orbits}, trace histogram = {dict(traces)}")

common_inter = set(normal_C2_3[0])
for T_sub in normal_C2_3[1:]:
    common_inter &= set(T_sub)
print("Common intersection of the normal C_2^3 subgroups:", [
    "ident" if g == ident else ("s" if g == s else g) for g in common_inter
])

# =============================================================================
# 4. GATE 4: OPS CORRESPONDENCE CHECKS (PAIRING, OMEGA, ADJACENCY, PROJECTION)
# =============================================================================
print("\n=== GATE 4A & 4B: PAIRING P AND SYMPLECTIC FORM OMEGA VS. OCTONIONS ===")
Omega_np = np.array(Omega, dtype=int)
s_bits = coord[2]
p_bits = coord[1]
h_bits = coord[7]

omega_matches_L = 0
omega_matches_R = 0
for eps in valid_tables:
    L_tab, R_tab = build_LR_vertex_basis(eps)
    if np.array_equal(L_tab[s_bits], Omega_np) or np.array_equal(L_tab[s_bits], -Omega_np):
        omega_matches_L += 1
    if np.array_equal(R_tab[s_bits], Omega_np) or np.array_equal(R_tab[s_bits], -Omega_np):
        omega_matches_R += 1

print(f"Out of 16 octonion tables on F_2^3: Omega == +/- L_{{e_s}} count = {omega_matches_L}, Omega == +/- R_{{e_s}} count = {omega_matches_R}")

P_np = np.array(P_mat, dtype=int)
comm_P_count = 0
for eps in valid_tables:
    L_tab, R_tab = build_LR_vertex_basis(eps)
    for a_bits in range(1, 8):
        if np.array_equal(L_tab[a_bits] @ P_np, P_np @ L_tab[a_bits]):
            comm_P_count += 1
        if np.array_equal(R_tab[a_bits] @ P_np, P_np @ R_tab[a_bits]):
            comm_P_count += 1
print(f"Number of (table, imaginary unit a) pairs where L_{{e_a}} or R_{{e_a}} commutes with P: {comm_P_count} (out of {16*14})")

line_no_p = [line for line in fano_lines if 1 not in line][0]
a_b, b_b, c_b = line_no_p
D_line = -L0[a_b] @ L0[b_b] @ L0[c_b]
print(f"Grading operator D = -L_{a_b} L_{b_b} L_{c_b} for Fano line {line_no_p}: diagonal =", np.diag(D_line))
assert np.array_equal(D_line @ D_line, np.eye(8, dtype=int))
assert np.trace(D_line) == 0

# =============================================================================
# 5. GATE 4C: FROZEN C_8 ADJACENCY LOCALITY & TRIALITY TRANSPORT
# =============================================================================
print("\n=== GATE 4C: FROZEN C_8 ADJACENCY LOCALITY & TRIALITY TRANSPORT ===")
def is_c8_local_np(M):
    for i in range(8):
        for j in range(8):
            if i != j and M[i, j] != 0:
                if tuple(sorted((i, j))) not in EDGES:
                    return False
    return True

local_L_units = [a for a in range(1, 8) if is_c8_local_np(L0[a])]
local_R_units = [a for a in range(1, 8) if is_c8_local_np(R0[a])]
print(f"C_8-local imaginary octonion units for L_a: {local_L_units} (bit 4 == s!)")
print(f"C_8-local imaginary octonion units for R_a: {local_R_units} (bit 4 == s!)")

c8_edge_indices = [so8_pair_to_idx[tuple(sorted(e))] for e in sorted(EDGES)]
C8_subspace = sp.zeros(28, len(c8_edge_indices))
for col, idx in enumerate(c8_edge_indices):
    C8_subspace[idx, col] = 1

forbidden_indices = [idx for idx in range(28) if idx not in c8_edge_indices]
Proj_forbidden = sp.zeros(len(forbidden_indices), 28)
for row, idx in enumerate(forbidden_indices):
    Proj_forbidden[row, idx] = 1

local_vs_null = (Proj_forbidden * Pi_vs * C8_subspace).nullspace()
local_vc_null = (Proj_forbidden * Pi_vc * C8_subspace).nullspace()
local_all_three = (sp.Matrix.vstack(
    Proj_forbidden * Pi_vs * C8_subspace,
    Proj_forbidden * Pi_vc * C8_subspace
)).nullspace()

print(f"Dim of C_8-local subspace in so(8)_v: {len(c8_edge_indices)}")
print(f"Dim of C_8-local D_v whose D_s is ALSO C_8-local: {len(local_vs_null)}")
print(f"Dim of C_8-local D_v whose D_c is ALSO C_8-local: {len(local_vc_null)}")
print(f"Dim of C_8-local D_v where (D_v, D_s, D_c) are ALL C_8-local: {len(local_all_three)}")
for v_w in local_all_three:
    Dv_mat = so8_vec_to_mat(C8_subspace * v_w)
    print("  Surviving triality-local generator support:", [
        (i, j, Dv_mat[i, j]) for i in range(8) for j in range(i+1, 8) if Dv_mat[i, j] != 0
    ])

# =============================================================================
# 6. GATE 4D: SYMPLECTIC COMPATIBILITY & TRIALITY TRANSPORT OF OMEGA
# =============================================================================
print("\n=== GATE 4D: SYMPLECTIC FORM OMEGA & TRIALITY TRANSPORT ===")
j0_vec = mat_to_so8_vec(J0)
js_vec = Pi_vs * j0_vec
jc_vec = Pi_vc * j0_vec
Js_mat = so8_vec_to_mat(js_vec)
Jc_mat = so8_vec_to_mat(jc_vec)

print(f"J_0 (on 8_v): rank = {J0.rank()}, J_0^2 == -I_8 is {J0*J0 == -I8}")
print(f"J_s = Pi_vs(J_0) (on 8_s): rank = {Js_mat.rank()}, J_s^2 == -I_8 is {Js_mat*Js_mat == -I8}")
print(f"J_c = Pi_vc(J_0) (on 8_c): rank = {Jc_mat.rank()}, J_c^2 == -I_8 is {Jc_mat*Jc_mat == -I8}")
print("Eigenvalues of J_0^2:", (J0*J0).eigenvals())
print("Eigenvalues of J_s^2:", (Js_mat*Js_mat).eigenvals())
print("Eigenvalues of J_c^2:", (Jc_mat*Jc_mat).eigenvals())

comm_omega_rows = []
for m in range(28):
    Bm = sp.Matrix(so8_basis[m])
    C_m = Bm * Omega - Omega * Bm
    comm_omega_rows.append(list(C_m))
Comm_Omega_op = sp.Matrix(comm_omega_rows).T
u4_v_basis = Comm_Omega_op.nullspace()
print(f"Dim of u(4)_v = Stab_{{so(8)_v}}(Omega): {len(u4_v_basis)} (expected 16)")

Simul_Omega_op = sp.Matrix.vstack(
    Comm_Omega_op,
    Comm_Omega_op * Pi_vs,
    Comm_Omega_op * Pi_vc
)
simul_u4_basis = Simul_Omega_op.nullspace()
print(f"Dim of simultaneous stabilizer of Omega across (8_v, 8_s, 8_c): {len(simul_u4_basis)}")

# Check universality across ALL 16 valid octonion tables on F_2^3:
ranks_across_16 = []
comm_P_across_16 = []
for eps in valid_tables:
    L_tab, R_tab = build_LR_vertex_basis(eps)
    T_ten = np.zeros((8, 8, 8), dtype=int)
    for i in range(8):
        Li = L_tab[coord[i]]
        for j in range(8):
            for k in range(8):
                T_ten[i, j, k] = Li[k, j] * eps[coord[k]][coord[k]]
    Tv = np.array([np.einsum('ai,ajk->ijk', Bm, T_ten).reshape(-1) for Bm in so8_basis])
    Ts = np.array([np.einsum('bj,ibk->ijk', Bm, T_ten).reshape(-1) for Bm in so8_basis])
    Tc = np.array([np.einsum('ck,ijc->ijk', Bm, T_ten).reshape(-1) for Bm in so8_basis])
    Msc = np.hstack([Ts.T, Tc.T])
    sol = sp.Matrix(Msc.T @ Msc).LUsolve(sp.Matrix(-Msc.T @ Tv.T))
    Pvs, Pvc = sol[:28, :], sol[28:, :]
    Js_t = so8_vec_to_mat(Pvs * j0_vec)
    Jc_t = so8_vec_to_mat(Pvc * j0_vec)
    ranks_across_16.append((J0.rank(), Js_t.rank(), Jc_t.rank()))
    comm_P_across_16.append((
        J0 * P_mat == P_mat * J0,
        Js_t * P_mat == P_mat * Js_t,
        Jc_t * P_mat == P_mat * Jc_t
    ))
print("Universal ranks of (J_v, J_s, J_c) across all 16 octonion tables:", set(ranks_across_16))
print("Universal [J_rep, P] == 0 across all 16 octonion tables:", set(comm_P_across_16))
assert set(ranks_across_16) == {(8, 8, 2), (8, 2, 8)}
assert set(comm_P_across_16) == {(True, False, False)}

# =============================================================================
# 7. GATE 4E: PROJECTION INTERFACE DESCENT
# =============================================================================
print("\n=== GATE 4E: PROJECTION INTERFACE B DESCENT ACROSS SPIN(8) ===")
comm_P_rows = [list(sp.Matrix(so8_basis[m]) * P_mat - P_mat * sp.Matrix(so8_basis[m])) for m in range(28)]
Comm_P_op = sp.Matrix(comm_P_rows).T
descend_v = Comm_P_op.nullspace()
descend_vs = (sp.Matrix.vstack(Comm_P_op, Comm_P_op * Pi_vs)).nullspace()
descend_all = (sp.Matrix.vstack(Comm_P_op, Comm_P_op * Pi_vs, Comm_P_op * Pi_vc)).nullspace()
descend_g2 = (sp.Matrix.vstack(Comm_P_op, Pi_vs - sp.eye(28))).nullspace()

print(f"Dim of so(8)_v descending through projection B ([D_v, P] = 0): {len(descend_v)} (so(4)_+ + so(4)_-, 28 - 12 = 16 obstructed)")
print(f"Dim of spin(8) where both D_v and D_s descend through B: {len(descend_vs)}")
print(f"Dim of spin(8) where ALL THREE (D_v, D_s, D_c) descend through B: {len(descend_all)}")
print(f"Dim of g_2 = Der(O) descending through projection B: {len(descend_g2)}")

# =============================================================================
# 8. GATE 6: NONASSOCIATIVITY AND LCO COMPATIBILITY
# =============================================================================
print("\n=== GATE 6: NONASSOCIATIVITY, ASSOCIATOR [L_a, R_b], AND LCO CLOSURE ===")
for a_b in range(1, 8):
    for b_b in range(1, 8):
        comm_LR = L0[a_b] @ R0[b_b] - R0[b_b] @ L0[a_b]
        assoc_mat = R0[b_b] @ L0[a_b] - L0[a_b] @ R0[b_b]
        assert np.array_equal(comm_LR, -assoc_mat)
        if a_b == b_b:
            assert np.array_equal(comm_LR, np.zeros((8, 8), dtype=int))
        else:
            evals = np.linalg.eigvalsh(assoc_mat.T @ assoc_mat)
            assert sorted(np.round(evals).astype(int)) == [0, 0, 0, 0, 4, 4, 4, 4]
print("Verified: For all 42 distinct pairs (a != b), [L_a, R_b](x) = -[e_a, x, e_b] has rank 4 (0 on H_{a,b}, +/-2i on H_{a,b}^perp)!")

for a_b in range(1, 8):
    for b_b in range(a_b + 1, 8):
        comm_LL = L0[a_b] @ L0[b_b] - L0[b_b] @ L0[a_b]
        L_prod = 2 * eps0[a_b][b_b] * L0[a_b ^ b_b]
        diff_rank = np.linalg.matrix_rank(comm_LL - L_prod)
        assert diff_rank == 4
print("Verified: For all 21 pairs (a < b), [L_a, L_b] != 2 L_{e_a e_b}; discrepancy has rank 4 on H_{a,b}^perp due to nonassociativity!")
print("=== ALL GATES COMPLETED SUCCESSFULLY ===")
