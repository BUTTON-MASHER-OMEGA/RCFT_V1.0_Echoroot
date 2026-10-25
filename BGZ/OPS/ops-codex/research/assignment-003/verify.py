"""OPS-003: exact octonionic triality and one frozen, torsor-based OPS mapping.

No historical simulation is imported. The worker has a 25-second timeout.
The Cayley-Dickson convention is (a,b)(c,d)=(ac-conj(d)b, da+b conj(c)).
"""
from pathlib import Path
import ast
import hashlib
import itertools
import json
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCES = [
    "projects/Operational_Phase_Space/SIM16.4.py",
    "projects/Operational_Phase_Space/SIM16.3.py",
    "projects/Operational_Phase_Space/SIM15.2.py",
    "projects/Operational_Phase_Space/recorded_output/SIM15.2.txt",
    "projects/Operational_Phase_Space/V4_Orbit-Bridge_Regularity_Theorem.md",
    "projects/ops-codex/research/assignment-001/report.md",
    "projects/ops-codex/research/assignment-002/report.md",
    "rcf/modules/lco/lco-primitives.md",
    "rcf/modules/lco/lco-invariants.md",
    "rcf/modules/lco/lco-operator.md",
    "rcf/modules/lco/lco-update-policy.md",
    "rcmt/local/invariants.md",
    "rcmt/local/constraints/symmetry_chain.md",
    "rcmt/local/constraints/projection.md",
]


def hashes():
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}


def worker():
    import sympy as sp
    I, Z = sp.eye(8), sp.zeros(8)
    frozen = {}
    tree = ast.parse((ROOT/SOURCES[0]).read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id in {"PAIRS", "EDGES", "p", "h", "s"}:
                frozen[t.id] = ast.literal_eval(node.value)
    omega, adjacency, B = Z.copy(), Z.copy(), sp.zeros(4, 8)
    for a, b in frozen["PAIRS"]:
        omega[a, b], omega[b, a] = 1, -1
    for a, b in frozen["EDGES"]:
        adjacency[a, b] = adjacency[b, a] = 1
    for i in range(4):
        B[i, 2*i] = B[i, 2*i+1] = 1
    N = sp.Matrix.hstack(*B.nullspace())

    def perm(g):
        a = Z.copy()
        for i, j in enumerate(g):
            a[j, i] = 1
        return a

    P, J0 = perm(frozen["p"]), -omega

    def conj(a):
        return [a[0]]+[-x for x in a[1:]]

    def add(a, b):
        return [x+y for x, y in zip(a, b)]

    def sub(a, b):
        return [x-y for x, y in zip(a, b)]

    def cd(x, y):
        if len(x) == 1:
            return [x[0]*y[0]]
        k = len(x)//2
        a, b, c, d = x[:k], x[k:], y[:k], y[k:]
        return sub(cd(a, c), cd(conj(d), b))+add(cd(d, a), cd(b, conj(c)))

    units = [list(I[:, i]) for i in range(8)]
    canonical_L = [sp.Matrix.hstack(*(sp.Matrix(cd(a, b)) for b in units)) for a in units]

    # The frozen regular torsor provides labels, but no preferred origin/unit.
    label = {}
    for a, b, c in itertools.product((0, 1), repeat=3):
        state = 0
        for exponent, name in ((a, "p"), (b, "s"), (c, "h")):
            if exponent:
                state = frozen[name][state]
        assert state not in label
        label[state] = a+2*b+4*c
    assert len(label) == 8
    F0 = Z.copy()
    for state in range(8):
        F0[label[state], state] = 1
    # Adapt four pair orientations to Ω. This is a declared frame choice,
    # not an octonionic prediction of Ω. Pair start signs are fixed to +1.
    raw = F0.T*canonical_L[2]*F0
    signs = [1]*8
    for a, b in frozen["PAIRS"]:
        assert abs(raw[b, a]) == 1
        signs[b] = int(raw[b, a])
    F = F0*sp.diag(*signs)
    assert F.T*F == I and F[:, 0] == I[:, 0]
    L = [signs[i]*F.T*canonical_L[label[i]]*F for i in range(8)]
    assert L[0] == I and L[2] == J0

    def mul(a, b):
        return sum((a[i]*L[i]*b for i in range(8)), sp.zeros(8, 1))

    def T(v, s, c):
        return (c.T*mul(v, s))[0]

    def comm(a, b):
        return a*b-b*a

    def local(a):
        return all(a[i, j] == 0 for i in range(8) for j in range(8)
                   if i != j and tuple(sorted((i, j))) not in frozen["EDGES"])

    def descends(a):
        return B*a*N == sp.zeros(4)

    def records(a):
        return [[i, j, str(a[i, j])] for i in range(a.rows) for j in range(a.cols) if a[i, j]]

    def kernel_basis(basis, maps):
        columns = [sp.Matrix([z for f in maps for z in f(x)]) for x in basis]
        return [sum((x*c for x, c in zip(basis, v)), Z.copy())
                for v in sp.Matrix.hstack(*columns).nullspace()]

    # Polarized composition norm: L_v^T L_v=||v||²I for every v.
    for i, j in itertools.product(range(8), repeat=2):
        assert L[i].T*L[j]+L[j].T*L[i] == (2*I if i == j else Z)
    # Real half-spin actions obtained from the even positive Clifford action.
    vector, spin_s, spin_c = {}, {}, {}
    for i in range(8):
        for j in range(i+1, 8):
            a = Z.copy()
            a[i, j], a[j, i] = 1, -1
            ss, cc = L[i].T*L[j]/2, L[i]*L[j].T/2
            assert ss.T == -ss and cc.T == -cc
            vector[i, j], spin_s[i, j], spin_c[i, j] = a, ss, cc
            for k in range(8):
                expected = sum((a[r, k]*L[r] for r in range(8)), Z.copy())
                assert cc*L[k]-L[k]*ss == expected
    # A few explicit bracket checks; general closure also follows from Clifford.
    bracket_pairs = [((0, 1), (1, 2)), ((0, 1), (2, 3)),
                     ((0, 2), (1, 2)), ((1, 4), (4, 7))]
    for u, v in bracket_pairs:
        bracket = comm(vector[u], vector[v])
        for representation in (spin_s, spin_c):
            represented = sum((bracket[i, j]*representation[i, j]
                               for i in range(8) for j in range(i+1, 8)), Z.copy())
            assert comm(representation[u], representation[v]) == represented
    assert spin_s[0, 1]**2 == spin_c[0, 1]**2 == -I/4

    # p: unsigned octonion multiplication, versus a genuine Spin lift.
    assert L[1].applyfunc(abs) == P and L[1]**2 == -I and P**2 == I
    assert T(I[:, 0], I[:, 0], I[:, 0]) == 1
    assert T(P*I[:, 0], P*I[:, 0], P*I[:, 0]) == 0
    assert P.T*omega*P == omega and P.T*adjacency*P == adjacency and B*P == B
    # p is four commuting hyperplane reflections. Their even Clifford product
    # yields two *different* half-spin matrices (normalization (sqrt(2))^4=4).
    d = [L[a]-L[b] for a, b in ((0, 1), (2, 3), (4, 5), (6, 7))]
    Ps = d[0].T*d[1]*d[2].T*d[3]/4
    Pc = d[0]*d[1].T*d[2]*d[3].T/4
    assert Ps.T*Ps == Pc.T*Pc == I and Ps**2 == Pc**2 == I
    for k in range(8):
        assert Pc*L[k]*Ps.T == L[frozen["p"][k]]

    # Natural graph checks: matching support and two-translation graph.
    assert perm(frozen["s"]) == L[2].applyfunc(abs)
    assert perm(frozen["s"]).T*adjacency*perm(frozen["s"]) != adjacency
    two_translation = L[1].applyfunc(abs)+L[2].applyfunc(abs)
    def components(a):
        seen, sizes = set(), []
        for start in range(8):
            if start in seen:
                continue
            frontier, found = [start], {start}
            while frontier:
                u = frontier.pop()
                for v in range(8):
                    if a[u, v] and v not in found:
                        found.add(v)
                        frontier.append(v)
            seen.update(found)
            sizes.append(len(found))
        return sorted(sizes)
    assert components(two_translation) == [4, 4] and components(adjacency) == [8]
    table = []
    for a in range(8):
        row = []
        for b in range(8):
            product = L[a]*I[:, b]
            k = next(k for k in range(8) if product[k])
            assert label[k] == label[a]^label[b]
            row.append({"sign": int(product[k]), "state": k})
        table.append(row)

    # Symplectic and projection negative controls on the vector action.
    skew = list(vector.values())
    symplectic = kernel_basis(skew, [lambda x: x.T*omega+omega*x])
    sparse = kernel_basis(symplectic, [lambda x: sp.Matrix([
        x[i, j] for i in range(8) for j in range(8)
        if i != j and tuple(sorted((i, j))) not in frozen["EDGES"]])])
    projected = kernel_basis(sparse, [lambda x: B*x*N])
    assert (len(symplectic), len(sparse), len(projected)) == (16, 4, 2)
    X_bad_omega, X_bad_B = vector[0, 1], vector[0, 2]
    assert X_bad_omega.T*omega+omega*X_bad_omega != Z
    w = I[:, 0]-I[:, 1]
    assert B*w == sp.zeros(4, 1) and B*X_bad_B*w != sp.zeros(4, 1)
    assert local(X_bad_B) and X_bad_B.T*omega+omega*X_bad_B == Z
    assert descends(J0)
    # T cannot descend jointly through the same B on all three slots.
    assert T(w, I[:, 0], I[:, 0]) == 1

    # Nonassociativity: a small explicit witness and an alternative subalgebra.
    a, b, c = I[:, 1], I[:, 2], I[:, 7]
    left, right = mul(mul(a, b), c), mul(a, mul(b, c))
    assert left != right
    assert mul(mul(a, a), b) == mul(a, mul(a, b))
    assert (L[1]*L[2])*L[7] == L[1]*(L[2]*L[7])
    Lab = sum((mul(a, b)[i]*L[i] for i in range(8)), Z.copy())
    assert (L[1]*L[2]-Lab)*c != sp.zeros(8, 1)
    assert mul(a, a) == -I[:, 0]  # Eight positive basis states not closed.
    # Same Gram data, different T: not a dot-product-only compatibility score.
    v, s = F.T*I[:, 1], F.T*I[:, 2]
    c1, c2 = F.T*I[:, 3], F.T*I[:, 4]
    assert sp.Matrix.hstack(v, s, c1).T*sp.Matrix.hstack(v, s, c1) == sp.eye(3)
    assert sp.Matrix.hstack(v, s, c2).T*sp.Matrix.hstack(v, s, c2) == sp.eye(3)
    assert (T(v, s, c1), T(v, s, c2)) == (1, 0)

    def operator_audit(m):
        return {"preserves_omega": m.T*omega*m == omega,
                "C8_sparse_matrix": local(m), "descends_B": descends(m),
                "commutes_adjacency": comm(m, adjacency) == Z}

    return {"status": "PASS", "sympy": sp.__version__, "exact_arithmetic": True,
        "mapping": {"origin": 0, "ordered_torsor_generators": ["p", "s", "h"],
            "canonical_unit_index_by_OPS_state": [label[i] for i in range(8)],
            "basis_signs_by_OPS_state": signs,
            "F_columns_OPS_to_octonions": [[int(x) for x in row] for row in F.tolist()]},
        "triality": {"norm_polarization_checks": 64, "generator_triples": 28,
            "infinitesimal_covariance_matrix_checks": 224, "explicit_bracket_pairs": len(bracket_pairs),
            "F01_spin_generator_square": "-I/4; a 2pi vector rotation lifts to -I on both spinors"},
        "pairing": {"p_equals_unsigned_L1": True, "p_is_octonion_automorphism": False,
            "diagonal_p_triality_values_before_after": [1, 0],
            "Ps": [[str(x) for x in row] for row in Ps.tolist()],
            "Pc": [[str(x) for x in row] for row in Pc.tolist()],
            "vector_p_audit": operator_audit(P), "spin_s_p_audit": operator_audit(Ps),
            "spin_c_p_audit": operator_audit(Pc), "finite_lift_covariance_checks": 8},
        "adjacency": {"C8_components": components(adjacency),
            "two_unit_matching_components": components(two_translation),
            "s_is_graph_automorphism": False},
        "forms_and_projection": {"full_vector_symplectic_subalgebra_dimension": len(symplectic),
            "C8_sparse_dimension": len(sparse), "B_projectable_local_dimension": len(projected),
            "L_basis_audits": [operator_audit(m) for m in L],
            "nonprojectable_F02_kernel_vector": list(map(int, w)),
            "B_F02_w": list(map(int, B*X_bad_B*w)),
            "full_T_descends_B_in_all_slots": False, "T_kernel_witness": 1},
        "nonassociativity": {"OPS_state_triple": [1, 2, 7],
            "left_bracket_product": list(map(int, left)), "right_bracket_product": list(map(int, right)),
            "associator": list(map(int, left-right)),
            "L1_L2_minus_Lproduct_rank": int((L[1]*L[2]-Lab).rank()),
            "operator_composition_associative": True, "positive_eight_states_closed": False},
        "compatibility": {"identical_Gram_triality_values": [1, 0]},
        "multiplication_table": table}


if __name__ == "__main__":
    if "--worker" in sys.argv:
        print(json.dumps(worker()))
    else:
        before = hashes()
        started = time.perf_counter()
        proc = subprocess.run([sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()),
            "--worker"], capture_output=True, text=True, encoding="utf-8", timeout=25)
        if proc.returncode:
            raise RuntimeError(proc.stderr)
        result = json.loads(proc.stdout)
        result.update(seconds=round(time.perf_counter()-started, 3), python=platform.python_version(),
                      timeout_seconds=25)
        assert hashes() == before, "Recorded source changed during verification"
        result["recorded_inputs_unchanged"] = True
        (HERE/"results.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        commits = {}
        for repo in ("rcf", "rcmt"):
            path = ROOT/repo
            commits[repo] = subprocess.check_output(["git", "-c", f"safe.directory={path.as_posix()}",
                "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        manifest = {"date": "2026-10-24", "source_sha256": before, "commits": commits,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "non_repository_inputs": "Copied OPS artifacts and independent Codex OPS-001/002 reports",
            "external_reference": "https://math.ucr.edu/home/baez/octonions/node7.html"}
        (HERE/"source-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        compact = {k: v for k, v in result.items() if k not in {"multiplication_table"}}
        print(json.dumps(compact, indent=2))
