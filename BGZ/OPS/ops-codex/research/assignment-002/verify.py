"""Bounded OPS-002 toys; exact arithmetic, five edge states, three squeeze states.

Historical simulations are never imported or executed. A subprocess enforces a
20-second limit. Run with Python -X utf8 -B from any directory.
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
    "rcf/modules/temporal-interface/local-constraint-density.md",
    "rcf/modules/geometry/discrete/candidate-geometries/density-regimes.md",
    "rcmt/local/invariants.md",
    "rcmt/local/constraints/symmetry_chain.md",
    "rcmt/local/constraints/projection.md",
    "projects/Operational_Phase_Space/SIM16.1.py",
    "projects/Operational_Phase_Space/SIM16.3.py",
    "projects/Operational_Phase_Space/SIM16.4.py",
    "projects/ops-codex/research/assignment-001/report.md",
    "projects/ops-codex/research/assignment-001/results.json",
    "projects/ops-codex/implementation/ops-repair-verification.json",
]


def hashes():
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}


def worker():
    import sympy as sp
    frozen = {}
    tree = ast.parse((ROOT/SOURCES[7]).read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {"PAIRS", "EDGES", "p"}:
                frozen[target.id] = ast.literal_eval(node.value)
    pairs, edges = frozen["PAIRS"], frozen["EDGES"]
    I, Z = sp.eye(8), sp.zeros(8)
    omega = sp.zeros(8)
    for a, b in pairs:
        omega[a, b], omega[b, a] = 1, -1
    K = omega.applyfunc(abs)
    B = sp.zeros(4, 8)
    for i in range(4):
        B[i, 2*i] = B[i, 2*i+1] = 1
    R = B.T/2
    N = sp.Matrix.hstack(*B.nullspace())
    assert B*R == sp.eye(4) and B*N == sp.zeros(4)

    def comm(a, b):
        return a*b-b*a

    def basis_kernel(basis, maps):
        if not maps:
            return basis[:]
        if not basis:
            return []
        columns = [sp.Matrix([e for f in maps for e in f(x)]) for x in basis]
        null = sp.Matrix.hstack(*columns).nullspace()
        return [sum((c*x for c, x in zip(v, basis)), Z.copy()) for v in null]

    def rank_span(basis):
        return sp.Matrix.hstack(*(sp.Matrix(list(x)) for x in basis)).rank() if basis else 0

    def local(x):
        return all(x[i, j] == 0 for i in range(8) for j in range(8)
                   if i != j and tuple(sorted((i, j))) not in edges)

    skew, symmetric = [], []
    for i in range(8):
        m = Z.copy()
        m[i, i] = 1
        symmetric.append(m)
        for j in range(i+1, 8):
            a, h = Z.copy(), Z.copy()
            a[i, j], a[j, i] = 1, -1
            h[i, j] = h[j, i] = 1
            skew.append(a)
            symmetric.append(h)
    sp_basis = [-omega*h for h in symmetric]
    local_sp = basis_kernel(sp_basis, [lambda x: sp.Matrix([
        x[i, j] for i in range(8) for j in range(8)
        if i != j and tuple(sorted((i, j))) not in edges])])
    local_orthosp = basis_kernel(skew, [lambda x: x.T*omega+omega*x,
        lambda x: sp.Matrix([x[i, j] for i in range(8) for j in range(8)
                           if i != j and tuple(sorted((i, j))) not in edges])])
    assert len(local_sp) == 12 and len(local_orthosp) == 4

    # Toy 1: nested, individually retained relational equality constraints.
    states = [[], [(0, 1)], [(0, 1), (0, 2), (1, 3)],
              [(0, 1), (0, 2), (1, 3), (2, 4)],
              [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5)]]
    edge_results = []
    for active in states:
        assert all(e in edges for e in active)
        vectors = [I[:, a]-I[:, b] for a, b in active]
        tensors = [v*v.T for v in vectors]
        C = sp.Matrix.vstack(*(v.T for v in vectors)) if vectors else sp.zeros(0, 8)
        L = sum(tensors, Z.copy())
        maps = [lambda x, h=h: comm(x, h) for h in tensors]
        g_full = basis_kernel(skew, maps)
        g_local = basis_kernel(local_orthosp, maps)
        g_projectable = basis_kernel(g_local, [lambda x: B*x*N])
        survivor_basis = sp.Matrix.hstack(*C.nullspace())
        assert L*survivor_basis == sp.zeros(8, survivor_basis.cols)
        configurations = [x for x in itertools.product((0, 1), repeat=8)
                          if all(x[a] == x[b] for a, b in active)]
        outputs = {tuple(x[2*i]+x[2*i+1] for i in range(4)) for x in configurations}
        defect = B*L*N
        Lbar = B*L*R
        response_descends = defect == sp.zeros(4)
        assert response_descends == (B*L == Lbar*B)
        assert L == L.T and local(L) and C.rank() == len(active)
        assert len(configurations) == 2**(8-len(active))
        assert len(g_full) == (8-len(active))*(7-len(active))//2
        assert all(comm(a, b) == Z for a in g_local for b in g_local)
        assert any(any(x) for x in outputs)
        # A Hamiltonian response is a separate, explicitly proposed operation.
        H = -omega*L
        assert H.T*omega+omega*H == Z
        assert (B*H*N == sp.zeros(4)) == response_descends
        # Unit symplectic transport and complex descent persist independently.
        assert B*(-omega)*N == sp.zeros(4)
        edge_results.append({
            "active_edges": active, "toy_density": str(sp.Rational(len(active), 8)),
            "binary_configurations": len(configurations),
            "linear_survivor_dimension": survivor_basis.cols,
            "projected_linear_survivor_dimension": int((B*survivor_basis).rank()),
            "restricted_omega_rank": int((survivor_basis.T*omega*survivor_basis).rank()),
            "distinct_projected_binary_configurations": len(outputs),
            "orthogonal_constraint_algebra_dimension": len(g_full),
            "local_orthogonal_symplectic_constraint_dimension": len(g_local),
            "projectable_local_symmetry_dimension": len(g_projectable),
            "aggregate_response_descends_through_B": response_descends,
            "aggregate_response_defect_rank": int(defect.rank()),
            "hamiltonian_response_C8_sparse": local(H),
        })
    assert [r["orthogonal_constraint_algebra_dimension"] for r in edge_results] == [28, 21, 10, 6, 3]
    assert [r["local_orthogonal_symplectic_constraint_dimension"] for r in edge_results] == [4, 2, 2, 1, 0]
    assert [r["aggregate_response_descends_through_B"] for r in edge_results] == [True, True, True, False, True]
    # Failure witness at the fourth state, followed by restoration with its p-mate.
    Lbad = sum(((I[:, a]-I[:, b])*(I[:, a]-I[:, b]).T for a, b in states[3]), Z.copy())
    w = I[:, 2]-I[:, 3]
    assert B*w == sp.zeros(4, 1) and B*Lbad*w != sp.zeros(4, 1)
    # Same density and binary survivor count, different projection behavior.
    control_edges = [(0, 1), (0, 2), (1, 3), (6, 7)]
    Lcontrol = sum(((I[:, a]-I[:, b])*(I[:, a]-I[:, b]).T
                    for a, b in control_edges), Z.copy())
    control_configs = [x for x in itertools.product((0, 1), repeat=8)
                      if all(x[a] == x[b] for a, b in control_edges)]
    assert len(control_configs) == 16 and B*Lcontrol*N == sp.zeros(4)

    # Toy 2: geometric squeezing of the existing SIM16.1 family, not density.
    squeeze_results = []
    for q in (sp.Integer(1), sp.Integer(2), sp.Integer(4)):
        G = (q+1/q)*I/2+(q-1/q)*K/2
        J = -omega*G
        # q=exp(t); S=exp(-t*K/2). Exact radicals, no floating-point tolerance.
        S = (sp.sqrt(q)+1/sp.sqrt(q))*I/2-(sp.sqrt(q)-1/sp.sqrt(q))*K/2
        Sinv = (sp.sqrt(q)+1/sp.sqrt(q))*I/2+(sp.sqrt(q)-1/sp.sqrt(q))*K/2
        assert sp.simplify(S*Sinv) == I
        assert sp.simplify(S.T*omega*S) == omega
        assert sp.simplify(S.T*G*S) == I
        assert sp.simplify(S*(-omega)*Sinv) == J
        assert J**2 == -I and G*omega*G == omega and G.det() == 1
        assert B*J*N == sp.zeros(4) and sp.simplify(B*S*N) == sp.zeros(4)
        full = basis_kernel(sp_basis, [lambda x: x.T*G+G*x])
        moving_orthogonal = [sp.simplify(S*a*Sinv) for a in skew]
        assert rank_span(moving_orthogonal) == 28
        assert all(a.T*G+G*a == Z for a in moving_orthogonal)
        loc = basis_kernel(local_sp, [lambda x: x.T*G+G*x])
        proj = basis_kernel(loc, [lambda x: B*x*N])
        assert len(full) == 16 and len(loc) == 4 and len(proj) == 2
        assert all(comm(a, b) == Z for a in loc for b in loc)
        assert all(local(sp.simplify(S*a*Sinv)) for a in local_orthosp)
        squeeze_results.append({"q_exp_t": int(q), "metric_condition_number": int(q*q),
            "orthogonal_metric_stabilizer_dimension": len(moving_orthogonal),
            "symplectic_metric_stabilizer_dimension": len(full),
            "C8_sparse_stabilizer_dimension": len(loc),
            "projectable_local_stabilizer_dimension": len(proj),
            "J_and_transport_descend_B": True, "symplectically_conjugate_to_q1": True})
    return {"status": "PASS", "sympy": sp.__version__, "exact_arithmetic": True,
            "edge_constraint_states": edge_results, "squeeze_states": squeeze_results,
            "same_density_control": {"active_edges": control_edges, "toy_density": "1/2",
                "binary_configurations": len(control_configs), "aggregate_response_descends_B": True},
            "failure_witness": {"kernel_vector": list(map(int, w)),
                "B_L_w": list(map(int, B*Lbad*w))},
            "closure": {"constraint_algebras": "centralizer intersections in so(8)",
                        "local_algebras": "abelian subalgebras; all brackets checked"}}


if __name__ == "__main__":
    if "--worker" in sys.argv:
        print(json.dumps(worker()))
    else:
        before = hashes()
        started = time.perf_counter()
        run = subprocess.run([sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()),
                              "--worker"], capture_output=True, text=True, encoding="utf-8",
                             timeout=20)
        if run.returncode:
            raise RuntimeError(run.stderr)
        results = json.loads(run.stdout)
        results.update(seconds=round(time.perf_counter()-started, 3),
                       python=platform.python_version(), timeout_seconds=20)
        assert hashes() == before, "Recorded input changed during verification"
        results["recorded_inputs_unchanged"] = True
        (HERE/"results.json").write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")
        commits = {}
        for repo in ("rcf", "rcmt"):
            path = ROOT/repo
            commits[repo] = subprocess.check_output(["git", "-c", f"safe.directory={path.as_posix()}",
                "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        manifest = {"date": "2026-10-24", "commits": commits, "source_sha256": before,
            "non_repository_inputs": "Copied OPS research files and prior independent Codex report/results",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (HERE/"source-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(results, indent=2))
