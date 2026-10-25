"""Assignment 001: bounded exact matrix checks; no historical scripts executed.

Run from theory/: & C:/Users/blaur/anaconda3/envs/glyphnet/python.exe -X utf8 -B
    projects/ops-codex/research/assignment-001/verify.py
The parent enforces a 15-second limit on the symbolic worker.
"""
from pathlib import Path
import ast
import hashlib
import json
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCES = [
    "rcf/core/relational-constraints/geometry.md",
    "rcmt/local/constraints/symmetry_chain.md",
    "rcmt/local/constraints/mode_space.md",
    "rcmt/local/constraints/projection.md",
    "projects/Operational_Phase_Space/V4_Orbit-Bridge_Regularity_Theorem.md",
    "projects/Operational_Phase_Space/SIM14_Discussions.md",
    "projects/Operational_Phase_Space/SIM15_Discussions.md",
    "projects/Operational_Phase_Space/SIM14.3.py",
    "projects/Operational_Phase_Space/SIM15.2.py",
    "projects/Operational_Phase_Space/SIM16.0.py",
    "projects/Operational_Phase_Space/SIM16.1.py",
    "projects/Operational_Phase_Space/SIM16.4.py",
    "projects/Operational_Phase_Space/recorded_output/SIM14.3.txt",
    "projects/Operational_Phase_Space/recorded_output/SIM15.2.txt",
    "projects/Operational_Phase_Space/recorded_output/SIM15.4.txt",
    "projects/Operational_Phase_Space/recorded_output/SIM15.6.txt",
    "projects/ops-codex/implementation/ops-repair-verification.json",
    "projects/ops-codex/implementation/SIM16.4.py.stdout.txt",
]


def hashes():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES}


def worker():
    import sympy as sp

    # Read only literal assignments, without importing/executing SIM16.4.
    tree = ast.parse((ROOT / "projects/Operational_Phase_Space/SIM16.4.py")
                     .read_text(encoding="utf-8-sig"))
    frozen = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {"PAIRS", "EDGES", "p", "h", "s"}:
                frozen[target.id] = ast.literal_eval(node.value)
    pairs, edges = frozen["PAIRS"], frozen["EDGES"]
    omega = sp.zeros(8)
    for a, b in pairs:
        omega[a, b], omega[b, a] = 1, -1
    assert omega**2 == -sp.eye(8) and omega.det() == 1

    def rotation(n, a, b):
        x = sp.zeros(n)
        x[a, b], x[b, a] = 1, -1
        return x

    def so(n):
        return [rotation(n, a, b) for a in range(n) for b in range(a + 1, n)]

    def kernel(basis, maps):
        columns = [sp.Matrix([e for f in maps for e in f(x)]) for x in basis]
        coefficients = sp.Matrix.hstack(*columns).nullspace()
        return [sum((x*c for x, c in zip(basis, v)), sp.zeros(*basis[0].shape))
                for v in coefficients]

    def comm(x, y):
        return x*y-y*x

    b8, b9 = so(8), so(9)
    projection = sp.eye(9)[:8, :]
    inclusion = projection.T
    axis = sp.eye(9)[:, 8]
    compress = lambda x: projection*x*inclusion
    descended = kernel(b9, [lambda x: projection*x*axis])
    assert len(descended) == 28
    assert all(x*axis == sp.zeros(9, 1) for x in descended)
    assert all(compress(x).T == -compress(x) for x in descended)
    # An explicit bracket witness, using an actual OPS conjugate pair (0,2).
    x, y = rotation(9, 0, 8), rotation(9, 2, 8)
    defect = compress(comm(x, y))-comm(compress(x), compress(y))
    assert compress(x) == compress(y) == sp.zeros(8)
    assert defect == -rotation(8, 0, 2) and defect.rank() == 2
    assert projection*x*axis == sp.eye(8)[:, 0]
    assert all(projection*rotation(9, i, 8)*axis == sp.eye(8)[:, i]
               for i in range(8))
    # The full SO(8) block has exact linear descent, including brackets.
    embed = lambda a: sp.diag(a, sp.zeros(1))
    assert all(projection*embed(a) == a*projection for a in b8)
    assert all(compress(comm(embed(a), embed(b))) == comm(a, b)
               for a, b in zip(b8, reversed(b8)))

    symplectic = lambda x: x.T*omega+omega*x
    full_compatible = kernel(b8, [symplectic])
    local_compatible = kernel([rotation(8, a, b) for a, b in edges], [symplectic])
    omega9 = sp.diag(omega, sp.zeros(1))
    extended_compatible = kernel(b9, [lambda x: x.T*omega9+omega9*x])
    assert len(full_compatible) == len(extended_compatible) == 16
    assert len(local_compatible) == 4
    expected = [rotation(8, a, b) for a, b in pairs]
    assert sp.Matrix.hstack(*(sp.Matrix(list(a)) for a in local_compatible+expected)).rank() == 4
    assert all(comm(a, b) == sp.zeros(8) for a in expected for b in expected)

    # Negative control: a direct-product action on the same R8 must commute.
    local_sp = []
    for a, b in pairs:
        upper, lower, diagonal = sp.zeros(8), sp.zeros(8), sp.zeros(8)
        upper[a, b], lower[b, a] = 1, 1
        diagonal[a, a], diagonal[b, b] = 1, -1
        local_sp.extend([upper, lower, diagonal])
    assert all(symplectic(a) == sp.zeros(8) for a in local_sp)
    commuting_orthogonal = kernel(b8, [lambda x, a=a: comm(x, a) for a in local_sp])
    assert len(commuting_orthogonal) == 0

    # Orbit-bridge control: both ambient groups contain the same finite action.
    identity = tuple(range(8))
    compose = lambda a, b: tuple(a[b[i]] for i in range(8))
    group, pending = {identity}, [identity]
    while pending:
        g = pending.pop()
        for name in ("p", "h", "s"):
            q = compose(frozen[name], g)
            if q not in group:
                group.add(q)
                pending.append(q)
    determinants = []
    for g in group:
        matrix = sp.zeros(8)
        for i, j in enumerate(g):
            matrix[j, i] = 1
        determinants.append(int(matrix.det()))
        assert matrix.T*matrix == sp.eye(8)
    assert len(group) == len({g[0] for g in group}) == 8
    assert determinants == [1]*8
    return {
        "status": "PASS", "sympy": sp.__version__, "exact_arithmetic": True,
        "frozen": frozen,
        "standard_SO9_to_R8": {
            "descendable_lie_algebra_dimension": len(descended),
            "extra_generators_failing_kernel_invariance": 8,
            "bracket_defect_nonzero_entries": [[i, j, int(defect[i, j])]
                for i in range(8) for j in range(8) if defect[i, j]],
            "bracket_defect_rank": int(defect.rank()),
            "standard_SO8_block_descent": True,
        },
        "fixed_form_controls": {
            "so8_preserving_omega_dimension": len(full_compatible),
            "so9_preserving_omega_plus_zero_dimension": len(extended_compatible),
            "C8_sparse_so8_preserving_omega_dimension": len(local_compatible),
            "local_rotation_pairs": pairs, "local_rotation_algebra_abelian": True,
        },
        "same_R8_direct_product_control": {
            "local_sp_generators": len(local_sp),
            "commuting_so8_dimension": len(commuting_orthogonal),
        },
        "orbit_bridge_control": {"order": len(group), "regular": True,
            "all_determinants_plus_one": True,
            "same_action_embeds_in_SO8_and_axis_fixing_SO9": True},
    }


if __name__ == "__main__":
    if "--worker" in sys.argv:
        print(json.dumps(worker()))
    else:
        before = hashes()
        started = time.perf_counter()
        run = subprocess.run([sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()),
                              "--worker"], capture_output=True, text=True,
                             encoding="utf-8", timeout=15, check=True)
        result = json.loads(run.stdout)
        result.update(seconds=round(time.perf_counter()-started, 3),
                      python=platform.python_version(), worker_timeout_seconds=15)
        after = hashes()
        assert before == after, "A recorded input changed during the check"
        result["recorded_inputs_unchanged"] = True
        (HERE / "results.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        commits = {}
        for repo in ("rcf", "rcmt"):
            path = ROOT / repo
            git = ["git", "-c", f"safe.directory={path.as_posix()}", "-C", str(path)]
            commits[repo] = subprocess.check_output(git+["rev-parse", "HEAD"], text=True).strip()
        manifest = {"date": "2026-10-24", "commits": commits,
                    "non_repository_inputs": "OPS copied research record and prior Codex artifacts",
                    "source_sha256": before,
                    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (HERE / "source-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
