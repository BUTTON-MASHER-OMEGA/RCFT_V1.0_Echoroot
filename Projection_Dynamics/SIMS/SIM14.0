"""
SIM14.0 — SYMPLECTIC CARRIER CONSTRUCTION PROBE
================================================

Purpose
-------
Test whether the frozen 8-state C8 projection laboratory admits a clean,
non-arbitrary Sp(8,R)-type conjugate-pair realization BEFORE ISP/LCO
dynamics are introduced.

This is a carrier-validation calculation, not a dynamical simulation.

Frozen C8 architecture
----------------------
neighbors = {
    0: [1, 2],
    1: [0, 3],
    2: [0, 4],
    3: [1, 5],
    4: [2, 6],
    5: [3, 7],
    6: [4, 7],
    7: [5, 6],
}

Projection pairing:
    p = (01)(23)(45)(67)

Intrinsic graph half-turn:
    h = (07)(16)(25)(34)

SIM14.0 asks
------------
1. What does canonical symplectic conjugacy look like on eight states?
2. Can C8 adjacency itself be interpreted as canonical symplectic pairing?
3. How much of C8 is recovered by |Omega| alone?
4. What information is lost by passing from signed Omega to unsigned adjacency?
5. How do projection pairing p, graph half-turn h, and symplectic
   conjugation s_Omega compose?
6. What subgroup of S8 do they generate?
7. Which of those permutations are actual automorphisms of the frozen C8?
8. Is the construction robust to canonical relabelings?
9. Does an Sp(8,R) transformation preserve the pairing matrix exactly?

Hard methodological rule
------------------------
No optimizer is allowed to tune vectors to reproduce C8.

The primary embedding is the canonical symplectic basis itself.
A permutation scan is allowed ONLY as a classification diagnostic:
it asks whether some relabeling of that already-frozen basis can make
canonical symplectic conjugacy coincide with a C8 perfect matching.

No ISP.
No LCO.
No Fano constraint.
No H4.
No field.
No Dirac interpretation.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter, deque

import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

N = 8
TOL = 1.0e-10
RNG_SEED = 1400
N_SP8_TESTS = 250

NEIGHBORS = {
    0: [1, 2],
    1: [0, 3],
    2: [0, 4],
    3: [1, 5],
    4: [2, 6],
    5: [3, 7],
    6: [4, 7],
    7: [5, 6],
}

PROJ_STRUCTURED = {
    0: 0,
    1: 0,
    2: 1,
    3: 1,
    4: 2,
    5: 2,
    6: 3,
    7: 3,
}

DEFAULT_BASINS = {
    0: [0, 1],
    1: [2, 3],
    2: [4, 5],
    3: [6, 7],
}

# Permutations are represented by tuples:
# perm[i] = image of vertex i.

IDENTITY = tuple(range(N))

P_PROJECTION = (
    1, 0,
    3, 2,
    5, 4,
    7, 6,
)

H_GRAPH = (
    7, 6,
    5, 4,
    3, 2,
    1, 0,
)


# =============================================================================
# BASIC HELPERS
# =============================================================================

def banner(title: str, width: int = 88) -> None:
    print()
    print("=" * width)
    print(title)
    print("=" * width)


def section(title: str, width: int = 88) -> None:
    print()
    print(title)
    print("-" * width)


def matrix_string(matrix: np.ndarray, precision: int = 3) -> str:
    return np.array2string(
        matrix,
        precision=precision,
        suppress_small=True,
        max_line_width=140,
    )


def edge_set_from_neighbors(neighbors: dict[int, list[int]]) -> frozenset:
    edges = set()

    for i, js in neighbors.items():
        for j in js:
            if i == j:
                raise ValueError("Self-edge found in frozen C8.")
            edges.add(tuple(sorted((i, j))))

    return frozenset(edges)


def adjacency_matrix(edges: frozenset, n: int = N) -> np.ndarray:
    adjacency = np.zeros((n, n), dtype=int)

    for i, j in edges:
        adjacency[i, j] = 1
        adjacency[j, i] = 1

    return adjacency


def degree_sequence(edges: frozenset, n: int = N) -> tuple[int, ...]:
    degrees = [0] * n

    for i, j in edges:
        degrees[i] += 1
        degrees[j] += 1

    return tuple(degrees)


def format_edges(edges: frozenset) -> str:
    ordered = sorted(edges)
    return "{" + ", ".join(f"{i}-{j}" for i, j in ordered) + "}"


def compose(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    """
    Composition a o b:
        first apply b, then apply a.
    """
    return tuple(a[b[i]] for i in range(N))


def inverse_perm(p: tuple[int, ...]) -> tuple[int, ...]:
    inv = [0] * N

    for i, image in enumerate(p):
        inv[image] = i

    return tuple(inv)


def commutator(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    """
    Group commutator:
        a b a^{-1} b^{-1}
    """
    return compose(
        compose(
            compose(a, b),
            inverse_perm(a),
        ),
        inverse_perm(b),
    )


def commute(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    return compose(a, b) == compose(b, a)


def permutation_order(p: tuple[int, ...]) -> int:
    current = IDENTITY

    for order in range(1, 1000):
        current = compose(p, current)
        if current == IDENTITY:
            return order

    raise RuntimeError("Permutation order exceeded safety bound.")


def permutation_cycles(p: tuple[int, ...]) -> str:
    seen = set()
    cycles = []

    for start in range(N):
        if start in seen:
            continue

        cycle = []
        current = start

        while current not in seen:
            seen.add(current)
            cycle.append(current)
            current = p[current]

        if len(cycle) > 1:
            cycles.append("(" + " ".join(map(str, cycle)) + ")")

    if not cycles:
        return "()"

    return "".join(cycles)


def generate_group(generators: list[tuple[int, ...]]) -> set[tuple[int, ...]]:
    group = {IDENTITY}
    queue = deque([IDENTITY])

    while queue:
        current = queue.popleft()

        for generator in generators:
            candidate = compose(generator, current)

            if candidate not in group:
                group.add(candidate)
                queue.append(candidate)

    return group


def is_graph_automorphism(
    p: tuple[int, ...],
    edges: frozenset,
) -> bool:
    mapped = {
        tuple(sorted((p[i], p[j])))
        for i, j in edges
    }
    return frozenset(mapped) == edges


def induced_edge_overlap(
    source_edges: frozenset,
    target_edges: frozenset,
) -> tuple[int, float, float]:
    intersection = len(source_edges & target_edges)
    union = len(source_edges | target_edges)

    recall = intersection / len(target_edges) if target_edges else math.nan
    jaccard = intersection / union if union else math.nan

    return intersection, recall, jaccard


def perfect_matching_from_perm(
    p: tuple[int, ...],
) -> frozenset:
    edges = set()

    for i in range(N):
        j = p[i]

        if i != j:
            edges.add(tuple(sorted((i, j))))

    return frozenset(edges)


# =============================================================================
# FROZEN C8 VALIDATION
# =============================================================================

C8_EDGES = edge_set_from_neighbors(NEIGHBORS)
C8_ADJ = adjacency_matrix(C8_EDGES)


def validate_c8() -> None:
    assert len(C8_EDGES) == 8
    assert degree_sequence(C8_EDGES) == (2,) * N

    for i, js in NEIGHBORS.items():
        for j in js:
            assert i in NEIGHBORS[j]

    assert set(PROJ_STRUCTURED) == set(range(N))

    basin_vertices = []
    for basin in sorted(DEFAULT_BASINS):
        basin_vertices.extend(DEFAULT_BASINS[basin])

    assert sorted(basin_vertices) == list(range(N))


# =============================================================================
# CANONICAL SYMPLECTIC STRUCTURE
# =============================================================================

def canonical_omega() -> np.ndarray:
    """
    Coordinate order:
        (q1, q2, q3, q4, p1, p2, p3, p4)

    Omega =
        [ 0   I ]
        [-I   0 ]
    """
    identity4 = np.eye(4)
    zero4 = np.zeros((4, 4))

    return np.block(
        [
            [zero4, identity4],
            [-identity4, zero4],
        ]
    )


OMEGA = canonical_omega()


def canonical_basis_embedding() -> np.ndarray:
    """
    Row i is the vector assigned to microscopic state i.

    No fitting:
        v_0 = q1
        v_1 = q2
        v_2 = q3
        v_3 = q4
        v_4 = p1
        v_5 = p2
        v_6 = p3
        v_7 = p4
    """
    return np.eye(N)


V_CANONICAL = canonical_basis_embedding()


def symplectic_pairing_matrix(
    vectors: np.ndarray,
    omega: np.ndarray,
) -> np.ndarray:
    return vectors @ omega @ vectors.T


S_CANONICAL = symplectic_pairing_matrix(
    V_CANONICAL,
    OMEGA,
)


def validate_symplectic_form() -> None:
    antisymmetry_error = np.max(np.abs(OMEGA.T + OMEGA))
    determinant = np.linalg.det(OMEGA)
    rank = np.linalg.matrix_rank(OMEGA)

    assert antisymmetry_error < TOL
    assert abs(determinant) > TOL
    assert rank == N


def signed_support_edges(
    pairing: np.ndarray,
    threshold: float = 0.5,
) -> frozenset:
    edges = set()

    for i in range(N):
        for j in range(i + 1, N):
            if abs(pairing[i, j]) >= threshold:
                edges.add((i, j))

    return frozenset(edges)


def canonical_symplectic_conjugation_perm() -> tuple[int, ...]:
    """
    q_i <-> p_i in canonical coordinate order.
    """
    return (
        4, 5, 6, 7,
        0, 1, 2, 3,
    )


S_OMEGA_CANONICAL = canonical_symplectic_conjugation_perm()
OMEGA_SUPPORT = signed_support_edges(S_CANONICAL)


# =============================================================================
# C8 CYCLE COORDINATES
# =============================================================================

def recover_cycle_order(
    neighbors: dict[int, list[int]],
    start: int = 0,
    first_neighbor: int = 1,
) -> tuple[int, ...]:
    order = [start]
    previous = None
    current = start
    chosen_next = first_neighbor

    while True:
        if previous is None:
            nxt = chosen_next
        else:
            candidates = [
                vertex
                for vertex in neighbors[current]
                if vertex != previous
            ]

            if not candidates:
                raise RuntimeError("Cycle traversal terminated unexpectedly.")

            nxt = candidates[0]

        if nxt == start:
            break

        if nxt in order:
            raise RuntimeError("Graph traversal revisited a vertex early.")

        order.append(nxt)
        previous, current = current, nxt

    if len(order) != N:
        raise RuntimeError("Recovered traversal is not an 8-cycle.")

    return tuple(order)


CYCLE_ORDER = recover_cycle_order(
    NEIGHBORS,
    start=0,
    first_neighbor=1,
)


def graph_half_turn_from_cycle(
    cycle_order: tuple[int, ...],
) -> tuple[int, ...]:
    p = [0] * N

    for coordinate, vertex in enumerate(cycle_order):
        opposite = cycle_order[(coordinate + 4) % N]
        p[vertex] = opposite

    return tuple(p)


H_FROM_CYCLE = graph_half_turn_from_cycle(CYCLE_ORDER)


# =============================================================================
# CANONICAL RELABELING CLASSIFICATION
# =============================================================================

def conjugate_perm_by_labeling(
    canonical_perm: tuple[int, ...],
    labeling: tuple[int, ...],
) -> tuple[int, ...]:
    """
    labeling[k] = graph vertex receiving canonical basis vector k.

    Push canonical permutation forward into graph-label coordinates:
        labeling o canonical_perm o labeling^{-1}
    """
    inverse_labeling = inverse_perm(labeling)

    return compose(
        compose(labeling, canonical_perm),
        inverse_labeling,
    )


def scan_basis_relabelings() -> dict:
    """
    Exhaustively scan 8! = 40320 relabelings of the already-frozen
    canonical symplectic basis.

    This is NOT vector fitting.

    It classifies whether canonical q_i <-> p_i conjugacy can be placed
    onto a perfect matching contained in C8, and how many symmetry-
    equivalent ways this can occur.
    """
    overlap_histogram = Counter()
    best_overlap = -1
    best_labelings = []
    exact_labelings = []
    unique_best_matchings = set()

    for labeling in itertools.permutations(range(N)):
        pushed = conjugate_perm_by_labeling(
            S_OMEGA_CANONICAL,
            labeling,
        )

        matching = perfect_matching_from_perm(pushed)
        overlap = len(matching & C8_EDGES)

        overlap_histogram[overlap] += 1

        if overlap > best_overlap:
            best_overlap = overlap
            best_labelings = [labeling]
            unique_best_matchings = {matching}

        elif overlap == best_overlap:
            best_labelings.append(labeling)
            unique_best_matchings.add(matching)

        if matching.issubset(C8_EDGES):
            exact_labelings.append(labeling)

    return {
        "histogram": overlap_histogram,
        "best_overlap": best_overlap,
        "best_labelings": best_labelings,
        "exact_labelings": exact_labelings,
        "unique_best_matchings": unique_best_matchings,
    }


# =============================================================================
# GRAPH AUTOMORPHISMS
# =============================================================================

def all_graph_automorphisms() -> list[tuple[int, ...]]:
    automorphisms = []

    for p in itertools.permutations(range(N)):
        if is_graph_automorphism(p, C8_EDGES):
            automorphisms.append(p)

    return automorphisms


# =============================================================================
# SP(8,R) GENERATION AND INVARIANCE TEST
# =============================================================================

def random_hamiltonian_generator(
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Every X in sp(8,R) can be written as:
        X = -Omega @ H
    for symmetric H, because canonical Omega^{-1} = -Omega.

    Then:
        X^T Omega + Omega X = 0.
    """
    raw = rng.normal(size=(N, N))
    hessian = 0.5 * (raw + raw.T)

    generator = -OMEGA @ hessian

    return generator


def matrix_exponential_symmetric_hamiltonian(
    generator: np.ndarray,
    scale: float,
) -> np.ndarray:
    """
    Compute exp(scale * generator) through eigendecomposition.

    scipy is deliberately avoided so the script remains portable
    in lightweight GPT console environments.
    """
    values, vectors = np.linalg.eig(scale * generator)
    inverse_vectors = np.linalg.inv(vectors)

    exponential = (
        vectors
        @ np.diag(np.exp(values))
        @ inverse_vectors
    )

    return np.real_if_close(exponential, tol=1000).real


def sp8_invariance_trials(
    n_trials: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)

    generator_errors = []
    symplectic_errors = []
    pairing_errors = []
    determinant_errors = []

    for _ in range(n_trials):
        generator = random_hamiltonian_generator(rng)

        generator_error = np.max(
            np.abs(
                generator.T @ OMEGA
                + OMEGA @ generator
            )
        )
        generator_errors.append(generator_error)

        # Keep scale moderate for numerical stability.
        scale = rng.uniform(0.01, 0.20)

        transformation = matrix_exponential_symmetric_hamiltonian(
            generator,
            scale,
        )

        symplectic_error = np.max(
            np.abs(
                transformation.T
                @ OMEGA
                @ transformation
                - OMEGA
            )
        )
        symplectic_errors.append(symplectic_error)

        transformed_vectors = (
            V_CANONICAL
            @ transformation.T
        )

        transformed_pairing = symplectic_pairing_matrix(
            transformed_vectors,
            OMEGA,
        )

        pairing_error = np.max(
            np.abs(
                transformed_pairing
                - S_CANONICAL
            )
        )
        pairing_errors.append(pairing_error)

        determinant_error = abs(
            np.linalg.det(transformation) - 1.0
        )
        determinant_errors.append(determinant_error)

    return {
        "max_generator_error": max(generator_errors),
        "max_symplectic_error": max(symplectic_errors),
        "max_pairing_error": max(pairing_errors),
        "max_determinant_error": max(determinant_errors),
    }


# =============================================================================
# ALGEBRA REPORTING
# =============================================================================

def report_pairwise_permutation_relation(
    name_a: str,
    a: tuple[int, ...],
    name_b: str,
    b: tuple[int, ...],
) -> None:
    print(
        f"{name_a:>12s} with {name_b:<12s}: "
        f"commute={str(commute(a, b)):<5s}  "
        f"commutator={permutation_cycles(commutator(a, b))}"
    )


def group_order_histogram(
    group: set[tuple[int, ...]],
) -> Counter:
    return Counter(
        permutation_order(element)
        for element in group
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    validate_c8()
    validate_symplectic_form()

    if H_FROM_CYCLE != H_GRAPH:
        raise RuntimeError(
            "Frozen H_GRAPH does not equal intrinsic C8 half-turn."
        )

    banner(
        "SIM14.0 — SYMPLECTIC CARRIER CONSTRUCTION PROBE"
    )

    print("NO ISP / LCO dynamics are run in SIM14.0.")
    print("Primary carrier: canonical R^8 symplectic basis.")
    print("No vector optimizer or C8-fitting procedure is permitted.")
    print()
    print("Frozen microscopic states :", tuple(range(N)))
    print("Frozen C8 cycle order      :", CYCLE_ORDER)
    print("Frozen projection basins   :", DEFAULT_BASINS)
    print("C8 edge count              :", len(C8_EDGES))
    print("C8 degree sequence         :", degree_sequence(C8_EDGES))

    # -------------------------------------------------------------------------
    section("1. FROZEN C8 ARCHITECTURE")

    print("C8 edges:")
    print(format_edges(C8_EDGES))
    print()
    print("Adjacency matrix:")
    print(matrix_string(C8_ADJ, precision=0))
    print()
    print(
        "Projection pairing p :",
        permutation_cycles(P_PROJECTION),
    )
    print(
        "Graph half-turn h    :",
        permutation_cycles(H_GRAPH),
    )
    print(
        "Recovered from cycle :",
        permutation_cycles(H_FROM_CYCLE),
    )

    # -------------------------------------------------------------------------
    section("2. CANONICAL Sp(8,R)-TYPE CARRIER")

    print(
        "Coordinate order: "
        "(q1, q2, q3, q4, p1, p2, p3, p4)"
    )
    print()
    print("Canonical Omega:")
    print(matrix_string(OMEGA))
    print()
    print(
        "max |Omega^T + Omega| :",
        f"{np.max(np.abs(OMEGA.T + OMEGA)):.3e}",
    )
    print(
        "rank(Omega)           :",
        np.linalg.matrix_rank(OMEGA),
    )
    print(
        "det(Omega)            :",
        f"{np.linalg.det(OMEGA):.6f}",
    )

    print()
    print("Canonical state-vector embedding V:")
    print(matrix_string(V_CANONICAL, precision=0))

    print()
    print("Signed symplectic pairing S_ij = v_i^T Omega v_j:")
    print(matrix_string(S_CANONICAL, precision=0))

    print()
    print(
        "Canonical conjugation s_Omega :",
        permutation_cycles(S_OMEGA_CANONICAL),
    )
    print(
        "Unsigned |Omega| support       :",
        format_edges(OMEGA_SUPPORT),
    )

    # -------------------------------------------------------------------------
    section("3. DOES CANONICAL |Omega| REPRODUCE C8?")

    intersection, recall, jaccard = induced_edge_overlap(
        OMEGA_SUPPORT,
        C8_EDGES,
    )

    print("Canonical |Omega| edges :", format_edges(OMEGA_SUPPORT))
    print("Frozen C8 edges         :", format_edges(C8_EDGES))
    print()
    print(
        "Shared edges           :",
        f"{intersection}/{len(C8_EDGES)}",
    )
    print(
        "C8 edge recall         :",
        f"{recall:.6f}",
    )
    print(
        "Jaccard similarity     :",
        f"{jaccard:.6f}",
    )
    print()
    print(
        "Interpretation: the raw canonical coordinate assignment is "
        "NOT allowed to be tuned."
    )

    # -------------------------------------------------------------------------
    section("4. CANONICAL-BASIS RELABELING CLASSIFICATION")

    print(
        "Scanning all 8! = 40320 graph-label assignments of the "
        "same frozen canonical basis..."
    )

    scan = scan_basis_relabelings()

    print()
    print("Overlap histogram:")
    for overlap in sorted(scan["histogram"]):
        count = scan["histogram"][overlap]
        print(
            f"  {overlap}/4 symplectic matching edges inside C8 : "
            f"{count:5d} labelings"
        )

    print()
    print(
        "Maximum matching-edge overlap :",
        f"{scan['best_overlap']}/4",
    )
    print(
        "Number of best labelings      :",
        len(scan["best_labelings"]),
    )
    print(
        "Exact C8-contained labelings  :",
        len(scan["exact_labelings"]),
    )
    print(
        "Distinct best matchings       :",
        len(scan["unique_best_matchings"]),
    )

    best_matchings_sorted = sorted(
        scan["unique_best_matchings"],
        key=lambda edges: sorted(edges),
    )

    print()
    print("Distinct maximum-overlap conjugate matchings:")

    for index, matching in enumerate(
        best_matchings_sorted,
        start=1,
    ):
        print(
            f"  M{index}: {format_edges(matching)}"
        )

    if scan["exact_labelings"]:
        representative_labeling = scan["exact_labelings"][0]
    else:
        representative_labeling = scan["best_labelings"][0]

    S_OMEGA_GRAPH = conjugate_perm_by_labeling(
        S_OMEGA_CANONICAL,
        representative_labeling,
    )

    representative_matching = perfect_matching_from_perm(
        S_OMEGA_GRAPH
    )

    print()
    print("Representative frozen labeling:")
    print(
        "  canonical basis index -> graph vertex =",
        representative_labeling,
    )
    print(
        "  induced s_Omega                  =",
        permutation_cycles(S_OMEGA_GRAPH),
    )
    print(
        "  induced matching                 =",
        format_edges(representative_matching),
    )

    # -------------------------------------------------------------------------
    section("5. PERMUTATION / COMMUTATION TABLE")

    permutations = {
        "p": P_PROJECTION,
        "h": H_GRAPH,
        "s_Omega": S_OMEGA_GRAPH,
    }

    for name, perm in permutations.items():
        print(
            f"{name:>8s}: "
            f"{permutation_cycles(perm):<22s} "
            f"order={permutation_order(perm)}  "
            f"C8 automorphism={is_graph_automorphism(perm, C8_EDGES)}"
        )

    print()

    report_pairwise_permutation_relation(
        "p",
        P_PROJECTION,
        "h",
        H_GRAPH,
    )
    report_pairwise_permutation_relation(
        "p",
        P_PROJECTION,
        "s_Omega",
        S_OMEGA_GRAPH,
    )
    report_pairwise_permutation_relation(
        "h",
        H_GRAPH,
        "s_Omega",
        S_OMEGA_GRAPH,
    )

    generated_group = generate_group(
        [
            P_PROJECTION,
            H_GRAPH,
            S_OMEGA_GRAPH,
        ]
    )

    print()
    print(
        "|<p, h, s_Omega>|       :",
        len(generated_group),
    )
    print(
        "Element-order histogram :",
        dict(sorted(group_order_histogram(generated_group).items())),
    )

    all_generated_are_graph_autos = all(
        is_graph_automorphism(element, C8_EDGES)
        for element in generated_group
    )

    print(
        "Entire generated group preserves C8:",
        all_generated_are_graph_autos,
    )

    # -------------------------------------------------------------------------
    section("6. FULL C8 AUTOMORPHISM GROUP")

    automorphisms = all_graph_automorphisms()

    print(
        "Number of C8 graph automorphisms:",
        len(automorphisms),
    )
    print(
        "Expected for an 8-cycle         :",
        16,
    )
    print(
        "Automorphism count PASS         :",
        len(automorphisms) == 16,
    )

    projection_is_auto = is_graph_automorphism(
        P_PROJECTION,
        C8_EDGES,
    )
    halfturn_is_auto = is_graph_automorphism(
        H_GRAPH,
        C8_EDGES,
    )
    symplectic_is_auto = is_graph_automorphism(
        S_OMEGA_GRAPH,
        C8_EDGES,
    )

    print()
    print(
        "p is C8 automorphism       :",
        projection_is_auto,
    )
    print(
        "h is C8 automorphism       :",
        halfturn_is_auto,
    )
    print(
        "s_Omega is C8 automorphism :",
        symplectic_is_auto,
    )

    # -------------------------------------------------------------------------
    section("7. Sp(8,R) INVARIANCE CONTROL")

    print(
        f"Running {N_SP8_TESTS} random Hamiltonian-generator "
        "invariance trials..."
    )

    sp8_results = sp8_invariance_trials(
        n_trials=N_SP8_TESTS,
        seed=RNG_SEED,
    )

    print()
    print(
        "max |X^T Omega + Omega X| :",
        f"{sp8_results['max_generator_error']:.3e}",
    )
    print(
        "max |M^T Omega M - Omega| :",
        f"{sp8_results['max_symplectic_error']:.3e}",
    )
    print(
        "max pairing-matrix error  :",
        f"{sp8_results['max_pairing_error']:.3e}",
    )
    print(
        "max |det(M)-1|            :",
        f"{sp8_results['max_determinant_error']:.3e}",
    )

    sp8_pass = (
        sp8_results["max_generator_error"] < 1.0e-8
        and sp8_results["max_symplectic_error"] < 1.0e-8
        and sp8_results["max_pairing_error"] < 1.0e-8
        and sp8_results["max_determinant_error"] < 1.0e-8
    )

    print()
    print(
        "Sp(8) numerical invariance PASS:",
        sp8_pass,
    )

    # -------------------------------------------------------------------------
    section("8. INFORMATION-LOSS LEDGER")

    signed_nonzero = {
        (i, j): S_CANONICAL[i, j]
        for i in range(N)
        for j in range(N)
        if abs(S_CANONICAL[i, j]) > TOL
    }

    positive_count = sum(
        value > 0
        for value in signed_nonzero.values()
    )
    negative_count = sum(
        value < 0
        for value in signed_nonzero.values()
    )

    print(
        "Signed nonzero ordered pairings :",
        len(signed_nonzero),
    )
    print(
        "Positive orientations           :",
        positive_count,
    )
    print(
        "Negative orientations           :",
        negative_count,
    )
    print(
        "Unsigned conjugate edges        :",
        len(OMEGA_SUPPORT),
    )

    print()
    print("Passing S -> |S| preserves:")
    print("  * which canonical states are symplectically conjugate")
    print("  * absolute pairing magnitude")
    print()
    print("Passing S -> |S| discards:")
    print("  * symplectic orientation")
    print("  * distinction Omega(i,j) = -Omega(j,i)")
    print()
    print("Passing |S| -> ordinary unweighted adjacency additionally discards:")
    print("  * pairing magnitude, if non-unit weights are later introduced")

    # -------------------------------------------------------------------------
    section("9. SIM14.0 TRUTH PACKET")

    exact_matching_exists = len(scan["exact_labelings"]) > 0

    print(
        "Canonical Omega valid                     :",
        True,
    )
    print(
        "Canonical conjugation is four 2-cycles    :",
        permutation_order(S_OMEGA_CANONICAL) == 2,
    )
    print(
        "Raw canonical |Omega| equals frozen C8    :",
        OMEGA_SUPPORT == C8_EDGES,
    )
    print(
        "A relabeling embeds conjugate matching in C8:",
        exact_matching_exists,
    )
    print(
        "Representative s_Omega preserves C8       :",
        symplectic_is_auto,
    )
    print(
        "Projection pairing p preserves C8         :",
        projection_is_auto,
    )
    print(
        "Graph half-turn h preserves C8            :",
        halfturn_is_auto,
    )
    print(
        "Sp(8) transformations preserve Omega      :",
        sp8_pass,
    )

    print()
    print("Classification rule:")
    print()
    print(
        "A. If no canonical-basis relabeling places all four "
        "symplectic conjugate pairs on C8 edges, then ordinary C8 "
        "adjacency is not even compatible with conjugacy as a "
        "perfect matching under this minimal construction."
    )
    print()
    print(
        "B. If such relabelings exist but many inequivalent choices "
        "remain, then C8 is compatible with symplectic conjugacy but "
        "does not uniquely determine it."
    )
    print()
    print(
        "C. If a conjugate matching is uniquely selected up to C8 "
        "automorphism, that is stronger structural evidence."
    )
    print()
    print(
        "D. If signed Omega later changes ISP/LCO behavior while "
        "|Omega| does not, orientation carries information that the "
        "ordinary C8 graph suppresses."
    )
    print()
    print(
        "E. SIM14.0 alone makes NO physical claim about RCFT, "
        "Fano structure, H4, spinors, or quantum mechanics."
    )

    # -------------------------------------------------------------------------
    banner("SIM14.0 COMPLETE")

    if not exact_matching_exists:
        status = (
            "OBSTRUCTION: canonical symplectic conjugacy cannot be "
            "embedded as a four-edge matching of frozen C8."
        )
    elif exact_matching_exists and len(
        scan["unique_best_matchings"]
    ) > 1:
        status = (
            "COMPATIBLE BUT NON-UNIQUE: C8 admits canonical "
            "symplectic conjugate matchings, but C8 alone does not "
            "select a unique one."
        )
    else:
        status = (
            "STRUCTURALLY SELECTIVE: the canonical construction "
            "produces a unique conjugate matching up to the tested "
            "classification."
        )

    print("STATUS:")
    print(status)
    print()
    print(
        "Next allowed question: SIM14.1 may compare the frozen "
        "ordinary C8 carrier against a frozen symplectic surrogate "
        "without changing ISP/LCO for any other reason."
    )


if __name__ == "__main__":
    main()






~~~~~~~~~~~~~~~~~~~~~~~~~~~







RESULTS:




134 |

========================================================================================

SIM14.0 — SYMPLECTIC CARRIER CONSTRUCTION PROBE

========================================================================================

NO ISP / LCO dynamics are run in SIM14.0.

Primary carrier: canonical R^8 symplectic basis.

No vector optimizer or C8-fitting procedure is permitted.

754 |

Frozen microscopic states : (0, 1, 2, 3, 4, 5, 6, 7)

Frozen C8 cycle order      : (0, 1, 3, 5, 7, 6, 4, 2)

Frozen projection basins   : {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7]}

C8 edge count              : 8

C8 degree sequence         : (2, 2, 2, 2, 2, 2, 2, 2)

141 |

1. FROZEN C8 ARCHITECTURE

----------------------------------------------------------------------------------------

C8 edges:

{0-1, 0-2, 1-3, 2-4, 3-5, 4-6, 5-7, 6-7}

766 |

Adjacency matrix:

[[0 1 1 0 0 0 0 0]
 [1 0 0 1 0 0 0 0]
 [1 0 0 0 1 0 0 0]
 [0 1 0 0 0 1 0 0]
 [0 0 1 0 0 0 1 0]
 [0 0 0 1 0 0 0 1]
 [0 0 0 0 1 0 0 1]
 [0 0 0 0 0 1 1 0]]
769 |

Projection pairing p : (0 1)(2 3)(4 5)(6 7)

Graph half-turn h    : (0 7)(1 6)(2 5)(3 4)

Recovered from cycle : (0 7)(1 6)(2 5)(3 4)

141 |

2. CANONICAL Sp(8,R)-TYPE CARRIER

----------------------------------------------------------------------------------------

Coordinate order: (q1, q2, q3, q4, p1, p2, p3, p4)

790 |

Canonical Omega:

[[ 0.  0.  0.  0.  1.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  1.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  1.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  1.]
 [-1. -0. -0. -0.  0.  0.  0.  0.]
 [-0. -1. -0. -0.  0.  0.  0.  0.]
 [-0. -0. -1. -0.  0.  0.  0.  0.]
 [-0. -0. -0. -1.  0.  0.  0.  0.]]
793 |

max |Omega^T + Omega| : 0.000e+00

rank(Omega)           : 8

det(Omega)            : 1.000000

807 |

Canonical state-vector embedding V:

[[1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1.]]
811 |

Signed symplectic pairing S_ij = v_i^T Omega v_j:

[[ 0.  0.  0.  0.  1.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  1.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  1.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  1.]
 [-1.  0.  0.  0.  0.  0.  0.  0.]
 [ 0. -1.  0.  0.  0.  0.  0.  0.]
 [ 0.  0. -1.  0.  0.  0.  0.  0.]
 [ 0.  0.  0. -1.  0.  0.  0.  0.]]
815 |

Canonical conjugation s_Omega : (0 4)(1 5)(2 6)(3 7)

Unsigned |Omega| support       : {0-4, 1-5, 2-6, 3-7}

141 |

3. DOES CANONICAL |Omega| REPRODUCE C8?

----------------------------------------------------------------------------------------

Canonical |Omega| edges : {0-4, 1-5, 2-6, 3-7}

Frozen C8 edges         : {0-1, 0-2, 1-3, 2-4, 3-5, 4-6, 5-7, 6-7}

835 |

Shared edges           : 0/8

C8 edge recall         : 0.000000

Jaccard similarity     : 0.000000

848 |

Interpretation: the raw canonical coordinate assignment is NOT allowed to be tuned.

141 |

4. CANONICAL-BASIS RELABELING CLASSIFICATION

----------------------------------------------------------------------------------------

Scanning all 8! = 40320 graph-label assignments of the same frozen canonical basis...

864 |

Overlap histogram:

  0/4 symplectic matching edges inside C8 : 11904 labelings

  1/4 symplectic matching edges inside C8 : 15360 labelings

  2/4 symplectic matching edges inside C8 :  9216 labelings

  3/4 symplectic matching edges inside C8 :  3072 labelings

  4/4 symplectic matching edges inside C8 :   768 labelings

873 |

Maximum matching-edge overlap : 4/4

Number of best labelings      : 768

Exact C8-contained labelings  : 768

Distinct best matchings       : 2

896 |

Distinct maximum-overlap conjugate matchings:

  M1: {0-1, 2-4, 3-5, 6-7}

  M2: {0-2, 1-3, 4-6, 5-7}

921 |

Representative frozen labeling:

  canonical basis index -> graph vertex = (0, 1, 4, 5, 2, 3, 6, 7)

  induced s_Omega                  = (0 2)(1 3)(4 6)(5 7)

  induced matching                 = {0-2, 1-3, 4-6, 5-7}

141 |

5. PERMUTATION / COMMUTATION TABLE

----------------------------------------------------------------------------------------

       p: (0 1)(2 3)(4 5)(6 7)   order=2  C8 automorphism=True

       h: (0 7)(1 6)(2 5)(3 4)   order=2  C8 automorphism=True

 s_Omega: (0 2)(1 3)(4 6)(5 7)   order=2  C8 automorphism=False

953 |

           p with h           : commute=True   commutator=()

           p with s_Omega     : commute=True   commutator=()

           h with s_Omega     : commute=True   commutator=()

982 |

|<p, h, s_Omega>|       : 8

Element-order histogram : {1: 1, 2: 7}

Entire generated group preserves C8: False

141 |

6. FULL C8 AUTOMORPHISM GROUP

----------------------------------------------------------------------------------------

Number of C8 graph automorphisms: 16

Expected for an 8-cycle         : 16

Automorphism count PASS         : True

1033 |

p is C8 automorphism       : True

h is C8 automorphism       : True

s_Omega is C8 automorphism : False

141 |

7. Sp(8,R) INVARIANCE CONTROL

----------------------------------------------------------------------------------------

Running 250 random Hamiltonian-generator invariance trials...

1060 |

max |X^T Omega + Omega X| : 0.000e+00

max |M^T Omega M - Omega| : 5.519e-15

max pairing-matrix error  : 5.519e-15

max |det(M)-1|            : 4.441e-15

1085 |

Sp(8) numerical invariance PASS: True

141 |

8. INFORMATION-LOSS LEDGER

----------------------------------------------------------------------------------------

Signed nonzero ordered pairings : 8

Positive orientations           : 4

Negative orientations           : 4

Unsigned conjugate edges        : 4

1127 |

Passing S -> |S| preserves:

  * which canonical states are symplectically conjugate

  * absolute pairing magnitude

1131 |

Passing S -> |S| discards:

  * symplectic orientation

  * distinction Omega(i,j) = -Omega(j,i)

1135 |

Passing |S| -> ordinary unweighted adjacency additionally discards:

  * pairing magnitude, if non-unit weights are later introduced

141 |

9. SIM14.0 TRUTH PACKET

----------------------------------------------------------------------------------------

Canonical Omega valid                     : True

Canonical conjugation is four 2-cycles    : True

Raw canonical |Omega| equals frozen C8    : False

A relabeling embeds conjugate matching in C8: True

Representative s_Omega preserves C8       : False

Projection pairing p preserves C8         : True

Graph half-turn h preserves C8            : True

Sp(8) transformations preserve Omega      : True

1177 |

Classification rule:

1179 |

A. If no canonical-basis relabeling places all four symplectic conjugate pairs on C8 edges, then ordinary C8 adjacency is not even compatible with conjugacy as a perfect matching under this minimal construction.

1186 |

B. If such relabelings exist but many inequivalent choices remain, then C8 is compatible with symplectic conjugacy but does not uniquely determine it.

1192 |

C. If a conjugate matching is uniquely selected up to C8 automorphism, that is stronger structural evidence.

1197 |

D. If signed Omega later changes ISP/LCO behavior while |Omega| does not, orientation carries information that the ordinary C8 graph suppresses.

1203 |

E. SIM14.0 alone makes NO physical claim about RCFT, Fano structure, H4, spinors, or quantum mechanics.

134 |

========================================================================================

SIM14.0 COMPLETE

========================================================================================

STATUS:

COMPATIBLE BUT NON-UNIQUE: C8 admits canonical symplectic conjugate matchings, but C8 alone does not select a unique one.

1234 |

Next allowed question: SIM14.1 may compare the frozen ordinary C8 carrier against a frozen symplectic surrogate without changing ISP/LCO for any other reason.
