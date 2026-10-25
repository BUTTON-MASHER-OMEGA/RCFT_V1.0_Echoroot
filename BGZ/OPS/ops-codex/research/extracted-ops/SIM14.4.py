"""

SIM14.4 — Affine Phase-Space Structure Map 
==========================================

Purpose
-------
Freeze the successful M2 branch from SIM14.0–14.3 and exhaustively map the
static eight-state affine carrier before returning to ISP × LCO.

This script deliberately does NOT search for a target algebra/group beyond
structures already earned in the SIM14 lineage. In particular it does not
construct J, fit a metric, search for U(4), introduce octonionic multiplication,
or run any dynamics.

Frozen objects
--------------
X8 = {0,...,7}
C8 edges = {01,02,13,24,35,46,57,67}
p = (01)(23)(45)(67)
h = (07)(16)(25)(34)
s = M2 = (02)(13)(46)(57)
K = <p,h>
G = <p,h,s>

Primary outputs
---------------
1. Verify the regular C2^3 action.
2. Derive all affine 4-point planes and parallel classes without hard-coding.
3. Map plane-plane incidence/intersection structure.
4. Map p,h,s and all seven nonidentity G-directions on the 14 planes.
5. Diagnose how C8 intersects each affine plane.
6. Diagnose M2/symplectic-matching content of each affine plane.
7. Exhaustively classify carrier permutations preserving progressively enriched
   finite structures.
8. Classify those finite symmetries against a fixed signed symplectic form as
   Omega-preserving, Omega-reversing, or neither.
9. Verify which affine-plane facts survive all eight choices of torsor origin.

Interpretive ceiling
--------------------
Finite affine/symplectic carrier mathematics only. No ISP, no LCO, no U(4)
claim, no octonion claim, no H4 claim, no physical identification.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple, FrozenSet

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("SIM14.4 requires numpy.") from exc


# =============================================================================
# 0. FROZEN CARRIER
# =============================================================================

X: Tuple[int, ...] = tuple(range(8))

C8_EDGES: FrozenSet[FrozenSet[int]] = frozenset(
    frozenset(e)
    for e in [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    ]
)

# Permutations are tuples perm[x] = image of x.
p: Tuple[int, ...] = (1, 0, 3, 2, 5, 4, 7, 6)
h: Tuple[int, ...] = (7, 6, 5, 4, 3, 2, 1, 0)
s: Tuple[int, ...] = (2, 3, 0, 1, 6, 7, 4, 5)  # M2

ID: Tuple[int, ...] = X

M2_EDGES: FrozenSet[FrozenSet[int]] = frozenset(
    frozenset(e)
    for e in [
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
    ]
)

# Signed symplectic form in the carrier labeling whose unsigned support is M2.
# This is a permutation-conjugate copy of the canonical 8D symplectic form.
OMEGA = np.zeros((8, 8), dtype=int)

for a, b in [(0, 2), (1, 3), (4, 6), (5, 7)]:
    OMEGA[a, b] = +1
    OMEGA[b, a] = -1


# =============================================================================
# 1. BASIC PERMUTATION / GROUP HELPERS
# =============================================================================

def compose(
    a: Tuple[int, ...],
    b: Tuple[int, ...],
) -> Tuple[int, ...]:
    """a ∘ b: apply b first, then a."""
    return tuple(a[b[x]] for x in X)


def inv(a: Tuple[int, ...]) -> Tuple[int, ...]:
    out = [None] * 8

    for i, j in enumerate(a):
        out[j] = i

    return tuple(out)  # type: ignore


def perm_order(a: Tuple[int, ...]) -> int:
    cur = ID

    for k in range(1, 100):
        cur = compose(a, cur)

        if cur == ID:
            return k

    raise RuntimeError("Permutation order exceeded bound")


def cycles(
    a: Tuple[int, ...],
    include_fixed: bool = False,
) -> Tuple[Tuple[int, ...], ...]:

    seen = set()
    out = []

    for x in X:
        if x in seen:
            continue

        cyc = []
        y = x

        while y not in seen:
            seen.add(y)
            cyc.append(y)
            y = a[y]

        if include_fixed or len(cyc) > 1:
            out.append(tuple(cyc))

    return tuple(out)


def cycle_notation(a: Tuple[int, ...]) -> str:
    cs = cycles(a, include_fixed=False)

    if not cs:
        return "e"

    return "".join(
        "(" + " ".join(map(str, c)) + ")"
        for c in cs
    )


def generated_group(
    gens: Sequence[Tuple[int, ...]],
) -> Set[Tuple[int, ...]]:

    group = {ID}
    frontier = [ID]

    while frontier:
        g = frontier.pop()

        for a in gens:
            for z in (compose(a, g), compose(g, a)):
                if z not in group:
                    group.add(z)
                    frontier.append(z)

    return group


def orbit(
    group: Iterable[Tuple[int, ...]],
    x: int,
) -> FrozenSet[int]:

    return frozenset(g[x] for g in group)


def apply_set(
    a: Tuple[int, ...],
    S: Iterable[int],
) -> FrozenSet[int]:

    return frozenset(a[x] for x in S)


def mapped_edges(
    a: Tuple[int, ...],
    E: Iterable[FrozenSet[int]],
) -> FrozenSet[FrozenSet[int]]:

    out = []

    for e in E:
        u, v = tuple(e)
        out.append(frozenset((a[u], a[v])))

    return frozenset(out)


def commutes(a, b) -> bool:
    return compose(a, b) == compose(b, a)


def is_involution(a) -> bool:
    return compose(a, a) == ID


# =============================================================================
# 2. DERIVED GROUPS AND DIRECTION LABELS
# =============================================================================

K = generated_group([p, h])
G = generated_group([p, h, s])

# Unique binary word p^a h^b s^c.
WORD_TO_G: Dict[
    Tuple[int, int, int],
    Tuple[int, ...],
] = {}

G_TO_WORD: Dict[
    Tuple[int, ...],
    Tuple[int, int, int],
] = {}

for a, b, c in itertools.product((0, 1), repeat=3):

    g = ID

    if a:
        g = compose(p, g)

    if b:
        g = compose(h, g)

    if c:
        g = compose(s, g)

    WORD_TO_G[(a, b, c)] = g

    if g in G_TO_WORD:
        raise AssertionError(
            "Generator words are not unique"
        )

    G_TO_WORD[g] = (a, b, c)


DIRECTIONS = tuple(
    sorted(
        (g for g in G if g != ID),
        key=lambda z: G_TO_WORD[z],
    )
)


# =============================================================================
# 3. AFFINE PLANES DERIVED FROM V4 SUBGROUPS
# =============================================================================

def order4_subgroups_of_G() -> List[
    FrozenSet[Tuple[int, ...]]
]:
    subs = set()

    nonid = [
        g for g in G
        if g != ID
    ]

    for a, b in itertools.combinations(nonid, 2):
        H = generated_group([a, b])

        if len(H) == 4:
            subs.add(frozenset(H))

    return sorted(
        subs,
        key=lambda H: sorted(
            G_TO_WORD[g]
            for g in H
        ),
    )


V4S = order4_subgroups_of_G()


@dataclass(frozen=True)
class Plane:
    points: FrozenSet[int]
    direction_subgroup: FrozenSet[
        Tuple[int, ...]
    ]


plane_by_points: Dict[
    FrozenSet[int],
    Plane,
] = {}

parallel_classes: List[
    Tuple[
        FrozenSet[int],
        FrozenSet[int],
        FrozenSet[Tuple[int, ...]],
    ]
] = []


for H in V4S:

    cosets = []
    unseen = set(X)

    while unseen:

        x = min(unseen)

        P = frozenset(
            g[x]
            for g in H
        )

        cosets.append(P)
        unseen -= set(P)

        plane_by_points.setdefault(
            P,
            Plane(P, H),
        )

    assert (
        len(cosets) == 2
        and all(len(P) == 4 for P in cosets)
    )

    parallel_classes.append(
        (
            cosets[0],
            cosets[1],
            H,
        )
    )


PLANES: Tuple[
    FrozenSet[int], ...
] = tuple(
    sorted(
        plane_by_points,
        key=lambda P: tuple(sorted(P)),
    )
)

PLANE_INDEX = {
    P: i
    for i, P in enumerate(PLANES)
}


# =============================================================================
# 4. GRAPH HELPERS FOR C8-INDUCED SUBGRAPHS
# =============================================================================

def induced_edges(
    P: FrozenSet[int],
    E=C8_EDGES,
) -> FrozenSet[FrozenSet[int]]:

    return frozenset(
        e for e in E
        if e <= P
    )


def degree_sequence(
    P: FrozenSet[int],
    Esub: FrozenSet[FrozenSet[int]],
) -> Tuple[int, ...]:

    deg = {
        x: 0
        for x in P
    }

    for e in Esub:
        u, v = tuple(e)
        deg[u] += 1
        deg[v] += 1

    return tuple(
        sorted(
            deg.values(),
            reverse=True,
        )
    )


def component_sizes(
    P: FrozenSet[int],
    Esub: FrozenSet[FrozenSet[int]],
) -> Tuple[int, ...]:

    adj = {
        x: set()
        for x in P
    }

    for e in Esub:
        u, v = tuple(e)

        adj[u].add(v)
        adj[v].add(u)

    unseen = set(P)
    sizes = []

    while unseen:

        root = next(iter(unseen))

        stack = [root]
        seen = {root}

        while stack:
            u = stack.pop()

            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)

        unseen -= seen
        sizes.append(len(seen))

    return tuple(
        sorted(
            sizes,
            reverse=True,
        )
    )


def graph_type(P: FrozenSet[int]) -> str:

    Esub = induced_edges(P)

    m = len(Esub)

    ds = degree_sequence(P, Esub)
    cs = component_sizes(P, Esub)

    if m == 0:
        return "4K1"

    if m == 1:
        return "K2+2K1"

    if m == 2 and ds == (1, 1, 1, 1):
        return "2K2"

    if m == 2 and ds == (2, 1, 1, 0):
        return "P3+K1"

    if m == 3 and ds == (2, 2, 1, 1):
        return "P4"

    if m == 3 and ds == (3, 1, 1, 1):
        return "K1,3"

    if m == 3 and ds == (2, 2, 2, 0):
        return "K3+K1"

    if m == 4 and ds == (2, 2, 2, 2):
        return "C4"

    return (
        f"m={m},deg={ds},comp={cs}"
    )


# =============================================================================
# 5. PLANE INCIDENCE
# =============================================================================

NPLANES = len(PLANES)

INTERSECTION = np.zeros(
    (NPLANES, NPLANES),
    dtype=int,
)

for i, P in enumerate(PLANES):
    for j, Q in enumerate(PLANES):
        INTERSECTION[i, j] = len(P & Q)


# Off-diagonal weighted incidence matrix.
INCIDENCE = INTERSECTION.copy()
np.fill_diagonal(INCIDENCE, 0)


# =============================================================================
# 6. PLANE ACTION AND DIRECTION FINGERPRINTS
# =============================================================================

def induced_plane_perm(
    g: Tuple[int, ...],
) -> Tuple[int, ...]:

    out = []

    for P in PLANES:

        Q = apply_set(g, P)

        if Q not in PLANE_INDEX:
            raise AssertionError(
                "g does not preserve "
                "affine-plane family: "
                f"{cycle_notation(g)}"
            )

        out.append(
            PLANE_INDEX[Q]
        )

    return tuple(out)


def generic_cycles(
    perm: Sequence[int],
    include_fixed=True,
) -> Tuple[Tuple[int, ...], ...]:

    seen = set()
    out = []

    for i in range(len(perm)):

        if i in seen:
            continue

        cyc = []
        j = i

        while j not in seen:
            seen.add(j)
            cyc.append(j)
            j = perm[j]

        if include_fixed or len(cyc) > 1:
            out.append(tuple(cyc))

    return tuple(out)


def plane_cycle_type(g) -> Tuple[int, ...]:

    pp = induced_plane_perm(g)

    return tuple(
        sorted(
            (
                len(c)
                for c in generic_cycles(pp)
            ),
            reverse=True,
        )
    )


def fixed_plane_count(g) -> int:

    pp = induced_plane_perm(g)

    return sum(
        i == j
        for i, j in enumerate(pp)
    )


def parallel_pair_action(
    g,
) -> Tuple[int, int]:
    """
    Return:
      (# parallel classes preserved plane-by-plane,
       # classes whose two planes are exchanged)
    """

    fixed = 0
    swapped = 0

    for P, Q, H in parallel_classes:

        gP = apply_set(g, P)
        gQ = apply_set(g, Q)

        if {gP, gQ} == {P, Q}:

            if gP == P and gQ == Q:
                fixed += 1

            elif gP == Q and gQ == P:
                swapped += 1

    return fixed, swapped


def c8_edge_overlap(g) -> int:
    """
    Number of C8 edges that are also
    displacement pairs {x,gx}.
    """

    pairs = set()

    for x in X:

        y = g[x]

        if x != y:
            pairs.add(
                frozenset((x, y))
            )

    return len(
        pairs & set(C8_EDGES)
    )


def direction_fingerprint(g) -> Tuple:

    return (
        c8_edge_overlap(g),
        fixed_plane_count(g),
        plane_cycle_type(g),
        parallel_pair_action(g),
        g in K,
        commutes(g, s),
    )


# =============================================================================
# 7. SYMPLECTIC MATCHING DIAGNOSTICS ON PLANES
# =============================================================================

def matching_plane_stats(
    P: FrozenSet[int],
) -> Tuple[int, int, int]:

    internal = 0
    crossing = 0

    for e in M2_EDGES:

        n = len(e & P)

        if n == 2:
            internal += 1

        elif n == 1:
            crossing += 1

    return (
        internal,
        crossing,
        4 - 2 * internal - crossing,
    )


def matching_bridge_matrix(
    P: FrozenSet[int],
    Q: FrozenSet[int],
) -> Tuple[
    Tuple[int, int],
    Tuple[int, int],
]:

    assert P.isdisjoint(Q)
    assert P | Q == frozenset(X)

    return (
        (
            sum(
                1 for x in P
                if s[x] in P
            ),
            sum(
                1 for x in P
                if s[x] in Q
            ),
        ),
        (
            sum(
                1 for x in Q
                if s[x] in P
            ),
            sum(
                1 for x in Q
                if s[x] in Q
            ),
        ),
    )


# =============================================================================
# 8. EXHAUSTIVE S8 AUTOMORPHISM / STABILIZER MAP
# =============================================================================

def preserves_edge_set(sig) -> bool:

    return (
        mapped_edges(
            sig,
            C8_EDGES,
        )
        == C8_EDGES
    )


def conjugate(sig, g):

    return compose(
        compose(sig, g),
        inv(sig),
    )


def preserves_named_perm(
    sig,
    g,
) -> bool:

    return (
        conjugate(sig, g)
        == g
    )


def preserves_group_set(
    sig,
    H,
) -> bool:

    return {
        conjugate(sig, g)
        for g in H
    } == set(H)


def preserves_plane_family(sig) -> bool:

    return all(
        apply_set(sig, P)
        in PLANE_INDEX
        for P in PLANES
    )


def preserves_parallel_classes(sig) -> bool:

    classes = {
        frozenset((P, Q))
        for P, Q, _ in parallel_classes
    }

    mapped = {
        frozenset(
            (
                apply_set(sig, P),
                apply_set(sig, Q),
            )
        )
        for P, Q, _ in parallel_classes
    }

    return mapped == classes


def perm_matrix(sig) -> np.ndarray:
    """
    P e_j = e_{sig(j)}
    """

    P = np.zeros(
        (8, 8),
        dtype=int,
    )

    for j, i in enumerate(sig):
        P[i, j] = 1

    return P


def omega_class(sig) -> str:

    P = perm_matrix(sig)

    A = (
        P.T
        @ OMEGA
        @ P
    )

    if np.array_equal(
        A,
        OMEGA,
    ):
        return "+Omega"

    if np.array_equal(
        A,
        -OMEGA,
    ):
        return "-Omega"

    return "neither"


ALL_PERMS = itertools.permutations(X)

AUT_PLANE = []
AUT_C8 = []
STAB_PH = []
STAB_PHS = []
AUT_FULL_FINITE = []


for sig in ALL_PERMS:

    sig = tuple(sig)

    pp = preserves_plane_family(sig)

    if pp:
        AUT_PLANE.append(sig)

    c8 = preserves_edge_set(sig)

    if c8:

        AUT_C8.append(sig)

        ph = (
            preserves_named_perm(sig, p)
            and
            preserves_named_perm(sig, h)
        )

        if ph:

            STAB_PH.append(sig)

            phs = preserves_named_perm(
                sig,
                s,
            )

            if phs:
                STAB_PHS.append(sig)

    # "full finite" =
    # preserve C8,
    # plane family,
    # and named p,h,s.

    if (
        c8
        and pp
        and preserves_named_perm(sig, p)
        and preserves_named_perm(sig, h)
        and preserves_named_perm(sig, s)
    ):
        AUT_FULL_FINITE.append(sig)


# =============================================================================
# 9. ORIGIN ROBUSTNESS
# =============================================================================

def coords_from_origin(
    origin: int,
) -> Dict[
    int,
    Tuple[int, int, int],
]:

    # Regular action:
    # unique g maps origin to x.

    out = {}

    for g in G:

        x = g[origin]

        if x in out:
            raise AssertionError(
                "Nonregular action "
                "during origin map"
            )

        out[x] = G_TO_WORD[g]

    assert len(out) == 8

    return out


def xor3(a, b):

    return tuple(
        x ^ y
        for x, y in zip(a, b)
    )


def planes_from_coords(
    coord: Dict[
        int,
        Tuple[int, int, int],
    ],
) -> FrozenSet[FrozenSet[int]]:

    invc = {
        v: k
        for k, v in coord.items()
    }

    dirs = [
        w
        for w in itertools.product(
            (0, 1),
            repeat=3,
        )
        if w != (0, 0, 0)
    ]

    out = set()

    for xw in invc:

        for a, b in itertools.combinations(
            dirs,
            2,
        ):

            pts = frozenset(
                invc[w]
                for w in (
                    xw,
                    xor3(xw, a),
                    xor3(xw, b),
                    xor3(
                        xor3(xw, a),
                        b,
                    ),
                )
            )

            if len(pts) == 4:
                out.add(pts)

    return frozenset(out)


ORIGIN_PLANE_FAMILIES = {
    o: planes_from_coords(
        coords_from_origin(o)
    )
    for o in X
}

ORIGIN_FAMILY_INVARIANT = (
    len(
        set(
            ORIGIN_PLANE_FAMILIES.values()
        )
    )
    == 1
)


# =============================================================================
# 10. REPORTING HELPERS
# =============================================================================

def fmt_points(P):

    return (
        "{"
        + ",".join(
            map(
                str,
                sorted(P),
            )
        )
        + "}"
    )


def fmt_word(w):

    return "".join(
        map(str, w)
    )


def fmt_g(g):

    return (
        f"{fmt_word(G_TO_WORD[g])}:"
        f"{cycle_notation(g)}"
    )


def print_header(title):

    print(
        "\n"
        + "=" * 88
    )

    print(title)

    print(
        "=" * 88
    )


# =============================================================================
# 11. REPORT
# =============================================================================

print_header(
    "SIM14.4 — AFFINE PHASE-SPACE STRUCTURE MAP"
)

print(
    "Interpretive ceiling: "
    "finite affine/symplectic carrier mathematics only."
)

print(
    "No ISP | No LCO | "
    "No imposed metric/J/U(4) | "
    "No octonion/H4/physics claim"
)


# -----------------------------------------------------------------------------
# A. Frozen carrier
# -----------------------------------------------------------------------------

print_header(
    "A. FROZEN CARRIER / SANITY"
)

print("X8 =", X)

print(
    "C8 edges =",
    sorted(
        tuple(sorted(e))
        for e in C8_EDGES
    ),
)

print(
    "p =",
    cycle_notation(p),
)

print(
    "h =",
    cycle_notation(h),
)

print(
    "s =",
    cycle_notation(s),
    " [M2]",
)

print(
    "|K=<p,h>| =",
    len(K),
    " orders =",
    dict(
        sorted(
            Counter(
                perm_order(g)
                for g in K
            ).items()
        )
    ),
)

print(
    "|G=<p,h,s>| =",
    len(G),
    " orders =",
    dict(
        sorted(
            Counter(
                perm_order(g)
                for g in G
            ).items()
        )
    ),
)

print(
    "pairwise generator commutation =",
    commutes(p, h),
    commutes(p, s),
    commutes(h, s),
)

print(
    "G vertex orbit sizes =",
    sorted(
        {
            len(orbit(G, x))
            for x in X
        }
    ),
)

print(
    "G stabilizer sizes =",
    sorted(
        {
            sum(
                g[x] == x
                for g in G
            )
            for x in X
        }
    ),
)

print(
    "G action regular =",
    all(
        len(orbit(G, x)) == 8
        and
        sum(
            g[x] == x
            for g in G
        ) == 1
        for x in X
    ),
)

print(
    "Omega antisymmetric =",
    np.array_equal(
        OMEGA.T,
        -OMEGA,
    ),
    " rank =",
    np.linalg.matrix_rank(OMEGA),
    " det =",
    round(np.linalg.det(OMEGA)),
)

print(
    "unsigned Omega support == M2 =",
    frozenset(
        frozenset((i, j))
        for i in X
        for j in range(i + 1, 8)
        if abs(OMEGA[i, j]) > 0
    )
    == M2_EDGES,
)


# -----------------------------------------------------------------------------
# B. Affine planes
# -----------------------------------------------------------------------------

print_header(
    "B. AFFINE PLANES / PARALLEL CLASSES"
)

print(
    "number of V4 direction subgroups =",
    len(V4S),
)

print(
    "number of distinct affine 4-point planes =",
    len(PLANES),
)

print(
    "number of parallel classes =",
    len(parallel_classes),
)

for idx, (P, Q, H) in enumerate(
    sorted(
        parallel_classes,
        key=lambda z: (
            tuple(sorted(z[0])),
            tuple(sorted(z[1])),
        ),
    )
):

    dirs = sorted(
        fmt_word(G_TO_WORD[g])
        for g in H
        if g != ID
    )

    print(
        f"class {idx:02d}: "
        f"{fmt_points(P)} || "
        f"{fmt_points(Q)}   "
        f"directions={dirs}"
    )


# -----------------------------------------------------------------------------
# C. Plane-plane incidence
# -----------------------------------------------------------------------------

print_header(
    "C. PLANE–PLANE INCIDENCE"
)

off = [
    len(
        PLANES[i]
        & PLANES[j]
    )
    for i in range(NPLANES)
    for j in range(
        i + 1,
        NPLANES,
    )
]

print(
    "off-diagonal intersection histogram =",
    dict(
        sorted(
            Counter(off).items()
        )
    ),
)

for k in sorted(set(off)):

    degs = []

    for i in range(NPLANES):

        degs.append(
            sum(
                1
                for j in range(NPLANES)
                if (
                    i != j
                    and
                    len(
                        PLANES[i]
                        & PLANES[j]
                    ) == k
                )
            )
        )

    print(
        f"intersection={k}: "
        "per-plane degree histogram = "
        f"{dict(sorted(Counter(degs).items()))}"
    )


eigs = np.linalg.eigvalsh(
    INCIDENCE.astype(float)
)

print(
    "weighted incidence eigenvalues "
    "(rounded 10dp) =",
    [
        round(float(x), 10)
        for x in eigs
    ],
)


# -----------------------------------------------------------------------------
# D. C8 content
# -----------------------------------------------------------------------------

print_header(
    "D. C8 CONTENT OF AFFINE PLANES"
)

gt = Counter()

for i, P in enumerate(PLANES):

    Esub = induced_edges(P)
    typ = graph_type(P)

    gt[typ] += 1

    print(
        f"P{i:02d} "
        f"{fmt_points(P)}: "
        f"edges={len(Esub)} "
        f"deg={degree_sequence(P, Esub)} "
        f"comps={component_sizes(P, Esub)} "
        f"type={typ}"
    )

print(
    "C8 induced graph-type histogram =",
    dict(
        sorted(gt.items())
    ),
)


# -----------------------------------------------------------------------------
# E. M2 matching content
# -----------------------------------------------------------------------------

print_header(
    "E. M2 / SYMPLECTIC-MATCHING CONTENT OF PLANES"
)

stat_hist = Counter()

for i, P in enumerate(PLANES):

    st = matching_plane_stats(P)
    stat_hist[st] += 1

    print(
        f"P{i:02d} "
        f"{fmt_points(P)}: "
        f"complete_pairs={st[0]} "
        f"crossing_pairs={st[1]}"
    )

print(
    "plane matching-stat histogram =",
    dict(
        sorted(
            stat_hist.items()
        )
    ),
)


bridge_hist = Counter()

for idx, (P, Q, H) in enumerate(
    parallel_classes
):

    B = matching_bridge_matrix(P, Q)

    bridge_hist[B] += 1

    print(
        f"parallel class {idx:02d}: "
        f"{fmt_points(P)} || "
        f"{fmt_points(Q)}  "
        f"B_s={B}"
    )

print(
    "parallel-class M2 "
    "bridge-matrix histogram =",
    dict(bridge_hist),
)


# -----------------------------------------------------------------------------
# F. p,h,s action
# -----------------------------------------------------------------------------

print_header(
    "F. ACTION OF p, h, s ON THE 14 PLANES"
)

for name, g in [
    ("p", p),
    ("h", h),
    ("s", s),
]:

    pp = induced_plane_perm(g)

    print(
        f"{name}: "
        f"fixed_planes={fixed_plane_count(g)} "
        f"plane_cycle_type={plane_cycle_type(g)} "
        "parallel(fixed,swapped)="
        f"{parallel_pair_action(g)}"
    )

    print(
        "   cycles =",
        generic_cycles(pp),
    )


# -----------------------------------------------------------------------------
# G. Seven direction fingerprints
# -----------------------------------------------------------------------------

print_header(
    "G. ALL SEVEN NONIDENTITY DIRECTION FINGERPRINTS"
)

fp_classes = defaultdict(list)

for g in DIRECTIONS:

    fp = direction_fingerprint(g)

    fp_classes[fp].append(g)

    print(
        f"{fmt_g(g):28s}  "
        f"C8_overlap={fp[0]} "
        f"fixed_planes={fp[1]} "
        f"plane_cycles={fp[2]} "
        f"parallel={fp[3]} "
        f"in_K={fp[4]} "
        f"commutes_s={fp[5]}"
    )


print(
    "exact fingerprint class count =",
    len(fp_classes),
)

for i, (fp, gs) in enumerate(
    fp_classes.items()
):

    print(
        f"class {i}: "
        f"{[fmt_g(g) for g in gs]} "
        f"-> {fp}"
    )


# -----------------------------------------------------------------------------
# H. Symmetry hierarchy
# -----------------------------------------------------------------------------

print_header(
    "H. EXHAUSTIVE CARRIER SYMMETRY HIERARCHY "
    "(ALL 8! PERMUTATIONS)"
)

print(
    "|S8| =",
    math.factorial(8),
)

print(
    "|Aut(affine-plane family)| =",
    len(AUT_PLANE),
)

print(
    "|Aut(C8)| =",
    len(AUT_C8),
)

print(
    "|Stab_C8(p,h)| =",
    len(STAB_PH),
)

print(
    "|Stab_C8(p,h,s)| =",
    len(STAB_PHS),
)

print(
    "|preserve C8 + plane family "
    "+ named p,h,s| =",
    len(AUT_FULL_FINITE),
)

print(
    "plane-family automorphisms "
    "preserve parallel classes =",
    all(
        preserves_parallel_classes(sig)
        for sig in AUT_PLANE
    ),
)


for label, collection in [

    (
        "Aut(plane family)",
        AUT_PLANE,
    ),

    (
        "Aut(C8)",
        AUT_C8,
    ),

    (
        "Stab_C8(p,h)",
        STAB_PH,
    ),

    (
        "Stab_C8(p,h,s)",
        STAB_PHS,
    ),

    (
        "full finite",
        AUT_FULL_FINITE,
    ),

]:

    oc = Counter(
        omega_class(sig)
        for sig in collection
    )

    print(
        f"{label:20s} "
        "Omega classification = "
        f"{dict(sorted(oc.items()))}"
    )


# -----------------------------------------------------------------------------
# I. Origin robustness
# -----------------------------------------------------------------------------

print_header(
    "I. ORIGIN ROBUSTNESS"
)

print(
    "all 8 origin choices recover "
    "identical 14-plane family =",
    ORIGIN_FAMILY_INVARIANT,
)

for o in X:

    fam = ORIGIN_PLANE_FAMILIES[o]

    print(
        f"origin {o}: "
        f"plane_count={len(fam)} "
        "identical_to_intrinsic="
        f"{fam == frozenset(PLANES)}"
    )

print(
    "NOTE: which seven planes "
    "'contain coordinate 000' "
    "depends on chosen origin; "
    "the full 14-plane family does not."
)


# -----------------------------------------------------------------------------
# J. Machine truth packet
# -----------------------------------------------------------------------------

print_header(
    "J. MACHINE TRUTH PACKET"
)

print("FROZEN:")
print(
    "  X8, C8, p, h, M2/s, "
    "K=<p,h>, G=<p,h,s>"
)

print("DERIVED:")
print(
    f"  |K|={len(K)}"
)

print(
    f"  |G|={len(G)}"
)

print(
    "  G action regular="
    f"{all(
        len(orbit(G, x)) == 8
        and
        sum(
            g[x] == x
            for g in G
        ) == 1
        for x in X
    )}"
)

print(
    f"  affine planes={len(PLANES)}"
)

print(
    "  parallel classes="
    f"{len(parallel_classes)}"
)


print("AFFINE INCIDENCE:")

print(
    "  intersection histogram="
    f"{dict(
        sorted(
            Counter(off).items()
        )
    )}"
)

print(
    "  weighted incidence spectrum="
    f"{[
        round(float(x), 8)
        for x in eigs
    ]}"
)


print("C8 RELATION:")

print(
    "  induced plane graph types="
    f"{dict(sorted(gt.items()))}"
)


print(
    "SYMPLECTIC MATCHING RELATION:"
)

print(
    "  plane matching stats="
    f"{dict(
        sorted(
            stat_hist.items()
        )
    )}"
)

print(
    "  parallel bridge matrices="
    f"{dict(bridge_hist)}"
)


print("DIRECTION STRUCTURE:")

print(
    "  exact direction "
    "fingerprint classes="
    f"{len(fp_classes)}"
)


print("SYMMETRY:")

print(
    "  |Aut(plane family)|="
    f"{len(AUT_PLANE)}"
)

print(
    "  |Aut(C8)|="
    f"{len(AUT_C8)}"
)

print(
    "  |Stab_C8(p,h)|="
    f"{len(STAB_PH)}"
)

print(
    "  |Stab_C8(p,h,s)|="
    f"{len(STAB_PHS)}"
)

print(
    "  |full finite stabilizer|="
    f"{len(AUT_FULL_FINITE)}"
)


print(
    "CONTINUOUS OMEGA CHECK:"
)

print(
    "  full finite Omega classes="
    f"{dict(
        sorted(
            Counter(
                omega_class(sig)
                for sig in AUT_FULL_FINITE
            ).items()
        )
    )}"
)


print("ORIGIN ROBUSTNESS:")

print(
    "  all origins recover same "
    "affine-plane family="
    f"{ORIGIN_FAMILY_INVARIANT}"
)


print("INTERPRETIVE CEILING:")

print(
    "  finite affine/symplectic "
    "carrier mathematics only"
)

print(
    "  no ISP; no LCO; "
    "no imposed J/g/U(4); "
    "no octonion; no H4; "
    "no physical identification"
)


# =============================================================================
# 12. HARD BOOKKEEPING SANITY CHECKS
# =============================================================================

assert len(K) == 4

assert len(G) == 8

assert all(
    perm_order(g) in (1, 2)
    for g in G
)

assert len(V4S) == 7

assert len(PLANES) == 14

assert len(parallel_classes) == 7

assert ORIGIN_FAMILY_INVARIANT

assert all(
    fam == frozenset(PLANES)
    for fam
    in ORIGIN_PLANE_FAMILIES.values()
)

assert np.array_equal(
    OMEGA.T,
    -OMEGA,
)

assert (
    np.linalg.matrix_rank(OMEGA)
    == 8
)

print(
    "\nSIM14.4 COMPLETE — "
    "bookkeeping sanity checks PASS"
)








