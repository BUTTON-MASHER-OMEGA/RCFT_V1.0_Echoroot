# =============================================================================
# SIM14.3 — V4 ORBIT-BRIDGE CLASSIFICATION
# =============================================================================
#
# PURPOSE
# -------
# SIM14.2 established:
#
#   K = <p,h> ~= V4
#
# and showed that the two SIM14.0 C8-contained symplectic-compatible
# conjugate matchings behave differently:
#
#   M1 = {0-1, 2-4, 3-5, 6-7}
#   M2 = {0-2, 1-3, 4-6, 5-7}
#
# Both satisfy
#
#   [p,s] = [h,s] = e
#
# and both generate an abstract group
#
#   <p,h,s> ~= (Z_2)^3.
#
# But:
#
#   M1 -> nonregular action on X8, with two 4-state orbits
#   M2 -> regular action on all eight states.
#
# SIM14.3 asks the smallest unresolved question:
#
#   WHAT STRUCTURAL PROPERTY OF s RELATIVE TO K=<p,h>
#   DETERMINES REGULARITY?
#
# The principal candidate is the way s acts on the pre-existing K-orbits.
# We therefore define the orbit-transition matrix
#
#   B_ab(s) = #{x in O_a : s(x) in O_b}.
#
# IMPORTANT:
# ----------
# The code does NOT define regularity using B(s).
#
# It independently:
#
#   1. constructs K and its orbits,
#   2. enumerates every fixed-point-free involution s on X8,
#   3. retains only those commuting with p and h,
#   4. computes B(s),
#   5. independently computes <K,s> and its action,
#   6. asks afterward whether B(s) predicts regularity.
#
# C8 containment is recorded only as a secondary restriction.
#
# NO ISP.
# NO LCO.
# NO stochastic sampling.
# NO fitting.
# NO optimization.
# NO PG(2,2) reconstruction.
# NO octonionic / E8 / H4 / spin / QM interpretation.
#
# =============================================================================

from itertools import combinations
from collections import Counter, defaultdict, deque


# =============================================================================
# GLOBALS
# =============================================================================

WIDTH = 112

N = 8
X8 = tuple(range(N))
IDENTITY = tuple(range(N))


def banner(title):
    print("\n" + "=" * WIDTH)
    print(title)
    print("=" * WIDTH)


def section(title):
    print("\n" + "-" * WIDTH)
    print(title)
    print("-" * WIDTH)


# =============================================================================
# PERMUTATION UTILITIES
# =============================================================================
#
# Permutation g is represented by
#
#   g = (g(0), ..., g(7))
#
# compose(g,h) means
#
#   (g o h)(x) = g(h(x)).
#
# =============================================================================

def compose(g, h):
    return tuple(g[h[i]] for i in X8)


def inverse(g):
    out = [None] * N

    for i, j in enumerate(g):
        out[j] = i

    return tuple(out)


def perm_order(g):
    x = IDENTITY

    for k in range(1, 100):
        x = compose(g, x)

        if x == IDENTITY:
            return k

    raise RuntimeError("Permutation order exceeded search bound.")


def commute(g, h):
    return compose(g, h) == compose(h, g)


def cycles(g, include_fixed=False):
    seen = set()
    out = []

    for start in X8:

        if start in seen:
            continue

        cyc = []
        x = start

        while x not in seen:
            seen.add(x)
            cyc.append(x)
            x = g[x]

        if len(cyc) > 1 or include_fixed:
            out.append(tuple(cyc))

    return tuple(out)


def cycle_string(g):
    cs = cycles(g)

    if not cs:
        return "()"

    return "".join(
        "(" + " ".join(map(str, cyc)) + ")"
        for cyc in cs
    )


def generated_group(generators):
    group = {IDENTITY}
    queue = deque([IDENTITY])

    gens = tuple(generators)

    while queue:

        x = queue.popleft()

        for g in gens:

            for y in (
                compose(g, x),
                compose(x, g),
            ):

                if y not in group:
                    group.add(y)
                    queue.append(y)

    return frozenset(group)


# =============================================================================
# GRAPH UTILITIES
# =============================================================================

C8_CYCLE_ORDER = (0, 1, 3, 5, 7, 6, 4, 2)

C8_EDGES = frozenset(
    frozenset(
        (
            C8_CYCLE_ORDER[i],
            C8_CYCLE_ORDER[(i + 1) % N],
        )
    )
    for i in range(N)
)


def edge_tuple(edge):
    return tuple(sorted(edge))


def edge_set_string(edges):
    return "{" + ", ".join(
        f"{a}-{b}"
        for a, b in sorted(
            edge_tuple(e)
            for e in edges
        )
    ) + "}"


def image_edge(edge, g):
    a, b = tuple(edge)
    return frozenset((g[a], g[b]))


def image_edge_set(edges, g):
    return frozenset(
        image_edge(edge, g)
        for edge in edges
    )


def is_c8_automorphism(g):
    return image_edge_set(C8_EDGES, g) == C8_EDGES


# =============================================================================
# FROZEN PROJECTION STRUCTURE
# =============================================================================

PROJECTION_BASINS = (
    frozenset((0, 1)),
    frozenset((2, 3)),
    frozenset((4, 5)),
    frozenset((6, 7)),
)


def image_partition(partition, g):
    return frozenset(
        frozenset(g[v] for v in block)
        for block in partition
    )


def preserves_projection_setwise(g):
    return (
        image_partition(PROJECTION_BASINS, g)
        ==
        frozenset(PROJECTION_BASINS)
    )


# =============================================================================
# FROZEN SIM14 OPERATIONS
# =============================================================================

# p = (0 1)(2 3)(4 5)(6 7)
p = (1, 0, 3, 2, 5, 4, 7, 6)

# h = (0 7)(1 6)(2 5)(3 4)
h = (7, 6, 5, 4, 3, 2, 1, 0)

# M1 = {0-1, 2-4, 3-5, 6-7}
# s1 = (0 1)(2 4)(3 5)(6 7)
s1 = (1, 0, 4, 5, 2, 3, 7, 6)

# M2 = {0-2, 1-3, 4-6, 5-7}
# s2 = (0 2)(1 3)(4 6)(5 7)
s2 = (2, 3, 0, 1, 6, 7, 4, 5)


# =============================================================================
# MATCHING UTILITIES
# =============================================================================

def matching_from_involution(s):

    if perm_order(s) != 2:
        return None

    if any(s[v] == v for v in X8):
        return None

    return frozenset(
        frozenset((v, s[v]))
        for v in X8
    )


M1 = matching_from_involution(s1)
M2 = matching_from_involution(s2)


# =============================================================================
# ENUMERATE ALL PERFECT MATCHINGS / FIXED-POINT-FREE INVOLUTIONS
# =============================================================================
#
# Number expected on eight labeled vertices:
#
#   (8-1)!! = 105.
#
# =============================================================================

def all_perfect_matchings(vertices):

    vertices = tuple(sorted(vertices))

    if not vertices:
        yield tuple()
        return

    a = vertices[0]

    for idx in range(1, len(vertices)):

        b = vertices[idx]

        remaining = (
            vertices[1:idx]
            +
            vertices[idx + 1:]
        )

        for rest in all_perfect_matchings(remaining):

            yield ((a, b),) + rest


def matching_to_involution(matching):

    s = list(range(N))

    for a, b in matching:
        s[a] = b
        s[b] = a

    return tuple(s)


ALL_MATCHINGS_RAW = tuple(
    all_perfect_matchings(X8)
)

ALL_INVOLUTIONS = tuple(
    matching_to_involution(matching)
    for matching in ALL_MATCHINGS_RAW
)


# =============================================================================
# GROUP ACTION UTILITIES
# =============================================================================

def group_orbits(G):

    unseen = set(X8)
    out = []

    while unseen:

        v = min(unseen)

        orbit = frozenset(
            g[v]
            for g in G
        )

        out.append(orbit)
        unseen -= set(orbit)

    return tuple(
        sorted(
            out,
            key=lambda orbit: (
                min(orbit),
                len(orbit),
                tuple(sorted(orbit)),
            )
        )
    )


def stabilizer(G, v):
    return frozenset(
        g
        for g in G
        if g[v] == v
    )


def action_is_transitive(G):
    return len(group_orbits(G)) == 1


def action_is_free(G):

    for g in G:

        if g == IDENTITY:
            continue

        if any(g[v] == v for v in X8):
            return False

    return True


def action_is_regular(G):
    return (
        action_is_transitive(G)
        and
        action_is_free(G)
    )


def action_kernel(G):
    """
    Kernel of the literal permutation action on X8.

    Since G is represented as a permutation subgroup, this should
    always be trivial. Included as an implementation control.
    """

    return frozenset(
        g
        for g in G
        if all(g[v] == v for v in X8)
    )


# =============================================================================
# ORBIT-TRANSITION MATRIX
# =============================================================================

def orbit_transition_matrix(s, K_orbits):

    r = len(K_orbits)

    orbit_index = {}

    for idx, orbit in enumerate(K_orbits):

        for v in orbit:
            orbit_index[v] = idx

    B = [
        [0 for _ in range(r)]
        for _ in range(r)
    ]

    for v in X8:

        a = orbit_index[v]
        b = orbit_index[s[v]]

        B[a][b] += 1

    return tuple(
        tuple(row)
        for row in B
    )


def matrix_string(B):
    return "[" + "; ".join(
        ",".join(map(str, row))
        for row in B
    ) + "]"


def matrix_type(B):
    """
    Purely descriptive canonical label.

    No regularity information enters this label.
    """

    return matrix_string(B)


# =============================================================================
# HEADER
# =============================================================================

banner("SIM14.3 — V4 ORBIT-BRIDGE CLASSIFICATION")

print()
print("NO ISP / LCO dynamics are run in SIM14.3.")
print("NO stochastic sampling, fitting, optimization, or new geometry.")
print()
print("Primary question:")
print()
print("    What property of s relative to K=<p,h> determines regularity")
print("    of G_s=<K,s> on the eight-state microscopic carrier?")
print()
print("The orbit-transition matrix B(s) is measured independently of")
print("the subsequent regularity classification.")


# =============================================================================
# 1. FROZEN LABORATORY
# =============================================================================

section("1. FROZEN SIM14 LABORATORY")

print("X8                  :", X8)
print("C8 cycle order      :", C8_CYCLE_ORDER)
print("C8 edges            :", edge_set_string(C8_EDGES))

print()
print(
    "Projection basins   :",
    {
        i: sorted(block)
        for i, block in enumerate(PROJECTION_BASINS)
    }
)

print()
print("p                   :", cycle_string(p))
print("h                   :", cycle_string(h))

print()
print("M1                  :", edge_set_string(M1))
print("s1                  :", cycle_string(s1))

print()
print("M2                  :", edge_set_string(M2))
print("s2                  :", cycle_string(s2))


# =============================================================================
# 2. PRE-SYMPLECTIC BASE GROUP K=<p,h>
# =============================================================================

section("2. PRE-SYMPLECTIC BASE GROUP K = <p,h>")

K = generated_group([p, h])

print("|K|                 :", len(K))

print(
    "Element orders      :",
    dict(
        sorted(
            Counter(
                perm_order(g)
                for g in K
            ).items()
        )
    )
)

print("p,h commute         :", commute(p, h))

K_IS_V4 = (
    len(K) == 4
    and
    commute(p, h)
    and
    all(
        perm_order(g) == 2
        for g in K
        if g != IDENTITY
    )
)

print("K ~= V4             :", K_IS_V4)

print()
print("K elements:")

for g in sorted(K):
    print("   ", cycle_string(g))


# =============================================================================
# 3. K ACTION ON X8
# =============================================================================

section("3. K ACTION ON X8")

K_ORBITS = group_orbits(K)

print("Number of K-orbits  :", len(K_ORBITS))
print(
    "Orbit-size multiset :",
    sorted(len(O) for O in K_ORBITS)
)

print()

for idx, orbit in enumerate(K_ORBITS):
    print(
        f"O{idx} = {sorted(orbit)}"
    )

print()
print("vertex   orbit-size   stabilizer-size")

for v in X8:

    orbit = frozenset(
        g[v]
        for g in K
    )

    stab = stabilizer(K, v)

    print(
        f"{v:>3d}"
        f"{len(orbit):>12d}"
        f"{len(stab):>18d}"
    )


# =============================================================================
# 4. M1 / M2 ORBIT-TRANSITION MATRICES
# =============================================================================

section("4. M1 / M2 RELATION TO THE PRE-EXISTING K-ORBITS")

B1 = orbit_transition_matrix(
    s1,
    K_ORBITS
)

B2 = orbit_transition_matrix(
    s2,
    K_ORBITS
)

print("B(s1) =", matrix_string(B1))
print("B(s2) =", matrix_string(B2))

print()
print("Rows = source K-orbits.")
print("Columns = destination K-orbits.")
print()
print(
    "No interpretation of these matrices is used to define "
    "regularity."
)


# =============================================================================
# 5. REPRODUCE THE SIM14.2 ACTION DISTINCTION
# =============================================================================

section("5. SIM14.2 REPRODUCTION THROUGH THE K+s LENS")


def action_record(s):

    G = generated_group([p, h, s])

    orbits = group_orbits(G)

    stabilizer_sizes = tuple(
        sorted(
            len(stabilizer(G, v))
            for v in X8
        )
    )

    matching = matching_from_involution(s)

    return {
        "s": s,
        "matching": matching,
        "G": G,
        "group_order": len(G),
        "group_order_hist": dict(
            sorted(
                Counter(
                    perm_order(g)
                    for g in G
                ).items()
            )
        ),
        "orbits": orbits,
        "orbit_sizes": tuple(
            sorted(len(O) for O in orbits)
        ),
        "stabilizer_sizes": stabilizer_sizes,
        "kernel_size": len(action_kernel(G)),
        "transitive": action_is_transitive(G),
        "free": action_is_free(G),
        "regular": action_is_regular(G),
        "B": orbit_transition_matrix(s, K_ORBITS),
        "in_K": s in K,
        "c8_contained": (
            matching is not None
            and
            matching.issubset(C8_EDGES)
        ),
        "c8_auto": is_c8_automorphism(s),
        "projection_setwise": preserves_projection_setwise(s),
    }


R1 = action_record(s1)
R2 = action_record(s2)


def print_action_record(label, R):

    print(label)
    print("-" * 72)

    print("s                     :", cycle_string(R["s"]))
    print("matching              :", edge_set_string(R["matching"]))
    print("B(s)                  :", matrix_string(R["B"]))

    print()
    print("|<K,s>|               :", R["group_order"])
    print("element-order hist    :", R["group_order_hist"])

    print()
    print(
        "orbits                :",
        [
            sorted(O)
            for O in R["orbits"]
        ]
    )

    print("orbit-size multiset   :", R["orbit_sizes"])
    print("stabilizer sizes      :", R["stabilizer_sizes"])
    print("kernel size           :", R["kernel_size"])

    print()
    print("transitive            :", R["transitive"])
    print("free                  :", R["free"])
    print("regular               :", R["regular"])

    print()
    print("matching in C8        :", R["c8_contained"])
    print("s is C8 automorphism  :", R["c8_auto"])
    print(
        "preserves Pi setwise :",
        R["projection_setwise"]
    )

    print()


print_action_record("M1", R1)
print_action_record("M2", R2)


# =============================================================================
# 6. ENUMERATION CONTROL: ALL 105 PERFECT MATCHINGS
# =============================================================================

section("6. ENUMERATION CONTROL")

print(
    "All perfect matchings on 8 labeled vertices:",
    len(ALL_INVOLUTIONS)
)

print("Expected                            : 105")

ALL_UNIQUE = (
    len(set(ALL_INVOLUTIONS))
    ==
    len(ALL_INVOLUTIONS)
)

print("All enumerated involutions unique   :", ALL_UNIQUE)

ALL_VALID_FPF = all(
    perm_order(s) == 2
    and
    all(s[v] != v for v in X8)
    for s in ALL_INVOLUTIONS
)

print("All are fixed-point-free involutions:", ALL_VALID_FPF)


# =============================================================================
# 7. RESTRICT TO THE COMMUTING CLASS S_K
# =============================================================================

section("7. COMMUTING FIXED-POINT-FREE INVOLUTION CLASS S_K")

S_K = tuple(
    s
    for s in ALL_INVOLUTIONS
    if commute(s, p)
    and commute(s, h)
)

print("|S_K|                        :", len(S_K))

print()
print("Definition:")
print()
print("  S_K = {s : s^2=e, Fix(s)=empty, [s,p]=[s,h]=e}")
print()

print("M1 included                  :", s1 in S_K)
print("M2 included                  :", s2 in S_K)


# =============================================================================
# 8. COMPLETE CLASSIFICATION OF S_K
# =============================================================================

section("8. COMPLETE STRUCTURAL CLASSIFICATION OF S_K")

records = []

for idx, s in enumerate(sorted(S_K), start=1):

    R = action_record(s)

    R["id"] = idx

    if s == s1:
        R["special"] = "M1"
    elif s == s2:
        R["special"] = "M2"
    else:
        R["special"] = ""

    records.append(R)


print(
    "ID   tag   s                              "
    "inK  C8   B(s)              |G|   orbits       free  trans  regular"
)

print("-" * WIDTH)

for R in records:

    print(
        f"{R['id']:>2d}   "
        f"{R['special']:<3s}   "
        f"{cycle_string(R['s']):30s} "
        f"{str(R['in_K']):>4s} "
        f"{str(R['c8_contained']):>4s} "
        f"{matrix_string(R['B']):18s} "
        f"{R['group_order']:>3d}   "
        f"{str(R['orbit_sizes']):12s} "
        f"{str(R['free']):>5s} "
        f"{str(R['transitive']):>6s} "
        f"{str(R['regular']):>7s}"
    )


# =============================================================================
# 9. B(s) TYPE -> ACTION TYPE
# =============================================================================

section("9. DOES B(s) CLASSIFY THE ACTION TYPE?")

by_B = defaultdict(list)

for R in records:
    by_B[R["B"]].append(R)


print("Number of distinct B(s) types:", len(by_B))
print()

B_summary = {}

for B, Rs in sorted(
    by_B.items(),
    key=lambda item: matrix_string(item[0])
):

    regular_values = {
        R["regular"]
        for R in Rs
    }

    free_values = {
        R["free"]
        for R in Rs
    }

    trans_values = {
        R["transitive"]
        for R in Rs
    }

    orbit_types = {
        R["orbit_sizes"]
        for R in Rs
    }

    group_orders = {
        R["group_order"]
        for R in Rs
    }

    homogeneous_regular = (
        len(regular_values) == 1
    )

    B_summary[B] = {
        "count": len(Rs),
        "regular_values": regular_values,
        "free_values": free_values,
        "trans_values": trans_values,
        "orbit_types": orbit_types,
        "group_orders": group_orders,
        "homogeneous_regular": homogeneous_regular,
    }

    print("B(s) =", matrix_string(B))

    print(
        "  count              :",
        len(Rs)
    )

    print(
        "  group orders       :",
        sorted(group_orders)
    )

    print(
        "  orbit types        :",
        sorted(orbit_types)
    )

    print(
        "  free values        :",
        sorted(free_values)
    )

    print(
        "  transitive values  :",
        sorted(trans_values)
    )

    print(
        "  regular values     :",
        sorted(regular_values)
    )

    print(
        "  regularity uniform :",
        homogeneous_regular
    )

    tags = [
        R["special"]
        for R in Rs
        if R["special"]
    ]

    if tags:
        print(
            "  contains           :",
            tags
        )

    print()


B_PERFECTLY_CLASSIFIES_REGULARITY = all(
    summary["homogeneous_regular"]
    for summary in B_summary.values()
)

print(
    "B(s) type perfectly classifies regularity:",
    B_PERFECTLY_CLASSIFIES_REGULARITY
)


# =============================================================================
# 10. REGULAR VS NONREGULAR COUNTS
# =============================================================================

section("10. REGULARITY LEDGER INSIDE S_K")

regular_records = [
    R
    for R in records
    if R["regular"]
]

nonregular_records = [
    R
    for R in records
    if not R["regular"]
]

print("Total commuting candidates     :", len(records))
print("Regular candidates             :", len(regular_records))
print("Nonregular candidates          :", len(nonregular_records))

print()

print(
    "Regular fraction              :",
    (
        len(regular_records) / len(records)
        if records
        else float("nan")
    )
)

print()

print(
    "Regular B(s) types            :",
    sorted(
        {
            matrix_string(R["B"])
            for R in regular_records
        }
    )
)

print(
    "Nonregular B(s) types         :",
    sorted(
        {
            matrix_string(R["B"])
            for R in nonregular_records
        }
    )
)


# =============================================================================
# 11. TEST THE SIMPLE ORBIT-BRIDGE CANDIDATE
# =============================================================================
#
# This section defines a purely orbit-theoretic predicate AFTER B(s)
# has already been computed.
#
# "Complete orbit exchange" means:
#
#   s maps every point of each K-orbit into a different K-orbit.
#
# This is not assumed to imply regularity.
#
# We test whether it does.
#
# =============================================================================

section("11. CANDIDATE ORBIT-BRIDGE LAW")


def complete_orbit_exchange(s, K_orbits):

    orbit_index = {}

    for idx, O in enumerate(K_orbits):

        for v in O:
            orbit_index[v] = idx

    for v in X8:

        if orbit_index[v] == orbit_index[s[v]]:
            return False

    return True


for R in records:
    R["complete_bridge"] = complete_orbit_exchange(
        R["s"],
        K_ORBITS
    )


bridge_records = [
    R
    for R in records
    if R["complete_bridge"]
]

nonbridge_records = [
    R
    for R in records
    if not R["complete_bridge"]
]


print(
    "Complete orbit-exchange candidates:",
    len(bridge_records)
)

print(
    "Non-exchange candidates            :",
    len(nonbridge_records)
)

print()

print(
    "Regular among complete exchanges   :",
    sum(R["regular"] for R in bridge_records),
    "/",
    len(bridge_records)
)

print(
    "Regular among non-exchanges        :",
    sum(R["regular"] for R in nonbridge_records),
    "/",
    len(nonbridge_records)
)

print()


BRIDGE_NECESSARY = all(
    R["complete_bridge"]
    for R in regular_records
)

BRIDGE_SUFFICIENT = all(
    R["regular"]
    for R in bridge_records
)

BRIDGE_IFF_REGULAR = (
    BRIDGE_NECESSARY
    and
    BRIDGE_SUFFICIENT
    and
    len(regular_records) > 0
    and
    len(bridge_records) > 0
)


print(
    "Complete exchange necessary for regularity:",
    BRIDGE_NECESSARY
)

print(
    "Complete exchange sufficient for regularity:",
    BRIDGE_SUFFICIENT
)

print(
    "Complete exchange iff regularity            :",
    BRIDGE_IFF_REGULAR
)


# =============================================================================
# 12. CHECK FOR COUNTEREXAMPLES
# =============================================================================

section("12. COUNTEREXAMPLE SEARCH")

regular_nonbridges = [
    R
    for R in records
    if R["regular"]
    and
    not R["complete_bridge"]
]

bridging_nonregular = [
    R
    for R in records
    if R["complete_bridge"]
    and
    not R["regular"]
]


print(
    "Regular but NOT complete exchange:",
    len(regular_nonbridges)
)

for R in regular_nonbridges:
    print(
        "   ",
        cycle_string(R["s"]),
        matrix_string(R["B"])
    )


print()

print(
    "Complete exchange but NOT regular:",
    len(bridging_nonregular)
)

for R in bridging_nonregular:
    print(
        "   ",
        cycle_string(R["s"]),
        matrix_string(R["B"])
    )


# =============================================================================
# 13. GROUP-ORDER / ORBIT-STRUCTURE CLASSIFICATION
# =============================================================================

section("13. ACTION-TYPE CLASSIFICATION")

action_classes = defaultdict(list)

for R in records:

    key = (
        R["group_order"],
        R["orbit_sizes"],
        R["free"],
        R["transitive"],
        R["regular"],
    )

    action_classes[key].append(R)


print(
    "Number of distinct action classes:",
    len(action_classes)
)

print()

for idx, (key, Rs) in enumerate(
    sorted(
        action_classes.items(),
        key=lambda item: str(item[0])
    ),
    start=1
):

    group_order, orbit_sizes, free, trans, regular = key

    print(f"CLASS {idx}")

    print("  |G|         :", group_order)
    print("  orbit sizes :", orbit_sizes)
    print("  free        :", free)
    print("  transitive  :", trans)
    print("  regular     :", regular)
    print("  count       :", len(Rs))

    print(
        "  B types     :",
        sorted(
            {
                matrix_string(R["B"])
                for R in Rs
            }
        )
    )

    tags = [
        R["special"]
        for R in Rs
        if R["special"]
    ]

    if tags:
        print(
            "  contains    :",
            tags
        )

    print()


# =============================================================================
# 14. C8 RESTRICTION
# =============================================================================

section("14. WHAT ADDITIONAL RESTRICTION DOES C8 CONTRIBUTE?")

C8_records = [
    R
    for R in records
    if R["c8_contained"]
]

nonC8_records = [
    R
    for R in records
    if not R["c8_contained"]
]


print(
    "Commuting candidates with matching fully in C8:",
    len(C8_records)
)

print(
    "Commuting candidates outside C8             :",
    len(nonC8_records)
)

print()

print(
    "Regular among C8-contained                  :",
    sum(R["regular"] for R in C8_records),
    "/",
    len(C8_records)
)

print(
    "Regular outside C8                          :",
    sum(R["regular"] for R in nonC8_records),
    "/",
    len(nonC8_records)
)

print()


print("C8-contained candidates:")
print()

for R in C8_records:

    print(
        f"{R['special'] or '-':>3s}   "
        f"{cycle_string(R['s']):30s} "
        f"B={matrix_string(R['B']):18s} "
        f"orbits={str(R['orbit_sizes']):10s} "
        f"regular={R['regular']}"
    )


# =============================================================================
# 15. M1 / M2 LOCATION IN THE COMPLETE CLASSIFICATION
# =============================================================================

section("15. M1 / M2 LOCATION IN S_K")


def locate_special(tag):

    for R in records:

        if R["special"] == tag:
            return R

    raise RuntimeError(
        f"{tag} not found."
    )


RM1 = locate_special("M1")
RM2 = locate_special("M2")


for tag, R in (
    ("M1", RM1),
    ("M2", RM2),
):

    print(tag)

    print("  s                 :", cycle_string(R["s"]))
    print("  B(s)              :", matrix_string(R["B"]))
    print("  complete bridge   :", R["complete_bridge"])
    print("  in K              :", R["in_K"])
    print("  C8-contained      :", R["c8_contained"])
    print("  |G|               :", R["group_order"])
    print("  orbit sizes       :", R["orbit_sizes"])
    print("  free              :", R["free"])
    print("  transitive        :", R["transitive"])
    print("  regular           :", R["regular"])

    print()


# =============================================================================
# 16. OPTIONAL ALGEBRAIC COMPRESSION CHECK
# =============================================================================
#
# Because every retained s commutes with K, if s is not already in K then
#
#   <K,s>
#
# has at most the simple direct-product-like size expected from adjoining
# one independent involution.
#
# We do not assume the result; we tabulate it.
#
# =============================================================================

section("16. ALGEBRAIC COMPRESSION CHECK")

inK_records = [
    R
    for R in records
    if R["in_K"]
]

outsideK_records = [
    R
    for R in records
    if not R["in_K"]
]


print("Candidates already in K :", len(inK_records))
print("Candidates outside K    :", len(outsideK_records))

print()

print(
    "|G| for s in K          :",
    sorted(
        {
            R["group_order"]
            for R in inK_records
        }
    )
)

print(
    "|G| for s outside K     :",
    sorted(
        {
            R["group_order"]
            for R in outsideK_records
        }
    )
)

print()

print(
    "All outside-K candidates produce abstract order-8 groups:",
    all(
        R["group_order"] == 8
        for R in outsideK_records
    )
)

print(
    "All outside-K group elements have order <=2:",
    all(
        all(
            perm_order(g) in (1, 2)
            for g in R["G"]
        )
        for R in outsideK_records
    )
)


# =============================================================================
# 17. REGULARITY VS FIXED POINTS
# =============================================================================

section("17. NONIDENTITY FIXED-POINT LEDGER")


def nonidentity_fixed_point_counts(G):

    counts = []

    for g in G:

        if g == IDENTITY:
            continue

        count = sum(
            g[v] == v
            for v in X8
        )

        counts.append(count)

    return tuple(sorted(counts))


for R in records:
    R["nonidentity_fixed_counts"] = (
        nonidentity_fixed_point_counts(R["G"])
    )


fixed_count_classes = Counter(
    R["nonidentity_fixed_counts"]
    for R in records
)

for pattern, count in sorted(
    fixed_count_classes.items(),
    key=lambda item: str(item[0])
):

    print(
        f"{count:>3d} x fixed-point pattern "
        f"{pattern}"
    )


print()

print(
    "M1 nonidentity fixed-point counts:",
    RM1["nonidentity_fixed_counts"]
)

print(
    "M2 nonidentity fixed-point counts:",
    RM2["nonidentity_fixed_counts"]
)


# =============================================================================
# 18. PREREGISTERED OUTCOME CLASSIFICATION
# =============================================================================

section("18. PREREGISTERED OUTCOME CLASSIFICATION")

if BRIDGE_IFF_REGULAR:

    STATUS = (
        "OUTCOME A — EXACT ORBIT-BRIDGE LAW: "
        "within the complete commuting fixed-point-free class S_K, "
        "complete exchange of the pre-existing K-orbits is equivalent "
        "to regularity of <K,s> on X8."
    )

elif BRIDGE_NECESSARY and not BRIDGE_SUFFICIENT:

    STATUS = (
        "OUTCOME B — BRIDGING NECESSARY BUT INSUFFICIENT: "
        "every regular candidate bridges the K-orbits, but some "
        "bridging candidates remain nonregular."
    )

else:

    regular_B_types = {
        R["B"]
        for R in regular_records
    }

    nonregular_B_types = {
        R["B"]
        for R in nonregular_records
    }

    if regular_B_types & nonregular_B_types:

        STATUS = (
            "OUTCOME C — ORBIT-TRANSITION TYPE DOES NOT CLEANLY "
            "DISCRIMINATE REGULARITY: regular and nonregular candidates "
            "share at least one B(s) type."
        )

    else:

        STATUS = (
            "OUTCOME D — B(s) SEPARATES ACTION TYPES, BUT NOT THROUGH "
            "THE SIMPLE COMPLETE-EXCHANGE IFF CRITERION. "
            "A more precise structural condition is required."
        )


print(STATUS)


# =============================================================================
# 19. C8-SPECIFIC CONTEXT
# =============================================================================

section("19. C8-SPECIFIC CONTEXT")

C8_regular = [
    R
    for R in C8_records
    if R["regular"]
]

C8_nonregular = [
    R
    for R in C8_records
    if not R["regular"]
]


print(
    "C8-contained commuting candidates:",
    len(C8_records)
)

print(
    "C8-contained regular candidates  :",
    len(C8_regular)
)

print(
    "C8-contained nonregular candidates:",
    len(C8_nonregular)
)

print()


if len(C8_records) > 0:

    if len(C8_regular) == 1:

        C8_CONTEXT = (
            "Within S_K, the C8 containment restriction leaves exactly "
            "one regular matching."
        )

    elif len(C8_regular) == 0:

        C8_CONTEXT = (
            "Within S_K, no C8-contained matching is regular."
        )

    else:

        C8_CONTEXT = (
            "Within S_K, multiple C8-contained matchings are regular."
        )

else:

    C8_CONTEXT = (
        "No commuting candidate is C8-contained."
    )


print(C8_CONTEXT)


# =============================================================================
# 20. TRUTH-PACKET LEDGER
# =============================================================================

section("20. SIM14.3 TRUTH-PACKET LEDGER")

print(f"{'K=<p,h> has order 4':54s}: {len(K) == 4}")
print(f"{'K ~= V4':54s}: {K_IS_V4}")

print()

print(
    f"{'Number of pre-symplectic K-orbits':54s}: "
    f"{len(K_ORBITS)}"
)

print(
    f"{'K-orbit size multiset':54s}: "
    f"{sorted(len(O) for O in K_ORBITS)}"
)

print()

print(
    f"{'All perfect matchings enumerated':54s}: "
    f"{len(ALL_INVOLUTIONS)}"
)

print(
    f"{'Expected perfect matching count':54s}: "
    f"105"
)

print(
    f"{'Commuting class |S_K|':54s}: "
    f"{len(S_K)}"
)

print()

print(
    f"{'M1 belongs to S_K':54s}: "
    f"{s1 in S_K}"
)

print(
    f"{'M2 belongs to S_K':54s}: "
    f"{s2 in S_K}"
)

print()

print(
    f"{'M1 B(s)':54s}: "
    f"{matrix_string(RM1['B'])}"
)

print(
    f"{'M2 B(s)':54s}: "
    f"{matrix_string(RM2['B'])}"
)

print()

print(
    f"{'M1 regular':54s}: "
    f"{RM1['regular']}"
)

print(
    f"{'M2 regular':54s}: "
    f"{RM2['regular']}"
)

print()

print(
    f"{'B(s) perfectly classifies regularity':54s}: "
    f"{B_PERFECTLY_CLASSIFIES_REGULARITY}"
)

print(
    f"{'Complete orbit exchange necessary':54s}: "
    f"{BRIDGE_NECESSARY}"
)

print(
    f"{'Complete orbit exchange sufficient':54s}: "
    f"{BRIDGE_SUFFICIENT}"
)

print(
    f"{'Complete orbit exchange iff regularity':54s}: "
    f"{BRIDGE_IFF_REGULAR}"
)

print()

print(
    f"{'C8-contained candidates inside S_K':54s}: "
    f"{len(C8_records)}"
)

print(
    f"{'C8-contained regular candidates':54s}: "
    f"{len(C8_regular)}"
)

print()

print("STATUS:")
print()
print(STATUS)

print()
print("C8 CONTEXT:")
print()
print(C8_CONTEXT)


# =============================================================================
# 21. INTERPRETATION RULES
# =============================================================================

section("21. INTERPRETATION RULES")

print("""
A. SIM14.3 first derives the orbit structure of K=<p,h>. It does not
   hard-code a two-orbit decomposition.

B. B(s) records how a candidate conjugate matching acts relative to those
   pre-existing K-orbits. B(s) is computed without using the regularity
   result.

C. Regularity is independently determined from the actual permutation
   action of G_s=<K,s>.

D. If complete K-orbit exchange is equivalent to regularity throughout
   S_K, SIM14.3 has isolated an exact finite structural discriminator
   inside the tested class.

E. If complete exchange is only necessary, only sufficient, or neither,
   the stronger iff statement must be rejected.

F. C8 containment is a secondary filter. A regularity law found throughout
   S_K must not be falsely attributed to C8.

G. M1 must not be discarded merely because its action is nonregular.
   SIM14.3 classifies the mathematical distinction; it does not decide
   which embedding is physically preferred.

H. No PG(2,2) reconstruction is needed here. The abstract Fano incidence
   downstream of (Z_2)^3 was already established in SIM14.1/14.2.

I. SIM14.3 makes no physical claim about octonions, nonassociativity,
   E8/RE8/4_21, H4, spinors, quantum mechanics, or physical spacetime.

J. If an exact iff criterion appears, the next step is an analytic proof,
   not immediately another simulation.
""")


# =============================================================================
# 22. IMPLEMENTATION SANITY CHECKS
# =============================================================================

section("22. IMPLEMENTATION SANITY CHECKS")

assert len(C8_EDGES) == 8

assert perm_order(p) == 2
assert perm_order(h) == 2
assert commute(p, h)

assert K_IS_V4

assert len(ALL_INVOLUTIONS) == 105
assert ALL_UNIQUE
assert ALL_VALID_FPF

assert s1 in ALL_INVOLUTIONS
assert s2 in ALL_INVOLUTIONS

assert s1 in S_K
assert s2 in S_K

assert M1 == frozenset(
    frozenset(e)
    for e in (
        (0, 1),
        (2, 4),
        (3, 5),
        (6, 7),
    )
)

assert M2 == frozenset(
    frozenset(e)
    for e in (
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
    )
)

assert M1.issubset(C8_EDGES)
assert M2.issubset(C8_EDGES)

# Reproduce the key SIM14.2 distinction.
assert not RM1["regular"]
assert RM2["regular"]

assert RM1["orbit_sizes"] == (4, 4)
assert RM2["orbit_sizes"] == (8,)

# Literal permutation groups have trivial action kernel.
assert all(
    R["kernel_size"] == 1
    for R in records
)

print("All implementation sanity checks PASS.")


# =============================================================================
# 23. FINAL CLASSIFICATION TABLE
# =============================================================================

section("23. COMPACT FINAL CLASSIFICATION")

print(
    "ID   tag   inK   C8   bridge   B(s)              "
    "|G|   orbit sizes    free   trans   regular"
)

print("-" * WIDTH)

for R in records:

    print(
        f"{R['id']:>2d}   "
        f"{R['special']:<3s}   "
        f"{str(R['in_K']):>4s} "
        f"{str(R['c8_contained']):>4s} "
        f"{str(R['complete_bridge']):>7s}   "
        f"{matrix_string(R['B']):18s} "
        f"{R['group_order']:>3d}   "
        f"{str(R['orbit_sizes']):13s} "
        f"{str(R['free']):>5s}   "
        f"{str(R['transitive']):>5s}   "
        f"{str(R['regular']):>7s}"
    )


# =============================================================================
# 24. FINAL TRUTH PACKET
# =============================================================================

section("24. SIM14.3 FINAL TRUTH PACKET")

print("Question:")
print()
print(
    "    What distinguishes the regular M2 realization from the "
    "nonregular M1 realization relative to K=<p,h>?"
)

print()
print("Frozen:")
print()
print("    X8")
print("    C8")
print("    p")
print("    h")
print("    projection partition")
print("    M1")
print("    M2")

print()
print("Candidate class:")
print()
print(
    "    all fixed-point-free involutions s commuting with both p and h"
)

print()
print("Primary diagnostic:")
print()
print(
    "    B_ab(s) = #{x in O_a : s(x) in O_b}"
)

print()
print("Independent endpoint:")
print()
print(
    "    regularity of <K,s> on X8"
)

print()
print("RESULT:")
print()
print(STATUS)

print()
print("C8 SECONDARY RESULT:")
print()
print(C8_CONTEXT)

print()
print("Interpretive ceiling:")
print()
print(
    "Finite carrier mathematics only. "
    "No downstream physical identification is licensed by SIM14.3."
)


# =============================================================================
# COMPLETE
# =============================================================================

banner("SIM14.3 COMPLETE")

print("STATUS:")
print()
print(STATUS)

print()
print("If an exact iff criterion has appeared, stop the numerical sequence")
print("and prove the criterion analytically before promoting it.")




~~~~~~~~~~~~~~~~~~~~~~~~




RESULTS:



================================================================================================================
SIM14.3 — V4 ORBIT-BRIDGE CLASSIFICATION

================================================================================================================

502 |

NO ISP / LCO dynamics are run in SIM14.3.

NO stochastic sampling, fitting, optimization, or new geometry.

505 |

Primary question:

507 |

    What property of s relative to K=<p,h> determines regularity

    of G_s=<K,s> on the eight-state microscopic carrier?

510 |

The orbit-transition matrix B(s) is measured independently of

the subsequent regularity classification.


----------------------------------------------------------------------------------------------------------------
1. FROZEN SIM14 LABORATORY

----------------------------------------------------------------------------------------------------------------

X8                  : (0, 1, 2, 3, 4, 5, 6, 7)

C8 cycle order      : (0, 1, 3, 5, 7, 6, 4, 2)

C8 edges            : {0-1, 0-2, 1-3, 2-4, 3-5, 4-6, 5-7, 6-7}

525 |

Projection basins   : {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7]}

534 |

p                   : (0 1)(2 3)(4 5)(6 7)

h                   : (0 7)(1 6)(2 5)(3 4)

538 |

M1                  : {0-1, 2-4, 3-5, 6-7}

s1                  : (0 1)(2 4)(3 5)(6 7)

542 |

M2                  : {0-2, 1-3, 4-6, 5-7}

s2                  : (0 2)(1 3)(4 6)(5 7)


----------------------------------------------------------------------------------------------------------------
2. PRE-SYMPLECTIC BASE GROUP K = <p,h>

----------------------------------------------------------------------------------------------------------------

|K|                 : 4

Element orders      : {1: 1, 2: 3}

p,h commute         : True

K ~= V4             : True

585 |

K elements:

    ()

    (0 1)(2 3)(4 5)(6 7)

    (0 6)(1 7)(2 4)(3 5)

    (0 7)(1 6)(2 5)(3 4)


----------------------------------------------------------------------------------------------------------------
3. K ACTION ON X8

----------------------------------------------------------------------------------------------------------------

Number of K-orbits  : 2

Orbit-size multiset : [4, 4]

606 |

O0 = [0, 1, 6, 7]

O1 = [2, 3, 4, 5]

613 |

vertex   orbit-size   stabilizer-size

  0           4                 1

  1           4                 1

  2           4                 1

  3           4                 1

  4           4                 1

  5           4                 1

  6           4                 1

  7           4                 1


----------------------------------------------------------------------------------------------------------------
4. M1 / M2 RELATION TO THE PRE-EXISTING K-ORBITS

----------------------------------------------------------------------------------------------------------------

B(s1) = [4,0; 0,4]

B(s2) = [0,4; 4,0]

651 |

Rows = source K-orbits.

Columns = destination K-orbits.

654 |

No interpretation of these matrices is used to define regularity.


----------------------------------------------------------------------------------------------------------------
5. SIM14.2 REPRODUCTION THROUGH THE K+s LENS

----------------------------------------------------------------------------------------------------------------

M1

------------------------------------------------------------------------

s                     : (0 1)(2 4)(3 5)(6 7)

matching              : {0-1, 2-4, 3-5, 6-7}

B(s)                  : [4,0; 0,4]

730 |

|<K,s>|               : 8

element-order hist    : {1: 1, 2: 7}

734 |

orbits                : [[0, 1, 6, 7], [2, 3, 4, 5]]

orbit-size multiset   : (4, 4)

stabilizer sizes      : (2, 2, 2, 2, 2, 2, 2, 2)

kernel size           : 1

747 |

transitive            : False

free                  : False

regular               : False

752 |

matching in C8        : True

s is C8 automorphism  : False

preserves Pi setwise : True

760 |

M2

------------------------------------------------------------------------

s                     : (0 2)(1 3)(4 6)(5 7)

matching              : {0-2, 1-3, 4-6, 5-7}

B(s)                  : [0,4; 4,0]

730 |

|<K,s>|               : 8

element-order hist    : {1: 1, 2: 7}

734 |

orbits                : [[0, 1, 2, 3, 4, 5, 6, 7]]

orbit-size multiset   : (8,)

stabilizer sizes      : (1, 1, 1, 1, 1, 1, 1, 1)

kernel size           : 1

747 |

transitive            : True

free                  : True

regular               : True

752 |

matching in C8        : True

s is C8 automorphism  : False

preserves Pi setwise : True

760 |


----------------------------------------------------------------------------------------------------------------
6. ENUMERATION CONTROL

----------------------------------------------------------------------------------------------------------------

All perfect matchings on 8 labeled vertices: 105

Expected                            : 105

All enumerated involutions unique   : True

All are fixed-point-free involutions: True


----------------------------------------------------------------------------------------------------------------
7. COMMUTING FIXED-POINT-FREE INVOLUTION CLASS S_K

----------------------------------------------------------------------------------------------------------------

|S_K|                        : 13

813 |

Definition:

815 |

  S_K = {s : s^2=e, Fix(s)=empty, [s,p]=[s,h]=e}

817 |

M1 included                  : True

M2 included                  : True


----------------------------------------------------------------------------------------------------------------
8. COMPLETE STRUCTURAL CLASSIFICATION OF S_K

----------------------------------------------------------------------------------------------------------------

ID   tag   s                              inK  C8   B(s)              |G|   orbits       free  trans  regular

----------------------------------------------------------------------------------------------------------------

 1         (0 1)(2 3)(4 5)(6 7)           True False [4,0; 0,4]           4   (4, 4)        True  False   False

 2   M1    (0 1)(2 4)(3 5)(6 7)           False True [4,0; 0,4]           8   (4, 4)       False  False   False

 3         (0 1)(2 5)(3 4)(6 7)           False False [4,0; 0,4]           8   (4, 4)       False  False   False

 4   M2    (0 2)(1 3)(4 6)(5 7)           False True [0,4; 4,0]           8   (8,)          True   True    True

 5         (0 3)(1 2)(4 7)(5 6)           False False [0,4; 4,0]           8   (8,)          True   True    True

 6         (0 4)(1 5)(2 6)(3 7)           False False [0,4; 4,0]           8   (8,)          True   True    True

 7         (0 5)(1 4)(2 7)(3 6)           False False [0,4; 4,0]           8   (8,)          True   True    True

 8         (0 6)(1 7)(2 3)(4 5)           False False [4,0; 0,4]           8   (4, 4)       False  False   False

 9         (0 6)(1 7)(2 4)(3 5)           True False [4,0; 0,4]           4   (4, 4)        True  False   False

10         (0 6)(1 7)(2 5)(3 4)           False False [4,0; 0,4]           8   (4, 4)       False  False   False

11         (0 7)(1 6)(2 3)(4 5)           False False [4,0; 0,4]           8   (4, 4)       False  False   False

12         (0 7)(1 6)(2 4)(3 5)           False False [4,0; 0,4]           8   (4, 4)       False  False   False

13         (0 7)(1 6)(2 5)(3 4)           True False [4,0; 0,4]           4   (4, 4)        True  False   False


----------------------------------------------------------------------------------------------------------------
9. DOES B(s) CLASSIFY THE ACTION TYPE?

----------------------------------------------------------------------------------------------------------------

Number of distinct B(s) types: 2

884 |

B(s) = [0,4; 4,0]

  count              : 4

  group orders       : [8]

  orbit types        : [(8,)]

  free values        : [True]

  transitive values  : [True]

  regular values     : [True]

  regularity uniform : True

  contains           : ['M2']

981 |

B(s) = [4,0; 0,4]

  count              : 9

  group orders       : [4, 8]

  orbit types        : [(4, 4)]

  free values        : [False, True]

  transitive values  : [False]

  regular values     : [False]

  regularity uniform : True

  contains           : ['M1']

981 |

B(s) type perfectly classifies regularity: True


----------------------------------------------------------------------------------------------------------------
10. REGULARITY LEDGER INSIDE S_K

----------------------------------------------------------------------------------------------------------------

Total commuting candidates     : 13

Regular candidates             : 4

Nonregular candidates          : 9

1017 |

Regular fraction              : 0.3076923076923077

1028 |

Regular B(s) types            : ['[0,4; 4,0]']

Nonregular B(s) types         : ['[4,0; 0,4]']


----------------------------------------------------------------------------------------------------------------
11. CANDIDATE ORBIT-BRIDGE LAW

----------------------------------------------------------------------------------------------------------------

Complete orbit-exchange candidates: 4

Non-exchange candidates            : 9

1118 |

Regular among complete exchanges   : 4 / 4

Regular among non-exchanges        : 0 / 9

1134 |

Complete exchange necessary for regularity: True

Complete exchange sufficient for regularity: True

Complete exchange iff regularity            : True


----------------------------------------------------------------------------------------------------------------
12. COUNTEREXAMPLE SEARCH

----------------------------------------------------------------------------------------------------------------

Regular but NOT complete exchange: 0

1210 |

Complete exchange but NOT regular: 0


----------------------------------------------------------------------------------------------------------------
13. ACTION-TYPE CLASSIFICATION

----------------------------------------------------------------------------------------------------------------

Number of distinct action classes: 3

1251 |

CLASS 1

  |G|         : 4

  orbit sizes : (4, 4)

  free        : True

  transitive  : False

  regular     : False

  count       : 3

  B types     : ['[4,0; 0,4]']

1294 |

CLASS 2

  |G|         : 8

  orbit sizes : (4, 4)

  free        : False

  transitive  : False

  regular     : False

  count       : 6

  B types     : ['[4,0; 0,4]']

  contains    : ['M1']

1294 |

CLASS 3

  |G|         : 8

  orbit sizes : (8,)

  free        : True

  transitive  : True

  regular     : True

  count       : 4

  B types     : ['[0,4; 4,0]']

  contains    : ['M2']

1294 |


----------------------------------------------------------------------------------------------------------------
14. WHAT ADDITIONAL RESTRICTION DOES C8 CONTRIBUTE?

----------------------------------------------------------------------------------------------------------------

Commuting candidates with matching fully in C8: 2

Commuting candidates outside C8             : 11

1326 |

Regular among C8-contained                  : 1 / 2

Regular outside C8                          : 3 / 11

1342 |

C8-contained candidates:

1346 |

 M1   (0 1)(2 4)(3 5)(6 7)           B=[4,0; 0,4]         orbits=(4, 4)     regular=False

 M2   (0 2)(1 3)(4 6)(5 7)           B=[0,4; 4,0]         orbits=(8,)       regular=True


----------------------------------------------------------------------------------------------------------------
15. M1 / M2 LOCATION IN S_K

----------------------------------------------------------------------------------------------------------------

M1

  s                 : (0 1)(2 4)(3 5)(6 7)

  B(s)              : [4,0; 0,4]

  complete bridge   : False

  in K              : False

  C8-contained      : True

  |G|               : 8

  orbit sizes       : (4, 4)

  free              : False

  transitive        : False

  regular           : False

1400 |

M2

  s                 : (0 2)(1 3)(4 6)(5 7)

  B(s)              : [0,4; 4,0]

  complete bridge   : True

  in K              : False

  C8-contained      : True

  |G|               : 8

  orbit sizes       : (8,)

  free              : True

  transitive        : True

  regular           : True

1400 |


----------------------------------------------------------------------------------------------------------------
16. ALGEBRAIC COMPRESSION CHECK

----------------------------------------------------------------------------------------------------------------

Candidates already in K : 3

Candidates outside K    : 10

1436 |

|G| for s in K          : [4]

|G| for s outside K     : [8]

1458 |

All outside-K candidates produce abstract order-8 groups: True

All outside-K group elements have order <=2: True


----------------------------------------------------------------------------------------------------------------
17. NONIDENTITY FIXED-POINT LEDGER

----------------------------------------------------------------------------------------------------------------

  3 x fixed-point pattern (0, 0, 0)

  4 x fixed-point pattern (0, 0, 0, 0, 0, 0, 0)

  6 x fixed-point pattern (0, 0, 0, 0, 0, 4, 4)

1528 |

M1 nonidentity fixed-point counts: (0, 0, 0, 0, 0, 4, 4)

M2 nonidentity fixed-point counts: (0, 0, 0, 0, 0, 0, 0)


----------------------------------------------------------------------------------------------------------------
18. PREREGISTERED OUTCOME CLASSIFICATION

----------------------------------------------------------------------------------------------------------------

OUTCOME A — EXACT ORBIT-BRIDGE LAW: within the complete commuting fixed-point-free class S_K, complete exchange of the pre-existing K-orbits is equivalent to regularity of <K,s> on X8.


----------------------------------------------------------------------------------------------------------------
19. C8-SPECIFIC CONTEXT

----------------------------------------------------------------------------------------------------------------

C8-contained commuting candidates: 2

C8-contained regular candidates  : 1

C8-contained nonregular candidates: 1

1630 |

Within S_K, the C8 containment restriction leaves exactly one regular matching.


----------------------------------------------------------------------------------------------------------------
20. SIM14.3 TRUTH-PACKET LEDGER

----------------------------------------------------------------------------------------------------------------

K=<p,h> has order 4                                   : True

K ~= V4                                               : True

1673 |

Number of pre-symplectic K-orbits                     : 2

K-orbit size multiset                                 : [4, 4]

1685 |

All perfect matchings enumerated                      : 105

Expected perfect matching count                       : 105

Commuting class |S_K|                                 : 13

1702 |

M1 belongs to S_K                                     : True

M2 belongs to S_K                                     : True

1714 |

M1 B(s)                                               : [4,0; 0,4]

M2 B(s)                                               : [0,4; 4,0]

1726 |

M1 regular                                            : False

M2 regular                                            : True

1738 |

B(s) perfectly classifies regularity                  : True

Complete orbit exchange necessary                     : True

Complete orbit exchange sufficient                    : True

Complete orbit exchange iff regularity                : True

1760 |

C8-contained candidates inside S_K                    : 2

C8-contained regular candidates                       : 1

1772 |

STATUS:

1775 |

OUTCOME A — EXACT ORBIT-BRIDGE LAW: within the complete commuting fixed-point-free class S_K, complete exchange of the pre-existing K-orbits is equivalent to regularity of <K,s> on X8.

1778 |

C8 CONTEXT:

1780 |

Within S_K, the C8 containment restriction leaves exactly one regular matching.


----------------------------------------------------------------------------------------------------------------
21. INTERPRETATION RULES

----------------------------------------------------------------------------------------------------------------


A. SIM14.3 first derives the orbit structure of K=<p,h>. It does not
   hard-code a two-orbit decomposition.

B. B(s) records how a candidate conjugate matching acts relative to those
   pre-existing K-orbits. B(s) is computed without using the regularity
   result.

C. Regularity is independently determined from the actual permutation
   action of G_s=<K,s>.

D. If complete K-orbit exchange is equivalent to regularity throughout
   S_K, SIM14.3 has isolated an exact finite structural discriminator
   inside the tested class.

E. If complete exchange is only necessary, only sufficient, or neither,
   the stronger iff statement must be rejected.

F. C8 containment is a secondary filter. A regularity law found throughout
   S_K must not be falsely attributed to C8.

G. M1 must not be discarded merely because its action is nonregular.
   SIM14.3 classifies the mathematical distinction; it does not decide
   which embedding is physically preferred.

H. No PG(2,2) reconstruction is needed here. The abstract Fano incidence
   downstream of (Z_2)^3 was already established in SIM14.1/14.2.

I. SIM14.3 makes no physical claim about octonions, nonassociativity,
   E8/RE8/4_21, H4, spinors, quantum mechanics, or physical spacetime.

J. If an exact iff criterion appears, the next step is an analytic proof,
   not immediately another simulation.


----------------------------------------------------------------------------------------------------------------
22. IMPLEMENTATION SANITY CHECKS

----------------------------------------------------------------------------------------------------------------

All implementation sanity checks PASS.


----------------------------------------------------------------------------------------------------------------
23. COMPACT FINAL CLASSIFICATION

----------------------------------------------------------------------------------------------------------------

ID   tag   inK   C8   bridge   B(s)              |G|   orbit sizes    free   trans   regular

----------------------------------------------------------------------------------------------------------------

 1         True False   False   [4,0; 0,4]           4   (4, 4)         True   False     False

 2   M1    False True   False   [4,0; 0,4]           8   (4, 4)        False   False     False

 3         False False   False   [4,0; 0,4]           8   (4, 4)        False   False     False

 4   M2    False True    True   [0,4; 4,0]           8   (8,)           True    True      True

 5         False False    True   [0,4; 4,0]           8   (8,)           True    True      True

 6         False False    True   [0,4; 4,0]           8   (8,)           True    True      True

 7         False False    True   [0,4; 4,0]           8   (8,)           True    True      True

 8         False False   False   [4,0; 0,4]           8   (4, 4)        False   False     False

 9         True False   False   [4,0; 0,4]           4   (4, 4)         True   False     False

10         False False   False   [4,0; 0,4]           8   (4, 4)        False   False     False

11         False False   False   [4,0; 0,4]           8   (4, 4)        False   False     False

12         False False   False   [4,0; 0,4]           8   (4, 4)        False   False     False

13         True False   False   [4,0; 0,4]           4   (4, 4)         True   False     False


----------------------------------------------------------------------------------------------------------------
24. SIM14.3 FINAL TRUTH PACKET

----------------------------------------------------------------------------------------------------------------

Question:

1926 |

    What distinguishes the regular M2 realization from the nonregular M1 realization relative to K=<p,h>?

1932 |

Frozen:

1934 |

    X8

    C8

    p

    h

    projection partition

    M1

    M2

1943 |

Candidate class:

1945 |

    all fixed-point-free involutions s commuting with both p and h

1950 |

Primary diagnostic:

1952 |

    B_ab(s) = #{x in O_a : s(x) in O_b}

1957 |

Independent endpoint:

1959 |

    regularity of <K,s> on X8

1964 |

RESULT:

1966 |

OUTCOME A — EXACT ORBIT-BRIDGE LAW: within the complete commuting fixed-point-free class S_K, complete exchange of the pre-existing K-orbits is equivalent to regularity of <K,s> on X8.

1969 |

C8 SECONDARY RESULT:

1971 |

Within S_K, the C8 containment restriction leaves exactly one regular matching.

1974 |

Interpretive ceiling:

1976 |

Finite carrier mathematics only. No downstream physical identification is licensed by SIM14.3.


================================================================================================================
SIM14.3 COMPLETE

================================================================================================================

STATUS:

1990 |

OUTCOME A — EXACT ORBIT-BRIDGE LAW: within the complete commuting fixed-point-free class S_K, complete exchange of the pre-existing K-orbits is equivalent to regularity of <K,s> on X8.

1993 |

If an exact iff criterion has appeared, stop the numerical sequence

and prove the criterion analytically before promoting it.
