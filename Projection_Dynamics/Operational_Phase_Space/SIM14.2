# =============================================================================
# SIM14.2 — SYMPLECTIC EMBEDDING INVARIANCE PROBE
# =============================================================================
#
# PURPOSE
# -------
# SIM14.0 found two distinct C8-contained canonical symplectic conjugate
# matchings:
#
#   M1 = {0-1, 2-4, 3-5, 6-7}
#   M2 = {0-2, 1-3, 4-6, 5-7}
#
# SIM14.1 used M2 through
#
#   s_Omega,2 = (0 2)(1 3)(4 6)(5 7)
#
# and found
#
#   <p,h,s_Omega,2> ~= (Z_2)^3
#
# acting regularly on the eight microscopic states, followed by the
# standard seven-direction / seven-V4 / PG(2,2) incidence structure.
#
# SIM14.2 changes ONE structural input:
#
#   s_Omega,2  --->  s_Omega,1
#
# with
#
#   s_Omega,1 = (0 1)(2 4)(3 5)(6 7).
#
# Everything else remains frozen.
#
# PRIMARY QUESTION
# ----------------
# Does the alternative SIM14.0-compatible symplectic embedding also generate
# a regular elementary abelian group of order eight?
#
#   G1 = <p,h,s_Omega,1>  ?~=  (Z_2)^3
#
# compared with the SIM14.1 reference
#
#   G2 = <p,h,s_Omega,2> ~= (Z_2)^3.
#
# SECONDARY QUESTIONS
# -------------------
# If both branches produce regular (Z_2)^3:
#
#   * do both produce 7 nonzero directions?
#   * do both produce 7 V4 subgroups?
#   * do both produce 7 distinct 4+4 decompositions?
#   * do both recover PG(2,2)?
#   * are they merely abstractly equivalent, or also equivalent relative
#     to the frozen C8 / projection laboratory?
#
# IMPORTANT INTERPRETIVE CEILING
# ------------------------------
# SIM14.2 does NOT establish:
#
#   * octonions
#   * nonassociativity
#   * E8 / RE8 / 4_21
#   * H4
#   * spin / spinors
#   * quantum mechanics
#   * physical Fano structure
#
# It tests only embedding invariance of the finite carrier structure already
# encountered in SIM14.0 / SIM14.1.
#
# NO ISP.
# NO LCO.
# NO stochastic sampling.
# NO fitting.
# NO optimization.
# NO new carrier structures.
#
# =============================================================================

from itertools import permutations, combinations, product
from collections import Counter, defaultdict, deque


# =============================================================================
# GLOBALS / PRINTING
# =============================================================================

WIDTH = 108
N = 8
VERTICES = tuple(range(N))
IDENTITY = tuple(range(N))


def banner(title):
    print("\n" + "=" * WIDTH)
    print(title)
    print("=" * WIDTH)


def section(title):
    print("\n" + "-" * WIDTH)
    print(title)
    print("-" * WIDTH)


def bitstr(w):
    return "".join(map(str, w))


# =============================================================================
# PERMUTATION UTILITIES
# =============================================================================
#
# A permutation g is represented by
#
#     g = (g(0), g(1), ..., g(7)).
#
# compose(g,h) means
#
#     (g o h)(x) = g(h(x)).
#
# =============================================================================

def compose(g, h):
    return tuple(g[h[i]] for i in VERTICES)


def inverse(g):
    inv = [None] * N
    for i, j in enumerate(g):
        inv[j] = i
    return tuple(inv)


def conjugate(a, g):
    return compose(compose(a, g), inverse(a))


def perm_order(g):
    x = IDENTITY

    for k in range(1, 100):
        x = compose(g, x)
        if x == IDENTITY:
            return k

    raise RuntimeError("Permutation order exceeded search bound.")


def commute(g, h):
    return compose(g, h) == compose(h, g)


def permutation_cycles(g, include_fixed=False):
    seen = set()
    cycles = []

    for start in VERTICES:
        if start in seen:
            continue

        cyc = []
        x = start

        while x not in seen:
            seen.add(x)
            cyc.append(x)
            x = g[x]

        if len(cyc) > 1 or include_fixed:
            cycles.append(tuple(cyc))

    return cycles


def cycle_string(g):
    cycles = permutation_cycles(g, include_fixed=False)

    if not cycles:
        return "()"

    return "".join(
        "(" + " ".join(map(str, cyc)) + ")"
        for cyc in cycles
    )


def generated_group(generators):
    gens = list(generators)

    group = {IDENTITY}
    queue = deque([IDENTITY])

    while queue:
        x = queue.popleft()

        for g in gens:
            for y in (compose(g, x), compose(x, g)):
                if y not in group:
                    group.add(y)
                    queue.append(y)

    return frozenset(group)


def subgroup_generated(elements):
    return generated_group(elements)


# =============================================================================
# GRAPH / PARTITION UTILITIES
# =============================================================================

C8_CYCLE_ORDER = (0, 1, 3, 5, 7, 6, 4, 2)

C8_EDGES = frozenset(
    frozenset(
        (
            C8_CYCLE_ORDER[i],
            C8_CYCLE_ORDER[(i + 1) % N]
        )
    )
    for i in range(N)
)

PROJECTION_BASINS = (
    frozenset((0, 1)),
    frozenset((2, 3)),
    frozenset((4, 5)),
    frozenset((6, 7)),
)


def edge_tuple(edge):
    return tuple(sorted(edge))


def edge_set_string(edges):
    return "{" + ", ".join(
        f"{a}-{b}"
        for a, b in sorted(edge_tuple(e) for e in edges)
    ) + "}"


def image_edge(edge, g):
    a, b = tuple(edge)
    return frozenset((g[a], g[b]))


def image_edge_set(edges, g):
    return frozenset(
        image_edge(e, g)
        for e in edges
    )


def is_graph_automorphism(g, edges=C8_EDGES):
    return image_edge_set(edges, g) == edges


def matching_from_involution(g):
    if perm_order(g) != 2:
        return None

    if any(g[v] == v for v in VERTICES):
        return None

    return frozenset(
        frozenset((v, g[v]))
        for v in VERTICES
    )


def image_partition(partition, g):
    return frozenset(
        frozenset(g[v] for v in block)
        for block in partition
    )


def preserves_partition_setwise(g):
    return (
        image_partition(PROJECTION_BASINS, g)
        ==
        frozenset(PROJECTION_BASINS)
    )


def preserves_partition_labeled(g):
    for block in PROJECTION_BASINS:
        image = frozenset(g[v] for v in block)

        if image != block:
            return False

    return True


def basin_permutation(g):
    block_to_idx = {
        block: idx
        for idx, block in enumerate(PROJECTION_BASINS)
    }

    out = []

    for block in PROJECTION_BASINS:
        image = frozenset(g[v] for v in block)

        if image not in block_to_idx:
            return None

        out.append(block_to_idx[image])

    return tuple(out)


# =============================================================================
# FROZEN SIM14 OPERATIONS
# =============================================================================

# Projection pairing
#
# p = (0 1)(2 3)(4 5)(6 7)
#
p = (1, 0, 3, 2, 5, 4, 7, 6)


# C8 graph half-turn
#
# h = (0 7)(1 6)(2 5)(3 4)
#
h = (7, 6, 5, 4, 3, 2, 1, 0)


# ---------------------------------------------------------------------------
# M1 BRANCH — NEW SIM14.2 INTERVENTION
# ---------------------------------------------------------------------------
#
# M1 = {0-1, 2-4, 3-5, 6-7}
#
# s1 = (0 1)(2 4)(3 5)(6 7)
#
s1 = (1, 0, 4, 5, 2, 3, 7, 6)


# ---------------------------------------------------------------------------
# M2 BRANCH — FROZEN SIM14.1 REFERENCE
# ---------------------------------------------------------------------------
#
# M2 = {0-2, 1-3, 4-6, 5-7}
#
# s2 = (0 2)(1 3)(4 6)(5 7)
#
s2 = (2, 3, 0, 1, 6, 7, 4, 5)


M1 = matching_from_involution(s1)
M2 = matching_from_involution(s2)


# =============================================================================
# 0. HEADER
# =============================================================================

banner("SIM14.2 — SYMPLECTIC EMBEDDING INVARIANCE PROBE")

print()
print("NO ISP / LCO dynamics are run in SIM14.2.")
print("NO stochastic sampling, fitting, or optimization is permitted.")
print()
print("Only structural intervention:")
print()
print("    s_Omega,2  --->  s_Omega,1")
print()
print("Everything else remains frozen from SIM14.0 / SIM14.1.")


# =============================================================================
# 1. FROZEN INPUTS
# =============================================================================

section("1. FROZEN C8 / PROJECTION LABORATORY")

print("Microscopic states :", VERTICES)
print("C8 cycle order     :", C8_CYCLE_ORDER)
print("C8 edges           :", edge_set_string(C8_EDGES))
print()
print(
    "Projection basins  :",
    {
        i: sorted(block)
        for i, block in enumerate(PROJECTION_BASINS)
    }
)

print()
print("p                 :", cycle_string(p))
print("h                 :", cycle_string(h))

print()
print("M1 / new branch   :", edge_set_string(M1))
print("s_Omega,1         :", cycle_string(s1))

print()
print("M2 / reference    :", edge_set_string(M2))
print("s_Omega,2         :", cycle_string(s2))

print()
print("M1 subset of C8   :", M1.issubset(C8_EDGES))
print("M2 subset of C8   :", M2.issubset(C8_EDGES))


# =============================================================================
# 2. PRIMARY COMMUTATION / INDEPENDENCE GATE
# =============================================================================

section("2. PRIMARY COMMUTATION / INDEPENDENCE GATE")

branches = {
    "M1": s1,
    "M2": s2,
}

primary_gate = {}

for name, s in branches.items():

    G = generated_group([p, h, s])

    pairwise_commuting = (
        commute(p, h)
        and commute(p, s)
        and commute(h, s)
    )

    all_nonidentity_order2 = all(
        perm_order(g) == 2
        for g in G
        if g != IDENTITY
    )

    group_order_8 = len(G) == 8

    z2cubed = (
        pairwise_commuting
        and group_order_8
        and all_nonidentity_order2
    )

    primary_gate[name] = {
        "G": G,
        "pairwise_commuting": pairwise_commuting,
        "order8": group_order_8,
        "all_nonidentity_order2": all_nonidentity_order2,
        "z2cubed": z2cubed,
    }

    print(name)
    print("-" * 48)

    print("[p,h] = e       :", commute(p, h))
    print("[p,s] = e       :", commute(p, s))
    print("[h,s] = e       :", commute(h, s))

    print()
    print("|<p,h,s>|       :", len(G))
    print(
        "Element orders  :",
        dict(
            sorted(
                Counter(
                    perm_order(g)
                    for g in G
                ).items()
            )
        )
    )

    print(
        "(Z_2)^3 gate    :",
        "PASS" if z2cubed else "FAIL"
    )

    print()


# =============================================================================
# 3. REGULAR ACTION GATE
# =============================================================================

section("3. REGULAR ACTION GATE")

regularity = {}

for name, info in primary_gate.items():

    G = info["G"]

    rows = []
    branch_regular = True

    for v in VERTICES:

        orbit = frozenset(
            g[v]
            for g in G
        )

        stabilizer = frozenset(
            g
            for g in G
            if g[v] == v
        )

        rows.append(
            (
                v,
                len(orbit),
                len(stabilizer),
                sorted(orbit),
            )
        )

        if len(orbit) != 8 or len(stabilizer) != 1:
            branch_regular = False

    regularity[name] = branch_regular

    print(name)
    print("-" * 76)
    print("vertex   orbit size   stabilizer size   orbit")

    for v, osize, ssize, orbit in rows:
        print(
            f"{v:>3d}        "
            f"{osize:>3d}             "
            f"{ssize:>3d}          "
            f"{orbit}"
        )

    print()
    print("Regular action PASS:", branch_regular)
    print()


# =============================================================================
# 4. BINARY WORD REPRESENTATION
# =============================================================================

section("4. BINARY WORD REPRESENTATION")

branch_word_maps = {}
branch_perm_words = {}

for name, s in branches.items():

    word_to_perm = {}
    perm_to_words = defaultdict(list)

    for a, b, c in product((0, 1), repeat=3):

        g = IDENTITY

        if a:
            g = compose(p, g)

        if b:
            g = compose(h, g)

        if c:
            g = compose(s, g)

        word = (a, b, c)

        word_to_perm[word] = g
        perm_to_words[g].append(word)

    injective = (
        len(word_to_perm) == 8
        and
        len(set(word_to_perm.values())) == 8
    )

    branch_word_maps[name] = word_to_perm
    branch_perm_words[name] = perm_to_words

    print(name)
    print("-" * 52)
    print("Eight binary words distinct:", injective)
    print()
    print("word     permutation")

    for word in sorted(word_to_perm):
        print(
            f"{bitstr(word):>3s}      "
            f"{cycle_string(word_to_perm[word])}"
        )

    print()


# =============================================================================
# 5. F_2^3 TORSOR COORDINATES
# =============================================================================

section("5. F_2^3 TORSOR COORDINATES")

ORIGIN = 0

branch_vertex_coords = {}
branch_coord_vertices = {}
coordinate_validity = {}


def xor3(a, b):
    return tuple(
        x ^ y
        for x, y in zip(a, b)
    )


for name in branches:

    word_to_perm = branch_word_maps[name]

    if (
        primary_gate[name]["z2cubed"]
        and
        regularity[name]
        and
        len(set(word_to_perm.values())) == 8
    ):

        vertex_to_coord = {}
        coord_to_vertex = {}

        for word, g in word_to_perm.items():

            v = g[ORIGIN]

            if v in vertex_to_coord:
                raise RuntimeError(
                    f"{name}: coordinate collision."
                )

            vertex_to_coord[v] = word
            coord_to_vertex[word] = v

        valid = (
            len(vertex_to_coord) == 8
            and
            len(coord_to_vertex) == 8
        )

    else:

        vertex_to_coord = {}
        coord_to_vertex = {}
        valid = False

    branch_vertex_coords[name] = vertex_to_coord
    branch_coord_vertices[name] = coord_to_vertex
    coordinate_validity[name] = valid

    print(name)
    print("-" * 44)

    if valid:

        print("Conventional origin:", ORIGIN)
        print()
        print("vertex   coordinate")

        for v in VERTICES:
            print(
                f"{v:>3d}      "
                f"{bitstr(vertex_to_coord[v])}"
            )

    else:

        print(
            "F_2^3 coordinates unavailable because "
            "the primary / regularity gates failed."
        )

    print()


# =============================================================================
# 6. TRANSLATION LAW
# =============================================================================

section("6. GENERATOR TRANSLATION LAW")

translation_pass = {}

for name, s in branches.items():

    if not coordinate_validity[name]:
        translation_pass[name] = False
        print(name, ": unavailable")
        continue

    vertex_to_coord = branch_vertex_coords[name]

    generator_vectors = {
        "p": (1, 0, 0),
        "h": (0, 1, 0),
        "s": (0, 0, 1),
    }

    generator_perms = {
        "p": p,
        "h": h,
        "s": s,
    }

    results = {}

    print(name)
    print("-" * 52)

    for gname in ("p", "h", "s"):

        d = generator_vectors[gname]
        g = generator_perms[gname]

        passed = True

        for v in VERTICES:

            lhs = vertex_to_coord[g[v]]
            rhs = xor3(
                vertex_to_coord[v],
                d
            )

            if lhs != rhs:
                passed = False
                break

        results[gname] = passed

        print(
            f"{gname:4s}: x -> x + "
            f"{bitstr(d)}   PASS={passed}"
        )

    translation_pass[name] = all(results.values())

    print()
    print(
        "Full translation law PASS:",
        translation_pass[name]
    )

    print()


# =============================================================================
# 7. C8 ADJACENCY IN EACH BINARY COORDINATE SYSTEM
# =============================================================================

section("7. C8 ADJACENCY IN EACH BINARY COORDINATE SYSTEM")

branch_displacement_hist = {}

for name in branches:

    if not coordinate_validity[name]:
        branch_displacement_hist[name] = None
        print(name, ": unavailable")
        continue

    coords = branch_vertex_coords[name]

    counter = Counter()

    print(name)
    print("-" * 70)
    print(
        "edge     x_i    x_j    displacement"
    )

    for edge in sorted(C8_EDGES, key=edge_tuple):

        i, j = edge_tuple(edge)

        xi = coords[i]
        xj = coords[j]

        d = xor3(xi, xj)

        counter[d] += 1

        print(
            f"{i}-{j:<3d}    "
            f"{bitstr(xi)}    "
            f"{bitstr(xj)}        "
            f"{bitstr(d)}"
        )

    branch_displacement_hist[name] = counter

    print()
    print("Displacement histogram:")

    for d, count in sorted(counter.items()):
        print(
            f"  {bitstr(d)} : {count}"
        )

    print()


# =============================================================================
# 8. ENUMERATE SEVEN NONIDENTITY DIRECTIONS
# =============================================================================

section("8. SEVEN NONIDENTITY DIRECTIONS")

branch_direction_records = {}

for name, s in branches.items():

    if not primary_gate[name]["z2cubed"]:
        branch_direction_records[name] = []
        print(name, ": unavailable")
        continue

    word_to_perm = branch_word_maps[name]

    records = []

    print(name)
    print("-" * 100)

    print(
        "word   permutation                 "
        "C8-auto   matching-in-C8   setwise-Pi   labeled-Pi   basin action"
    )

    for word in sorted(word_to_perm):

        if word == (0, 0, 0):
            continue

        g = word_to_perm[word]

        matching = matching_from_involution(g)

        matching_in_c8 = (
            matching is not None
            and
            matching.issubset(C8_EDGES)
        )

        record = {
            "word": word,
            "perm": g,
            "matching": matching,
            "c8_auto": is_graph_automorphism(g),
            "matching_in_c8": matching_in_c8,
            "setwise_pi": preserves_partition_setwise(g),
            "labeled_pi": preserves_partition_labeled(g),
            "basin_perm": basin_permutation(g),
        }

        records.append(record)

        print(
            f"{bitstr(word):>3s}    "
            f"{cycle_string(g):27s} "
            f"{str(record['c8_auto']):>7s}   "
            f"{str(record['matching_in_c8']):>14s}   "
            f"{str(record['setwise_pi']):>10s}   "
            f"{str(record['labeled_pi']):>10s}   "
            f"{record['basin_perm']}"
        )

    branch_direction_records[name] = records

    print()


# =============================================================================
# 9. ENUMERATE ORDER-4 / V4 SUBGROUPS
# =============================================================================

section("9. ORDER-4 / V4 SUBGROUPS")


def enumerate_order4_subgroups(G):

    nonidentity = [
        g
        for g in G
        if g != IDENTITY
    ]

    subgroups = set()

    for a, b in combinations(nonidentity, 2):

        H = subgroup_generated([a, b])

        if len(H) == 4:
            subgroups.add(frozenset(H))

    return frozenset(subgroups)


branch_v4s = {}
branch_v4_orbits = {}

for name in branches:

    G = primary_gate[name]["G"]

    if not primary_gate[name]["z2cubed"]:
        branch_v4s[name] = frozenset()
        branch_v4_orbits[name] = {}
        print(name, ": unavailable")
        continue

    v4s = enumerate_order4_subgroups(G)

    branch_v4s[name] = v4s

    perm_to_words = branch_perm_words[name]

    sorted_v4s = sorted(
        v4s,
        key=lambda H: sorted(
            min(perm_to_words[g])
            for g in H
        )
    )

    orbit_map = {}

    print(name)
    print("-" * 84)

    print(
        "Number of distinct order-4 subgroups:",
        len(v4s)
    )

    for idx, H in enumerate(sorted_v4s, start=1):

        words = sorted(
            min(perm_to_words[g])
            for g in H
        )

        nonzero_words = [
            w
            for w in words
            if w != (0, 0, 0)
        ]

        unseen = set(VERTICES)
        vertex_orbits = []

        while unseen:

            v = min(unseen)

            orb = frozenset(
                g[v]
                for g in H
            )

            vertex_orbits.append(orb)
            unseen -= set(orb)

        vertex_orbits = tuple(
            sorted(
                vertex_orbits,
                key=lambda x: tuple(sorted(x))
            )
        )

        orbit_map[H] = vertex_orbits

        print(
            f"H{idx}: "
            f"nonzero="
            f"{[bitstr(w) for w in nonzero_words]}"
            f"   orbits="
            f"{[sorted(o) for o in vertex_orbits]}"
        )

    branch_v4_orbits[name] = orbit_map

    print()


# =============================================================================
# 10. DISTINCT 4+4 SECTOR PARTITIONS
# =============================================================================

section("10. V4-INDUCED 4+4 SECTOR PARTITIONS")

branch_sector_partitions = {}

for name in branches:

    partitions = set()

    for H, orbits in branch_v4_orbits[name].items():

        partition = frozenset(orbits)
        partitions.add(partition)

    branch_sector_partitions[name] = frozenset(partitions)

    all_4plus4 = all(
        sorted(len(o) for o in partition) == [4, 4]
        for partition in partitions
    )

    print(name)
    print("-" * 60)
    print(
        "V4 subgroup count             :",
        len(branch_v4s[name])
    )
    print(
        "Distinct 4+4 partitions       :",
        len(partitions)
    )
    print(
        "Every partition is 4+4        :",
        all_4plus4
    )

    print()


# =============================================================================
# 11. PG(2,2) INCIDENCE TEST
# =============================================================================

section("11. PG(2,2) INCIDENCE TEST")

branch_pg22 = {}

for name in branches:

    if not primary_gate[name]["z2cubed"]:

        branch_pg22[name] = False
        print(name, ": unavailable")
        continue

    word_to_perm = branch_word_maps[name]
    perm_to_words = branch_perm_words[name]

    points = tuple(
        word
        for word in sorted(word_to_perm)
        if word != (0, 0, 0)
    )

    lines = []

    for H in branch_v4s[name]:

        line = frozenset(
            min(perm_to_words[g])
            for g in H
            if g != IDENTITY
        )

        lines.append(line)

    lines = tuple(lines)

    point_line_count = Counter()

    for line in lines:
        for point in line:
            point_line_count[point] += 1

    pair_line_counts = Counter()

    for a, b in combinations(points, 2):

        pair_line_counts[(a, b)] = sum(
            1
            for line in lines
            if a in line and b in line
        )

    line_intersections = [
        len(L1 & L2)
        for L1, L2 in combinations(lines, 2)
    ]

    pg22 = (
        len(points) == 7
        and
        len(lines) == 7
        and
        all(len(line) == 3 for line in lines)
        and
        all(point_line_count[pnt] == 3 for pnt in points)
        and
        all(v == 1 for v in pair_line_counts.values())
        and
        all(v == 1 for v in line_intersections)
    )

    branch_pg22[name] = pg22

    print(name)
    print("-" * 60)

    print("Number of points             :", len(points))
    print("Number of lines              :", len(lines))
    print(
        "Every line has 3 points      :",
        all(len(line) == 3 for line in lines)
    )
    print(
        "Every point lies on 3 lines  :",
        all(point_line_count[pnt] == 3 for pnt in points)
    )
    print(
        "Every point pair has 1 line  :",
        all(v == 1 for v in pair_line_counts.values())
    )
    print(
        "Every line pair meets once   :",
        all(v == 1 for v in line_intersections)
    )
    print()
    print(
        "PG(2,2) PASS                 :",
        pg22
    )

    print()


# =============================================================================
# 12. FULL C8 AUTOMORPHISM GROUP
# =============================================================================

section("12. FULL C8 AUTOMORPHISM GROUP")

AUT_C8 = frozenset(
    g
    for g in permutations(VERTICES)
    if is_graph_automorphism(g)
)

print("|Aut(C8)| =", len(AUT_C8))
print("Expected  = 16")
print(
    "PASS      =",
    len(AUT_C8) == 16
)


# =============================================================================
# 13. FROZEN-LAB STABILIZER HIERARCHY
# =============================================================================

section("13. FROZEN-LAB STABILIZER HIERARCHY")

G0 = AUT_C8

G1 = frozenset(
    g
    for g in G0
    if conjugate(g, p) == p
)

G2 = frozenset(
    g
    for g in G1
    if conjugate(g, h) == h
)

G3_setwise = frozenset(
    g
    for g in G2
    if preserves_partition_setwise(g)
)

G3_labeled = frozenset(
    g
    for g in G2
    if preserves_partition_labeled(g)
)

STABILIZER_LEVELS = (
    ("Aut(C8)", G0),
    ("Stab(p)", G1),
    ("Stab(p,h)", G2),
    ("Stab_setwise(p,h,Pi)", G3_setwise),
    ("Stab_labeled(p,h,Pi)", G3_labeled),
)

print("level                               size")
print("-" * 52)

for name, H in STABILIZER_LEVELS:
    print(
        f"{name:36s}"
        f"{len(H):4d}"
    )


# =============================================================================
# 14. DIRECT M1 <-> M2 EXCHANGE TEST
# =============================================================================

section("14. DIRECT M1 / M2 EXCHANGE TEST")


def matching_exchange_elements(group, A, B):

    return tuple(
        g
        for g in group
        if image_edge_set(A, g) == B
    )


exchange_results = {}

print(
    "level                               "
    "exchange?    number"
)

print("-" * 68)

for level_name, H in STABILIZER_LEVELS:

    exchangers = matching_exchange_elements(
        H,
        M1,
        M2
    )

    equivalent = len(exchangers) > 0

    exchange_results[level_name] = {
        "equivalent": equivalent,
        "exchangers": exchangers,
    }

    print(
        f"{level_name:36s}"
        f"{str(equivalent):>8s}"
        f"{len(exchangers):>10d}"
    )

    if exchangers:

        representative = sorted(exchangers)[0]

        print(
            "    representative:",
            cycle_string(representative)
        )


# =============================================================================
# 15. CONJUGACY OF THE ENTIRE DERIVED GROUPS
# =============================================================================

section("15. WHOLE-GROUP EQUIVALENCE UNDER FROZEN-LAB SYMMETRY")

G_M1 = primary_gate["M1"]["G"]
G_M2 = primary_gate["M2"]["G"]


def conjugate_group(a, G):
    return frozenset(
        conjugate(a, g)
        for g in G
    )


whole_group_equivalence = {}

print(
    "level                               "
    "G1 -> G2?    number"
)

print("-" * 68)

for level_name, H in STABILIZER_LEVELS:

    exchangers = tuple(
        a
        for a in H
        if conjugate_group(a, G_M1) == G_M2
    )

    equivalent = len(exchangers) > 0

    whole_group_equivalence[level_name] = {
        "equivalent": equivalent,
        "exchangers": exchangers,
    }

    print(
        f"{level_name:36s}"
        f"{str(equivalent):>8s}"
        f"{len(exchangers):>10d}"
    )

    if exchangers:

        representative = sorted(exchangers)[0]

        print(
            "    representative:",
            cycle_string(representative)
        )


# =============================================================================
# 16. ABSTRACT GROUP COMPARISON
# =============================================================================

section("16. ABSTRACT GROUP COMPARISON")

abstract_equivalent = (
    primary_gate["M1"]["z2cubed"]
    and
    primary_gate["M2"]["z2cubed"]
    and
    regularity["M1"]
    and
    regularity["M2"]
)

print(
    "M1 branch regular (Z_2)^3 :",
    (
        primary_gate["M1"]["z2cubed"]
        and regularity["M1"]
    )
)

print(
    "M2 branch regular (Z_2)^3 :",
    (
        primary_gate["M2"]["z2cubed"]
        and regularity["M2"]
    )
)

print()
print(
    "Abstract carrier equivalence:",
    abstract_equivalent
)


# =============================================================================
# 17. DIRECTION-SIGNATURE COMPARISON
# =============================================================================

section("17. DIRECTION SIGNATURE COMPARISON")


def direction_signature(record):
    """
    Deliberately ignores the binary word label itself.

    This records how a direction sits relative to the frozen laboratory.
    """
    return (
        record["c8_auto"],
        record["matching_in_c8"],
        record["setwise_pi"],
        record["labeled_pi"],
        record["basin_perm"],
    )


branch_signature_counters = {}

for name in branches:

    counter = Counter(
        direction_signature(rec)
        for rec in branch_direction_records[name]
    )

    branch_signature_counters[name] = counter

    print(name)
    print("-" * 84)

    for signature, count in sorted(
        counter.items(),
        key=lambda item: str(item[0])
    ):
        print(
            f"{count:2d} x "
            f"{signature}"
        )

    print()


same_direction_signature_multiset = (
    branch_signature_counters["M1"]
    ==
    branch_signature_counters["M2"]
)

print(
    "Same frozen-lab direction-signature multiset:",
    same_direction_signature_multiset
)


# =============================================================================
# 18. 4+4 PARTITION COMPARISON
# =============================================================================

section("18. 4+4 SECTOR-PARTITION COMPARISON")

P1 = branch_sector_partitions["M1"]
P2 = branch_sector_partitions["M2"]

print(
    "M1 distinct 4+4 partitions:",
    len(P1)
)

print(
    "M2 distinct 4+4 partitions:",
    len(P2)
)

print()
print(
    "Literal partition sets equal:",
    P1 == P2
)

print(
    "Shared literal partitions   :",
    len(P1 & P2)
)

print(
    "M1-only partitions          :",
    len(P1 - P2)
)

print(
    "M2-only partitions          :",
    len(P2 - P1)
)


# =============================================================================
# 19. PARTITION EQUIVALENCE UNDER STABILIZER LEVELS
# =============================================================================

section("19. SECTOR-FAMILY EQUIVALENCE UNDER FROZEN-LAB SYMMETRY")


def image_sector_partition(partition, g):

    return frozenset(
        frozenset(
            g[v]
            for v in block
        )
        for block in partition
    )


def image_sector_family(family, g):

    return frozenset(
        image_sector_partition(partition, g)
        for partition in family
    )


sector_family_equivalence = {}

print(
    "level                               "
    "P1 -> P2?    number"
)

print("-" * 68)

for level_name, H in STABILIZER_LEVELS:

    exchangers = tuple(
        g
        for g in H
        if image_sector_family(P1, g) == P2
    )

    equivalent = len(exchangers) > 0

    sector_family_equivalence[level_name] = {
        "equivalent": equivalent,
        "exchangers": exchangers,
    }

    print(
        f"{level_name:36s}"
        f"{str(equivalent):>8s}"
        f"{len(exchangers):>10d}"
    )

    if exchangers:

        representative = sorted(exchangers)[0]

        print(
            "    representative:",
            cycle_string(representative)
        )


# =============================================================================
# 20. EMBEDDING-INVARIANCE LEDGER
# =============================================================================

section("20. EMBEDDING-INVARIANCE LEDGER")

print(
    f"{'M1 pairwise commuting':48s}: "
    f"{primary_gate['M1']['pairwise_commuting']}"
)

print(
    f"{'M2 pairwise commuting':48s}: "
    f"{primary_gate['M2']['pairwise_commuting']}"
)

print()

print(
    f"{'M1 generates order 8':48s}: "
    f"{primary_gate['M1']['order8']}"
)

print(
    f"{'M2 generates order 8':48s}: "
    f"{primary_gate['M2']['order8']}"
)

print()

print(
    f"{'M1 regular (Z_2)^3':48s}: "
    f"{primary_gate['M1']['z2cubed'] and regularity['M1']}"
)

print(
    f"{'M2 regular (Z_2)^3':48s}: "
    f"{primary_gate['M2']['z2cubed'] and regularity['M2']}"
)

print()

print(
    f"{'M1 PG(2,2)':48s}: "
    f"{branch_pg22['M1']}"
)

print(
    f"{'M2 PG(2,2)':48s}: "
    f"{branch_pg22['M2']}"
)

print()

print(
    f"{'M1 number of V4 subgroups':48s}: "
    f"{len(branch_v4s['M1'])}"
)

print(
    f"{'M2 number of V4 subgroups':48s}: "
    f"{len(branch_v4s['M2'])}"
)

print()

print(
    f"{'M1 distinct 4+4 partitions':48s}: "
    f"{len(P1)}"
)

print(
    f"{'M2 distinct 4+4 partitions':48s}: "
    f"{len(P2)}"
)

print()

print(
    f"{'Same direction-signature multiset':48s}: "
    f"{same_direction_signature_multiset}"
)

print(
    f"{'Literal 4+4 sector families equal':48s}: "
    f"{P1 == P2}"
)

print()

print(
    f"{'M1~M2 under bare Aut(C8)':48s}: "
    f"{exchange_results['Aut(C8)']['equivalent']}"
)

print(
    f"{'M1~M2 under Stab(p)':48s}: "
    f"{exchange_results['Stab(p)']['equivalent']}"
)

print(
    f"{'G1~G2 under bare Aut(C8)':48s}: "
    f"{whole_group_equivalence['Aut(C8)']['equivalent']}"
)

print(
    f"{'G1~G2 under Stab(p)':48s}: "
    f"{whole_group_equivalence['Stab(p)']['equivalent']}"
)

print(
    f"{'Sector families ~ under bare Aut(C8)':48s}: "
    f"{sector_family_equivalence['Aut(C8)']['equivalent']}"
)

print(
    f"{'Sector families ~ under Stab(p)':48s}: "
    f"{sector_family_equivalence['Stab(p)']['equivalent']}"
)


# =============================================================================
# 21. PREREGISTERED CLASSIFICATION
# =============================================================================

section("21. PREREGISTERED CLASSIFICATION")

M1_core = (
    primary_gate["M1"]["z2cubed"]
    and
    regularity["M1"]
)

M2_core = (
    primary_gate["M2"]["z2cubed"]
    and
    regularity["M2"]
)


if not M2_core:

    STATUS = (
        "INVALID REFERENCE: SIM14.1 M2 STRUCTURE "
        "FAILED TO REPRODUCE"
    )

elif not M1_core:

    STATUS = (
        "OUTCOME C — EMBEDDING DEPENDENT: "
        "M1 DOES NOT REPRODUCE THE REGULAR (Z_2)^3 CARRIER"
    )

else:

    both_downstream = (
        branch_pg22["M1"]
        and
        branch_pg22["M2"]
        and
        len(branch_v4s["M1"]) == 7
        and
        len(branch_v4s["M2"]) == 7
        and
        len(P1) == 7
        and
        len(P2) == 7
    )

    if not both_downstream:

        STATUS = (
            "PARTIAL — BOTH EMBEDDINGS GIVE REGULAR (Z_2)^3, "
            "BUT A DOWNSTREAM FINITE-STRUCTURE CHECK FAILED"
        )

    else:

        frozen_lab_equivalent = (
            whole_group_equivalence["Stab(p)"]["equivalent"]
            and
            sector_family_equivalence["Stab(p)"]["equivalent"]
            and
            same_direction_signature_multiset
        )

        if frozen_lab_equivalent:

            STATUS = (
                "OUTCOME A — STRONG EMBEDDING INVARIANCE: "
                "BOTH SYMPLECTIC EMBEDDINGS PRODUCE THE SAME "
                "REGULAR FINITE CARRIER UP TO THE FROZEN LABORATORY"
            )

        else:

            STATUS = (
                "OUTCOME B — ABSTRACT INVARIANCE / EMBEDDED INEQUIVALENCE: "
                "BOTH EMBEDDINGS PRODUCE REGULAR (Z_2)^3 -> PG(2,2), "
                "BUT THEIR RELATION TO THE FROZEN LABORATORY DIFFERS"
            )


print(STATUS)


# =============================================================================
# 22. INTERPRETATION RULES
# =============================================================================

section("22. INTERPRETATION RULES")

print("""
A. If M1 fails to generate a regular (Z_2)^3 action while M2 succeeds,
   then SIM14.1's finite carrier is embedding-dependent.

B. If both M1 and M2 generate regular (Z_2)^3 actions, then the abstract
   binary carrier and its PG(2,2) incidence structure survive the
   SIM14.0 symplectic embedding ambiguity.

C. If both branches are abstractly equivalent but differ relative to the
   frozen C8 / projection laboratory, classify the result as:

       abstract invariance + embedded inequivalence.

   Do NOT collapse those two statements.

D. PG(2,2) is downstream of a regular F_2^3 carrier. Once the regular
   elementary-abelian action is established, the Fano incidence result is
   standard finite geometry and should not be treated as an independent
   surprise.

E. SIM14.2 does not decide whether either embedding is physically preferred.

F. If the embeddings remain dynamically distinguishable later under a
   separately frozen ISP/LCO law, that belongs to a future dynamical SIM.

G. No octonionic, nonassociative, E8, H4, spinorial, quantum, or physical
   interpretation is permitted from SIM14.2 alone.
""")


# =============================================================================
# 23. IMPLEMENTATION SANITY CHECKS
# =============================================================================

section("23. IMPLEMENTATION SANITY CHECKS")

assert len(C8_EDGES) == 8

assert perm_order(p) == 2
assert perm_order(h) == 2
assert perm_order(s1) == 2
assert perm_order(s2) == 2

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

assert len(AUT_C8) == 16

assert G3_labeled.issubset(G3_setwise)
assert G3_setwise.issubset(G2)
assert G2.issubset(G1)
assert G1.issubset(G0)

# SIM14.1 reference reproduction must remain exact.
assert primary_gate["M2"]["z2cubed"]
assert regularity["M2"]
assert branch_pg22["M2"]
assert len(branch_v4s["M2"]) == 7
assert len(P2) == 7

print("All implementation sanity checks PASS.")


# =============================================================================
# 24. FINAL TRUTH PACKET
# =============================================================================

section("24. SIM14.2 TRUTH PACKET")

print("Intervention:")
print()
print("    M2 symplectic conjugate embedding")
print("              versus")
print("    M1 symplectic conjugate embedding")
print()
print("Frozen:")
print()
print("    C8")
print("    p")
print("    h")
print("    projection basins")
print("    finite-group analysis rules")
print()
print("Primary endpoint:")
print()
print("    Does M1 reproduce the regular (Z_2)^3 carrier?")
print()
print("Secondary endpoints:")
print()
print("    7 nonzero directions")
print("    7 V4 subgroups")
print("    7 distinct 4+4 decompositions")
print("    PG(2,2)")
print()
print("Critical distinction:")
print()
print("    abstract group/incidence equivalence")
print("              versus")
print("    equivalence relative to frozen C8/projection structure")
print()
print("STATUS:")
print()
print(STATUS)


# =============================================================================
# COMPLETE
# =============================================================================

banner("SIM14.2 COMPLETE")

print("STATUS:")
print(STATUS)

print()
print("Interpretive ceiling:")
print()
print(
    "SIM14.2 tests whether the finite carrier structure recovered in "
    "SIM14.1 survives the alternative symplectic-compatible embedding "
    "already discovered in SIM14.0. It introduces no new physical or "
    "algebraic structure."
)





~~~~~~~~~~~~~~~~~~~~






RESULTS:



============================================================================================================
SIM14.2 — SYMPLECTIC EMBEDDING INVARIANCE PROBE

============================================================================================================

368 |

NO ISP / LCO dynamics are run in SIM14.2.

NO stochastic sampling, fitting, or optimization is permitted.

371 |

Only structural intervention:

373 |

    s_Omega,2  --->  s_Omega,1

375 |

Everything else remains frozen from SIM14.0 / SIM14.1.


------------------------------------------------------------------------------------------------------------
1. FROZEN C8 / PROJECTION LABORATORY

------------------------------------------------------------------------------------------------------------

Microscopic states : (0, 1, 2, 3, 4, 5, 6, 7)

C8 cycle order     : (0, 1, 3, 5, 7, 6, 4, 2)

C8 edges           : {0-1, 0-2, 1-3, 2-4, 3-5, 4-6, 5-7, 6-7}

388 |

Projection basins  : {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7]}

397 |

p                 : (0 1)(2 3)(4 5)(6 7)

h                 : (0 7)(1 6)(2 5)(3 4)

401 |

M1 / new branch   : {0-1, 2-4, 3-5, 6-7}

s_Omega,1         : (0 1)(2 4)(3 5)(6 7)

405 |

M2 / reference    : {0-2, 1-3, 4-6, 5-7}

s_Omega,2         : (0 2)(1 3)(4 6)(5 7)

409 |

M1 subset of C8   : True

M2 subset of C8   : True


------------------------------------------------------------------------------------------------------------
2. PRIMARY COMMUTATION / INDEPENDENCE GATE

------------------------------------------------------------------------------------------------------------

M1

------------------------------------------------

[p,h] = e       : True

[p,s] = e       : True

[h,s] = e       : True

466 |

|<p,h,s>|       : 8

Element orders  : {1: 1, 2: 7}

(Z_2)^3 gate    : PASS

485 |

M2

------------------------------------------------

[p,h] = e       : True

[p,s] = e       : True

[h,s] = e       : True

466 |

|<p,h,s>|       : 8

Element orders  : {1: 1, 2: 7}

(Z_2)^3 gate    : PASS

485 |


------------------------------------------------------------------------------------------------------------
3. REGULAR ACTION GATE

------------------------------------------------------------------------------------------------------------

M1

----------------------------------------------------------------------------

vertex   orbit size   stabilizer size   orbit

  0          4               2          [0, 1, 6, 7]

  1          4               2          [0, 1, 6, 7]

  2          4               2          [2, 3, 4, 5]

  3          4               2          [2, 3, 4, 5]

  4          4               2          [2, 3, 4, 5]

  5          4               2          [2, 3, 4, 5]

  6          4               2          [0, 1, 6, 7]

  7          4               2          [0, 1, 6, 7]

542 |

Regular action PASS: False

544 |

M2

----------------------------------------------------------------------------

vertex   orbit size   stabilizer size   orbit

  0          8               1          [0, 1, 2, 3, 4, 5, 6, 7]

  1          8               1          [0, 1, 2, 3, 4, 5, 6, 7]

  2          8               1          [0, 1, 2, 3, 4, 5, 6, 7]

  3          8               1          [0, 1, 2, 3, 4, 5, 6, 7]

  4          8               1          [0, 1, 2, 3, 4, 5, 6, 7]

  5          8               1          [0, 1, 2, 3, 4, 5, 6, 7]

  6          8               1          [0, 1, 2, 3, 4, 5, 6, 7]

  7          8               1          [0, 1, 2, 3, 4, 5, 6, 7]

542 |

Regular action PASS: True

544 |


------------------------------------------------------------------------------------------------------------
4. BINARY WORD REPRESENTATION

------------------------------------------------------------------------------------------------------------

M1

----------------------------------------------------

Eight binary words distinct: True

591 |

word     permutation

000      ()

001      (0 1)(2 4)(3 5)(6 7)

010      (0 7)(1 6)(2 5)(3 4)

011      (0 6)(1 7)(2 3)(4 5)

100      (0 1)(2 3)(4 5)(6 7)

101      (2 5)(3 4)

110      (0 6)(1 7)(2 4)(3 5)

111      (0 7)(1 6)

600 |

M2

----------------------------------------------------

Eight binary words distinct: True

591 |

word     permutation

000      ()

001      (0 2)(1 3)(4 6)(5 7)

010      (0 7)(1 6)(2 5)(3 4)

011      (0 5)(1 4)(2 7)(3 6)

100      (0 1)(2 3)(4 5)(6 7)

101      (0 3)(1 2)(4 7)(5 6)

110      (0 6)(1 7)(2 4)(3 5)

111      (0 4)(1 5)(2 6)(3 7)

600 |


------------------------------------------------------------------------------------------------------------
5. F_2^3 TORSOR COORDINATES

------------------------------------------------------------------------------------------------------------

M1

--------------------------------------------

F_2^3 coordinates unavailable because the primary / regularity gates failed.

688 |

M2

--------------------------------------------

Conventional origin: 0

672 |

vertex   coordinate

  0      000

  1      100

  2      001

  3      101

  4      111

  5      011

  6      110

  7      010

688 |


------------------------------------------------------------------------------------------------------------
6. GENERATOR TRANSLATION LAW

------------------------------------------------------------------------------------------------------------

M1 : unavailable

M2

----------------------------------------------------

p   : x -> x + 100   PASS=True

h   : x -> x + 010   PASS=True

s   : x -> x + 001   PASS=True

753 |

Full translation law PASS: True

759 |


------------------------------------------------------------------------------------------------------------
7. C8 ADJACENCY IN EACH BINARY COORDINATE SYSTEM

------------------------------------------------------------------------------------------------------------

M1 : unavailable

M2

----------------------------------------------------------------------

edge     x_i    x_j    displacement

0-1      000    100        100

0-2      000    001        001

1-3      100    101        001

2-4      001    111        110

3-5      101    011        110

4-6      111    110        001

5-7      011    010        001

6-7      110    010        100

807 |

Displacement histogram:

  001 : 4

  100 : 2

  110 : 2

815 |


------------------------------------------------------------------------------------------------------------
8. SEVEN NONIDENTITY DIRECTIONS

------------------------------------------------------------------------------------------------------------

M1

----------------------------------------------------------------------------------------------------

word   permutation                 C8-auto   matching-in-C8   setwise-Pi   labeled-Pi   basin action

001    (0 1)(2 4)(3 5)(6 7)          False             True         True        False   (0, 2, 1, 3)

010    (0 7)(1 6)(2 5)(3 4)           True            False         True        False   (3, 2, 1, 0)

011    (0 6)(1 7)(2 3)(4 5)          False            False         True        False   (3, 1, 2, 0)

100    (0 1)(2 3)(4 5)(6 7)           True            False         True         True   (0, 1, 2, 3)

101    (2 5)(3 4)                    False            False         True        False   (0, 2, 1, 3)

110    (0 6)(1 7)(2 4)(3 5)           True            False         True        False   (3, 2, 1, 0)

111    (0 7)(1 6)                    False            False         True        False   (3, 1, 2, 0)

885 |

M2

----------------------------------------------------------------------------------------------------

word   permutation                 C8-auto   matching-in-C8   setwise-Pi   labeled-Pi   basin action

001    (0 2)(1 3)(4 6)(5 7)          False             True         True        False   (1, 0, 3, 2)

010    (0 7)(1 6)(2 5)(3 4)           True            False         True        False   (3, 2, 1, 0)

011    (0 5)(1 4)(2 7)(3 6)          False            False         True        False   (2, 3, 0, 1)

100    (0 1)(2 3)(4 5)(6 7)           True            False         True         True   (0, 1, 2, 3)

101    (0 3)(1 2)(4 7)(5 6)          False            False         True        False   (1, 0, 3, 2)

110    (0 6)(1 7)(2 4)(3 5)           True            False         True        False   (3, 2, 1, 0)

111    (0 4)(1 5)(2 6)(3 7)          False            False         True        False   (2, 3, 0, 1)

885 |


------------------------------------------------------------------------------------------------------------
9. ORDER-4 / V4 SUBGROUPS

------------------------------------------------------------------------------------------------------------

M1

------------------------------------------------------------------------------------

Number of distinct order-4 subgroups: 7

H1: nonzero=['001', '010', '011']   orbits=[[0, 1, 6, 7], [2, 3, 4, 5]]

H2: nonzero=['001', '100', '101']   orbits=[[0, 1], [2, 3, 4, 5], [6, 7]]

H3: nonzero=['001', '110', '111']   orbits=[[0, 1, 6, 7], [2, 4], [3, 5]]

H4: nonzero=['010', '100', '110']   orbits=[[0, 1, 6, 7], [2, 3, 4, 5]]

H5: nonzero=['010', '101', '111']   orbits=[[0, 7], [1, 6], [2, 5], [3, 4]]

H6: nonzero=['011', '100', '111']   orbits=[[0, 1, 6, 7], [2, 3], [4, 5]]

H7: nonzero=['011', '101', '110']   orbits=[[0, 6], [1, 7], [2, 3, 4, 5]]

999 |

M2

------------------------------------------------------------------------------------

Number of distinct order-4 subgroups: 7

H1: nonzero=['001', '010', '011']   orbits=[[0, 2, 5, 7], [1, 3, 4, 6]]

H2: nonzero=['001', '100', '101']   orbits=[[0, 1, 2, 3], [4, 5, 6, 7]]

H3: nonzero=['001', '110', '111']   orbits=[[0, 2, 4, 6], [1, 3, 5, 7]]

H4: nonzero=['010', '100', '110']   orbits=[[0, 1, 6, 7], [2, 3, 4, 5]]

H5: nonzero=['010', '101', '111']   orbits=[[0, 3, 4, 7], [1, 2, 5, 6]]

H6: nonzero=['011', '100', '111']   orbits=[[0, 1, 4, 5], [2, 3, 6, 7]]

H7: nonzero=['011', '101', '110']   orbits=[[0, 3, 5, 6], [1, 2, 4, 7]]

999 |


------------------------------------------------------------------------------------------------------------
10. V4-INDUCED 4+4 SECTOR PARTITIONS

------------------------------------------------------------------------------------------------------------

M1

------------------------------------------------------------

V4 subgroup count             : 7

Distinct 4+4 partitions       : 6

Every partition is 4+4        : False

1041 |

M2

------------------------------------------------------------

V4 subgroup count             : 7

Distinct 4+4 partitions       : 7

Every partition is 4+4        : True

1041 |


------------------------------------------------------------------------------------------------------------
11. PG(2,2) INCIDENCE TEST

------------------------------------------------------------------------------------------------------------

M1

------------------------------------------------------------

Number of points             : 7

Number of lines              : 7

Every line has 3 points      : True

Every point lies on 3 lines  : True

Every point pair has 1 line  : True

Every line pair meets once   : True

1141 |

PG(2,2) PASS                 : True

1147 |

M2

------------------------------------------------------------

Number of points             : 7

Number of lines              : 7

Every line has 3 points      : True

Every point lies on 3 lines  : True

Every point pair has 1 line  : True

Every line pair meets once   : True

1141 |

PG(2,2) PASS                 : True

1147 |


------------------------------------------------------------------------------------------------------------
12. FULL C8 AUTOMORPHISM GROUP

------------------------------------------------------------------------------------------------------------

|Aut(C8)| = 16

Expected  = 16

PASS      = True


------------------------------------------------------------------------------------------------------------
13. FROZEN-LAB STABILIZER HIERARCHY

------------------------------------------------------------------------------------------------------------

level                               size

----------------------------------------------------

Aut(C8)                               16

Stab(p)                                4

Stab(p,h)                              4

Stab_setwise(p,h,Pi)                   4

Stab_labeled(p,h,Pi)                   2


------------------------------------------------------------------------------------------------------------
14. DIRECT M1 / M2 EXCHANGE TEST

------------------------------------------------------------------------------------------------------------

level                               exchange?    number

--------------------------------------------------------------------

Aut(C8)                                 True         8

    representative: (1 2)(3 4)(5 6)

Stab(p)                                False         0

Stab(p,h)                              False         0

Stab_setwise(p,h,Pi)                   False         0

Stab_labeled(p,h,Pi)                   False         0


------------------------------------------------------------------------------------------------------------
15. WHOLE-GROUP EQUIVALENCE UNDER FROZEN-LAB SYMMETRY

------------------------------------------------------------------------------------------------------------

level                               G1 -> G2?    number

--------------------------------------------------------------------

Aut(C8)                                False         0

Stab(p)                                False         0

Stab(p,h)                              False         0

Stab_setwise(p,h,Pi)                   False         0

Stab_labeled(p,h,Pi)                   False         0


------------------------------------------------------------------------------------------------------------
16. ABSTRACT GROUP COMPARISON

------------------------------------------------------------------------------------------------------------

M1 branch regular (Z_2)^3 : False

M2 branch regular (Z_2)^3 : True

1365 |

Abstract carrier equivalence: False


------------------------------------------------------------------------------------------------------------
17. DIRECTION SIGNATURE COMPARISON

------------------------------------------------------------------------------------------------------------

M1

------------------------------------------------------------------------------------

 1 x (False, False, True, False, (0, 2, 1, 3))

 2 x (False, False, True, False, (3, 1, 2, 0))

 1 x (False, True, True, False, (0, 2, 1, 3))

 2 x (True, False, True, False, (3, 2, 1, 0))

 1 x (True, False, True, True, (0, 1, 2, 3))

1417 |

M2

------------------------------------------------------------------------------------

 1 x (False, False, True, False, (1, 0, 3, 2))

 2 x (False, False, True, False, (2, 3, 0, 1))

 1 x (False, True, True, False, (1, 0, 3, 2))

 2 x (True, False, True, False, (3, 2, 1, 0))

 1 x (True, False, True, True, (0, 1, 2, 3))

1417 |

Same frozen-lab direction-signature multiset: False


------------------------------------------------------------------------------------------------------------
18. 4+4 SECTOR-PARTITION COMPARISON

------------------------------------------------------------------------------------------------------------

M1 distinct 4+4 partitions: 6

M2 distinct 4+4 partitions: 7

1451 |

Literal partition sets equal: False

Shared literal partitions   : 1

M1-only partitions          : 5

M2-only partitions          : 6


------------------------------------------------------------------------------------------------------------
19. SECTOR-FAMILY EQUIVALENCE UNDER FROZEN-LAB SYMMETRY

------------------------------------------------------------------------------------------------------------

level                               P1 -> P2?    number

--------------------------------------------------------------------

Aut(C8)                                False         0

Stab(p)                                False         0

Stab(p,h)                              False         0

Stab_setwise(p,h,Pi)                   False         0

Stab_labeled(p,h,Pi)                   False         0


------------------------------------------------------------------------------------------------------------
20. EMBEDDING-INVARIANCE LEDGER

------------------------------------------------------------------------------------------------------------

M1 pairwise commuting                           : True

M2 pairwise commuting                           : True

1555 |

M1 generates order 8                            : True

M2 generates order 8                            : True

1567 |

M1 regular (Z_2)^3                              : False

M2 regular (Z_2)^3                              : True

1579 |

M1 PG(2,2)                                      : True

M2 PG(2,2)                                      : True

1591 |

M1 number of V4 subgroups                       : 7

M2 number of V4 subgroups                       : 7

1603 |

M1 distinct 4+4 partitions                      : 6

M2 distinct 4+4 partitions                      : 7

1615 |

Same direction-signature multiset               : False

Literal 4+4 sector families equal               : False

1627 |

M1~M2 under bare Aut(C8)                        : True

M1~M2 under Stab(p)                             : False

G1~G2 under bare Aut(C8)                        : False

G1~G2 under Stab(p)                             : False

Sector families ~ under bare Aut(C8)            : False

Sector families ~ under Stab(p)                 : False


------------------------------------------------------------------------------------------------------------
21. PREREGISTERED CLASSIFICATION

------------------------------------------------------------------------------------------------------------

OUTCOME C — EMBEDDING DEPENDENT: M1 DOES NOT REPRODUCE THE REGULAR (Z_2)^3 CARRIER


------------------------------------------------------------------------------------------------------------
22. INTERPRETATION RULES

------------------------------------------------------------------------------------------------------------


A. If M1 fails to generate a regular (Z_2)^3 action while M2 succeeds,
   then SIM14.1's finite carrier is embedding-dependent.

B. If both M1 and M2 generate regular (Z_2)^3 actions, then the abstract
   binary carrier and its PG(2,2) incidence structure survive the
   SIM14.0 symplectic embedding ambiguity.

C. If both branches are abstractly equivalent but differ relative to the
   frozen C8 / projection laboratory, classify the result as:

       abstract invariance + embedded inequivalence.

   Do NOT collapse those two statements.

D. PG(2,2) is downstream of a regular F_2^3 carrier. Once the regular
   elementary-abelian action is established, the Fano incidence result is
   standard finite geometry and should not be treated as an independent
   surprise.

E. SIM14.2 does not decide whether either embedding is physically preferred.

F. If the embeddings remain dynamically distinguishable later under a
   separately frozen ISP/LCO law, that belongs to a future dynamical SIM.

G. No octonionic, nonassociative, E8, H4, spinorial, quantum, or physical
   interpretation is permitted from SIM14.2 alone.


------------------------------------------------------------------------------------------------------------
23. IMPLEMENTATION SANITY CHECKS

------------------------------------------------------------------------------------------------------------

All implementation sanity checks PASS.


------------------------------------------------------------------------------------------------------------
24. SIM14.2 TRUTH PACKET

------------------------------------------------------------------------------------------------------------

Intervention:

1842 |

    M2 symplectic conjugate embedding

              versus

    M1 symplectic conjugate embedding

1846 |

Frozen:

1848 |

    C8

    p

    h

    projection basins

    finite-group analysis rules

1854 |

Primary endpoint:

1856 |

    Does M1 reproduce the regular (Z_2)^3 carrier?

1858 |

Secondary endpoints:

1860 |

    7 nonzero directions

    7 V4 subgroups

    7 distinct 4+4 decompositions

    PG(2,2)

1865 |

Critical distinction:

1867 |

    abstract group/incidence equivalence

              versus

    equivalence relative to frozen C8/projection structure

1871 |

STATUS:

1873 |

OUTCOME C — EMBEDDING DEPENDENT: M1 DOES NOT REPRODUCE THE REGULAR (Z_2)^3 CARRIER


============================================================================================================
SIM14.2 COMPLETE

============================================================================================================

STATUS:

OUTCOME C — EMBEDDING DEPENDENT: M1 DOES NOT REPRODUCE THE REGULAR (Z_2)^3 CARRIER

1886 |

Interpretive ceiling:

1888 |

SIM14.2 tests whether the finite carrier structure recovered in SIM14.1 survives the alternative symplectic-compatible embedding already discovered in SIM14.0. It introduces no new physical or algebraic structure.
