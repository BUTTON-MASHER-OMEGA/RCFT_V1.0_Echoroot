# =============================================================================
# SIM14.1 — BINARY CARRIER / ORBIT CLASSIFICATION PROBE
# =============================================================================
#
# PURPOSE
# -------
# Extend SIM14.0 using exact finite mathematics only.
#
# SIM14.1 asks:
#
#   What finite structure is induced on the frozen eight-state carrier by
#
#       G = <p, h, s_Omega> ?
#
# where
#
#   p       = frozen projection pairing
#   h       = intrinsic C8 graph half-turn
#   s_Omega = representative C8-compatible symplectic conjugation
#
# NO ISP.
# NO LCO.
# NO stochastic dynamics.
# NO fitting.
# NO optimization.
# NO Fano structure is assumed.
#
# Main tests
# ----------
# 1. Reconstruct G exactly from the three frozen involutions.
# 2. Verify G ~= (Z_2)^3 from its computed structure.
# 3. Test whether G acts regularly on the eight microscopic states.
# 4. If regular, construct F_2^3 coordinates from the action.
# 5. Express frozen C8 adjacency in those coordinates.
# 6. Enumerate all seven nonidentity transformations and their matchings.
# 7. Enumerate all order-4 subgroups and their vertex orbits.
# 8. Construct the induced point-line incidence structure.
# 9. Test whether that incidence structure is PG(2,2).
# 10. Compute the frozen-laboratory stabilizer hierarchy.
# 11. Test M1 <-> M2 equivalence at every stabilizer level.
# 12. Classify the seven transformations / V4 subgroups under frozen-lab
#     symmetry.
#
# INTERPRETATION CEILING
# ----------------------
# Even if PG(2,2) is recovered, SIM14.1 establishes only a finite
# mathematical incidence structure induced by the derived transformation
# algebra. It does NOT establish octonionic multiplication, nonassociativity,
# E8, H4, spin, quantum mechanics, or a physical Fano substrate.
#
# =============================================================================

from itertools import permutations, combinations, product
from collections import Counter, defaultdict, deque

import numpy as np


# =============================================================================
# PRINTING UTILITIES
# =============================================================================

WIDTH = 104


def banner(title):
    print("\n" + "=" * WIDTH)
    print(title)
    print("=" * WIDTH)


def section(title):
    print("\n" + "-" * WIDTH)
    print(title)
    print("-" * WIDTH)


def fmt_bool(x):
    return "True" if bool(x) else "False"


# =============================================================================
# PERMUTATION UTILITIES
# =============================================================================
#
# A permutation g is represented as a tuple
#
#       g = (g(0), g(1), ..., g(7)).
#
# Composition compose(g,h) means
#
#       (g o h)(x) = g(h(x)).
#
# =============================================================================

N = 8
VERTICES = tuple(range(N))
IDENTITY = tuple(range(N))


def compose(g, h):
    """Return g o h."""
    return tuple(g[h[i]] for i in VERTICES)


def inverse(g):
    inv = [None] * N
    for i, j in enumerate(g):
        inv[j] = i
    return tuple(inv)


def perm_power(g, n):
    out = IDENTITY
    for _ in range(n):
        out = compose(g, out)
    return out


def perm_order(g):
    x = IDENTITY
    for k in range(1, 100):
        x = compose(g, x)
        if x == IDENTITY:
            return k
    raise RuntimeError("Permutation order exceeded search bound.")


def commute(g, h):
    return compose(g, h) == compose(h, g)


def conjugate(a, g):
    """Return a g a^{-1}."""
    return compose(compose(a, g), inverse(a))


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
    return "".join("(" + " ".join(map(str, c)) + ")" for c in cycles)


def generated_group(generators):
    """
    Exact closure of a finite permutation generating set.
    """
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
    frozenset((C8_CYCLE_ORDER[i], C8_CYCLE_ORDER[(i + 1) % N]))
    for i in range(N)
)

PROJECTION_BASINS = (
    frozenset((0, 1)),
    frozenset((2, 3)),
    frozenset((4, 5)),
    frozenset((6, 7)),
)

BASIN_INDEX = {}
for bidx, block in enumerate(PROJECTION_BASINS):
    for v in block:
        BASIN_INDEX[v] = bidx


def edge_tuple(edge):
    a, b = sorted(edge)
    return (a, b)


def edge_set_string(edges):
    return "{" + ", ".join(
        f"{a}-{b}" for a, b in sorted(edge_tuple(e) for e in edges)
    ) + "}"


def image_edge(edge, g):
    a, b = tuple(edge)
    return frozenset((g[a], g[b]))


def image_edge_set(edges, g):
    return frozenset(image_edge(e, g) for e in edges)


def is_graph_automorphism(g, edges=C8_EDGES):
    return image_edge_set(edges, g) == edges


def matching_from_involution(g):
    """
    Return unordered perfect-matching edges if g is fixed-point-free involution.
    Otherwise return None.
    """
    if perm_order(g) != 2:
        return None

    if any(g[v] == v for v in VERTICES):
        return None

    edges = {
        frozenset((v, g[v]))
        for v in VERTICES
    }

    return frozenset(edges)


def preserves_matching_set(g, matching):
    return image_edge_set(matching, g) == matching


def image_partition(partition, g):
    return frozenset(
        frozenset(g[v] for v in block)
        for block in partition
    )


def preserves_partition_setwise(g, partition=PROJECTION_BASINS):
    """
    Whole blocks may be permuted.
    """
    return image_partition(partition, g) == frozenset(partition)


def preserves_partition_labeled(g, partition=PROJECTION_BASINS):
    """
    Every named basin block must map to itself.
    """
    for block in partition:
        image = frozenset(g[v] for v in block)
        if image != block:
            return False
    return True


def basin_permutation(g):
    """
    If g maps whole projection blocks to whole projection blocks,
    return induced permutation of basin indices.
    Otherwise return None.
    """
    blocks = list(PROJECTION_BASINS)
    block_to_idx = {block: i for i, block in enumerate(blocks)}

    out = []

    for block in blocks:
        image = frozenset(g[v] for v in block)
        if image not in block_to_idx:
            return None
        out.append(block_to_idx[image])

    return tuple(out)


# =============================================================================
# FROZEN SIM14.0 OPERATIONS
# =============================================================================

# Projection pairing:
# p = (01)(23)(45)(67)
p = (1, 0, 3, 2, 5, 4, 7, 6)

# Graph half-turn:
# h = (07)(16)(25)(34)
h = (7, 6, 5, 4, 3, 2, 1, 0)

# Representative SIM14.0 symplectic conjugation:
# s_Omega = (02)(13)(46)(57)
sO = (2, 3, 0, 1, 6, 7, 4, 5)


# Two maximum-overlap / C8-contained conjugate matchings from SIM14.0
M1 = frozenset(
    frozenset(e)
    for e in ((0, 1), (2, 4), (3, 5), (6, 7))
)

M2 = frozenset(
    frozenset(e)
    for e in ((0, 2), (1, 3), (4, 6), (5, 7))
)


# =============================================================================
# 0. HEADER / FROZEN ARCHITECTURE
# =============================================================================

banner("SIM14.1 — BINARY CARRIER / ORBIT CLASSIFICATION PROBE")

print()
print("NO ISP / LCO dynamics are run in SIM14.1.")
print("NO stochastic sampling, fitting, or optimization is permitted.")
print("NO Fano structure is assumed.")
print()
print("SIM14.1 extends the exact static carrier analysis of SIM14.0.")

section("0. FROZEN SIM14.0 INPUTS")

print("Microscopic states       :", VERTICES)
print("C8 cycle order           :", C8_CYCLE_ORDER)
print("C8 edges                 :", edge_set_string(C8_EDGES))
print("Projection basins        :", {
    i: sorted(block) for i, block in enumerate(PROJECTION_BASINS)
})
print()
print("p       =", cycle_string(p))
print("h       =", cycle_string(h))
print("s_Omega =", cycle_string(sO))
print()
print("M1      =", edge_set_string(M1))
print("M2      =", edge_set_string(M2))


# =============================================================================
# 1. RECONSTRUCT THE GENERATED GROUP
# =============================================================================

section("1. EXACT GROUP CLOSURE")

G = generated_group([p, h, sO])

print("|<p,h,s_Omega>| =", len(G))
print()

for name, g in (("p", p), ("h", h), ("s_Omega", sO)):
    print(
        f"{name:9s}: {cycle_string(g):24s} "
        f"order={perm_order(g):2d} "
        f"C8 automorphism={is_graph_automorphism(g)}"
    )

print()
print("Pairwise commutation:")
print("  [p,h]       = e :", commute(p, h))
print("  [p,s_Omega] = e :", commute(p, sO))
print("  [h,s_Omega] = e :", commute(h, sO))

orders = Counter(perm_order(g) for g in G)
print()
print("Element-order histogram:", dict(sorted(orders.items())))

is_abelian = all(commute(a, b) for a in G for b in G)
all_nonidentity_order2 = all(
    perm_order(g) == 2 for g in G if g != IDENTITY
)

z2_cubed_pass = (
    len(G) == 8
    and is_abelian
    and all_nonidentity_order2
)

print("Group abelian            :", is_abelian)
print("All nonidentity order 2  :", all_nonidentity_order2)
print("(Z_2)^3 identification   :", "PASS" if z2_cubed_pass else "FAIL")


# =============================================================================
# 2. EXPLICIT BINARY WORD REPRESENTATION
# =============================================================================

section("2. GENERATOR-WORD REPRESENTATION")

word_to_perm = {}
perm_to_words = defaultdict(list)

for a, b, c in product((0, 1), repeat=3):
    g = IDENTITY

    if a:
        g = compose(p, g)
    if b:
        g = compose(h, g)
    if c:
        g = compose(sO, g)

    word = (a, b, c)
    word_to_perm[word] = g
    perm_to_words[g].append(word)

word_map_injective = (
    len(word_to_perm) == 8
    and len(set(word_to_perm.values())) == 8
)

print("Eight binary words produce eight distinct permutations:",
      word_map_injective)

print()
print(" word     permutation")
print("-" * 52)

for word in sorted(word_to_perm):
    g = word_to_perm[word]
    print(f" {''.join(map(str, word))}      {cycle_string(g)}")


# =============================================================================
# 3. REGULAR ACTION TEST
# =============================================================================

section("3. REGULAR ACTION ON THE EIGHT MICROSCOPIC STATES")

regular_action = True

print("vertex   orbit size   stabilizer size   orbit")
print("-" * 80)

for v in VERTICES:
    orbit = sorted({g[v] for g in G})
    stabilizer = [g for g in G if g[v] == v]

    print(
        f"{v:>3d}        {len(orbit):>3d}             "
        f"{len(stabilizer):>3d}          {orbit}"
    )

    if len(orbit) != 8 or len(stabilizer) != 1:
        regular_action = False

print()
print("Regular action PASS:", regular_action)


# =============================================================================
# 4. F_2^3 / TORSOR COORDINATES
# =============================================================================

section("4. BINARY CARRIER COORDINATES")

ORIGIN = 0

vertex_to_coord = {}
coord_to_vertex = {}

if regular_action and word_map_injective:
    for word, g in word_to_perm.items():
        v = g[ORIGIN]

        if v in vertex_to_coord:
            raise RuntimeError("Coordinate collision despite regular action.")

        vertex_to_coord[v] = word
        coord_to_vertex[word] = v

    print("Conventional origin vertex:", ORIGIN)
    print()
    print("vertex   coordinate")
    print("-" * 28)

    for v in VERTICES:
        coord = vertex_to_coord[v]
        print(f"{v:>3d}      {''.join(map(str, coord))}")

    print()
    print("Interpretation:")
    print("  p       acts as translation by 100")
    print("  h       acts as translation by 010")
    print("  s_Omega acts as translation by 001")
    print()
    print("The vertex set is therefore tested as a torsor for F_2^3.")

else:
    print("Binary carrier coordinates NOT constructed because regularity failed.")


# =============================================================================
# 5. VERIFY TRANSLATION LAW
# =============================================================================

section("5. BINARY TRANSLATION LAW")

def xor3(a, b):
    return tuple((x ^ y) for x, y in zip(a, b))


translation_tests = {}

if regular_action:
    generator_vectors = {
        "p": (1, 0, 0),
        "h": (0, 1, 0),
        "s_Omega": (0, 0, 1),
    }

    generator_perms = {
        "p": p,
        "h": h,
        "s_Omega": sO,
    }

    for name in generator_vectors:
        d = generator_vectors[name]
        g = generator_perms[name]

        passed = True

        for v in VERTICES:
            lhs = vertex_to_coord[g[v]]
            rhs = xor3(vertex_to_coord[v], d)

            if lhs != rhs:
                passed = False
                break

        translation_tests[name] = passed

        print(
            f"{name:9s}: x -> x + "
            f"{''.join(map(str, d))}   PASS={passed}"
        )

    full_translation_pass = all(translation_tests.values())

else:
    full_translation_pass = False
    print("Translation test unavailable.")

print()
print("Full generator translation law PASS:", full_translation_pass)


# =============================================================================
# 6. EXPRESS C8 ADJACENCY IN BINARY COORDINATES
# =============================================================================

section("6. FROZEN C8 ADJACENCY IN BINARY COORDINATES")

edge_displacement_counter = Counter()
edge_binary_rows = []

if regular_action:
    for edge in sorted(C8_EDGES, key=edge_tuple):
        i, j = edge_tuple(edge)

        xi = vertex_to_coord[i]
        xj = vertex_to_coord[j]

        d = xor3(xi, xj)

        edge_displacement_counter[d] += 1
        edge_binary_rows.append((i, j, xi, xj, d))

    print("edge     x_i    x_j    displacement x_i+x_j")
    print("-" * 62)

    for i, j, xi, xj, d in edge_binary_rows:
        print(
            f"{i}-{j:<3d}    "
            f"{''.join(map(str, xi))}    "
            f"{''.join(map(str, xj))}          "
            f"{''.join(map(str, d))}"
        )

    print()
    print("C8 edge-displacement histogram:")

    for d, count in sorted(edge_displacement_counter.items()):
        print(f"  {''.join(map(str, d))} : {count}")

    print()
    print("Number of distinct C8 displacement vectors:",
          len(edge_displacement_counter))

else:
    print("Unavailable because binary coordinates were not constructed.")


# =============================================================================
# 7. ENUMERATE ALL SEVEN NONIDENTITY TRANSFORMATIONS
# =============================================================================

section("7. SEVEN NONIDENTITY TRANSFORMATIONS")

nonzero_words = [
    w for w in sorted(word_to_perm)
    if w != (0, 0, 0)
]

transformation_records = []

header = (
    "word   permutation                 "
    "C8-auto   matching-in-C8   setwise-Pi   labeled-Pi"
)

print(header)
print("-" * len(header))

for word in nonzero_words:
    g = word_to_perm[word]
    matching = matching_from_involution(g)

    matching_in_c8 = (
        matching is not None
        and matching.issubset(C8_EDGES)
    )

    setwise_pi = preserves_partition_setwise(g)
    labeled_pi = preserves_partition_labeled(g)

    record = {
        "word": word,
        "perm": g,
        "matching": matching,
        "c8_auto": is_graph_automorphism(g),
        "matching_in_c8": matching_in_c8,
        "setwise_pi": setwise_pi,
        "labeled_pi": labeled_pi,
    }

    transformation_records.append(record)

    print(
        f"{''.join(map(str, word)):>3s}    "
        f"{cycle_string(g):27s} "
        f"{str(record['c8_auto']):>7s}   "
        f"{str(matching_in_c8):>14s}   "
        f"{str(setwise_pi):>10s}   "
        f"{str(labeled_pi):>10s}"
    )

print()
print("Perfect matchings generated by the seven nonzero transformations:")

for rec in transformation_records:
    print(
        f"  {''.join(map(str, rec['word']))}: "
        f"{edge_set_string(rec['matching'])}"
    )


# =============================================================================
# 8. ENUMERATE ALL ORDER-4 SUBGROUPS
# =============================================================================

section("8. ORDER-4 SUBGROUP / V4 ENUMERATION")

order4_subgroups = set()

nonidentity_elements = [g for g in G if g != IDENTITY]

for a, b in combinations(nonidentity_elements, 2):
    H = subgroup_generated([a, b])

    if len(H) == 4:
        order4_subgroups.add(frozenset(H))

order4_subgroups = sorted(
    order4_subgroups,
    key=lambda H: sorted(
        min(perm_to_words[g]) for g in H
    )
)

print("Number of distinct order-4 subgroups:", len(order4_subgroups))
print()

subgroup_records = []

for idx, H in enumerate(order4_subgroups, start=1):
    words = sorted(
        min(perm_to_words[g])
        for g in H
    )

    nonzero_H_words = [
        w for w in words if w != (0, 0, 0)
    ]

    # Vertex orbits under H
    unseen = set(VERTICES)
    orbits = []

    while unseen:
        v = min(unseen)
        orb = frozenset(g[v] for g in H)
        orbits.append(orb)
        unseen -= set(orb)

    orbits = sorted(orbits, key=lambda x: tuple(sorted(x)))

    subgroup_records.append({
        "index": idx,
        "group": H,
        "words": words,
        "nonzero_words": nonzero_H_words,
        "orbits": orbits,
    })

    word_strings = [
        "".join(map(str, w))
        for w in nonzero_H_words
    ]

    orbit_strings = [
        "{" + ",".join(map(str, sorted(o))) + "}"
        for o in orbits
    ]

    print(
        f"H{idx}: nonzero={word_strings}   "
        f"vertex orbits={orbit_strings}"
    )


# =============================================================================
# 9. INDUCED INCIDENCE STRUCTURE
# =============================================================================

section("9. INDUCED POINT-LINE INCIDENCE STRUCTURE")

# "Points" = seven nonzero F_2^3 directions.
# "Lines"  = nonidentity elements of each order-4 subgroup.
#
# This construction is derived from the group first.
# We do NOT assume the Fano plane.

points = tuple(nonzero_words)

lines = tuple(
    frozenset(rec["nonzero_words"])
    for rec in subgroup_records
)

point_line_count = Counter()

for line in lines:
    for point in line:
        point_line_count[point] += 1

num_points = len(points)
num_lines = len(lines)
points_per_line = sorted(len(line) for line in lines)
lines_per_point = sorted(point_line_count[p] for p in points)

pair_line_counts = Counter()

for a, b in combinations(points, 2):
    count = sum(
        1 for line in lines
        if a in line and b in line
    )
    pair_line_counts[(a, b)] = count

line_intersection_sizes = []

for L1, L2 in combinations(lines, 2):
    line_intersection_sizes.append(len(L1 & L2))

print("Number of points             :", num_points)
print("Number of lines              :", num_lines)
print("Points per line              :", points_per_line)
print("Lines through each point     :", lines_per_point)
print()
print("Every point pair on 1 line   :",
      all(v == 1 for v in pair_line_counts.values()))
print("Every line pair meets in 1 pt:",
      all(v == 1 for v in line_intersection_sizes))

pg22_pass = (
    num_points == 7
    and num_lines == 7
    and all(x == 3 for x in points_per_line)
    and all(x == 3 for x in lines_per_point)
    and all(v == 1 for v in pair_line_counts.values())
    and all(v == 1 for v in line_intersection_sizes)
)

print()
print("PG(2,2) incidence axioms PASS:", pg22_pass)

print()
print("Derived incidence lines:")

for idx, line in enumerate(lines, start=1):
    labels = sorted("".join(map(str, w)) for w in line)
    print(f"  L{idx}: {labels}")


# =============================================================================
# 10. FULL C8 AUTOMORPHISM GROUP
# =============================================================================

section("10. FULL C8 AUTOMORPHISM GROUP")

AUT_C8 = frozenset(
    g for g in permutations(VERTICES)
    if is_graph_automorphism(g)
)

print("|Aut(C8)| =", len(AUT_C8))
print("Expected  =", 16)
print("PASS      =", len(AUT_C8) == 16)


# =============================================================================
# 11. FROZEN-LAB STABILIZER HIERARCHY
# =============================================================================

section("11. FROZEN-LAB STABILIZER HIERARCHY")

def centralizes(g, x):
    return commute(g, x)


G0 = AUT_C8

# Preserve the projection involution p as an operation.
G1 = frozenset(
    g for g in G0
    if conjugate(g, p) == p
)

# Preserve p and h as named operations.
G2 = frozenset(
    g for g in G1
    if conjugate(g, h) == h
)

# Preserve the basin partition as an unlabeled collection of blocks.
G3_setwise = frozenset(
    g for g in G2
    if preserves_partition_setwise(g)
)

# Preserve every basin block individually.
G3_labeled = frozenset(
    g for g in G2
    if preserves_partition_labeled(g)
)

stabilizer_levels = [
    ("Aut(C8)", G0),
    ("Stab(p)", G1),
    ("Stab(p,h)", G2),
    ("Stab_setwise(p,h,Pi)", G3_setwise),
    ("Stab_labeled(p,h,Pi)", G3_labeled),
]

print("level                         size")
print("-" * 48)

for name, H in stabilizer_levels:
    print(f"{name:30s} {len(H):4d}")


# =============================================================================
# 12. M1 <-> M2 EQUIVALENCE
# =============================================================================

section("12. M1 / M2 EQUIVALENCE UNDER THE STABILIZER HIERARCHY")

def matching_exchange_elements(group, A, B):
    return [
        g for g in group
        if image_edge_set(A, g) == B
    ]


matching_equivalence_results = {}

print("level                         exchange?   number of exchangers")
print("-" * 70)

for name, H in stabilizer_levels:
    exchangers = matching_exchange_elements(H, M1, M2)
    equivalent = len(exchangers) > 0

    matching_equivalence_results[name] = equivalent

    print(
        f"{name:30s} "
        f"{str(equivalent):>8s}   "
        f"{len(exchangers):>8d}"
    )

    if exchangers:
        representative = sorted(exchangers)[0]
        print(
            f"    representative exchanger: "
            f"{cycle_string(representative)}"
        )


# =============================================================================
# 13. BASIN ACTION OF THE STABILIZERS
# =============================================================================

section("13. BASIN ACTION OF FROZEN-LAB SYMMETRIES")

for name, H in stabilizer_levels:
    induced = set()

    for g in H:
        bp = basin_permutation(g)
        if bp is not None:
            induced.add(bp)

    print(f"{name}:")
    print(f"  induced basin permutations = {len(induced)}")

    for bp in sorted(induced):
        print(f"    {bp}")

    print()


# =============================================================================
# 14. ACTION OF FROZEN-LAB SYMMETRIES ON THE SEVEN GROUP DIRECTIONS
# =============================================================================

section("14. ORBITS OF THE SEVEN NONZERO TRANSFORMATIONS")

# A graph automorphism acts on a permutation g by conjugation:
#
#       g -> a g a^{-1}.
#
# However, conjugation may take an element outside the derived group G.
# We therefore only define an internal action on the seven directions for
# stabilizer elements that normalize G.

def normalizes_group(a, group):
    return frozenset(conjugate(a, g) for g in group) == group


def normalizer_subset(H, group):
    return frozenset(a for a in H if normalizes_group(a, group))


def orbit_under_conjugation(element, acting_group):
    return frozenset(conjugate(a, element) for a in acting_group)


direction_orbit_results = {}

for level_name, H in stabilizer_levels:
    NORM = normalizer_subset(H, G)

    unseen = set(nonidentity_elements)
    orbits = []

    while unseen:
        g = next(iter(unseen))
        orb = orbit_under_conjugation(g, NORM)
        orb = frozenset(x for x in orb if x in G and x != IDENTITY)

        orbits.append(orb)
        unseen -= set(orb)

    # Stable readable sorting
    orbits = sorted(
        orbits,
        key=lambda orb: sorted(
            min(perm_to_words[g]) for g in orb
        )
    )

    direction_orbit_results[level_name] = orbits

    print(level_name)
    print(f"  stabilizer size             : {len(H)}")
    print(f"  subgroup normalizing G      : {len(NORM)}")
    print(f"  nonzero-direction orbit count: {len(orbits)}")

    for idx, orb in enumerate(orbits, start=1):
        words = sorted(
            "".join(map(str, min(perm_to_words[g])))
            for g in orb
        )
        print(f"    orbit {idx}: {words}")

    print()


# =============================================================================
# 15. ACTION ON THE SEVEN ORDER-4 SUBGROUPS
# =============================================================================

section("15. ORBITS OF THE ORDER-4 / V4 SUBGROUPS")

def conjugate_subgroup(a, H):
    return frozenset(conjugate(a, g) for g in H)


v4_orbit_results = {}

for level_name, Hacting in stabilizer_levels:
    NORM = normalizer_subset(Hacting, G)

    unseen = set(order4_subgroups)
    subgroup_orbits = []

    while unseen:
        H0 = next(iter(unseen))

        orb = frozenset(
            conjugate_subgroup(a, H0)
            for a in NORM
        )

        orb = frozenset(
            X for X in orb
            if X in set(order4_subgroups)
        )

        subgroup_orbits.append(orb)
        unseen -= set(orb)

    subgroup_orbits = sorted(
        subgroup_orbits,
        key=lambda orb: min(
            sorted(
                tuple(sorted(min(perm_to_words[g]) for g in H))
                for H in orb
            )
        )
    )

    v4_orbit_results[level_name] = subgroup_orbits

    print(level_name)
    print(f"  V4 orbit count: {len(subgroup_orbits)}")

    for idx, orb in enumerate(subgroup_orbits, start=1):
        readable = []

        for subgroup in orb:
            words = sorted(
                "".join(map(str, min(perm_to_words[g])))
                for g in subgroup
                if g != IDENTITY
            )
            readable.append("{" + ",".join(words) + "}")

        print(
            f"    orbit {idx}: "
            + ", ".join(sorted(readable))
        )

    print()


# =============================================================================
# 16. FOUR-STATE SECTOR DECOMPOSITIONS
# =============================================================================

section("16. FOUR-STATE SECTORS INDUCED BY ORDER-4 SUBGROUPS")

sector_partitions = []

for rec in subgroup_records:
    orbits = rec["orbits"]

    partition = frozenset(orbits)
    sector_partitions.append(partition)

    labels = [
        "".join(map(str, w))
        for w in rec["nonzero_words"]
    ]

    print(
        f"H{rec['index']} {labels}: "
        + " | ".join(
            "{" + ",".join(map(str, sorted(o))) + "}"
            for o in orbits
        )
    )

unique_sector_partitions = set(sector_partitions)

print()
print("Number of V4 subgroups             :", len(order4_subgroups))
print("Distinct induced 4+4 partitions    :", len(unique_sector_partitions))
print(
    "Every V4 gives two four-state orbits:",
    all(
        sorted(len(o) for o in rec["orbits"]) == [4, 4]
        for rec in subgroup_records
    )
)


# =============================================================================
# 17. C8 / GROUP RELATION LEDGER
# =============================================================================

section("17. C8 / DERIVED-GROUP RELATION LEDGER")

c8_auto_elements_in_G = sorted(
    [g for g in G if is_graph_automorphism(g)]
)

matching_in_c8_elements = sorted(
    [
        g for g in G
        if g != IDENTITY
        and matching_from_involution(g) is not None
        and matching_from_involution(g).issubset(C8_EDGES)
    ]
)

print("Elements of derived G preserving full C8:")
for g in c8_auto_elements_in_G:
    words = perm_to_words[g]
    word = min(words)
    print(
        f"  {''.join(map(str, word))}: {cycle_string(g)}"
    )

print()
print("Nonidentity G-elements whose conjugate matching lies entirely in C8:")
for g in matching_in_c8_elements:
    word = min(perm_to_words[g])
    print(
        f"  {''.join(map(str, word))}: "
        f"{edge_set_string(matching_from_involution(g))}"
    )

print()
print("|G ∩ Aut(C8)| =", len(c8_auto_elements_in_G))


# =============================================================================
# 18. INFORMATION / INTERPRETATION LEDGER
# =============================================================================

section("18. INTERPRETATION LEDGER")

print("What SIM14.1 is allowed to establish:")
print()
print("  * the exact finite group generated by p, h, and s_Omega")
print("  * whether that group acts regularly on the eight states")
print("  * whether an F_2^3 torsor coordinate system follows")
print("  * how frozen C8 adjacency appears in those coordinates")
print("  * the exact seven nonidentity transformations")
print("  * the exact order-4 subgroup structure")
print("  * the incidence structure induced by subgroup membership")
print("  * whether that incidence structure is PG(2,2)")
print("  * how frozen C8 / projection constraints break or preserve equivalences")
print()
print("What SIM14.1 is NOT allowed to establish:")
print()
print("  * octonionic multiplication")
print("  * nonassociativity")
print("  * a physical Fano substrate")
print("  * E8 / RE8 / 4_21")
print("  * H4")
print("  * spin or spinor physics")
print("  * quantum mechanics")
print("  * a physical RCFT carrier identification")


# =============================================================================
# 19. TRUTH PACKET
# =============================================================================

section("19. SIM14.1 TRUTH PACKET")

bare_equiv = matching_equivalence_results["Aut(C8)"]
setwise_equiv = matching_equivalence_results["Stab_setwise(p,h,Pi)"]
labeled_equiv = matching_equivalence_results["Stab_labeled(p,h,Pi)"]

print(f"{'Generated group order':48s}: {len(G)}")
print(f"{'(Z_2)^3 identification':48s}: {z2_cubed_pass}")
print(f"{'Regular action on 8 states':48s}: {regular_action}")
print(f"{'Binary F_2^3 coordinate ledger valid':48s}: {full_translation_pass}")
print(f"{'Nonidentity transformations':48s}: {len(nonidentity_elements)}")
print(f"{'Order-4 / V4 subgroups':48s}: {len(order4_subgroups)}")
print(f"{'Distinct V4-induced 4+4 sector partitions':48s}: {len(unique_sector_partitions)}")
print(f"{'PG(2,2) incidence identification':48s}: {pg22_pass}")
print()
print(f"{'M1 ~ M2 under bare Aut(C8)':48s}: {bare_equiv}")
print(f"{'M1 ~ M2 under setwise frozen lab':48s}: {setwise_equiv}")
print(f"{'M1 ~ M2 under labeled-basin frozen lab':48s}: {labeled_equiv}")
print()
print(f"{'|Aut(C8)|':48s}: {len(G0)}")
print(f"{'|Stab(p)|':48s}: {len(G1)}")
print(f"{'|Stab(p,h)|':48s}: {len(G2)}")
print(f"{'|Stab_setwise(p,h,Pi)|':48s}: {len(G3_setwise)}")
print(f"{'|Stab_labeled(p,h,Pi)|':48s}: {len(G3_labeled)}")


# =============================================================================
# 20. HARD STATUS CLASSIFICATION
# =============================================================================

section("20. SIM14.1 STATUS")

core_pass = (
    z2_cubed_pass
    and regular_action
    and full_translation_pass
)

if not core_pass:
    STATUS = (
        "FAIL / BINARY-CARRIER INTERPRETATION NOT ESTABLISHED"
    )

elif core_pass and pg22_pass:
    STATUS = (
        "PASS / REGULAR F_2^3 CARRIER WITH DERIVED PG(2,2) INCIDENCE"
    )

else:
    STATUS = (
        "PARTIAL / F_2^3 CARRIER ESTABLISHED BUT PG(2,2) INCIDENCE FAILED"
    )

print(STATUS)

print()
print("Classification rule:")
print()
print("A. If <p,h,s_Omega> does not act regularly, do not identify the")
print("   eight microscopic states with an F_2^3 torsor.")
print()
print("B. If regularity passes, binary coordinates are induced by the")
print("   already-frozen transformation algebra, up to conventional origin")
print("   and basis choices.")
print()
print("C. If the seven nonzero directions and seven order-4 subgroups")
print("   satisfy the finite-projective-plane incidence axioms, PG(2,2)")
print("   is a mathematical consequence of the derived binary algebra.")
print()
print("D. A PG(2,2) result does NOT establish octonions or physical Fano")
print("   structure. Those require additional independently motivated laws.")
print()
print("E. M1/M2 equivalence must be judged relative to the frozen laboratory,")
print("   not merely relative to the bare unlabeled C8 graph.")
print()
print("F. No future dynamical ingredient is inferred or inserted by SIM14.1.")


# =============================================================================
# 21. SANITY ASSERTIONS
# =============================================================================
#
# These assertions protect against silent implementation drift.
# They test only facts already frozen by SIM14.0 or internal consistency
# conditions required by the present implementation.
#
# =============================================================================

section("21. IMPLEMENTATION SANITY CHECKS")

assert len(C8_EDGES) == 8
assert all(len(e) == 2 for e in C8_EDGES)

assert perm_order(p) == 2
assert perm_order(h) == 2
assert perm_order(sO) == 2

assert commute(p, h)
assert commute(p, sO)
assert commute(h, sO)

assert len(G) == len(set(G))

assert M1.issubset(C8_EDGES)
assert M2.issubset(C8_EDGES)

assert len(AUT_C8) == 16

assert G3_labeled.issubset(G3_setwise)
assert G3_setwise.issubset(G2)
assert G2.issubset(G1)
assert G1.issubset(G0)

print("All implementation sanity checks PASS.")


# =============================================================================
# COMPLETE
# =============================================================================

banner("SIM14.1 COMPLETE")

print("STATUS:")
print(STATUS)

print()
print("Interpretive ceiling:")
print()
print(
    "SIM14.1 classifies exact finite structure induced by the frozen "
    "SIM14.0 operations. Any recovered PG(2,2) structure is an incidence "
    "property of the derived (Z_2)^3 algebra. No octonionic, nonassociative, "
    "E8, H4, spinorial, quantum, or physical identification is made."
)






~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~








RESULTS:




========================================================================================================
SIM14.1 — BINARY CARRIER / ORBIT CLASSIFICATION PROBE

========================================================================================================

342 |

NO ISP / LCO dynamics are run in SIM14.1.

NO stochastic sampling, fitting, or optimization is permitted.

NO Fano structure is assumed.

346 |

SIM14.1 extends the exact static carrier analysis of SIM14.0.


--------------------------------------------------------------------------------------------------------
0. FROZEN SIM14.0 INPUTS

--------------------------------------------------------------------------------------------------------

Microscopic states       : (0, 1, 2, 3, 4, 5, 6, 7)

C8 cycle order           : (0, 1, 3, 5, 7, 6, 4, 2)

C8 edges                 : {0-1, 0-2, 1-3, 2-4, 3-5, 4-6, 5-7, 6-7}

Projection basins        : {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7]}

357 |

p       = (0 1)(2 3)(4 5)(6 7)

h       = (0 7)(1 6)(2 5)(3 4)

s_Omega = (0 2)(1 3)(4 6)(5 7)

361 |

M1      = {0-1, 2-4, 3-5, 6-7}

M2      = {0-2, 1-3, 4-6, 5-7}


--------------------------------------------------------------------------------------------------------
1. EXACT GROUP CLOSURE

--------------------------------------------------------------------------------------------------------

|<p,h,s_Omega>| = 8

375 |

p        : (0 1)(2 3)(4 5)(6 7)     order= 2 C8 automorphism=True

h        : (0 7)(1 6)(2 5)(3 4)     order= 2 C8 automorphism=True

s_Omega  : (0 2)(1 3)(4 6)(5 7)     order= 2 C8 automorphism=False

384 |

Pairwise commutation:

  [p,h]       = e : True

  [p,s_Omega] = e : True

  [h,s_Omega] = e : True

391 |

Element-order histogram: {1: 1, 2: 7}

Group abelian            : True

All nonidentity order 2  : True

(Z_2)^3 identification   : PASS


--------------------------------------------------------------------------------------------------------
2. GENERATOR-WORD REPRESENTATION

--------------------------------------------------------------------------------------------------------

Eight binary words produce eight distinct permutations: True

441 |

 word     permutation

----------------------------------------------------

 000      ()

 001      (0 2)(1 3)(4 6)(5 7)

 010      (0 7)(1 6)(2 5)(3 4)

 011      (0 5)(1 4)(2 7)(3 6)

 100      (0 1)(2 3)(4 5)(6 7)

 101      (0 3)(1 2)(4 7)(5 6)

 110      (0 6)(1 7)(2 4)(3 5)

 111      (0 4)(1 5)(2 6)(3 7)


--------------------------------------------------------------------------------------------------------
3. REGULAR ACTION ON THE EIGHT MICROSCOPIC STATES

--------------------------------------------------------------------------------------------------------

vertex   orbit size   stabilizer size   orbit

--------------------------------------------------------------------------------

  0          8               1          [0, 1, 2, 3, 4, 5, 6, 7]

  1          8               1          [0, 1, 2, 3, 4, 5, 6, 7]

  2          8               1          [0, 1, 2, 3, 4, 5, 6, 7]

  3          8               1          [0, 1, 2, 3, 4, 5, 6, 7]

  4          8               1          [0, 1, 2, 3, 4, 5, 6, 7]

  5          8               1          [0, 1, 2, 3, 4, 5, 6, 7]

  6          8               1          [0, 1, 2, 3, 4, 5, 6, 7]

  7          8               1          [0, 1, 2, 3, 4, 5, 6, 7]

473 |

Regular action PASS: True


--------------------------------------------------------------------------------------------------------
4. BINARY CARRIER COORDINATES

--------------------------------------------------------------------------------------------------------

Conventional origin vertex: 0

499 |

vertex   coordinate

----------------------------

  0      000

  1      100

  2      001

  3      101

  4      111

  5      011

  6      110

  7      010

507 |

Interpretation:

  p       acts as translation by 100

  h       acts as translation by 010

  s_Omega acts as translation by 001

512 |

The vertex set is therefore tested as a torsor for F_2^3.


--------------------------------------------------------------------------------------------------------
5. BINARY TRANSLATION LAW

--------------------------------------------------------------------------------------------------------

p        : x -> x + 100   PASS=True

h        : x -> x + 010   PASS=True

s_Omega  : x -> x + 001   PASS=True

571 |

Full generator translation law PASS: True


--------------------------------------------------------------------------------------------------------
6. FROZEN C8 ADJACENCY IN BINARY COORDINATES

--------------------------------------------------------------------------------------------------------

edge     x_i    x_j    displacement x_i+x_j

--------------------------------------------------------------

0-1      000    100          100

0-2      000    001          001

1-3      100    101          001

2-4      001    111          110

3-5      101    011          110

4-6      111    110          001

5-7      011    010          001

6-7      110    010          100

607 |

C8 edge-displacement histogram:

  001 : 4

  100 : 2

  110 : 2

613 |

Number of distinct C8 displacement vectors: 3


--------------------------------------------------------------------------------------------------------
7. SEVEN NONIDENTITY TRANSFORMATIONS

--------------------------------------------------------------------------------------------------------

word   permutation                 C8-auto   matching-in-C8   setwise-Pi   labeled-Pi

-------------------------------------------------------------------------------------

001    (0 2)(1 3)(4 6)(5 7)          False             True         True        False

010    (0 7)(1 6)(2 5)(3 4)           True            False         True        False

011    (0 5)(1 4)(2 7)(3 6)          False            False         True        False

100    (0 1)(2 3)(4 5)(6 7)           True            False         True         True

101    (0 3)(1 2)(4 7)(5 6)          False            False         True        False

110    (0 6)(1 7)(2 4)(3 5)           True            False         True        False

111    (0 4)(1 5)(2 6)(3 7)          False            False         True        False

675 |

Perfect matchings generated by the seven nonzero transformations:

  001: {0-2, 1-3, 4-6, 5-7}

  010: {0-7, 1-6, 2-5, 3-4}

  011: {0-5, 1-4, 2-7, 3-6}

  100: {0-1, 2-3, 4-5, 6-7}

  101: {0-3, 1-2, 4-7, 5-6}

  110: {0-6, 1-7, 2-4, 3-5}

  111: {0-4, 1-5, 2-6, 3-7}


--------------------------------------------------------------------------------------------------------
8. ORDER-4 SUBGROUP / V4 ENUMERATION

--------------------------------------------------------------------------------------------------------

Number of distinct order-4 subgroups: 7

709 |

H1: nonzero=['001', '010', '011']   vertex orbits=['{0,2,5,7}', '{1,3,4,6}']

H2: nonzero=['001', '100', '101']   vertex orbits=['{0,1,2,3}', '{4,5,6,7}']

H3: nonzero=['001', '110', '111']   vertex orbits=['{0,2,4,6}', '{1,3,5,7}']

H4: nonzero=['010', '100', '110']   vertex orbits=['{0,1,6,7}', '{2,3,4,5}']

H5: nonzero=['010', '101', '111']   vertex orbits=['{0,3,4,7}', '{1,2,5,6}']

H6: nonzero=['011', '100', '111']   vertex orbits=['{0,1,4,5}', '{2,3,6,7}']

H7: nonzero=['011', '101', '110']   vertex orbits=['{0,3,5,6}', '{1,2,4,7}']


--------------------------------------------------------------------------------------------------------
9. INDUCED POINT-LINE INCIDENCE STRUCTURE

--------------------------------------------------------------------------------------------------------

Number of points             : 7

Number of lines              : 7

Points per line              : [3, 3, 3, 3, 3, 3, 3]

Lines through each point     : [3, 3, 3, 3, 3, 3, 3]

807 |

Every point pair on 1 line   : True

Every line pair meets in 1 pt: True

822 |

PG(2,2) incidence axioms PASS: True

825 |

Derived incidence lines:

  L1: ['001', '010', '011']

  L2: ['001', '100', '101']

  L3: ['001', '110', '111']

  L4: ['010', '100', '110']

  L5: ['010', '101', '111']

  L6: ['011', '100', '111']

  L7: ['011', '101', '110']


--------------------------------------------------------------------------------------------------------
10. FULL C8 AUTOMORPHISM GROUP

--------------------------------------------------------------------------------------------------------

|Aut(C8)| = 16

Expected  = 16

PASS      = True


--------------------------------------------------------------------------------------------------------
11. FROZEN-LAB STABILIZER HIERARCHY

--------------------------------------------------------------------------------------------------------

level                         size

------------------------------------------------

Aut(C8)                          16

Stab(p)                           4

Stab(p,h)                         4

Stab_setwise(p,h,Pi)              4

Stab_labeled(p,h,Pi)              2


--------------------------------------------------------------------------------------------------------
12. M1 / M2 EQUIVALENCE UNDER THE STABILIZER HIERARCHY

--------------------------------------------------------------------------------------------------------

level                         exchange?   number of exchangers

----------------------------------------------------------------------

Aut(C8)                            True          8

    representative exchanger: (1 2)(3 4)(5 6)

Stab(p)                           False          0

Stab(p,h)                         False          0

Stab_setwise(p,h,Pi)              False          0

Stab_labeled(p,h,Pi)              False          0


--------------------------------------------------------------------------------------------------------
13. BASIN ACTION OF FROZEN-LAB SYMMETRIES

--------------------------------------------------------------------------------------------------------

Aut(C8):

  induced basin permutations = 2

    (0, 1, 2, 3)

    (3, 2, 1, 0)

958 |

Stab(p):

  induced basin permutations = 2

    (0, 1, 2, 3)

    (3, 2, 1, 0)

958 |

Stab(p,h):

  induced basin permutations = 2

    (0, 1, 2, 3)

    (3, 2, 1, 0)

958 |

Stab_setwise(p,h,Pi):

  induced basin permutations = 2

    (0, 1, 2, 3)

    (3, 2, 1, 0)

958 |

Stab_labeled(p,h,Pi):

  induced basin permutations = 1

    (0, 1, 2, 3)

958 |


--------------------------------------------------------------------------------------------------------
14. ORBITS OF THE SEVEN NONZERO TRANSFORMATIONS

--------------------------------------------------------------------------------------------------------

Aut(C8)

  stabilizer size             : 16

  subgroup normalizing G      : 8

  nonzero-direction orbit count: 5

    orbit 1: ['001']

    orbit 2: ['010']

    orbit 3: ['011']

    orbit 4: ['100', '110']

    orbit 5: ['101', '111']

1025 |

Stab(p)

  stabilizer size             : 4

  subgroup normalizing G      : 4

  nonzero-direction orbit count: 7

    orbit 1: ['001']

    orbit 2: ['010']

    orbit 3: ['011']

    orbit 4: ['100']

    orbit 5: ['101']

    orbit 6: ['110']

    orbit 7: ['111']

1025 |

Stab(p,h)

  stabilizer size             : 4

  subgroup normalizing G      : 4

  nonzero-direction orbit count: 7

    orbit 1: ['001']

    orbit 2: ['010']

    orbit 3: ['011']

    orbit 4: ['100']

    orbit 5: ['101']

    orbit 6: ['110']

    orbit 7: ['111']

1025 |

Stab_setwise(p,h,Pi)

  stabilizer size             : 4

  subgroup normalizing G      : 4

  nonzero-direction orbit count: 7

    orbit 1: ['001']

    orbit 2: ['010']

    orbit 3: ['011']

    orbit 4: ['100']

    orbit 5: ['101']

    orbit 6: ['110']

    orbit 7: ['111']

1025 |

Stab_labeled(p,h,Pi)

  stabilizer size             : 2

  subgroup normalizing G      : 2

  nonzero-direction orbit count: 7

    orbit 1: ['001']

    orbit 2: ['010']

    orbit 3: ['011']

    orbit 4: ['100']

    orbit 5: ['101']

    orbit 6: ['110']

    orbit 7: ['111']

1025 |


--------------------------------------------------------------------------------------------------------
15. ORBITS OF THE ORDER-4 / V4 SUBGROUPS

--------------------------------------------------------------------------------------------------------

Aut(C8)

  V4 orbit count: 5

    orbit 1: {001,010,011}

    orbit 2: {001,100,101}, {001,110,111}

    orbit 3: {010,100,110}

    orbit 4: {010,101,111}

    orbit 5: {011,100,111}, {011,101,110}

1093 |

Stab(p)

  V4 orbit count: 7

    orbit 1: {001,010,011}

    orbit 2: {001,100,101}

    orbit 3: {001,110,111}

    orbit 4: {010,100,110}

    orbit 5: {010,101,111}

    orbit 6: {011,100,111}

    orbit 7: {011,101,110}

1093 |

Stab(p,h)

  V4 orbit count: 7

    orbit 1: {001,010,011}

    orbit 2: {001,100,101}

    orbit 3: {001,110,111}

    orbit 4: {010,100,110}

    orbit 5: {010,101,111}

    orbit 6: {011,100,111}

    orbit 7: {011,101,110}

1093 |

Stab_setwise(p,h,Pi)

  V4 orbit count: 7

    orbit 1: {001,010,011}

    orbit 2: {001,100,101}

    orbit 3: {001,110,111}

    orbit 4: {010,100,110}

    orbit 5: {010,101,111}

    orbit 6: {011,100,111}

    orbit 7: {011,101,110}

1093 |

Stab_labeled(p,h,Pi)

  V4 orbit count: 7

    orbit 1: {001,010,011}

    orbit 2: {001,100,101}

    orbit 3: {001,110,111}

    orbit 4: {010,100,110}

    orbit 5: {010,101,111}

    orbit 6: {011,100,111}

    orbit 7: {011,101,110}

1093 |


--------------------------------------------------------------------------------------------------------
16. FOUR-STATE SECTORS INDUCED BY ORDER-4 SUBGROUPS

--------------------------------------------------------------------------------------------------------

H1 ['001', '010', '011']: {0,2,5,7} | {1,3,4,6}

H2 ['001', '100', '101']: {0,1,2,3} | {4,5,6,7}

H3 ['001', '110', '111']: {0,2,4,6} | {1,3,5,7}

H4 ['010', '100', '110']: {0,1,6,7} | {2,3,4,5}

H5 ['010', '101', '111']: {0,3,4,7} | {1,2,5,6}

H6 ['011', '100', '111']: {0,1,4,5} | {2,3,6,7}

H7 ['011', '101', '110']: {0,3,5,6} | {1,2,4,7}

1125 |

Number of V4 subgroups             : 7

Distinct induced 4+4 partitions    : 7

Every V4 gives two four-state orbits: True


--------------------------------------------------------------------------------------------------------
17. C8 / DERIVED-GROUP RELATION LEDGER

--------------------------------------------------------------------------------------------------------

Elements of derived G preserving full C8:

  000: ()

  100: (0 1)(2 3)(4 5)(6 7)

  110: (0 6)(1 7)(2 4)(3 5)

  010: (0 7)(1 6)(2 5)(3 4)

1164 |

Nonidentity G-elements whose conjugate matching lies entirely in C8:

  001: {0-2, 1-3, 4-6, 5-7}

1173 |

|G ∩ Aut(C8)| = 4


--------------------------------------------------------------------------------------------------------
18. INTERPRETATION LEDGER

--------------------------------------------------------------------------------------------------------

What SIM14.1 is allowed to establish:

1184 |

  * the exact finite group generated by p, h, and s_Omega

  * whether that group acts regularly on the eight states

  * whether an F_2^3 torsor coordinate system follows

  * how frozen C8 adjacency appears in those coordinates

  * the exact seven nonidentity transformations

  * the exact order-4 subgroup structure

  * the incidence structure induced by subgroup membership

  * whether that incidence structure is PG(2,2)

  * how frozen C8 / projection constraints break or preserve equivalences

1194 |

What SIM14.1 is NOT allowed to establish:

1196 |

  * octonionic multiplication

  * nonassociativity

  * a physical Fano substrate

  * E8 / RE8 / 4_21

  * H4

  * spin or spinor physics

  * quantum mechanics

  * a physical RCFT carrier identification


--------------------------------------------------------------------------------------------------------
19. SIM14.1 TRUTH PACKET

--------------------------------------------------------------------------------------------------------

Generated group order                           : 8

(Z_2)^3 identification                          : True

Regular action on 8 states                      : True

Binary F_2^3 coordinate ledger valid            : True

Nonidentity transformations                     : 7

Order-4 / V4 subgroups                          : 7

Distinct V4-induced 4+4 sector partitions       : 7

PG(2,2) incidence identification                : True

1225 |

M1 ~ M2 under bare Aut(C8)                      : True

M1 ~ M2 under setwise frozen lab                : False

M1 ~ M2 under labeled-basin frozen lab          : False

1229 |

|Aut(C8)|                                       : 16

|Stab(p)|                                       : 4

|Stab(p,h)|                                     : 4

|Stab_setwise(p,h,Pi)|                          : 4

|Stab_labeled(p,h,Pi)|                          : 2


--------------------------------------------------------------------------------------------------------
20. SIM14.1 STATUS

--------------------------------------------------------------------------------------------------------

PASS / REGULAR F_2^3 CARRIER WITH DERIVED PG(2,2) INCIDENCE

1266 |

Classification rule:

1268 |

A. If <p,h,s_Omega> does not act regularly, do not identify the

   eight microscopic states with an F_2^3 torsor.

1271 |

B. If regularity passes, binary coordinates are induced by the

   already-frozen transformation algebra, up to conventional origin

   and basis choices.

1275 |

C. If the seven nonzero directions and seven order-4 subgroups

   satisfy the finite-projective-plane incidence axioms, PG(2,2)

   is a mathematical consequence of the derived binary algebra.

1279 |

D. A PG(2,2) result does NOT establish octonions or physical Fano

   structure. Those require additional independently motivated laws.

1282 |

E. M1/M2 equivalence must be judged relative to the frozen laboratory,

   not merely relative to the bare unlabeled C8 graph.

1285 |

F. No future dynamical ingredient is inferred or inserted by SIM14.1.


--------------------------------------------------------------------------------------------------------
21. IMPLEMENTATION SANITY CHECKS

--------------------------------------------------------------------------------------------------------

All implementation sanity checks PASS.


========================================================================================================
SIM14.1 COMPLETE

========================================================================================================

STATUS:

PASS / REGULAR F_2^3 CARRIER WITH DERIVED PG(2,2) INCIDENCE

1336 |

Interpretive ceiling:

1338 |

SIM14.1 classifies exact finite structure induced by the frozen SIM14.0 operations. Any recovered PG(2,2) structure is an incidence property of the derived (Z_2)^3 algebra. No octonionic, nonassociative, E8, H4, spinorial, quantum, or physical identification is made.
