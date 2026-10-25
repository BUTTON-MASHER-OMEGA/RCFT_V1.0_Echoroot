import numpy as np
from itertools import combinations, product, permutations

def generate_type1_roots():
    roots = set()
    # Base pattern is [±1, ±1, 0, 0, 0, 0, 0, 0]
    base_pattern = [1, -1]

    # Iterate over all permutations of [1, 1] and [1, -1] placed in 8 positions
    for signs in product(base_pattern, repeat=2):
        base = [signs[0], signs[1]] + [0] * 6

        for perm in set(permutations(base)):
            roots.add(perm)

    # This also misses half the roots because it doesn't account for the two non-zero positions being arbitrary.
    # The *correct* approach is iterating over indices, as in the first correction.

    # Let's revert to the most robust and accurate definition:
    roots = set()
    for i in range(8):
        for j in range(i + 1, 8):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    root = [0] * 8
                    root[i] = s1
                    root[j] = s2
                    roots.add(tuple(root))

    # Total count should be (8 choose 2) * 4 sign combinations = 28 * 4 = 112.
    return [np.array(r) for r in roots]


def generate_e8_roots():
    type1 = generate_type1_roots()
    type2 = []

    for signs in product([0.5, -0.5], repeat=8):
        if signs.count(-0.5) % 2 == 0:
            type2.append(np.array(signs))

    roots = np.array(type1 + type2)
    assert roots.shape == (240, 8), f"Expected 240 roots, got {roots.shape[0]}"
    return roots

def build_adjacency(roots, threshold=1.0):
    n = roots.shape[0]
    adj = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            dot = np.dot(roots[i], roots[j])
            if abs(dot) == threshold:
                adj[i, j] = 1
                adj[j, i] = 1

    return adj  # shape: (240, 240)

#as of now, this is the best performing sparsify function.  Effectively adds "anti-coherence" to the system.
def sparsify_adjacency_torsion_weighted(roots, adjacency, target_degree=32, preference_ratio=0.75):
    """
    Sparsifies the E8 adjacency matrix by prioritizing negative-dot links
    (Dot Product = -1) over positive-dot links (Dot Product = +1).

    preference_ratio: Fraction of the target_degree that MUST come from
                      the preferred (-1 dot product) connections.
    """
    n = adjacency.shape[0]
    sparse_adj = np.zeros_like(adjacency)

    # Target 75% of connections (24 out of 32) must be Torsion links
    preferred_count = int(target_degree * preference_ratio)

    for i in range(n):
        # Identify neighbors based on dot product sign
        torsion_neighbors = []
        coherence_neighbors = []

        for j in np.where(adjacency[i] == 1)[0]:
            dot = np.dot(roots[i], roots[j])

            if dot == -1.0:
                torsion_neighbors.append(j)  # NEW PRIORITY
            elif dot == 1.0:
                coherence_neighbors.append(j)

        # 2. Prioritize Torsion Links (Anti-Coherence)
        if len(torsion_neighbors) >= preferred_count:
            # Select the required number of torsion links
            selected_neg = np.random.choice(torsion_neighbors, preferred_count, replace=False)
        else:
            # Take all available torsion links
            selected_neg = np.array(torsion_neighbors)

        remaining_budget = target_degree - len(selected_neg)

        # 3. Fill remaining budget with Coherence Links (Necessary Purity)
        if remaining_budget > 0 and len(coherence_neighbors) > 0:
            # Select randomly from coherence neighbors
            num_pos_to_select = min(remaining_budget, len(coherence_neighbors))
            selected_pos = np.random.choice(coherence_neighbors, num_pos_to_select, replace=False)
        else:
            selected_pos = np.array([])

        final_selection = np.concatenate([selected_neg, selected_pos]).astype(int)

        sparse_adj[i, final_selection] = 1

    # 4. Enforce symmetry (Crucial for adjacency matrices)
    sparse_adj = np.maximum(sparse_adj, sparse_adj.T)

    return sparse_adj

def sparsify_adjacency_coherence_weighted(roots, adjacency, target_degree=32, preference_ratio=0.75):
    """
    Sparsifies the E8 adjacency matrix by prioritizing positive-dot links
    (Dot Product = +1) over negative-dot links (Dot Product = -1).

    preference_ratio: Fraction of the target_degree that MUST come from
                      the preferred (+1 dot product) connections.
    """
    n = adjacency.shape[0]
    sparse_adj = np.zeros_like(adjacency)

    # 1. Pre-calculate the Dot Product for all adjacent pairs
    dot_products = {}
    for i in range(n):
        for j in np.where(adjacency[i] == 1)[0]:
            if i < j:
                dot_products[(i, j)] = np.dot(roots[i], roots[j])

    preferred_count = int(target_degree * preference_ratio)

    for i in range(n):
        # Identify neighbors based on dot product sign
        positive_neighbors = []
        negative_neighbors = []

        for j in np.where(adjacency[i] == 1)[0]:
            dot = np.dot(roots[i], roots[j]) # Recalculate or use pre-calculated value
            if dot == 1.0:
                positive_neighbors.append(j)
            elif dot == -1.0:
                negative_neighbors.append(j)

        # 2. Prioritize Positive Links (High Coherence)
        if len(positive_neighbors) >= preferred_count:
            # Select the required number of positive links
            selected_pos = np.random.choice(positive_neighbors, preferred_count, replace=False)
        else:
            # Take all available positive links
            selected_pos = np.array(positive_neighbors)

        remaining_budget = target_degree - len(selected_pos)

        # 3. Fill remaining budget with Negative Links (Controlled Torsion)
        if remaining_budget > 0 and len(negative_neighbors) > 0:
            # Select randomly from remaining negative neighbors
            num_neg_to_select = min(remaining_budget, len(negative_neighbors))
            selected_neg = np.random.choice(negative_neighbors, num_neg_to_select, replace=False)
        else:
            selected_neg = np.array([])

        final_selection = np.concatenate([selected_pos, selected_neg]).astype(int)

        sparse_adj[i, final_selection] = 1

    # 4. Enforce symmetry (Crucial for adjacency matrices)
    sparse_adj = np.maximum(sparse_adj, sparse_adj.T)

    # Note: We need to ensure the final degree is consistent.
    # The symmetry enforcement will slightly increase the average degree,
    # but the core logic of prioritization is now implemented.
    return sparse_adj

def sparsify_adjacency(adjacency, target_degree=32):
    n = adjacency.shape[0]
    sparse_adj = np.zeros_like(adjacency)
    for i in range(n):
        neighbors = np.where(adjacency[i] > 0)[0]
        if len(neighbors) > target_degree:
            selected = np.random.choice(neighbors, target_degree, replace=False)
        else:
            selected = neighbors
        sparse_adj[i, selected] = 1
    # Enforce symmetry
    for i in range(n):
        for j in range(n):
            if sparse_adj[i, j] == 1:
                sparse_adj[j, i] = 1
    return sparse_adj

if __name__ == "__main__":
    roots = generate_e8_roots()
    adj = build_adjacency(roots)
    adj = sparsify_adjacency(adj, target_degree=32)

    print(f"E₈ roots shape: {roots.shape}")
    print(f"Adjacency matrix shape: {adj.shape}")
    print(f"Average degree: {np.sum(adj) / roots.shape[0]}")

    # Inject a signal at node 0
    signal = np.zeros(240)
    signal[0] = 1.0  # one-hot activation

    # Propagate through adjacency
    for step in range(5):
        signal = adj @ signal
        signal = np.clip(signal, 0, 1)  # containment via threshold
        print(f"Step {step+1} activation sum: {np.sum(signal)}")

