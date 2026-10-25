import numpy as np
from itertools import product

def generate_dual_5demicube_nodes():
    """
    Generates the 26 nodes of the Dual 5-Demicube.
    These correspond to the 26 facets of the primal 5-demicube.
    """
    nodes_A = []
    # Group A (10 nodes): The 10 demitesseract boundaries.
    # We represent these logically as unit vectors along the 5 axes.
    for i in range(5):
        for sign in [1, -1]:
            vec = np.zeros(5)
            vec[i] = sign
            nodes_A.append(('A', vec))

    nodes_B = []
    # Group B (16 nodes): The 16 5-cell (simplex) boundaries.
    # These correspond exactly to the odd-parity vertices of a 5-cube.
    for v in product([1, -1], repeat=5):
        if v.count(-1) % 2 != 0:
            nodes_B.append(('B', np.array(v)))

    return nodes_A + nodes_B

def build_dual_5demicube_adjacency(nodes):
    """
    Builds the 26x26 adjacency matrix.
    Total edges must mathematically equal 120.
    """
    n = len(nodes)
    adj = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        type_i, vec_i = nodes[i]
        for j in range(i + 1, n):
            type_j, vec_j = nodes[j]

            # Rule 1: A to A (Demitesseract to Demitesseract)
            # They share a 3-face if they do not lie on the exact same axis.
            if type_i == 'A' and type_j == 'A':
                if not np.all(np.abs(vec_i) == np.abs(vec_j)):
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

            # Rule 2: A to B (Demitesseract to 5-Cell)
            # They share a 3-face if the B-node's coordinate matches
            # the A-node's specific directional axis.
            elif type_i == 'A' and type_j == 'B':
                axis = np.argmax(np.abs(vec_i))
                if vec_i[axis] == vec_j[axis]:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
            elif type_i == 'B' and type_j == 'A':
                axis = np.argmax(np.abs(vec_j))
                if vec_j[axis] == vec_i[axis]:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

            # Rule 3: B to B (5-Cell to 5-Cell)
            # Mathematically, the odd-parity corners NEVER share a 3-face
            # with each other. They only attach to the A-group framework.
            elif type_i == 'B' and type_j == 'B':
                pass # Degree 0 within the B-group

    return adj

if __name__ == "__main__":
    nodes = generate_dual_5demicube_nodes()
    adj = build_dual_5demicube_adjacency(nodes)

    # Validation against the pure math
    print(f"Dual 5-Demicube Nodes: {len(nodes)} (Should be 26)")
    print(f"Adjacency Shape: {adj.shape}")

    # Calculate degrees
    degrees = adj.sum(axis=1)
    edges = int(adj.sum() / 2)
    print(f"Total Edges: {edges} (Should be 120)")

    # Validate the structural imbalance of the S1 Tier
    print(f"Group A Degree: {degrees[0]} (Should be 16)")
    print(f"Group B Degree: {degrees[10]} (Should be 5)")