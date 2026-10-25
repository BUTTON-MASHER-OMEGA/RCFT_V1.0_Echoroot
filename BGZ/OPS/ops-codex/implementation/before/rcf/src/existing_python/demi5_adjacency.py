import numpy as np
from itertools import product

def generate_5demicube_roots():
    """
    Generates vertices of the 5-demicube.
    Subset of 5-cube vertices {±1}^5 with an even number of minus signs.
    (Alternating vertices).
    """
    vertices = []
    # Iterate over all 32 vertices of the 5-cube
    for v in product([1, -1], repeat=5):
        # Check parity: count of -1s must be even
        if v.count(-1) % 2 == 0:
            vertices.append(np.array(v))
            
    return np.array(vertices)

def build_5demicube_adjacency(vertices):
    """
    Builds adjacency for the 5-demicube.
    In a 5-cube, edges connect vertices differing by 1 coordinate.
    In a demicube, those vertices are removed.
    New edges connect vertices that differ by exactly 2 coordinates.
    (This corresponds to the edges of the demicube itself).
    """
    n = len(vertices)
    adj = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            # Calculate Hamming distance (number of differing coordinates)
            diff = np.sum(np.abs(vertices[i] - vertices[j])) / 2 # Each diff is 2 (1 - -1)
            
            # In a demicube, adjacent vertices differ by exactly 2 coordinates
            if np.isclose(diff, 2.0):
                adj[i, j] = 1.0
                adj[j, i] = 1.0
                
    return adj

if __name__ == "__main__":
    roots = generate_5demicube_roots()
    adj = build_5demicube_adjacency(roots)
    print(f"5-Demicube Vertices: {roots.shape} (Should be 16)")
    print(f"Adjacency Shape: {adj.shape}")
    print(f"Average Degree: {adj.sum() / 16}") # Should be 10 (5 choose 2)