from itertools import product
import numpy as np
from rcf_core.geometry.simplicial import OrientedSimplex
from rcf_core.geometry.polytopes.base import BasePolytopeComplex

class Demi5(BasePolytopeComplex):
    """5-Demicube (Halved 5-Cube).
    
    16 Vertices (subset of 5-cube with even count of -1s), 80 Edges.
    """
    def __init__(self):
        super().__init__("5-Demicube")
        verts = []
        for v in product([1.0, -1.0], repeat=5):
            if v.count(-1.0) % 2 == 0:
                verts.append(v)
        
        self._vertices_coords = np.array(verts, dtype=np.float32)
        n = len(verts)

        self._simplices[0] = [OrientedSimplex((i,)) for i in range(n)]

        # Edges connect vertices differing by exactly 2 coordinates (Hamming distance 2, dot product 1.0)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = np.sum(np.abs(self._vertices_coords[i] - self._vertices_coords[j])) / 2.0
                if np.isclose(diff, 2.0):
                    edges.append(OrientedSimplex((i, j)))
        self._simplices[1] = edges


class DualDemi5(BasePolytopeComplex):
    """Dual 5-Demicube (Semistellated triacontaditeron).
    
    26 Vertices: 10 Group A (demitesseract boundaries) + 16 Group B (5-cell boundaries).
    120 Edges: Group A nodes have degree 16, Group B nodes have degree 5.
    """
    def __init__(self):
        super().__init__("Dual-5-Demicube")
        nodes_A = []
        # Group A (10 nodes): Unit vectors along 5 axes
        for i in range(5):
            for sign in [1.0, -1.0]:
                vec = np.zeros(5, dtype=np.float32)
                vec[i] = sign
                nodes_A.append(('A', vec))

        nodes_B = []
        # Group B (16 nodes): Odd-parity vertices of 5-cube
        for v in product([1.0, -1.0], repeat=5):
            if v.count(-1.0) % 2 != 0:
                nodes_B.append(('B', np.array(v, dtype=np.float32)))

        all_nodes = nodes_A + nodes_B
        self._vertices_coords = np.array([node[1] for node in all_nodes], dtype=np.float32)
        n = len(all_nodes)

        self._simplices[0] = [OrientedSimplex((i,)) for i in range(n)]

        edges = []
        for i in range(n):
            type_i, vec_i = all_nodes[i]
            for j in range(i + 1, n):
                type_j, vec_j = all_nodes[j]

                # Rule 1: A to A (Demitesseract to Demitesseract)
                if type_i == 'A' and type_j == 'A':
                    if not np.all(np.abs(vec_i) == np.abs(vec_j)):
                        edges.append(OrientedSimplex((i, j)))

                # Rule 2: A to B (Demitesseract to 5-Cell)
                elif (type_i == 'A' and type_j == 'B') or (type_i == 'B' and type_j == 'A'):
                    vec_a = vec_i if type_i == 'A' else vec_j
                    vec_b = vec_j if type_i == 'A' else vec_i
                    axis = np.argmax(np.abs(vec_a))
                    if vec_a[axis] == vec_b[axis]:
                        edges.append(OrientedSimplex((i, j)))

        self._simplices[1] = edges


class E8Roots(BasePolytopeComplex):
    """E8 Root Lattice.
    
    240 Vertices (112 Type 1 + 128 Type 2).
    Full adjacency connect pairs with dot product = ±1.
    """
    def __init__(self):
        super().__init__("E8-Roots")
        type1 = []
        for i in range(8):
            for j in range(i + 1, 8):
                for s1 in [1.0, -1.0]:
                    for s2 in [1.0, -1.0]:
                        root = [0.0] * 8
                        root[i] = s1
                        root[j] = s2
                        type1.append(root)

        type2 = []
        for s in product([0.5, -0.5], repeat=8):
            if s.count(-0.5) % 2 == 0:
                type2.append(list(s))

        all_roots = type1 + type2
        self._vertices_coords = np.array(all_roots, dtype=np.float32)
        n = len(all_roots)

        self._simplices[0] = [OrientedSimplex((i,)) for i in range(n)]

        # Full adjacency connects roots with dot product = ±1 (degree 112 per root)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                dot = np.dot(self._vertices_coords[i], self._vertices_coords[j])
                if np.isclose(abs(dot), 1.0):
                    edges.append(OrientedSimplex((i, j)))
        self._simplices[1] = edges
