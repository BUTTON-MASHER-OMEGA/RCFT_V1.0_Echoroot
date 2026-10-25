from itertools import combinations, permutations, product
import numpy as np
from rcf_core.geometry.simplicial import OrientedSimplex
from rcf_core.geometry.polytopes.base import BasePolytopeComplex

class FiveCell(BasePolytopeComplex):
    """5-Cell (4-simplex / Pentachoron).
    
    Self-dual 4D regular polytope.
    5 Vertices, 10 Edges, 10 Triangular Faces, 5 Tetrahedra.
    """
    def __init__(self):
        super().__init__("5-Cell")
        # Vertices of a standard 4-simplex in 4D space
        # (1, 1, 1, -1/sqrt(5)), etc., or Cartesian embedding:
        coords = np.zeros((5, 4), dtype=np.float32)
        r5 = np.sqrt(5)
        coords[0] = [1, 1, 1, -1/r5]
        coords[1] = [1, -1, -1, -1/r5]
        coords[2] = [-1, 1, -1, -1/r5]
        coords[3] = [-1, -1, 1, -1/r5]
        coords[4] = [0, 0, 0, r5 - 1/r5]
        self._vertices_coords = coords

        # All subsets of vertices form valid simplices
        v_idx = list(range(5))
        self._simplices[0] = [OrientedSimplex((i,)) for i in v_idx]
        self._simplices[1] = [OrientedSimplex(c) for c in combinations(v_idx, 2)]
        self._simplices[2] = [OrientedSimplex(c) for c in combinations(v_idx, 3)]
        self._simplices[3] = [OrientedSimplex(c) for c in combinations(v_idx, 4)]


class SixteenCell(BasePolytopeComplex):
    """16-Cell (Hexadecachoron / 4-Cross Polytope).
    
    8 Vertices, 24 Edges, 32 Triangular Faces, 16 Tetrahedra.
    Dual to the 8-cell (tesseract).
    """
    def __init__(self):
        super().__init__("16-Cell")
        # 8 Vertices: ±e1, ±e2, ±e3, ±e4
        coords = []
        for i in range(4):
            v_pos = [0.0] * 4
            v_neg = [0.0] * 4
            v_pos[i] = 1.0
            v_neg[i] = -1.0
            coords.append(v_pos)
            coords.append(v_neg)
        self._vertices_coords = np.array(coords, dtype=np.float32)

        # Vertices i and j are opposite iff j = i ^ 1 (indices (0,1), (2,3), (4,5), (6,7))
        def is_opposite(u, v):
            return (u ^ 1) == v

        # Generate simplices: any set of k vertices with no opposite pair forms a simplex
        v_idx = list(range(8))
        
        self._simplices[0] = [OrientedSimplex((i,)) for i in v_idx]

        edges = []
        for c in combinations(v_idx, 2):
            if not is_opposite(c[0], c[1]):
                edges.append(OrientedSimplex(c))
        self._simplices[1] = edges

        faces = []
        for c in combinations(v_idx, 3):
            if not (is_opposite(c[0], c[1]) or is_opposite(c[1], c[2]) or is_opposite(c[0], c[2])):
                faces.append(OrientedSimplex(c))
        self._simplices[2] = faces

        tets = []
        for c in combinations(v_idx, 4):
            valid = True
            for u, v in combinations(c, 2):
                if is_opposite(u, v):
                    valid = False
                    break
            if valid:
                tets.append(OrientedSimplex(c))
        self._simplices[3] = tets


class TwentyFourCell(BasePolytopeComplex):
    """24-Cell (Icositetrachoron / Octaplex).
    
    Self-dual 4D regular polytope with 24 Vertices, 96 Edges, 96 Faces, 24 Octahedral cells.
    """
    def __init__(self):
        super().__init__("24-Cell")
        # 24 Vertices: all permutations of (±1, ±1, 0, 0)
        verts = set()
        for p in set(permutations([1, 1, 0, 0])):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = list(p)
                    nonzeros = [i for i, x in enumerate(v) if x != 0]
                    v[nonzeros[0]] *= s1
                    v[nonzeros[1]] *= s2
                    verts.add(tuple(v))
        
        coords = sorted(list(verts))
        self._vertices_coords = np.array(coords, dtype=np.float32)
        n = len(coords)

        # 0-simplices
        self._simplices[0] = [OrientedSimplex((i,)) for i in range(n)]

        # Edges connect vertices at distance sqrt(2) (dot product == 1.0)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                dot = np.dot(self._vertices_coords[i], self._vertices_coords[j])
                if np.isclose(dot, 1.0):
                    edges.append(OrientedSimplex((i, j)))
        self._simplices[1] = edges

        # Triangular faces: triplets of vertices pairwise connected by edges
        edge_set = {e.vertices for e in edges}
        faces = []
        for c in combinations(range(n), 3):
            if (c[0], c[1]) in edge_set and (c[1], c[2]) in edge_set and (c[0], c[2]) in edge_set:
                faces.append(OrientedSimplex(c))
        self._simplices[2] = faces


class EightCell(BasePolytopeComplex):
    """8-Cell (Tesseract / 4-Cube).
    
    16 Vertices, 32 Edges, 24 Square Faces, 8 Cubic Cells.
    """
    def __init__(self):
        super().__init__("8-Cell")
        # 16 Vertices: (±1, ±1, ±1, ±1)
        coords = list(product([1.0, -1.0], repeat=4))
        self._vertices_coords = np.array(coords, dtype=np.float32)
        n = len(coords)

        self._simplices[0] = [OrientedSimplex((i,)) for i in range(n)]

        # Edges connect vertices at Hamming distance 1 (dot product == 2.0)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                dot = np.dot(self._vertices_coords[i], self._vertices_coords[j])
                if np.isclose(dot, 2.0):
                    edges.append(OrientedSimplex((i, j)))
        self._simplices[1] = edges


class SixHundredCell(BasePolytopeComplex):
    """600-Cell (Hexacosichoron / Tetraplex).
    
    120 Vertices, 720 Edges, 1200 Triangular Faces, 600 Tetrahedral Cells.
    Dual to 120-cell.
    """
    def __init__(self):
        super().__init__("600-Cell")
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        phi_inv = 1.0 / phi

        verts = set()
        # 8 vertices: (±2, 0, 0, 0) and permutations
        for p in set(permutations([2.0, 0.0, 0.0, 0.0])):
            for s in [1.0, -1.0]:
                verts.add(tuple(x * s for x in p))

        # 16 vertices: (±1, ±1, ±1, ±1)
        for s in product([1.0, -1.0], repeat=4):
            verts.add(s)

        # 96 vertices: even permutations of (±phi, ±1, ±1/phi, 0)
        indices = [0, 1, 2, 3]
        even_perms = []
        for p in permutations(indices):
            inv = sum(1 for i in range(4) for j in range(i+1, 4) if p[i] > p[j])
            if inv % 2 == 0:
                even_perms.append(p)

        base_pattern = [phi, 1.0, phi_inv, 0.0]
        for p_idx in even_perms:
            b = [base_pattern[i] for i in p_idx]
            for signs in product([1.0, -1.0], repeat=4):
                verts.add(tuple(b[i] * signs[i] for i in range(4)))

        coords = sorted(list(verts))
        self._vertices_coords = np.array(coords, dtype=np.float32)
        n = len(coords)

        self._simplices[0] = [OrientedSimplex((i,)) for i in range(n)]

        # Edges connect nearest neighbors (max non-self dot product)
        dots_0 = np.dot(self._vertices_coords, self._vertices_coords[0])
        max_dot = np.max(dots_0[1:])
        edges = []
        for i in range(n):
            dots = np.dot(self._vertices_coords, self._vertices_coords[i])
            for j in range(i + 1, n):
                if np.isclose(dots[j], max_dot, atol=1e-3):
                    edges.append(OrientedSimplex((i, j)))
        self._simplices[1] = edges

        # Triangular faces: 3-cliques in edge graph
        edge_set = {e.vertices for e in edges}
        faces = []
        for c in combinations(range(n), 3):
            if (c[0], c[1]) in edge_set and (c[1], c[2]) in edge_set and (c[0], c[2]) in edge_set:
                faces.append(OrientedSimplex(c))
        self._simplices[2] = faces

        # Tetrahedra: 4-cliques in edge graph
        tets = []
        for c in combinations(range(n), 4):
            if all((u, v) in edge_set for u, v in combinations(c, 2)):
                tets.append(OrientedSimplex(c))
        self._simplices[3] = tets


class OneTwentyCell(BasePolytopeComplex):
    """120-Cell (Hecatonacosachoron / Dodecaplex).
    
    600 Vertices, 1200 Edges, 720 Pentagonal Faces, 120 Dodecahedral Cells.
    Constructed via exact dual transformation of the 600-Cell tetrahedral basins.
    """
    def __init__(self):
        super().__init__("120-Cell")
        # Dual construction from 600-cell tetrahedral cells
        p600 = SixHundredCell()
        adj_600 = p600.adjacency_matrix()
        
        import networkx as nx
        G_600 = nx.from_numpy_array(adj_600)
        cliques = [tuple(sorted(c)) for c in nx.find_cliques(G_600) if len(c) == 4]
        
        # 600 Vertices correspond to 4-clique centroids
        p600_coords = p600.vertex_coordinates
        coords = np.array([np.mean(p600_coords[list(c)], axis=0) for c in cliques], dtype=np.float32)
        self._vertices_coords = coords
        n = len(coords)

        self._simplices[0] = [OrientedSimplex((i,)) for i in range(n)]

        # Two 120-cell vertices share an edge iff their 600-cell tetrahedra share 3 vertices (a face)
        sets = [set(c) for c in cliques]
        edges = []
        for i in range(n):
            set_i = sets[i]
            for j in range(i + 1, n):
                if len(set_i.intersection(sets[j])) == 3:
                    edges.append(OrientedSimplex((i, j)))
        self._simplices[1] = edges
