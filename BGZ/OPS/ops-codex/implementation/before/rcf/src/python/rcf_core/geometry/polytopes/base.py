from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np
from rcf_core.geometry.simplicial import OrientedSimplex, AbstractSimplicialComplex
from rcf_core.geometry.boundary import AbstractBoundaryOperator

class BasePolytopeComplex(AbstractSimplicialComplex[dict], AbstractBoundaryOperator[dict], ABC):
    """Base class for concrete polytope complexes inheriting from RCF core geometry interfaces."""

    def __init__(self, name: str):
        self.name = name
        self._vertices_coords: np.ndarray = np.array([])
        self._simplices: dict[int, list[OrientedSimplex]] = {}

    @property
    def num_vertices(self) -> int:
        return len(self._vertices_coords)

    @property
    def vertex_coordinates(self) -> np.ndarray:
        return self._vertices_coords

    def vertices(self, complex_: dict = None) -> set[int]:
        return set(range(self.num_vertices))

    def simplices(self, complex_: dict = None, dim: int = 0) -> Sequence[OrientedSimplex]:
        return self._simplices.get(dim, [])

    def max_dimension(self, complex_: dict = None) -> int:
        if not self._simplices:
            return 0
        return max(self._simplices.keys())

    def is_valid(self, complex_: dict = None) -> bool:
        # Verify face closure
        for dim, simp_list in self._simplices.items():
            if dim <= 0:
                continue
            lower_simplices = {s.vertices for s in self._simplices.get(dim - 1, [])}
            for simp in simp_list:
                for face in simp.faces():
                    # Unoriented vertex tuple set check
                    if face.vertices not in lower_simplices:
                        return False
        return True

    def adjacency_matrix(self) -> np.ndarray:
        """Computes the 0-1 unweighted vertex-vertex adjacency matrix."""
        n = self.num_vertices
        adj = np.zeros((n, n), dtype=np.float32)
        edges = self.simplices(dim=1)
        for e in edges:
            u, v = e.vertices[0], e.vertices[1]
            adj[u, v] = 1.0
            adj[v, u] = 1.0
        return adj

    def boundary_matrix(self, dim: int) -> np.ndarray:
        """Computes the sparse/dense boundary matrix ∂_k mapping k-simplices to (k-1)-simplices."""
        k_simplices = self.simplices(dim=dim)
        km1_simplices = self.simplices(dim=dim - 1)
        
        if not k_simplices or not km1_simplices:
            return np.zeros((0, 0), dtype=np.float32)

        km1_index = {s.vertices: i for i, s in enumerate(km1_simplices)}
        matrix = np.zeros((len(km1_simplices), len(k_simplices)), dtype=np.float32)

        for col, simp in enumerate(k_simplices):
            for face in simp.faces():
                if face.vertices in km1_index:
                    row = km1_index[face.vertices]
                    matrix[row, col] += face.orientation

        return matrix

    def boundary(self, chain: dict, dim: int) -> dict:
        """Applies boundary operator ∂_k to a chain dict mapping OrientedSimplex -> float."""
        result = {}
        for simplex, coeff in chain.items():
            for face in simplex.faces():
                result[face] = result.get(face, 0.0) + coeff * face.orientation
        return {s: c for s, c in result.items() if abs(c) > 1e-9}

    def euler_characteristic(self) -> int:
        """χ = Σ (-1)^k N_k"""
        chi = 0
        for dim, simp_list in self._simplices.items():
            sign = (-1) ** dim
            chi += sign * len(simp_list)
        return chi
