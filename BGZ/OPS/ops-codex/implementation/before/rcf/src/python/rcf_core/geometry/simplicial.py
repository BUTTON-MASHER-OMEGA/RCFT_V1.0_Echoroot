from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar

ComplexType = TypeVar("ComplexType")

@dataclass(frozen=True)
class OrientedSimplex:
    """Represents an oriented k-simplex defined by an ordered sequence of vertices up to even permutation.
    
    Attributes:
        vertices: Tuple of vertex identifiers (0-simplex: 1 vertex, 1-simplex: 2, etc.)
        orientation: Sign factor (+1 or -1) representing orientation.
    """
    vertices: tuple[int, ...]
    orientation: int = 1

    @property
    def dimension(self) -> int:
        """Dimension k of the simplex."""
        return len(self.vertices) - 1

    def faces(self) -> list["OrientedSimplex"]:
        """Returns the boundary faces of dimension k-1 with alternating orientation signs.
        
        ∂_k [v_0, ..., v_k] = Σ_i (-1)^i [v_0, ..., ^v_i, ..., v_k]
        """
        if self.dimension <= 0:
            return []
        
        face_list = []
        for i in range(len(self.vertices)):
            sub_vertices = self.vertices[:i] + self.vertices[i+1:]
            sign = self.orientation * ((-1) ** i)
            face_list.append(OrientedSimplex(vertices=sub_vertices, orientation=sign))
        return face_list


class AbstractSimplicialComplex(ABC, Generic[ComplexType]):
    """Abstract contract for discrete simplicial complex K.
    
    Defines structural constraints:
    - Closure under taking faces
    - Non-degeneracy (no repeated vertices in a simplex)
    - Orientation consistency across shared faces
    """

    @abstractmethod
    def vertices(self, complex_: ComplexType) -> set[int]:
        """Returns the set of 0-simplices (vertices) V."""
        pass

    @abstractmethod
    def simplices(self, complex_: ComplexType, dim: int) -> Sequence[OrientedSimplex]:
        """Returns all oriented k-simplices in the complex for dimension k."""
        pass

    @abstractmethod
    def max_dimension(self, complex_: ComplexType) -> int:
        """Returns the maximum dimension of simplices in the complex."""
        pass

    @abstractmethod
    def is_valid(self, complex_: ComplexType) -> bool:
        """Checks structural validity: face closure, non-degeneracy, orientation consistency."""
        pass
