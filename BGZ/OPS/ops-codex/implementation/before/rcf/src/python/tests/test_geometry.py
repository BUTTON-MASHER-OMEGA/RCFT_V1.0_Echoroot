import unittest
from rcf_core.geometry.simplicial import OrientedSimplex, AbstractSimplicialComplex
from rcf_core.geometry.boundary import AbstractBoundaryOperator
from rcf_core.geometry.coboundary import AbstractCoboundaryOperator
from rcf_core.geometry.topology import AbstractTopologyInvariants

class SimpleTriangleComplex(AbstractSimplicialComplex[dict]):
    """Reference implementation of a 2D oriented triangle complex."""

    def vertices(self, complex_: dict) -> set[int]:
        return {0, 1, 2}

    def simplices(self, complex_: dict, dim: int) -> list[OrientedSimplex]:
        if dim == 0:
            return [OrientedSimplex((0,)), OrientedSimplex((1,)), OrientedSimplex((2,))]
        elif dim == 1:
            return [OrientedSimplex((0, 1)), OrientedSimplex((1, 2)), OrientedSimplex((0, 2))]
        elif dim == 2:
            return [OrientedSimplex((0, 1, 2))]
        return []

    def max_dimension(self, complex_: dict) -> int:
        return 2

    def is_valid(self, complex_: dict) -> bool:
        return True

class SimpleBoundaryOperator(AbstractBoundaryOperator[list[tuple[OrientedSimplex, float]]]):
    """Reference boundary operator acting on formal chains."""

    def boundary(self, chain: list[tuple[OrientedSimplex, float]], dim: int) -> list[tuple[OrientedSimplex, float]]:
        result = {}
        for simplex, coeff in chain:
            for face in simplex.faces():
                v = face.vertices
                result[v] = result.get(v, 0.0) + coeff * face.orientation
        return [(OrientedSimplex(v), weight) for v, weight in result.items() if abs(weight) > 1e-9]

class TestGeometrySubstrate(unittest.TestCase):

    def test_simplex_faces(self):
        # 2-simplex [0, 1, 2]
        triangle = OrientedSimplex((0, 1, 2))
        faces = triangle.faces()
        self.assertEqual(len(faces), 3)
        # Check alternating sign orientation
        self.assertEqual(faces[0], OrientedSimplex((1, 2), orientation=1))
        self.assertEqual(faces[1], OrientedSimplex((0, 2), orientation=-1))
        self.assertEqual(faces[2], OrientedSimplex((0, 1), orientation=1))

    def test_boundary_of_boundary_is_zero(self):
        op = SimpleBoundaryOperator()
        # 2-chain containing 1 triangle: 1 * [0, 1, 2]
        chain_2 = [(OrientedSimplex((0, 1, 2)), 1.0)]
        
        # ∂_2 (chain_2) -> 1-chain
        chain_1 = op.boundary(chain_2, dim=2)
        self.assertTrue(len(chain_1) > 0)
        
        # ∂_1 (∂_2 (chain_2)) -> 0-chain
        chain_0 = op.boundary(chain_1, dim=1)
        # Must sum to 0 (Boundary of boundary is zero)
        self.assertEqual(len(chain_0), 0, f"Expected ∂^2 = 0, got {chain_0}")

if __name__ == "__main__":
    unittest.main()
