import unittest
import numpy as np
from rcf_core.geometry.polytopes import (
    FiveCell,
    SixteenCell,
    TwentyFourCell,
    EightCell,
    SixHundredCell,
    OneTwentyCell,
    Demi5,
    DualDemi5,
    E8Roots,
)

class TestPolytopeGeometries(unittest.TestCase):

    def test_five_cell(self):
        p = FiveCell()
        self.assertEqual(p.num_vertices, 5)
        self.assertEqual(len(p.simplices(dim=1)), 10)
        self.assertEqual(len(p.simplices(dim=2)), 10)
        self.assertEqual(len(p.simplices(dim=3)), 5)
        self.assertEqual(p.euler_characteristic(), 0) # χ = 5 - 10 + 10 - 5 = 0

        # Verify ∂_1 ∘ ∂_2 = 0
        d1 = p.boundary_matrix(dim=1)
        d2 = p.boundary_matrix(dim=2)
        d1_d2 = np.dot(d1, d2)
        self.assertTrue(np.allclose(d1_d2, 0.0), f"Expected ∂_1 ∘ ∂_2 = 0, got max diff {np.max(np.abs(d1_d2))}")

    def test_sixteen_cell(self):
        p = SixteenCell()
        self.assertEqual(p.num_vertices, 8)
        self.assertEqual(len(p.simplices(dim=1)), 24)
        self.assertEqual(len(p.simplices(dim=2)), 32)
        self.assertEqual(len(p.simplices(dim=3)), 16)
        self.assertEqual(p.euler_characteristic(), 0) # χ = 8 - 24 + 32 - 16 = 0

        # Verify ∂_1 ∘ ∂_2 = 0
        d1 = p.boundary_matrix(dim=1)
        d2 = p.boundary_matrix(dim=2)
        d1_d2 = np.dot(d1, d2)
        self.assertTrue(np.allclose(d1_d2, 0.0))

    def test_twenty_four_cell(self):
        p = TwentyFourCell()
        self.assertEqual(p.num_vertices, 24)
        self.assertEqual(len(p.simplices(dim=1)), 96)
        self.assertEqual(len(p.simplices(dim=2)), 96)

        # Verify ∂_1 ∘ ∂_2 = 0
        d1 = p.boundary_matrix(dim=1)
        d2 = p.boundary_matrix(dim=2)
        d1_d2 = np.dot(d1, d2)
        self.assertTrue(np.allclose(d1_d2, 0.0))

    def test_eight_cell(self):
        p = EightCell()
        self.assertEqual(p.num_vertices, 16)
        self.assertEqual(len(p.simplices(dim=1)), 32)
        adj = p.adjacency_matrix()
        self.assertEqual(adj.shape, (16, 16))
        self.assertTrue(np.allclose(adj, adj.T))

    def test_six_hundred_cell(self):
        p = SixHundredCell()
        self.assertEqual(p.num_vertices, 120)
        self.assertEqual(len(p.simplices(dim=1)), 720)
        self.assertEqual(len(p.simplices(dim=2)), 1200)
        self.assertEqual(len(p.simplices(dim=3)), 600)
        self.assertEqual(p.euler_characteristic(), 0) # χ = 120 - 720 + 1200 - 600 = 0

        # Verify ∂_1 ∘ ∂_2 = 0
        d1 = p.boundary_matrix(dim=1)
        d2 = p.boundary_matrix(dim=2)
        d1_d2 = np.dot(d1, d2)
        self.assertTrue(np.allclose(d1_d2, 0.0))

    def test_one_twenty_cell(self):
        p = OneTwentyCell()
        self.assertEqual(p.num_vertices, 600)
        self.assertEqual(len(p.simplices(dim=1)), 1200)

    def test_demi5(self):
        p = Demi5()
        self.assertEqual(p.num_vertices, 16)
        self.assertEqual(len(p.simplices(dim=1)), 80)
        adj = p.adjacency_matrix()
        self.assertEqual(adj.shape, (16, 16))
        # Average degree should be 10.0
        self.assertEqual(adj.sum() / 16.0, 10.0)

    def test_dual_demi5(self):
        p = DualDemi5()
        self.assertEqual(p.num_vertices, 26)
        self.assertEqual(len(p.simplices(dim=1)), 120)
        adj = p.adjacency_matrix()
        self.assertEqual(adj.shape, (26, 26))
        degrees = adj.sum(axis=1)
        self.assertEqual(degrees[0], 16.0) # Group A degree 16
        self.assertEqual(degrees[10], 5.0) # Group B degree 5

    def test_e8_roots(self):
        p = E8Roots()
        self.assertEqual(p.num_vertices, 240)
        adj = p.adjacency_matrix()
        self.assertEqual(adj.shape, (240, 240))
        self.assertTrue(np.allclose(adj, adj.T))

if __name__ == "__main__":
    unittest.main()
