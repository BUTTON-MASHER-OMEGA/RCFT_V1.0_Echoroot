import unittest
from rcf_core.calculus.operators import AbstractDiscreteOperators
from rcf_core.constraints.lyapunov import AbstractLyapunovDescentGate

class ReferenceDiscreteOperators(AbstractDiscreteOperators[dict, dict]):
    """Reference implementation of discrete gradient, curl, divergence."""

    def gradient(self, cochain_0: dict) -> dict:
        # cochain_0: map node -> scalar
        # grad(f) on edge (u, v) = f(v) - f(u)
        res = {}
        nodes = list(cochain_0.keys())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                res[(u, v)] = cochain_0[v] - cochain_0[u]
        return res

    def curl(self, cochain_1: dict) -> dict:
        # curl(A) on face (u, v, w) = A(u,v) + A(v,w) - A(u,w)
        res = {}
        edges = list(cochain_1.keys())
        for (u, v) in edges:
            for (v2, w) in edges:
                if v == v2 and (u, w) in cochain_1:
                    res[(u, v, w)] = cochain_1[(u, v)] + cochain_1[(v, w)] - cochain_1[(u, w)]
        return res

    def divergence(self, chain_1: dict) -> dict:
        res = {}
        for (u, v), val in chain_1.items():
            res[u] = res.get(u, 0.0) - val
            res[v] = res.get(v, 0.0) + val
        return res

class ReferenceLyapunovGate(AbstractLyapunovDescentGate[float]):
    def lyapunov_energy(self, state: float) -> float:
        return state

class TestCalculusAndConstraints(unittest.TestCase):

    def test_curl_grad_is_zero(self):
        ops = ReferenceDiscreteOperators()
        # 0-cochain: node values f(0)=1.0, f(1)=3.0, f(2)=7.0
        f = {0: 1.0, 1: 3.0, 2: 7.0}
        
        # ∇f
        grad_f = ops.gradient(f)
        self.assertEqual(grad_f[(0, 1)], 2.0)
        self.assertEqual(grad_f[(1, 2)], 4.0)
        self.assertEqual(grad_f[(0, 2)], 6.0)
        
        # curl(∇f) on triangle (0, 1, 2)
        curl_grad_f = ops.curl(grad_f)
        # 2.0 + 4.0 - 6.0 = 0.0
        self.assertEqual(curl_grad_f[(0, 1, 2)], 0.0, "Expected curl(grad(f)) == 0")

    def test_lyapunov_descent_gate(self):
        gate = ReferenceLyapunovGate()
        # Valid descent: energy drops from 10.0 -> 5.0
        self.assertTrue(gate.is_descent_valid(10.0, 5.0))
        # Invalid descent: energy increases from 5.0 -> 8.0
        self.assertFalse(gate.is_descent_valid(5.0, 8.0))

if __name__ == "__main__":
    unittest.main()
