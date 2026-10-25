from abc import ABC, abstractmethod
from typing import Generic, TypeVar

ComplexType = TypeVar("ComplexType")

class AbstractTopologyInvariants(ABC, Generic[ComplexType]):
    """Abstract contract for discrete topological invariants.
    
    Invariants:
    - Euler characteristic χ = Σ (-1)^k N_k
    - Betti numbers β_k = dim ker ∂_k - dim im ∂_{k+1}
    """

    @abstractmethod
    def euler_characteristic(self, complex_: ComplexType) -> int:
        """Computes Euler characteristic χ for the complex."""
        pass

    @abstractmethod
    def betti_number(self, complex_: ComplexType, dim: int) -> int:
        """Computes the k-th Betti number β_k for the complex."""
        pass
