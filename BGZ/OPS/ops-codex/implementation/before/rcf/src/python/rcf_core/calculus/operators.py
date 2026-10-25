from abc import ABC, abstractmethod
from typing import Generic, TypeVar

ChainType = TypeVar("ChainType")
CochainType = TypeVar("CochainType")

class AbstractDiscreteOperators(ABC, Generic[ChainType, CochainType]):
    """Abstract contract for discrete differential operators derived from ∂ and δ.
    
    Identities:
    - ∇ := δ_0 : C^0 → C^1  (Discrete Gradient)
    - curl := δ_1 : C^1 → C^2  (Discrete Curl)
    - div := -∂_1 : C_1 → C_0  (Discrete Divergence)
    
    Invariants:
    - curl(grad(f)) = 0
    - div(curl(v)) = 0
    """

    @abstractmethod
    def gradient(self, cochain_0: CochainType) -> CochainType:
        """Computes discrete gradient ∇f = δ_0(f)."""
        pass

    @abstractmethod
    def curl(self, cochain_1: CochainType) -> CochainType:
        """Computes discrete curl(A) = δ_1(A)."""
        pass

    @abstractmethod
    def divergence(self, chain_1: ChainType) -> ChainType:
        """Computes discrete divergence div(V) = -∂_1(V)."""
        pass
