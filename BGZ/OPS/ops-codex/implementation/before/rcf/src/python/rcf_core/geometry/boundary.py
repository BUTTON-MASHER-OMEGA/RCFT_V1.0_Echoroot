from abc import ABC, abstractmethod
from typing import Generic, TypeVar

ChainType = TypeVar("ChainType")

class AbstractBoundaryOperator(ABC, Generic[ChainType]):
    """Abstract contract for boundary operator ∂.
    
    Invariant Property:
        ∂_{k-1} ∘ ∂_k = 0  (Boundary of boundary is zero)
    """

    @abstractmethod
    def boundary(self, chain: ChainType, dim: int) -> ChainType:
        """Applies boundary operator ∂_k on a k-chain to yield a (k-1)-chain."""
        pass
