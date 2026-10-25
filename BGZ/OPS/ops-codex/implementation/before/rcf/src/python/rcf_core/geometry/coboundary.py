from abc import ABC, abstractmethod
from typing import Generic, TypeVar

CochainType = TypeVar("CochainType")

class AbstractCoboundaryOperator(ABC, Generic[CochainType]):
    """Abstract contract for coboundary operator δ.
    
    Invariant Property:
        δ_{k+1} ∘ δ_k = 0  (Coboundary of coboundary is zero)
    """

    @abstractmethod
    def coboundary(self, cochain: CochainType, dim: int) -> CochainType:
        """Applies coboundary operator δ_k on a k-cochain to yield a (k+1)-cochain."""
        pass
