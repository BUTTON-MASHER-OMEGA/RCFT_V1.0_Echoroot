from abc import ABC, abstractmethod
from typing import Generic, TypeVar

StateType = TypeVar("StateType")

class AbstractCoalgebra(ABC, Generic[StateType]):
    """Abstract contract for formal decomposition grammar (Coalgebra).
    
    Guarantees:
    - Comultiplication Δ: A → A ⊗ A (Coassociative decomposition)
    - Counit ε: A → K (Preservation of unit scalar)
    """

    @abstractmethod
    def comultiplication(self, state: StateType) -> tuple[StateType, StateType]:
        """Decomposes a state into a tensored pair (Δ(x))."""
        pass

    @abstractmethod
    def counit(self, state: StateType) -> float:
        """Evaluates the counit map ε(x)."""
        pass
