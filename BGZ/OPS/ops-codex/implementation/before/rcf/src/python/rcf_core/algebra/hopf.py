from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from rcf_core.algebra.coalgebra import AbstractCoalgebra

StateType = TypeVar("StateType")

class AbstractHopfAlgebra(AbstractCoalgebra[StateType], ABC, Generic[StateType]):
    """Abstract contract for reversible synthesis and analysis via Hopf algebra.
    
    Guarantees:
    - Antipode map S: A → A (Inversion / Reversibility operator)
    - Reversible synthesis (multiplication m) and analysis (comultiplication Δ)
    """

    @abstractmethod
    def antipode(self, state: StateType) -> StateType:
        """Applies the antipode operator S(x) to reverse/invert relational elements."""
        pass

    @abstractmethod
    def synthesize(self, left: StateType, right: StateType) -> StateType:
        """Combines two relational elements into a unified state (m: A ⊗ A → A)."""
        pass
