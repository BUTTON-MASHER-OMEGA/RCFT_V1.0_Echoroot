from abc import ABC, abstractmethod
from typing import Generic, TypeVar

StateType = TypeVar("StateType")

class AbstractGlobalCoherenceOperator(ABC, Generic[StateType]):
    """Abstract contract for Global Coherence Operator (GCO).
    
    Evaluates global structural viability and persistence detection.
    """

    @abstractmethod
    def detect_persistence(self, global_state: StateType) -> float:
        """Measures global persistence under constraint."""
        pass

    @abstractmethod
    def evaluate_structural_viability(self, global_state: StateType) -> bool:
        """Determines if the global state remains viable or requires elimination/reconfiguration."""
        pass
