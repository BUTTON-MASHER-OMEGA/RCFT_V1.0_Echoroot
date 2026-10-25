from abc import ABC, abstractmethod
from typing import Generic, TypeVar

StateType = TypeVar("StateType")
ModeType = TypeVar("ModeType")

class AbstractLocalCoherenceOperator(ABC, Generic[StateType, ModeType]):
    """Abstract contract for Local Coherence Operator (LCO).
    
    ADR-002: LCO evaluates ONLY locally available relational information.
    No global decision makers.
    """

    @abstractmethod
    def compute_local_coherence(self, local_state: StateType) -> float:
        """Evaluates local coherence score for a neighborhood."""
        pass

    @abstractmethod
    def extract_local_modes(self, local_state: StateType) -> ModeType:
        """Extracts local spectral mode bandwidth (including ε-allowance residual)."""
        pass

    @abstractmethod
    def apply_update_policy(self, local_state: StateType) -> StateType:
        """Applies local update policy to refine the state."""
        pass
