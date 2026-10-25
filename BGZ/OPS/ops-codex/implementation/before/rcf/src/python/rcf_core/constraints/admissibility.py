from abc import ABC, abstractmethod
from typing import Generic, TypeVar

StateType = TypeVar("StateType")

class AbstractRelationalAdmissibility(ABC, Generic[StateType]):
    """Abstract contract for pre-formal relational admissibility filtering.
    
    Structures that persist under relational pressure return higher admissibility;
    structures that collapse return zero or fall below the ε-allowance threshold.
    """

    @abstractmethod
    def evaluate_admissibility(self, state: StateType) -> float:
        """Returns an admissibility score in [0.0, 1.0]."""
        pass
