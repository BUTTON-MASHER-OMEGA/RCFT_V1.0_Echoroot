from abc import ABC, abstractmethod
from typing import Generic, TypeVar

StateType = TypeVar("StateType")

class AbstractRefinementEvent(ABC, Generic[StateType]):
    """Abstract contract for minimal refinement-atom transitions (Discrete Events).
    
    In RCF, time is event-indexed (ADR-003). A transition state_before -> state_after
    is an admissible event only if it satisfies refinement granularity constraints.
    """

    @abstractmethod
    def is_admissible_event(self, state_before: StateType, state_after: StateType) -> bool:
        """Evaluates whether a state transition qualifies as an admissible discrete event."""
        pass
