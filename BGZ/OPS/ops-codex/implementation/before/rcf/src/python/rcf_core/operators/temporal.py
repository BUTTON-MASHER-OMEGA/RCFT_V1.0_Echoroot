from abc import ABC, abstractmethod
from typing import Generic, TypeVar

StateType = TypeVar("StateType")

class AbstractEventIndexedTemporalInterface(ABC, Generic[StateType]):
    """Abstract contract for event-indexed temporal evolution (ADR-003).
    
    Time is represented by discrete event refinement rather than continuous background evolution.
    """

    @abstractmethod
    def current_event_index(self) -> int:
        """Returns the current discrete event index t."""
        pass

    @abstractmethod
    def step_event(self, state: StateType) -> tuple[int, StateType]:
        """Advances system by one discrete event step: t -> t+1."""
        pass
