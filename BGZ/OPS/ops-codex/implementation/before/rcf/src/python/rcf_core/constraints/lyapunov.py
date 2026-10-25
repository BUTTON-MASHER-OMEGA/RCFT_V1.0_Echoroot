from abc import ABC, abstractmethod
from typing import Generic, TypeVar

StateType = TypeVar("StateType")

class AbstractLyapunovDescentGate(ABC, Generic[StateType]):
    """Abstract contract for Lyapunov descent gating.
    
    Monotonicity constraint ensuring state transitions do not increase structural instability:
        L(x_{t+1}) <= L(x_t)
    """

    @abstractmethod
    def lyapunov_energy(self, state: StateType) -> float:
        """Returns the structural instability / energy measure L(x)."""
        pass

    def is_descent_valid(self, state_before: StateType, state_after: StateType) -> bool:
        """Evaluates monotonicity: L(state_after) <= L(state_before)."""
        return self.lyapunov_energy(state_after) <= self.lyapunov_energy(state_before)
