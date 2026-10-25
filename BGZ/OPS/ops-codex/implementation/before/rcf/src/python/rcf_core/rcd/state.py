from abc import ABC, abstractmethod
from typing import Generic, TypeVar

KernelType = TypeVar("KernelType")

class AbstractRelationalState(ABC, Generic[KernelType]):
    """Abstract contract for relational state representation under RCD.
    
    RCF Invariant:
    - Persistent structure is encoded spectrally, not configurationally.
    """

    @abstractmethod
    def kernel(self) -> KernelType:
        """Returns the relational kernel matrix/operator."""
        pass

    @abstractmethod
    def spectral_modes(self) -> KernelType:
        """Returns the persistent eigenstructure / spectral mode representation."""
        pass
