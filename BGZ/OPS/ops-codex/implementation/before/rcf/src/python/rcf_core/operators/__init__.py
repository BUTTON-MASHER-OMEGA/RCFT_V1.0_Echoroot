"""
Coherence Operators & Interfaces for RCF.
"""

from rcf_core.operators.lco import AbstractLocalCoherenceOperator
from rcf_core.operators.gco import AbstractGlobalCoherenceOperator
from rcf_core.operators.temporal import AbstractEventIndexedTemporalInterface

__all__ = [
    "AbstractLocalCoherenceOperator",
    "AbstractGlobalCoherenceOperator",
    "AbstractEventIndexedTemporalInterface",
]
