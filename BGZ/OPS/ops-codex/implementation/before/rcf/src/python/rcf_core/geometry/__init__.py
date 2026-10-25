"""
Discrete Geometry Substrate for RCF.
"""

from rcf_core.geometry.simplicial import OrientedSimplex, AbstractSimplicialComplex
from rcf_core.geometry.boundary import AbstractBoundaryOperator
from rcf_core.geometry.coboundary import AbstractCoboundaryOperator
from rcf_core.geometry.topology import AbstractTopologyInvariants
from rcf_core.geometry import polytopes

__all__ = [
    "OrientedSimplex",
    "AbstractSimplicialComplex",
    "AbstractBoundaryOperator",
    "AbstractCoboundaryOperator",
    "AbstractTopologyInvariants",
    "polytopes",
]
