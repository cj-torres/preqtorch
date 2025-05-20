# Import main classes and functions
from .core import PrequentialEncoder, ReplayStreams
from .encoders import BlockEncoder, MIRSEncoder
from .clustering import (
    prequential_clustering, 
    detect_codelength_boundaries,
    consensus_prequential_clustering,
    crp_clustering
)

__version__ = "0.0.1"

__all__ = [
    'PrequentialEncoder',
    'BlockEncoder',
    'MIRSEncoder',
    'ReplayStreams',
    'prequential_clustering',
    'detect_codelength_boundaries',
    'consensus_prequential_clustering',
    'crp_clustering',
    '__version__'
]