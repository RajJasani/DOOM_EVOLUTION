# perception/processors package
"""
Modular sensory processors for the DOOM autonomous agent.

Each processor takes one section of the raw obs dict and returns a typed
dataclass of clean, normalised, tensor-ready features.

Public API
----------
from perception.processors import (
    RaycastProcessor,   RayFeatures,
    DepthProcessor,     DepthFeatures,
    EntityProcessor,    EntityFeatures,
    VitalsProcessor,    VitalsFeatures,
    SensoryPipeline,    ProcessedObservation,
)
"""

from .raycast_processor import RaycastProcessor, RayFeatures
from .depth_processor   import DepthProcessor,   DepthFeatures
from .entity_processor  import EntityProcessor,  EntityFeatures
from .vitals_processor  import VitalsProcessor,  VitalsFeatures
from .pipeline          import SensoryPipeline,  ProcessedObservation

__all__ = [
    "RaycastProcessor",  "RayFeatures",
    "DepthProcessor",    "DepthFeatures",
    "EntityProcessor",   "EntityFeatures",
    "VitalsProcessor",   "VitalsFeatures",
    "SensoryPipeline",   "ProcessedObservation",
]
