"""Perception module for DOOM autonomous agent.

Member 2 — Spatial Intelligence & Sensory Processing.
Member 1 — Spatial Perception baseline & integration.

Public API
----------
SpatialPerception                  → geometric corridor, door, and threat parser (Member 1)
process_observation(obs, device)   → enriched obs dict ready for DoomNetwork (Member 2)
ObservationInspector               → developer tool: print / visualise any obs
SpatialAnalyzer                    → ray and entity statistics helpers
normalize_entity_features(feats)   → (32, 5) float32, inplace-safe normalisation

Integration contract (docs/ARCHITECTURE_MEMBER3.md §6)
------------------------------------------------------
Set NetworkConfig.visual_feature_dim = D before instantiating DoomNetwork.
Then add obs["visual_embedding"] = Tensor(..., D) in this module's output.
FeatureFusion will concatenate it automatically.

If visual_feature_dim == 0 (default), visual_embedding is never added and the
network runs purely on rays, vitals, and entity features (fastest / smallest).
"""

from __future__ import annotations

from .spatial import SpatialPerception
from .preprocessing import (
    normalize_entity_features,
    normalize_vitals_numpy,
    validate_observation,
    obs_to_device_tensors,
    process_observation,
)
from .observation_inspector import ObservationInspector
from .spatial_analyzer import SpatialAnalyzer
from .processors import (
    RaycastProcessor,  RayFeatures,
    DepthProcessor,    DepthFeatures,
    EntityProcessor,   EntityFeatures,
    VitalsProcessor,   VitalsFeatures,
    SensoryPipeline,   ProcessedObservation,
)

from .spatial_memory import (
    EgocentricSpatialMemory,
    MEMORY_FEATURE_DIM,
    ObjectMemorySlot,
    KeyCardTracker,
    MemorySnapshot,
    DirectionExplorationTracker,
)
from .bridge import PerceptionBridge, BridgeStats, MEMORY_KEY
from .adapter import ObservationAdapter, VectorObservationAdapter

__all__ = [
    # Member 1 spatial reasoning
    "SpatialPerception",
    # Pre-existing helpers
    "normalize_entity_features",
    "normalize_vitals_numpy",
    "validate_observation",
    "obs_to_device_tensors",
    "process_observation",
    "ObservationInspector",
    "SpatialAnalyzer",
    # New modular pipeline (Phase 2)
    "RaycastProcessor",  "RayFeatures",
    "DepthProcessor",    "DepthFeatures",
    "EntityProcessor",   "EntityFeatures",
    "VitalsProcessor",   "VitalsFeatures",
    "SensoryPipeline",   "ProcessedObservation",
    # Spatial memory (Phase 3)
    "EgocentricSpatialMemory",
    "MEMORY_FEATURE_DIM",
    "ObjectMemorySlot",
    "KeyCardTracker",
    "MemorySnapshot",
    "DirectionExplorationTracker",
    # Integration bridge & adapters (Phase 4)
    "PerceptionBridge",
    "BridgeStats",
    "MEMORY_KEY",
    "ObservationAdapter",
    "VectorObservationAdapter",
]
