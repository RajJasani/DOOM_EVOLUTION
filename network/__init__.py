"""DOOM Agent Neural Network and Memory Architecture package.

Exposes DoomNetwork, NetworkConfig, submodules, and output containers.
"""

from .config import NetworkConfig
from .encoder import FeatureFusion, MemoryEncoder, RaycastEncoder, VitalsEncoder
from .enemy_attention import EnemyAttentionModule
from .memory import RecurrentMemory
from .model import DoomNetwork, EvaluationResult, StepResult
from .policy import PolicyHead, ValueHead

__all__ = [
    "NetworkConfig",
    "VitalsEncoder",
    "RaycastEncoder",
    "MemoryEncoder",
    "EnemyAttentionModule",
    "FeatureFusion",
    "RecurrentMemory",
    "PolicyHead",
    "ValueHead",
    "DoomNetwork",
    "StepResult",
    "EvaluationResult",
]
