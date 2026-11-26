"""Configuration dataclasses for the DOOM neural network and memory architecture.

Specifies input dimensions matching docs/CONTRACT.md and env/doom_env/spaces.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class NetworkConfig:
    """Hyperparameters and architectural dimensions for DoomNetwork."""

    # ----------------- Observation Space Dimensions (from CONTRACT.md) -----------------
    # Vitals: 17 scalar features (health, armor, weapons, ammo, keys, velocities)
    vitals_dim: int = 17
    # Canonical vitals normalization: False by default because canonical perception pipeline
    # (PerceptionBridge / SensoryPipeline / VitalsProcessor) normalizes vitals to [0, 1] before network input.
    # Set to True if feeding raw unnormalized vitals directly into DoomNetwork.
    normalize_vitals: bool = False

    # Spatial Memory Features (Member 2 PerceptionBridge interface)
    # 64-dimensional egocentric spatial memory vector (doors, keys, threats, direction coverage)
    memory_feature_dim: int = 64
    memory_hidden_dim: int = 64

    # Rays: 64 rays across 360 degrees
    num_rays: int = 64
    # Ray hit types: 6 categories (none, wall, door, locked_door, enemy, item)
    num_ray_hit_types: int = 6
    ray_type_embed_dim: int = 16
    ray_hidden_dim: int = 128

    # Entities: up to 32 visible entities
    max_entities: int = 32
    # Entity kinematic features: 5 (dist, sin_rel_ang, cos_rel_ang, rel_vel_fwd, rel_vel_right)
    entity_features_dim: int = 5
    # Entity classes: 12 (pad=0, enemies, items, projectiles, etc.)
    num_entity_types: int = 12
    entity_type_embed_dim: int = 16
    entity_token_dim: int = 64

    # Enemy Attention Module
    attention_num_heads: int = 4
    attention_embed_dim: int = 64
    attention_dropout: float = 0.0

    # Optional Visual Embedding (Member 2 perception interface)
    # 0 indicates visual inputs are not processed directly or pre-encoded by perception module
    visual_feature_dim: int = 0

    # ----------------- Feature Fusion & Memory -----------------
    vitals_hidden_dim: int = 64
    fusion_hidden_dim: int = 256
    
    # Recurrent Memory (GRU)
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1
    rnn_dropout: float = 0.0

    # ----------------- Actor-Critic Heads -----------------
    # Action space: 9 MultiBinary buttons
    # (MOVE_FWD, MOVE_BACK, MOVE_LEFT, MOVE_RIGHT, TURN_LEFT, TURN_RIGHT, ATTACK, USE, NEXT_WEAPON)
    action_dim: int = 9
    critic_hidden_dim: int = 128

    # ----------------- General Architecture Options -----------------
    activation: str = "gelu"  # "relu", "gelu", "silu", "tanh"
    use_layer_norm: bool = True
    orthogonal_init: bool = True
