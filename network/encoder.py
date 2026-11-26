"""Observation encoders for numerical, spatial, and multi-modal DOOM features.

Processes player vitals, ego-centric circular raycasts, and feature fusion
according to docs/CONTRACT.md and env/doom_env/spaces.py.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NetworkConfig


def get_activation(name: str) -> nn.Module:
    """Return activation module by string name."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "silu":
        return nn.SiLU()
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation: {name}")


class VitalsEncoder(nn.Module):
    """Encodes the 17-dimensional player vitals vector into a compact latent vector.

    Vitals fields (from spaces.py):
      0: health (0-200)
      1: armor (0-200)
      2: selected_weapon (0-7)
      3: selected_weapon_ammo (count)
      4-7: ammo_bullets, ammo_shells, ammo_rockets, ammo_cells
      8-13: key_blue_card, key_yellow_card, key_red_card, key_blue_skull, key_yellow_skull, key_red_skull (0/1)
      14-16: vel_forward, vel_right, vel_up (ego velocities in map units/tic)
    """

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        in_dim = config.vitals_dim
        hidden_dim = config.vitals_hidden_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if config.use_layer_norm else nn.Identity(),
            get_activation(config.activation),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if config.use_layer_norm else nn.Identity(),
            get_activation(config.activation),
        )

    def normalize_vitals(self, vitals: torch.Tensor) -> torch.Tensor:
        """Apply numerically stable scaling to vitals components."""
        norm_vitals = vitals.clone()
        # Health & armor: scaled by 1/100 (normal range 0..2.0)
        norm_vitals[..., 0:2] = norm_vitals[..., 0:2] / 100.0
        # Selected weapon: scaled to 0..1
        norm_vitals[..., 2:3] = norm_vitals[..., 2:3] / 7.0
        # Ammo counts: safe log1p scaling to compress high counts smoothly
        ammo = torch.clamp(norm_vitals[..., 3:8], min=0.0)
        norm_vitals[..., 3:8] = torch.log1p(ammo) / 4.0
        # Key flags (indices 8-14) are already in {0, 1}
        # Ego-frame velocities (indices 14-17): scale ~20 units/tic
        norm_vitals[..., 14:17] = norm_vitals[..., 14:17] / 20.0
        return norm_vitals

    def forward(self, vitals: torch.Tensor, normalize: bool | None = None) -> torch.Tensor:
        """Args:
            vitals: Tensor of shape (..., 17)
            normalize: Optional override. If None, uses config.normalize_vitals.
        Returns:
            Tensor of shape (..., vitals_hidden_dim)
        """
        should_normalize = normalize if normalize is not None else getattr(self.config, "normalize_vitals", False)
        normed = self.normalize_vitals(vitals) if should_normalize else vitals
        return self.net(normed)


class RaycastEncoder(nn.Module):
    """Circular 1D Convolutional & MLP encoder for 360-degree ego-centric raycasts.

    The 64 rays start at 0 deg (straight ahead) and advance counter-clockwise.
    Ray 0 and Ray 63 are topologically adjacent in physical space.
    We leverage circular padding to preserve spatial continuity.
    """

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        self.num_rays = config.num_rays
        self.type_embed_dim = config.ray_type_embed_dim
        self.hidden_dim = config.ray_hidden_dim

        # Embedding for the 6 ray hit categories (none, wall, door, locked_door, enemy, item)
        self.type_embedding = nn.Embedding(config.num_ray_hit_types, self.type_embed_dim)

        per_ray_channels = 1 + self.type_embed_dim  # distance (1) + type embedding

        # Circular 1D convolutions across the 64 ray directions
        self.conv1 = nn.Conv1d(per_ray_channels, 32, kernel_size=3, padding=0)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=0)
        self.act = get_activation(config.activation)

        # Linear projection combining flattened conv features and direct linear representation
        flattened_dim = 64 * 64
        self.fc = nn.Sequential(
            nn.Linear(flattened_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim) if config.use_layer_norm else nn.Identity(),
            self.act,
        )

    def forward(self, distance: torch.Tensor, hit_type: torch.Tensor) -> torch.Tensor:
        """Args:
            distance: Tensor of shape (..., 64), float in [0.0, 1.0]
            hit_type: Tensor of shape (..., 64), integer in [0, 5]
        Returns:
            Tensor of shape (..., ray_hidden_dim)
        """
        leading_shape = distance.shape[:-1]
        dist_flat = distance.reshape(-1, self.num_rays, 1)
        type_flat = hit_type.reshape(-1, self.num_rays).long()

        # Embed categorical hit types: (B, 64, embed_dim)
        type_emb = self.type_embedding(type_flat)

        # Concatenate distance and type embeddings: (B, 64, 1 + embed_dim)
        ray_feats = torch.cat([dist_flat, type_emb], dim=-1)

        # Transpose for Conv1d: (B, channels, 64)
        x = ray_feats.transpose(1, 2)

        # Circular padding of 1 on each side preserves 360-degree circular topology
        x_pad1 = F.pad(x, (1, 1), mode="circular")
        h1 = self.act(self.conv1(x_pad1))

        x_pad2 = F.pad(h1, (1, 1), mode="circular")
        h2 = self.act(self.conv2(x_pad2))

        # Flatten and project to hidden representation
        h2_flat = h2.reshape(h2.shape[0], -1)
        out = self.fc(h2_flat)

        # Restore leading batch/sequence dimensions
        if leading_shape:
            out = out.reshape(*leading_shape, self.hidden_dim)
        return out


class MemoryEncoder(nn.Module):
    """Encodes the 64-dimensional spatial memory vector from PerceptionBridge.

    Projects doors, keys, threats, and direction exploration summaries into
    a compact latent embedding for feature fusion.
    """

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        in_dim = config.memory_feature_dim
        hidden_dim = config.memory_hidden_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if config.use_layer_norm else nn.Identity(),
            get_activation(config.activation),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if config.use_layer_norm else nn.Identity(),
            get_activation(config.activation),
        )

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """Args:
            memory: Tensor of shape (..., memory_feature_dim)
        Returns:
            Tensor of shape (..., memory_hidden_dim)
        """
        return self.net(memory)


class FeatureFusion(nn.Module):
    """Combines player vitals, spatial raycast, enemy attention, spatial memory, and optional visual representations.

    Produces a unified feature embedding for the recurrent memory.
    """

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config

        in_dim = config.vitals_hidden_dim + config.ray_hidden_dim + config.attention_embed_dim
        if config.memory_feature_dim > 0:
            in_dim += config.memory_hidden_dim
        if config.visual_feature_dim > 0:
            in_dim += config.visual_feature_dim

        out_dim = config.fusion_hidden_dim

        self.fusion_mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim) if config.use_layer_norm else nn.Identity(),
            get_activation(config.activation),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim) if config.use_layer_norm else nn.Identity(),
            get_activation(config.activation),
        )

    def forward(
        self,
        vitals_emb: torch.Tensor,
        ray_emb: torch.Tensor,
        enemy_emb: torch.Tensor,
        memory_emb: torch.Tensor | None = None,
        visual_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Args:
            vitals_emb: (..., vitals_hidden_dim)
            ray_emb: (..., ray_hidden_dim)
            enemy_emb: (..., attention_embed_dim)
            memory_emb: (..., memory_hidden_dim) optional
            visual_emb: (..., visual_feature_dim) optional
        Returns:
            (..., fusion_hidden_dim)
        """
        feats = [vitals_emb, ray_emb, enemy_emb]
        if self.config.memory_feature_dim > 0:
            if memory_emb is not None:
                feats.append(memory_emb)
            else:
                # Robust fallback if memory_emb is omitted: zero tensor matching batch shape
                leading_shape = vitals_emb.shape[:-1]
                zeros_mem = torch.zeros(
                    *leading_shape,
                    self.config.memory_hidden_dim,
                    dtype=vitals_emb.dtype,
                    device=vitals_emb.device,
                )
                feats.append(zeros_mem)

        if visual_emb is not None and self.config.visual_feature_dim > 0:
            feats.append(visual_emb)
        combined = torch.cat(feats, dim=-1)
        return self.fusion_mlp(combined)
