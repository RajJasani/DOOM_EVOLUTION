"""Canonical observation adapters for single and vectorized environment rollouts.

Ensures complete parity between PPO training and evaluation inference:
  Raw ViZDoom Observation -> ObservationAdapter -> DoomNetwork Observation
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence
import numpy as np
import torch

from .bridge import PerceptionBridge, MEMORY_KEY


class ObservationAdapter:
    """Canonical observation adapter for single-environment inference and rollouts.

    Wraps PerceptionBridge to provide a clean, standardized interface for converting
    raw ViZDoom observations into normalized, spatial-memory-enriched dictionaries
    ready for DoomNetwork.
    """

    def __init__(self, enable_depth: bool = False, warn_on_invalid: bool = False) -> None:
        self.bridge = PerceptionBridge(enable_depth=enable_depth, warn_on_invalid=warn_on_invalid)

    def reset(self) -> None:
        """Reset episode-scoped state and spatial memory."""
        self.bridge.reset()

    def process(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        """Convert a raw environment observation into a DoomNetwork-compatible dictionary."""
        return self.bridge.process(obs)

    def __call__(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        return self.process(obs)


class VectorObservationAdapter:
    """Vectorized observation adapter for N parallel environments (Member 4 PPO).

    Maintains independent spatial memory and sensory pipeline state for each
    environment in the vector, preventing state leakage across parallel rollout workers.
    """

    def __init__(
        self,
        num_envs: int,
        enable_depth: bool = False,
        warn_on_invalid: bool = False,
    ) -> None:
        self.num_envs = num_envs
        self.adapters = [
            ObservationAdapter(enable_depth=enable_depth, warn_on_invalid=warn_on_invalid)
            for _ in range(num_envs)
        ]

    def reset(self, env_idx: int | None = None) -> None:
        """Reset all environments or a specific environment at an episode boundary."""
        if env_idx is None:
            for a in self.adapters:
                a.reset()
        else:
            self.adapters[env_idx].reset()

    def reset_at(self, env_idx: int) -> None:
        """Reset spatial memory for a specific environment index."""
        self.adapters[env_idx].reset()

    def process_env(self, env_idx: int, obs: Mapping[str, Any]) -> dict[str, Any]:
        """Process observation for a single environment in the vector."""
        return self.adapters[env_idx].process(obs)

    def process_all(self, obs_list: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Process raw observations for all N parallel environments."""
        if len(obs_list) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} observations, got {len(obs_list)}")
        return [self.adapters[i].process(obs_list[i]) for i in range(self.num_envs)]

    def collate_for_network(
        self,
        processed_list: Sequence[Mapping[str, Any]],
        device: torch.device | str = "cpu",
    ) -> dict[str, Any]:
        """Collate N processed observation dicts into a single batched PyTorch dict.

        Batches vitals, rays, entities, and spatial memory features into tensors with
        leading dimension B = num_envs, ready for direct consumption by DoomNetwork.step().
        """
        batch_size = len(processed_list)
        if batch_size == 0:
            raise ValueError("Cannot collate empty observation list")

        # Stack vitals: (B, 17)
        vitals = torch.as_tensor(
            np.stack([p["vitals"] for p in processed_list]),
            dtype=torch.float32,
            device=device,
        )

        # Stack rays: distance (B, 64), hit_type (B, 64)
        ray_dist = torch.as_tensor(
            np.stack([p["rays"]["distance"] for p in processed_list]),
            dtype=torch.float32,
            device=device,
        )
        ray_hit = torch.as_tensor(
            np.stack([p["rays"]["hit_type"] for p in processed_list]),
            dtype=torch.long,
            device=device,
        )

        # Stack visible entities: features (B, 32, 5), entity_type (B, 32), mask (B, 32)
        ent_feats = torch.as_tensor(
            np.stack([p["visible_entities"]["features"] for p in processed_list]),
            dtype=torch.float32,
            device=device,
        )
        ent_type = torch.as_tensor(
            np.stack([p["visible_entities"]["entity_type"] for p in processed_list]),
            dtype=torch.long,
            device=device,
        )
        ent_mask = torch.as_tensor(
            np.stack([p["visible_entities"]["mask"] for p in processed_list]),
            dtype=torch.float32,
            device=device,
        )

        # Stack spatial memory features: (B, 64)
        memory = torch.as_tensor(
            np.stack([p[MEMORY_KEY] for p in processed_list]),
            dtype=torch.float32,
            device=device,
        )

        return {
            "vitals": vitals,
            "rays": {
                "distance": ray_dist,
                "hit_type": ray_hit,
            },
            "visible_entities": {
                "features": ent_feats,
                "entity_type": ent_type,
                "mask": ent_mask,
            },
            "memory_features": memory,
        }
