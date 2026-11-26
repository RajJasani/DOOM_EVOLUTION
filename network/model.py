"""Unified DoomNetwork combining feature encoders, enemy attention, recurrent memory, and actor-critic heads.

Provides a clean interface for inference, PPO rollouts, and action evaluation,
matching docs/CONTRACT.md and env/doom_env/spaces.py.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple
import numpy as np
import torch
import torch.nn as nn

from .config import NetworkConfig
from .encoder import FeatureFusion, RaycastEncoder, VitalsEncoder
from .enemy_attention import EnemyAttentionModule
from .memory import RecurrentMemory
from .policy import PolicyHead, ValueHead


class StepResult(NamedTuple):
    """Outputs from a single decision step rollout."""
    action: torch.Tensor
    value: torch.Tensor
    log_prob: torch.Tensor
    hx: torch.Tensor
    logits: torch.Tensor


class EvaluationResult(NamedTuple):
    """Outputs from action evaluation during PPO update."""
    values: torch.Tensor
    log_probs: torch.Tensor
    entropy: torch.Tensor
    hx: torch.Tensor
    logits: torch.Tensor


class DoomNetwork(nn.Module):
    """Deep Neural Network architecture for autonomous DOOM agents.

    Integrates:
      - VitalsEncoder: MLP for player health, armor, ammo, keys, velocities
      - RaycastEncoder: Circular 1D-CNN + MLP for 360-degree ego-centric depth/wall/door/hazard sensing
      - EnemyAttentionModule: Multi-head attention across visible threats with padding masks
      - FeatureFusion: Latent projection of spatial, vitals, and entity representations
      - RecurrentMemory: GRU tracking long-horizon state and spatial memory
      - PolicyHead: MultiBinary(9) action distribution (Bernoulli)
      - ValueHead: Scalar baseline reward estimator V(s)
    """

    def __init__(self, config: NetworkConfig | None = None):
        super().__init__()
        self.config = config or NetworkConfig()

        # 1. Component Encoders
        self.vitals_encoder = VitalsEncoder(self.config)
        self.ray_encoder = RaycastEncoder(self.config)
        self.enemy_attention = EnemyAttentionModule(self.config)
        self.fusion = FeatureFusion(self.config)

        # 2. Recurrent Memory
        self.memory = RecurrentMemory(self.config)

        # 3. Output Heads
        self.actor = PolicyHead(self.config)
        self.critic = ValueHead(self.config)

    @property
    def device(self) -> torch.device:
        """Return the device of model parameters."""
        return next(self.parameters()).device

    def init_hidden(self, batch_size: int = 1, device: torch.device | str | None = None) -> torch.Tensor:
        """Initialize zero hidden state for recurrent memory."""
        dev = device if device is not None else self.device
        return self.memory.init_hidden(batch_size=batch_size, device=dev)

    def _prepare_tensor(self, val: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Convert numpy array or tensor to appropriate PyTorch tensor on model device."""
        if isinstance(val, torch.Tensor):
            return val.to(device=self.device, dtype=dtype)
        elif isinstance(val, (np.ndarray, list, tuple)):
            return torch.as_tensor(val, device=self.device, dtype=dtype)
        else:
            raise TypeError(f"Cannot convert {type(val)} to tensor.")

    def _extract_inputs(self, obs: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Extract and prepare all required observation tensors."""
        # Vitals: (..., 17)
        vitals = self._prepare_tensor(obs["vitals"], dtype=torch.float32)

        # Rays: distance (..., 64), hit_type (..., 64)
        rays = obs["rays"]
        ray_dist = self._prepare_tensor(rays["distance"], dtype=torch.float32)
        ray_hit = self._prepare_tensor(rays["hit_type"], dtype=torch.long)

        # Visible entities: features (..., 32, 5), entity_type (..., 32), mask (..., 32)
        entities = obs["visible_entities"]
        ent_feats = self._prepare_tensor(entities["features"], dtype=torch.float32)
        ent_type = self._prepare_tensor(entities["entity_type"], dtype=torch.long)
        ent_mask = self._prepare_tensor(entities["mask"], dtype=torch.float32)

        # Optional visual embedding (Member 2 perception interface)
        visual = None
        if "visual_embedding" in obs and self.config.visual_feature_dim > 0:
            visual = self._prepare_tensor(obs["visual_embedding"], dtype=torch.float32)

        return vitals, ray_dist, ray_hit, ent_feats, ent_type, ent_mask, visual

    def encode_observation(self, obs: Mapping[str, Any]) -> torch.Tensor:
        """Run feature encoders and fusion to produce unified latent embeddings.

        Returns:
            Tensor of shape (..., fusion_hidden_dim)
        """
        vitals, ray_dist, ray_hit, ent_feats, ent_type, ent_mask, visual = self._extract_inputs(obs)

        vitals_emb = self.vitals_encoder(vitals)
        ray_emb = self.ray_encoder(ray_dist, ray_hit)
        enemy_emb = self.enemy_attention(
            features=ent_feats,
            entity_type=ent_type,
            mask=ent_mask,
            context_query=vitals_emb,
        )

        fused = self.fusion(
            vitals_emb=vitals_emb,
            ray_emb=ray_emb,
            enemy_emb=enemy_emb,
            visual_emb=visual,
        )
        return fused

    def forward(
        self,
        obs: Mapping[str, Any],
        hx: torch.Tensor | None = None,
        dones: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute action logits, scalar value estimates, and updated recurrent hidden states.

        Args:
            obs: Observation dict (matching CONTRACT.md). Can be unbatched or batched.
            hx: Hidden state tensor of shape (num_layers, batch_size, rnn_hidden_dim).
            dones: Optional termination flags tensor.

        Returns:
            logits: Action logits of shape (..., action_dim)
            values: Critic value estimate of shape (..., 1)
            next_hx: Updated hidden state of shape (num_layers, batch_size, rnn_hidden_dim)
        """
        fused = self.encode_observation(obs)
        rnn_out, next_hx = self.memory(fused, hx=hx, dones=dones)

        logits = self.actor(rnn_out)
        values = self.critic(rnn_out)
        return logits, values, next_hx

    @torch.no_grad()
    def step(
        self,
        obs: Mapping[str, Any],
        hx: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> StepResult:
        """Inference / Rollout step for agent execution.

        Args:
            obs: Observation dictionary.
            hx: Recurrent hidden state.
            deterministic: If True, selects greedy action without sampling.

        Returns:
            StepResult namedtuple: (action, value, log_prob, hx, logits)
        """
        # Ensure batch dimension exists if unbatched observation is passed
        is_unbatched = False
        vitals_val = obs["vitals"]
        if (isinstance(vitals_val, np.ndarray) and vitals_val.ndim == 1) or (
            isinstance(vitals_val, torch.Tensor) and vitals_val.ndim == 1
        ):
            is_unbatched = True

            def _expand(v):
                """Add a leading batch dimension of 1 to a numpy array or tensor."""
                if isinstance(v, np.ndarray):
                    return np.expand_dims(v, 0)
                return v.unsqueeze(0)

            # Wrap the three structured observation fields with batch dim 1.
            # All additional keys (e.g. "visual_embedding" from Member 2's perception
            # module, or "rgb"/"depth") are forwarded unchanged — they already carry
            # the correct shape because _extract_inputs() handles them directly.
            obs = {
                "vitals": _expand(obs["vitals"]),
                "rays": {
                    "distance": _expand(obs["rays"]["distance"]),
                    "hit_type": _expand(obs["rays"]["hit_type"]),
                },
                "visible_entities": {
                    "features": _expand(obs["visible_entities"]["features"]),
                    "entity_type": _expand(obs["visible_entities"]["entity_type"]),
                    "mask": _expand(obs["visible_entities"]["mask"]),
                },
                # Forward every extra key (visual_embedding, rgb, depth, …) unchanged.
                **{k: v for k, v in obs.items()
                   if k not in ("vitals", "rays", "visible_entities")},
            }

        logits, value, next_hx = self.forward(obs, hx=hx)
        action, log_prob = self.actor.sample_action(logits, deterministic=deterministic)

        if is_unbatched:
            action = action.squeeze(0)
            value = value.squeeze(0)
            log_prob = log_prob.squeeze(0)
            logits = logits.squeeze(0)

        return StepResult(
            action=action,
            value=value,
            log_prob=log_prob,
            hx=next_hx,
            logits=logits,
        )

    def evaluate_actions(
        self,
        obs: Mapping[str, Any],
        hx: torch.Tensor | None,
        actions: torch.Tensor,
        dones: torch.Tensor | None = None,
    ) -> EvaluationResult:
        """Evaluate actions for PPO updates across rollout sequences or batches.

        Args:
            obs: Observation dict with tensors of shape (batch_size, seq_len, ...) or (batch_size, ...).
            hx: Hidden state at the start of sequence (num_layers, batch_size, rnn_hidden_dim).
            actions: Action tensor of shape (batch_size, seq_len, action_dim) or (batch_size, action_dim).
            dones: Done mask tensor of shape (batch_size, seq_len).

        Returns:
            EvaluationResult namedtuple: (values, log_probs, entropy, next_hx, logits)
        """
        actions = self._prepare_tensor(actions, dtype=torch.float32)
        if dones is not None:
            dones = self._prepare_tensor(dones, dtype=torch.float32)

        logits, values, next_hx = self.forward(obs, hx=hx, dones=dones)
        log_probs, entropy = self.actor.evaluate_actions(logits, actions)

        return EvaluationResult(
            values=values,
            log_probs=log_probs,
            entropy=entropy,
            hx=next_hx,
            logits=logits,
        )

    def get_value(self, obs: Mapping[str, Any], hx: torch.Tensor | None = None) -> torch.Tensor:
        """Compute state value estimate V(s) only."""
        fused = self.encode_observation(obs)
        rnn_out, _ = self.memory(fused, hx=hx)
        return self.critic(rnn_out)

    def get_action(
        self,
        obs: Mapping[str, Any],
        hx: torch.Tensor | None = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Convenience method for evaluation agent execution (submission/agent.py).

        Returns:
            action: NumPy int8 array of shape (9,) matching action_space
            next_hx: Updated PyTorch hidden state tensor
        """
        step_res = self.step(obs, hx=hx, deterministic=deterministic)
        action_np = step_res.action.detach().cpu().numpy().astype(np.int8)
        return action_np, step_res.hx
