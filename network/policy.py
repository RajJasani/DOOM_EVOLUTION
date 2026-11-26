"""Actor-Critic heads for DOOM agent policy and value estimation.

Implements Bernoulli MultiBinary policy head (9 action buttons) and scalar Value head,
fully compatible with PPO reinforcement learning training pipelines and docs/CONTRACT.md §3.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Independent

from .config import NetworkConfig


class PolicyHead(nn.Module):
    """Actor (Policy) head producing MultiBinary(9) action distributions.

    Outputs logits for independent Bernoulli distributions over the 9 buttons:
      0: MOVE_FORWARD
      1: MOVE_BACKWARD
      2: MOVE_LEFT
      3: MOVE_RIGHT
      4: TURN_LEFT
      5: TURN_RIGHT
      6: ATTACK
      7: USE
      8: SELECT_NEXT_WEAPON
    """

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        self.action_dim = config.action_dim
        in_dim = config.rnn_hidden_dim

        self.actor_net = nn.Sequential(
            nn.Linear(in_dim, config.critic_hidden_dim),
            nn.LayerNorm(config.critic_hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Linear(config.critic_hidden_dim, self.action_dim),
        )

        if config.orthogonal_init:
            # Small gain for policy head output to encourage uniform initial exploration
            for m in self.actor_net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, rnn_features: torch.Tensor) -> torch.Tensor:
        """Compute action logits.

        Args:
            rnn_features: Tensor of shape (..., rnn_hidden_dim)
        Returns:
            logits: Tensor of shape (..., action_dim)
        """
        return self.actor_net(rnn_features)

    def get_distribution(self, logits: torch.Tensor) -> Independent:
        """Create an Independent Bernoulli distribution over binary buttons."""
        return Independent(Bernoulli(logits=logits), 1)

    def sample_action(
        self,
        logits: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action from logits or choose mode if deterministic.

        Args:
            logits: Tensor of shape (..., action_dim)
            deterministic: If True, threshold at logit > 0 (probability > 0.5)

        Returns:
            action: Tensor of shape (..., action_dim) with values in {0, 1}
            log_prob: Tensor of shape (...) containing total log probability
        """
        dist = self.get_distribution(logits)
        if deterministic:
            action = (logits > 0.0).float()
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_actions(
        self,
        logits: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probabilities and entropy of actions under current policy.

        Args:
            logits: Tensor of shape (..., action_dim)
            actions: Tensor of shape (..., action_dim) in {0, 1}

        Returns:
            log_prob: Tensor of shape (...)
            entropy: Tensor of shape (...)
        """
        dist = self.get_distribution(logits)
        log_prob = dist.log_prob(actions.float())
        entropy = dist.entropy()
        return log_prob, entropy


class ValueHead(nn.Module):
    """Critic (Value) head estimating expected discounted future reward V(s)."""

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        in_dim = config.rnn_hidden_dim
        hidden_dim = config.critic_hidden_dim

        self.critic_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        if config.orthogonal_init:
            for m in self.critic_net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, rnn_features: torch.Tensor) -> torch.Tensor:
        """Compute state value estimate.

        Args:
            rnn_features: Tensor of shape (..., rnn_hidden_dim)
        Returns:
            value: Tensor of shape (..., 1)
        """
        return self.critic_net(rnn_features)
