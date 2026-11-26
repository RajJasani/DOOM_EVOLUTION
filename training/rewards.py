"""Multi-tier shaped reward functions for reinforcement learning.

Supports:
  1. Combat & Survival (Tier 1)
  2. Multi-Room Exploration & Doors (Tier 2)
  3. Keycard Progression (Tier 3)
  4. Exit Completion (All tiers)
"""

from __future__ import annotations

from typing import Any, Mapping
import numpy as np


class DoomRewardShaper:
    """Computes shaped rewards during training rollouts."""

    def __init__(
        self,
        exit_bonus: float = 10.0,
        death_penalty: float = -2.0,
        kill_bonus: float = 1.0,
        door_open_bonus: float = 1.5,
        keycard_bonus: float = 3.0,
        exploration_step_bonus: float = 0.02,
        damage_taken_penalty: float = -0.05,
    ) -> None:
        self.exit_bonus = exit_bonus
        self.death_penalty = death_penalty
        self.kill_bonus = kill_bonus
        self.door_open_bonus = door_open_bonus
        self.keycard_bonus = keycard_bonus
        self.exploration_step_bonus = exploration_step_bonus
        self.damage_taken_penalty = damage_taken_penalty

        self.last_keys = 0
        self.last_health = 100.0

    def reset(self, initial_obs: Mapping[str, Any]) -> None:
        """Reset internal trackers at episode start."""
        vitals = initial_obs["vitals"]
        self.last_health = float(vitals[0])
        self.last_keys = int(np.sum(vitals[8:14]))

    def compute_reward(
        self,
        obs: Mapping[str, Any],
        raw_env_reward: float,
        terminated: bool,
        truncated: bool,
    ) -> float:
        """Calculate shaped reward combining milestones, survival, and exploration."""
        vitals = obs["vitals"]
        curr_health = float(vitals[0])
        curr_keys = int(np.sum(vitals[8:14]))

        shaped_reward = raw_env_reward

        # Milestone: exit reached vs death
        if terminated:
            if raw_env_reward > 0:
                shaped_reward += self.exit_bonus
            elif curr_health <= 0:
                shaped_reward += self.death_penalty

        # Keycard acquisition
        if curr_keys > self.last_keys:
            shaped_reward += self.keycard_bonus * (curr_keys - self.last_keys)
            self.last_keys = curr_keys

        # Damage taken
        if curr_health < self.last_health:
            damage = self.last_health - curr_health
            shaped_reward += self.damage_taken_penalty * damage
        self.last_health = curr_health

        # Small velocity/exploration incentive
        forward_vel = float(vitals[14])
        if abs(forward_vel) > 1.0:
            shaped_reward += self.exploration_step_bonus

        return shaped_reward
