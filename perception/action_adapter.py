"""Action adapter for handling edge-triggered ViZDoom button signals.

Provides optional edge-trigger filtering for buttons that require state transitions
(such as USE button 7 and SELECT_NEXT_WEAPON button 8) in the ViZDoom engine.
"""

from __future__ import annotations

import numpy as np


class ActionEdgeTriggerAdapter:
    """Edge-trigger filter for ViZDoom action buttons.

    In ViZDoom, buttons such as USE (index 7) and SELECT_NEXT_WEAPON (index 8)
    are edge-triggered:
      - Holding USE continuously does not re-trigger doors or switches.
      - Holding SELECT_NEXT_WEAPON continuously locks the player in constant cycling animations.

    This adapter converts consecutive '1' signals on configured button indices into pulses
    (1 -> 0 -> 1) so that the engine receives an edge transition on successive actions.

    IMPORTANT FOR PPO TRAINING:
    If enabled during inference, this adapter MUST ALSO be used during training rollouts,
    and stored actions / log-probabilities must correspond to the effective actions executed
    in the environment to prevent off-policy policy gradient distortion.
    """

    def __init__(self, pulse_indices: tuple[int, ...] = (7, 8), enabled: bool = False) -> None:
        self.pulse_indices = set(pulse_indices)
        self.enabled = enabled
        self._last_raw_action: np.ndarray | None = None

    def reset(self) -> None:
        """Reset action history at episode boundary."""
        self._last_raw_action = None

    def process(self, action: np.ndarray) -> np.ndarray:
        """Process action array and apply edge-triggering if enabled.

        Args:
            action: MultiBinary(9) action array with values in {0, 1}.
        Returns:
            MultiBinary(9) action array with edge transitions enforced.
        """
        action_out = np.asarray(action, dtype=np.int8).copy()
        if not self.enabled or self._last_raw_action is None:
            self._last_raw_action = action_out.copy()
            return action_out

        for idx in self.pulse_indices:
            # If button was pressed in previous step and is still pressed, suppress to 0
            # to force a release tic before the next press
            if self._last_raw_action[idx] == 1 and action_out[idx] == 1:
                action_out[idx] = 0

        self._last_raw_action = np.asarray(action, dtype=np.int8).copy()
        return action_out
