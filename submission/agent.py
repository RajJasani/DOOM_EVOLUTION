"""Evaluation entry point. The scorer instantiates DoomAgent from this file.

Contract: docs/CONTRACT.md §6.

Integrated Architecture (Members 1, 2, and 3):
----------------------------------------------
Member 1: System architecture, DoomAgent evaluation interface, weight loading, error handling.
Member 2: Sensory pipeline, raycast/vitals/entity preprocessing, spatial memory (PerceptionBridge).
Member 3: PyTorch neural network (DoomNetwork) with circular 1D-CNN, enemy attention, GRU memory, actor-critic heads.

Pipeline:
  Raw Obs -> PerceptionBridge.process() -> Enriched Obs -> DoomNetwork.get_action() -> 9-bit MultiBinary Action
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Any, Optional

import numpy as np

# Member 2: perception integration
from perception.bridge import PerceptionBridge, MEMORY_KEY
from perception.adapter import ObservationAdapter
from perception.action_adapter import ActionEdgeTriggerAdapter

SUBMISSION_DIR = Path(__file__).resolve().parent
REPO_ROOT = SUBMISSION_DIR.parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"

# Ensure repo root is available in module path for network and perception
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("DoomAgent")


class DoomAgent:
    """Autonomous DOOM agent integrating perception, recurrent memory, and neural policy.

    Lifecycle:
      __init__: Called ONCE per evaluation run. Loads weights if present.
      reset: Called before EVERY map. Clears recurrent hidden state and spatial memory.
      act: Called once per decision step (every 4 game ticks). Returns a MultiBinary(9) action.
    """

    def __init__(self, observation_space, action_space, device: str = "cpu"):
        """Called once per evaluation run."""
        self.observation_space = observation_space
        self.action_space = action_space
        self.device_str = device
        self.device = device
        self.rng = np.random.default_rng(0)
        self._rng = self.rng
        self.hx = None
        self.model = None

        # Member 2: Sensory pipeline & spatial memory (ObservationAdapter canonical pipeline)
        self.adapter = ObservationAdapter(enable_depth=False)
        self.bridge = self.adapter.bridge
        self.action_adapter = ActionEdgeTriggerAdapter(enabled=False)

        # Telemetry & fallback tracking
        self._total_steps = 0
        self._n_perception_fallbacks = 0
        self._n_network_fallbacks = 0
        self._last_error: Optional[str] = None

        # Member 1 & 3: Model and weights loading
        self._load_model(device)

    def reset(self) -> None:
        """Called before every map. Clear recurrent state, memory buffers and trackers."""
        self.hx = None
        if self.bridge is not None:
            self.bridge.reset()
        if self.action_adapter is not None:
            self.action_adapter.reset()
        self._n_perception_fallbacks = 0
        self._n_network_fallbacks = 0
        self._last_error = None
        logger.debug("DoomAgent.reset() completed.")

    def act(self, observation: dict) -> np.ndarray:
        """Process observation and return a 9-bit MultiBinary action: int8 array of shape (9,)."""
        self._total_steps += 1

        # --- Step 1: Member 2 Perception + Spatial Memory ---
        if self.bridge is not None:
            try:
                enriched = self.bridge.process(observation)
            except (ValueError, KeyError, AttributeError, TypeError) as exc:
                self._n_perception_fallbacks += 1
                self._last_error = f"perception: {exc}"
                logger.error(
                    "PerceptionBridge.process() failed (fallback #%d): %s",
                    self._n_perception_fallbacks,
                    exc,
                )
                return self._random_action()
        else:
            enriched = observation

        # --- Step 2: Member 3 Neural Network Inference ---
        if self.model is not None:
            try:
                import torch
                with torch.no_grad():
                    action_np, self.hx = self.model.get_action(
                        enriched,
                        hx=self.hx,
                        deterministic=True,
                    )
                if self.action_adapter is not None:
                    action_np = self.action_adapter.process(action_np)
                return action_np.astype(np.int8)
            except Exception as exc:
                self._n_network_fallbacks += 1
                self._last_error = f"network: {exc}"
                logger.error(
                    "DoomNetwork.get_action() failed (fallback #%d): %s",
                    self._n_network_fallbacks,
                    exc,
                )
                return self._random_action()

        # --- Fallback: random compliant action ---
        return self._random_action()

    def _load_model(self, device: str) -> None:
        """Attempt to load trained weights for DoomNetwork.

        If no weights checkpoint is found, self.model remains None and the agent
        falls back to random compliant actions (as verified by integration tests).
        """
        candidate_paths = [
            WEIGHTS_DIR / "model.pt",
            WEIGHTS_DIR / "emergency_beginner" / "starting_model.pt",
            WEIGHTS_DIR / "emergency_beginner" / "emergency_beginner_best_navigation.pt",
            WEIGHTS_DIR / "curriculum_phase5_2" / "best_curriculum_model.pt",
            WEIGHTS_DIR / "best_model.pt",
            WEIGHTS_DIR / "doom_network.pt",
        ]
        weights_path = None
        for p in candidate_paths:
            if p.is_file():
                weights_path = p
                break

        if weights_path is None:
            logger.info("No weights found in %s. Agent running in baseline mode.", WEIGHTS_DIR)
            self.model = None
            return

        try:
            import torch
            from network.model import DoomNetwork
            from network.config import NetworkConfig

            target_device = torch.device(
                device if torch.cuda.is_available() and device == "cuda" else "cpu"
            )
            self.device = target_device
            try:
                raw_checkpoint = torch.load(weights_path, map_location=self.device, weights_only=True)
            except TypeError:
                raw_checkpoint = torch.load(weights_path, map_location=self.device)

            if isinstance(raw_checkpoint, dict) and "model_state_dict" in raw_checkpoint:
                state_dict = raw_checkpoint["model_state_dict"]
            else:
                state_dict = raw_checkpoint

            # Detect whether checkpoint was trained with memory encoder (64-dim) or raw 1.19M baseline (0-dim)
            has_memory = (
                "memory_encoder.net.0.weight" in state_dict
                or (
                    "fusion.fusion_mlp.0.weight" in state_dict
                    and state_dict["fusion.fusion_mlp.0.weight"].shape[1] == 320
                )
            )
            config = NetworkConfig(
                memory_feature_dim=64 if has_memory else 0,
                normalize_vitals=False,  # PerceptionBridge normalizes vitals to [0, 1]
            )
            self.model = DoomNetwork(config=config)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            logger.info("DoomNetwork loaded from %s onto %s (memory_dim=%d).", weights_path, self.device, config.memory_feature_dim)
        except Exception as exc:
            logger.error("Failed to load DoomNetwork: %s. Falling back to random actions.", exc)
            self.model = None

    @property
    def n_perception_fallbacks(self) -> int:
        """Number of times perception failed and agent fell back to random."""
        return self._n_perception_fallbacks

    @property
    def n_network_fallbacks(self) -> int:
        """Number of times network inference failed and agent fell back to random."""
        return self._n_network_fallbacks

    @property
    def last_error(self) -> Optional[str]:
        """Last error message from perception or network failure. None if no failure."""
        return self._last_error

    @property
    def total_steps(self) -> int:
        """Total act() calls across all episodes."""
        return self._total_steps

    def _random_action(self) -> np.ndarray:
        """Return a random MultiBinary(9) action."""
        return self._rng.integers(0, 2, size=self.action_space.shape, dtype=np.int8)
