"""Evaluation entry point. The scorer instantiates DoomAgent from this file.

Contract: docs/CONTRACT.md §6.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import numpy as np

SUBMISSION_DIR = Path(__file__).resolve().parent
REPO_ROOT = SUBMISSION_DIR.parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"

# Ensure repo root is available in module path for network and perception
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("DoomAgent")


class DoomAgent:
    """Autonomous DOOM agent integrating perception, recurrent memory, and neural policy."""

    def __init__(self, observation_space, action_space, device: str):
        """Called once per evaluation run.

        Loads weights from WEIGHTS_DIR / 'model.pt' onto `device` ("cuda" or "cpu").
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.device_str = device
        self.rng = np.random.default_rng(0)
        self.model = None
        self.hx = None

        try:
            import torch
            from network import DoomNetwork, NetworkConfig

            self.torch = torch
            target_device = torch.device(
                device if torch.cuda.is_available() and device == "cuda" else "cpu"
            )
            self.device = target_device
            self.config = NetworkConfig()
            self.model = DoomNetwork(self.config).to(self.device)
            self.model.eval()

            weights_path = WEIGHTS_DIR / "model.pt"
            if weights_path.is_file():
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info("Loaded model weights from %s", weights_path)
            else:
                logger.info("No checkpoint found at %s; using initialized model weights.", weights_path)

        except Exception as e:
            logger.warning("Neural network initialization deferred: %s", e)
            self.model = None

    def reset(self) -> None:
        """Called before every map. Clear recurrent state, memory buffers and trackers."""
        self.hx = None

    def act(self, observation: dict) -> np.ndarray:
        """Return a MultiBinary action: int array of shape action_space.shape with values in {0, 1}."""
        if self.model is not None:
            try:
                with self.torch.no_grad():
                    action_np, self.hx = self.model.get_action(
                        obs=observation,
                        hx=self.hx,
                        deterministic=True,
                    )
                return action_np
            except Exception as e:
                logger.error("Error during model inference: %s", e)

        # Compliant fallback action
        return self.rng.integers(0, 2, size=self.action_space.shape, dtype=np.int8)
