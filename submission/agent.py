"""Evaluation entry point. The scorer instantiates DoomAgent from this file.

Contract: docs/CONTRACT.md §6.

Integration notes (Member 2 — Spatial Intelligence & Sensory Processing)
-------------------------------------------------------------------------
This file wires Member 2's PerceptionBridge into the agent loop:

    raw obs  →  PerceptionBridge.process()  →  enriched obs  →  DoomNetwork

PerceptionBridge runs:
  1. SensoryPipeline   — normalises rays, entities, vitals, depth.
  2. EgocentricSpatialMemory — updates cross-step spatial state.
  3. Injects obs["memory_features"] (64-dim float32) for the network.

Four-game-tick action schedule
-------------------------------
The environment applies each action for FRAME_SKIP=4 game ticks
(env/doom_env/spaces.py). DoomAgent.act() is called once per 4-tick block.
The 40 ms per-step budget (env/doom_env/limits.py) covers this whole block.

What to load / initialise
--------------------------
Replace the placeholder DoomNetwork instantiation below with the actual
trained model loaded from WEIGHTS_DIR once the network is trained.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

# Member 2: perception integration
from perception.bridge import PerceptionBridge, MEMORY_KEY

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR    = SUBMISSION_DIR / "weights"

logger = logging.getLogger(__name__)


class DoomAgent:
    """Autonomous DOOM-playing agent.

    Lifecycle
    ---------
    __init__  : Called ONCE per evaluation run. Load weights here.
    reset     : Called before EVERY map. Clear recurrent state and memory.
    act       : Called once per 4-game-tick block. Return 9-bit action array.

    Parameters
    ----------
    observation_space : gymnasium Space
        From the official environment wrapper.
    action_space : gymnasium Space
        MultiBinary(9) in the official wrapper.
    device : str
        "cuda" or "cpu" — from the scorer.
    """

    def __init__(self, observation_space, action_space, device: str):
        self.observation_space = observation_space
        self.action_space      = action_space
        self.device            = device

        # --- Member 2: perception pipeline + spatial memory ---------
        # PerceptionBridge is stateless across episodes except for spatial
        # memory, which is reset in reset().  Create once, reuse forever.
        self.bridge = PerceptionBridge(warn_on_invalid=True)

        # --- Network (Member 3) -------------------------------------
        # Attempt to load the trained network.  If weights are absent
        # (e.g. during development or unit testing), fall back to a
        # random baseline so the agent remains runnable.
        self.model  = None
        self.hx: Optional[object] = None   # GRU hidden state (torch.Tensor)
        self._load_model(device)

        # --- Fallback RNG (used when model is None) -----------------
        self._rng = np.random.default_rng(0)

        # --- Diagnostic counters (Section C audit) ------------------
        self._n_perception_fallbacks: int = 0
        self._n_network_fallbacks: int = 0
        self._last_error: Optional[str] = None
        self._total_steps: int = 0

        logger.info(
            "DoomAgent initialised (device=%s, model=%s).",
            device,
            "loaded" if self.model is not None else "RANDOM FALLBACK",
        )

    # ------------------------------------------------------------------
    # reset() — called before EVERY map
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all episode-scoped state.

        Called by the scorer before every map.  Must be fast (< 10 s budget).

        Resets:
          - PerceptionBridge (clears spatial memory, step counter)
          - GRU hidden state (hx → None, re-initialised on first act())
        """
        # Member 2: clear spatial memory and pipeline diagnostics
        self.bridge.reset()

        # Member 3: clear recurrent hidden state
        self.hx = None

        logger.debug("DoomAgent.reset() completed.")

    # ------------------------------------------------------------------
    # act() — called once per 4-game-tick block
    # ------------------------------------------------------------------

    def act(self, observation: dict) -> np.ndarray:
        """Process observation and return a 9-bit MultiBinary action.

        Pipeline
        --------
        1. PerceptionBridge.process(observation)
             → Runs SensoryPipeline + updates spatial memory.
             → Returns enriched obs dict with obs["memory_features"].
        2. DoomNetwork.get_action(enriched, hx)
             → Runs encoders + GRU + policy head.
             → Returns (action_np, next_hx).
        3. Update self.hx for the next step.
        4. Return action_np  (shape (9,), dtype int8).

        Parameters
        ----------
        observation : dict
            Raw obs dict from the official environment wrapper.

        Returns
        -------
        np.ndarray
            Shape (9,), dtype int8, values in {0, 1}.
        """
        self._total_steps += 1

        # --- Step 1: perception + spatial memory --------------------
        try:
            enriched = self.bridge.process(observation)
        except (ValueError, KeyError, AttributeError, TypeError) as exc:
            # Defensive: if perception fails, log and fall back to random.
            self._n_perception_fallbacks += 1
            self._last_error = f"perception: {exc}"
            logger.error(
                "PerceptionBridge.process() failed (fallback #%d): %s",
                self._n_perception_fallbacks, exc,
            )
            return self._random_action()

        # --- Step 2: network inference ------------------------------
        if self.model is not None:
            try:
                action_np, self.hx = self.model.get_action(
                    enriched,
                    hx=self.hx,
                    deterministic=True,
                )
                return action_np.astype(np.int8)
            except Exception as exc:
                self._n_network_fallbacks += 1
                self._last_error = f"network: {exc}"
                logger.error(
                    "DoomNetwork.get_action() failed (fallback #%d): %s",
                    self._n_network_fallbacks, exc,
                )
                return self._random_action()

        # --- Fallback: random action --------------------------------
        return self._random_action()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self, device: str) -> None:
        """Attempt to load the trained DoomNetwork from WEIGHTS_DIR.

        Does not raise on failure — sets self.model = None so the agent
        remains runnable during development without a trained checkpoint.
        """
        weights_path = WEIGHTS_DIR / "doom_network.pt"
        if not weights_path.exists():
            logger.warning(
                "No weights found at %s. Agent will use random actions until trained.",
                weights_path,
            )
            return

        try:
            import torch
            from network.model import DoomNetwork
            from network.config import NetworkConfig

            config = NetworkConfig()
            self.model = DoomNetwork(config=config)
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.to(device)
            self.model.eval()
            logger.info("DoomNetwork loaded from %s onto %s.", weights_path, device)

        except Exception as exc:
            logger.error(
                "Failed to load DoomNetwork: %s. Falling back to random actions.", exc
            )
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
