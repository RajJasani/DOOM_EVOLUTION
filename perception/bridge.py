"""PerceptionBridge — integrates sensory processing and spatial memory into DoomAgent.

This module is Member 2's integration point: it owns the full pipeline from raw
observation dict → processed features → spatial memory update → enriched obs dict
ready for DoomNetwork.

Architecture position
---------------------

    env.step() → raw obs dict
                     │
            ┌────────▼────────┐
            │ PerceptionBridge │   (this file — Member 2 owns)
            │                 │
            │  SensoryPipeline│   processes rays / entities / vitals / depth
            │  EgocentricSpatialMemory  accumulates cross-step spatial state
            │                 │
            └────────┬────────┘
                     │  enriched obs dict:
                     │    obs["rays"]              (unchanged keys, normalised values)
                     │    obs["vitals"]            (unchanged)
                     │    obs["visible_entities"]  (unchanged, mask as float32)
                     │    obs["memory_features"]   (64,) float32  ← NEW
                     ▼
            DoomNetwork.get_action(obs, hx)
            (Member 3 owns — do not modify)

Design constraints
------------------
- Never modifies env/doom_env/*, validate.py, docs/CONTRACT.md.
- Never holds a reference to DoomNetwork (no circular dep).
- Pure NumPy; zero PyTorch in this module.
- Must complete in << 5 ms per step (target < 1 ms).
- Thread-safe if called from a single thread (standard RL use-case).
- reset() MUST be called at the start of every episode.

Usage in DoomAgent
------------------
    from perception.bridge import PerceptionBridge

    class DoomAgent:
        def __init__(self, observation_space, action_space, device):
            ...
            self.bridge = PerceptionBridge()

        def reset(self):
            self.bridge.reset()      # clears spatial memory, pipeline caches
            self.hx = None           # clear GRU hidden state

        def act(self, observation):
            enriched = self.bridge.process(observation)
            action_np, self.hx = self.model.get_action(enriched, hx=self.hx)
            return action_np
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .processors.pipeline import SensoryPipeline, ProcessedObservation
from .spatial_memory import EgocentricSpatialMemory, MEMORY_FEATURE_DIM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

MEMORY_KEY = "memory_features"
"""Key used to inject the spatial memory vector into the enriched obs dict."""


# ---------------------------------------------------------------------------
# BridgeStats — lightweight telemetry for debugging
# ---------------------------------------------------------------------------

@dataclass
class BridgeStats:
    """Per-step diagnostic snapshot from PerceptionBridge.process().

    Useful for debugging and logging during development. Not used at inference
    time — call bridge.last_stats to read after each step.
    """
    step:              int    # episode-local step index
    n_active_enemies:  int    # enemies in spatial memory
    n_active_doors:    int    # doors in spatial memory
    n_locked_doors:    int    # locked doors in spatial memory
    n_keys_held:       int    # keycards currently possessed
    n_visible_entities: int   # entities visible this step (from ProcessedObservation)
    pipeline_valid:    bool   # whether the pipeline reported a valid observation
    memory_feature_dim: int   # sanity: should always equal MEMORY_FEATURE_DIM


# ---------------------------------------------------------------------------
# PerceptionBridge
# ---------------------------------------------------------------------------

class PerceptionBridge:
    """Integrates SensoryPipeline and EgocentricSpatialMemory for DoomAgent.

    Stateful across steps within an episode. Call reset() at the start of
    every episode (in DoomAgent.reset()).

    Parameters
    ----------
    pipeline : SensoryPipeline | None
        Pre-constructed pipeline. If None, a default one is created.
        Inject a custom pipeline for testing or non-default thresholds.
    memory : EgocentricSpatialMemory | None
        Pre-constructed memory. If None, default is created.
    warn_on_invalid : bool
        If True, log a warning when the pipeline reports observation issues.
        Default True.

    Attributes
    ----------
    last_stats : BridgeStats | None
        Diagnostic snapshot from the most recent process() call. None before
        the first call.
    last_processed : ProcessedObservation | None
        Full ProcessedObservation from the most recent pipeline run. None
        before the first call.
    """

    def __init__(
        self,
        pipeline:        Optional[SensoryPipeline]         = None,
        memory:          Optional[EgocentricSpatialMemory] = None,
        warn_on_invalid: bool                              = True,
    ) -> None:
        self._pipeline:        SensoryPipeline         = pipeline or SensoryPipeline()
        self._memory:          EgocentricSpatialMemory = memory   or EgocentricSpatialMemory()
        self._warn_on_invalid: bool                    = warn_on_invalid
        self._episode_step:    int                     = 0

        self.last_stats:     Optional[BridgeStats]        = None
        self.last_processed: Optional[ProcessedObservation] = None

    # ------------------------------------------------------------------
    # Public API — called from DoomAgent
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all episode-scoped state.

        MUST be called at the start of every new episode in DoomAgent.reset().
        Clears:
          - EgocentricSpatialMemory (all object slots, keys, direction tracker)
          - Episode step counter
          - Diagnostic snapshots

        The SensoryPipeline is stateless and needs no reset.
        """
        self._memory.reset()
        self._episode_step    = 0
        self.last_stats       = None
        self.last_processed   = None
        logger.debug("PerceptionBridge reset for new episode.")

    def process(self, observation: dict) -> dict:
        """Process one raw observation dict and return an enriched version.

        Steps
        -----
        1. Run SensoryPipeline → ProcessedObservation.
        2. Update EgocentricSpatialMemory from ProcessedObservation.
        3. Build the network_input dict from the pipeline (normalised keys).
        4. Inject ``obs["memory_features"]`` = (64,) float32.
        5. Update diagnostic stats.

        Parameters
        ----------
        observation : dict
            Raw obs dict from the environment (or the official wrapper).
            Must contain keys: "vitals", "rays", "visible_entities".
            Optionally: "rgb", "depth".

        Returns
        -------
        dict
            Enriched obs dict. Contains all keys expected by DoomNetwork plus
            the ``"memory_features"`` key (shape (64,) float32).
            Safe to pass directly to ``model.get_action(enriched, hx=self.hx)``.

        Notes
        -----
        - Never mutates the input ``observation`` dict.
        - Raises ValueError if the observation is structurally invalid (missing
          mandatory keys). Soft issues (wrong dtype, out-of-range) are logged.
        """
        # --- 1. Validate structure (fail fast on missing keys) -------
        self._check_mandatory_keys(observation)

        # --- 2. Run SensoryPipeline ----------------------------------
        processed: ProcessedObservation = self._pipeline.process(observation)

        if self._warn_on_invalid and not processed.is_valid:
            logger.warning(
                "PerceptionBridge step=%d: pipeline validation issues: %s",
                self._episode_step, processed.issues,
            )

        self.last_processed = processed

        # --- 3. Update spatial memory --------------------------------
        self._memory.update(processed)

        # --- 4. Build enriched obs dict ------------------------------
        # Start from the pipeline's network_input (already normalised, correct dtypes)
        enriched: dict = dict(processed.network_input)

        # Inject memory features — always (64,) float32
        mem_vec: np.ndarray = self._memory.to_feature_vector()
        enriched[MEMORY_KEY] = mem_vec   # (MEMORY_FEATURE_DIM,) float32

        # --- 5. Update diagnostics -----------------------------------
        snap = self._memory.snapshot()
        self.last_stats = BridgeStats(
            step=self._episode_step,
            n_active_enemies=snap.n_active_enemies,
            n_active_doors=snap.n_active_doors,
            n_locked_doors=snap.n_locked_doors,
            n_keys_held=snap.n_keys_held,
            n_visible_entities=int(processed.entities.n_visible),
            pipeline_valid=processed.is_valid,
            memory_feature_dim=mem_vec.shape[0],
        )

        self._episode_step += 1
        return enriched

    # ------------------------------------------------------------------
    # Properties — read-only access to sub-components for inspection
    # ------------------------------------------------------------------

    @property
    def memory(self) -> EgocentricSpatialMemory:
        """Direct access to the spatial memory (read-only intent)."""
        return self._memory

    @property
    def episode_step(self) -> int:
        """Episode-local step counter (increments each process() call)."""
        return self._episode_step

    @property
    def memory_feature_dim(self) -> int:
        """Dimension of the memory_features vector (always MEMORY_FEATURE_DIM)."""
        return MEMORY_FEATURE_DIM

    def __repr__(self) -> str:
        mem_repr = repr(self._memory)
        return f"PerceptionBridge(step={self._episode_step}, memory={mem_repr})"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_mandatory_keys(obs: dict) -> None:
        """Raise ValueError if mandatory top-level keys are absent."""
        mandatory = ("vitals", "rays", "visible_entities")
        missing = [k for k in mandatory if k not in obs]
        if missing:
            raise ValueError(
                f"PerceptionBridge.process(): observation missing mandatory keys: {missing}. "
                f"Present keys: {list(obs.keys())}"
            )
        ray_keys = ("distance", "hit_type")
        missing_ray = [k for k in ray_keys if k not in obs.get("rays", {})]
        if missing_ray:
            raise ValueError(
                f"PerceptionBridge.process(): obs['rays'] missing keys: {missing_ray}"
            )
        entity_keys = ("features", "entity_type", "mask")
        missing_ent = [k for k in entity_keys if k not in obs.get("visible_entities", {})]
        if missing_ent:
            raise ValueError(
                f"PerceptionBridge.process(): obs['visible_entities'] missing keys: {missing_ent}"
            )
