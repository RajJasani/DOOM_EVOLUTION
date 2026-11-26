"""SensoryPipeline — unified entry point for all sensory processing.

Runs all four processors (raycast, depth, entity, vitals) on a single
raw obs dict and returns a ``ProcessedObservation`` dataclass that exposes:

  1. All per-domain features (typed, normalised, tensor-ready).
  2. ``network_input`` — a plain dict that is a drop-in replacement for the
     raw obs dict in ``submission/agent.py``, with all Finding-3 fixes applied.
  3. ``is_valid`` and ``issues`` — validation results for diagnostics.

Integration example (submission/agent.py)
-----------------------------------------
    from perception.processors import SensoryPipeline

    class DoomAgent:
        def __init__(self, ...):
            self.pipeline = SensoryPipeline()      # stateless — create once

        def act(self, obs: dict, ...) -> np.ndarray:
            processed = self.pipeline.process(obs)
            # Drop-in: network_input has the same keys as the raw obs dict
            action_np, self.hx = self.model.get_action(
                processed.network_input, hx=self.hx
            )
            return action_np

Design rules
------------
- Pure NumPy; zero torch dependency.  Tensor conversion is handled by
  ``perception.preprocessing.obs_to_device_tensors()`` if needed.
- Never modifies the caller's obs dict.
- Validation issues are reported (not raised) so a noisy step never crashes.
- Must complete in well under 5 ms on a single CPU thread (target: < 1 ms).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .raycast_processor import RaycastProcessor, RayFeatures
from .depth_processor   import DepthProcessor,   DepthFeatures
from .entity_processor  import EntityProcessor,  EntityFeatures
from .vitals_processor  import VitalsProcessor,  VitalsFeatures

# Reuse the existing validate_observation() from the perception package
# (falls back gracefully if the import path is unusual)
try:
    import sys
    from pathlib import Path
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from perception.preprocessing import validate_observation
    _VALIDATOR_AVAILABLE = True
except ImportError:
    _VALIDATOR_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProcessedObservation:
    """Complete output of the SensoryPipeline for a single env step.

    Fields
    ------
    rays : RayFeatures
        Processed 64-ray ego-centric raycasts.

    depth : DepthFeatures
        Processed depth buffer.

    entities : EntityFeatures
        Processed visible entity table.

    vitals : VitalsFeatures
        Processed health / armor / ammo / velocity vector.

    is_valid : bool
        True if validate_observation() found no issues with the raw obs.
        A False value means the obs has schema violations; network_input
        is still populated with best-effort values (never raises).

    issues : list[str]
        Any validation warnings. Empty list when is_valid is True.

    network_input : dict
        A plain dict ready to pass directly to DoomNetwork.get_action().
        Layout matches the raw obs dict contract (docs/CONTRACT.md) with
        these changes applied:
          • visible_entities["features"] → normalised (Audit Finding 3 fix)
          • visible_entities["mask"]     → float32  (attention module needs this)
          • vitals                       → normalised float32
          • "visual_embedding"           → present only if visual_encoder provided
          • rgb / depth                  → passed through unchanged (numpy)
    """

    rays:          RayFeatures
    depth:         DepthFeatures
    entities:      EntityFeatures
    vitals:        VitalsFeatures

    is_valid:      bool
    issues:        list

    network_input: dict


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class SensoryPipeline:
    """Runs all four sensory processors and returns a ProcessedObservation.

    Usage
    -----
    pipeline = SensoryPipeline()           # create once per agent
    processed = pipeline.process(obs)      # call every step

    Parameters
    ----------
    visual_encoder : callable | None
        Optional visual encoder module.  If provided, it is called as:
            embedding = visual_encoder(obs["rgb"], obs["depth"], device)
        The return value must be a numpy array of shape (D,) float32.
        The embedding is added to network_input["visual_embedding"].
        ``NetworkConfig.visual_feature_dim`` must be set to D.
    device : str
        Device string passed through to visual_encoder. Default "cpu".
    open_threshold : float
        Passed to RaycastProcessor: rays beyond this normalised distance
        are considered "open". Default 0.9.
    low_health_threshold : float
        Passed to VitalsProcessor. Default 25.0.
    critical_health_threshold : float
        Passed to VitalsProcessor. Default 10.0.
    validate : bool
        Run validate_observation() on each step. Cheap (< 0.1 ms) but can
        be disabled in high-frequency training loops. Default True.
    """

    def __init__(
        self,
        visual_encoder:            Callable | None = None,
        device:                    str             = "cpu",
        open_threshold:            float           = 0.9,
        low_health_threshold:      float           = 25.0,
        critical_health_threshold: float           = 10.0,
        validate:                  bool            = True,
    ) -> None:
        self._visual_encoder = visual_encoder
        self._device         = device
        self._validate       = validate

        # Instantiate all processors once; they are stateless
        self._ray_proc    = RaycastProcessor(open_threshold=open_threshold)
        self._depth_proc  = DepthProcessor()
        self._entity_proc = EntityProcessor()
        self._vitals_proc = VitalsProcessor(
            low_health_threshold=low_health_threshold,
            critical_health_threshold=critical_health_threshold,
        )

    # ------------------------------------------------------------------
    def process(self, obs: dict) -> ProcessedObservation:
        """Process a raw DoomEnv obs dict into a ProcessedObservation.

        Parameters
        ----------
        obs : dict
            Raw observation returned by DoomEnv.step() or env.reset().
            Never modified in place.

        Returns
        -------
        ProcessedObservation
            Immutable dataclass. Always returned even if obs is malformed.
        """
        # --- Validation (optional, non-fatal) ---------------------------
        is_valid, issues = self._run_validation(obs)

        # --- Run all four processors ------------------------------------
        rays_feat    = self._ray_proc.process(obs)
        depth_feat   = self._depth_proc.process(obs)
        entity_feat  = self._entity_proc.process(obs)
        vitals_feat  = self._vitals_proc.process(obs)

        # --- Build network_input dict -----------------------------------
        network_input = self._build_network_input(obs, rays_feat, entity_feat, vitals_feat)

        return ProcessedObservation(
            rays=rays_feat,
            depth=depth_feat,
            entities=entity_feat,
            vitals=vitals_feat,
            is_valid=is_valid,
            issues=issues,
            network_input=network_input,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_validation(self, obs: dict) -> tuple[bool, list]:
        """Run validate_observation() if enabled; always returns (bool, list)."""
        if not self._validate or not _VALIDATOR_AVAILABLE:
            return True, []
        try:
            issues = validate_observation(obs)
            if issues:
                logger.debug("SensoryPipeline: obs validation issues: %s", issues)
            return (len(issues) == 0), issues
        except Exception as exc:  # pragma: no cover
            logger.warning("SensoryPipeline: validation raised %s", exc)
            return False, [str(exc)]

    def _build_network_input(
        self,
        obs:         dict,
        rays_feat:   RayFeatures,
        entity_feat: EntityFeatures,
        vitals_feat: VitalsFeatures,
    ) -> dict:
        """Assemble the network_input dict.

        Uses the already-processed, already-normalised arrays from the
        processor results so we don't repeat any computation.

        The structure exactly matches DoomNetwork._extract_inputs() expectations:
          network_input["vitals"]                             (17,)   float32
          network_input["rays"]["distance"]                   (64,)   float32
          network_input["rays"]["hit_type"]                   (64,)   int64
          network_input["visible_entities"]["features"]       (32, 5) float32
          network_input["visible_entities"]["entity_type"]    (32,)   int64
          network_input["visible_entities"]["mask"]           (32,)   float32  ← float!
          network_input["rgb"]                                (120,160,3) uint8
          network_input["depth"]                              (120,160)   uint8
          network_input["visual_embedding"]                   (D,)   float32   if encoder set
        """
        network_input = {
            # Normalised vitals (mirrors VitalsEncoder.normalize_vitals)
            "vitals": vitals_feat.normalized,

            # Rays: distance already [0,1]; hit_type clamped to [0,5]
            "rays": {
                "distance": rays_feat.distance_norm,
                "hit_type": rays_feat.hit_type_raw,
            },

            # Entities: features normalised (Finding 3 fix);
            # mask as float32 — EnemyAttentionModule expects float for masking
            "visible_entities": {
                "features":    entity_feat.features_norm,
                "entity_type": entity_feat.entity_type,
                "mask":        entity_feat.mask.astype(np.float32),
            },

            # Pass rgb / depth through unchanged (numpy uint8)
            # Network ignores them when visual_feature_dim==0
            "rgb":   obs.get("rgb",   np.zeros((120, 160, 3), dtype=np.uint8)),
            "depth": obs.get("depth", np.zeros((120, 160),    dtype=np.uint8)),
        }

        # Optional visual embedding (requires visual_encoder to be set)
        if self._visual_encoder is not None:
            rgb   = obs.get("rgb",   np.zeros((120, 160, 3), np.uint8))
            depth = obs.get("depth", np.zeros((120, 160),    np.uint8))
            try:
                embedding = self._visual_encoder(rgb, depth, self._device)
                network_input["visual_embedding"] = np.asarray(
                    embedding, dtype=np.float32
                )
            except Exception as exc:   # pragma: no cover
                logger.warning("SensoryPipeline: visual_encoder raised %s", exc)

        return network_input
