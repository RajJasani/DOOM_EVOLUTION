"""RaycastProcessor — ego-centric 64-ray perception.

Converts the raw ``obs["rays"]`` dict into a fully-typed, normalised
``RayFeatures`` dataclass that is safe to use directly in neural network
training or spatial analysis.

Observation contract (env/doom_env/spaces.py)
---------------------------------------------
obs["rays"]["distance"]   (64,)  float32   already normalised [0.0, 1.0]
                                           0 = obstacle touching player
                                           1 = nothing within 1024 map units
obs["rays"]["hit_type"]   (64,)  int64     {0=none, 1=wall, 2=door,
                                            3=locked_door, 4=enemy, 5=item}

Design rules
------------
- Pure NumPy; zero torch dependency.
- Never modifies the caller's obs dict.
- Clamps / fills invalid values rather than raising.
- All derived arrays are freshly allocated (no views into input arrays).
- Must complete in well under 1 ms on a single CPU thread.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants (inlined from doom_env.spaces so this file runs without the env)
# ---------------------------------------------------------------------------

NUM_RAYS       = 64          # ego-centric raycasts, evenly spaced 360°
NUM_HIT_TYPES  = 6           # {none, wall, door, locked_door, enemy, item}
RAY_RANGE      = 1024.0      # map units; distances are pre-divided by this

# Hit-type integer codes
RAY_NONE   = 0
RAY_WALL   = 1
RAY_DOOR   = 2
RAY_LOCKED = 3
RAY_ENEMY  = 4
RAY_ITEM   = 5

# Angle of each ray in degrees (CCW from forward = 0°)
_RAY_ANGLES_DEG = np.arange(NUM_RAYS, dtype=np.float32) * (360.0 / NUM_RAYS)
_RAY_ANGLES_RAD = np.deg2rad(_RAY_ANGLES_DEG)

# sin / cos look-up tables for all 64 ray angles (computed once at import time)
_RAY_SIN = np.sin(_RAY_ANGLES_RAD).astype(np.float32)
_RAY_COS = np.cos(_RAY_ANGLES_RAD).astype(np.float32)

# One-hot identity matrix for quick encoding
_ONEHOT_MATRIX = np.eye(NUM_HIT_TYPES, dtype=np.float32)  # (6, 6)

# Forward arc indices: rays within ±22.5° of straight ahead
#   Ray 0 = forward; rays 1-3 lean left, rays 61-63 lean right
_FORWARD_RAY_IDX = np.array([0, 1, 2, 3, 61, 62, 63], dtype=np.int64)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

class _RaySummaryTuple(NamedTuple):
    """Lightweight count summary (matches SpatialAnalyzer.RaySummary fields)."""
    n_wall:          int
    n_door:          int
    n_locked:        int
    n_enemy:         int
    n_item:          int
    n_open:          int
    mean_dist_norm:  float
    min_dist_norm:   float
    min_dist_angle:  float   # degrees CCW from forward


@dataclass(frozen=True)
class RayFeatures:
    """All derived raycast features, ready for neural network input.

    Fields
    ------
    distance_norm : (64,) float32
        Raw ray distances in [0.0, 1.0].  Exactly as the env provides;
        0 = surface touching player, 1 = open air at max range (1024 u).

    hit_type_raw : (64,) int64
        Raw hit_type integers, clamped to [0, 5].

    hit_type_onehot : (64, 6) float32
        One-hot encoding of hit_type. Each row sums to 1.0.

    combined_tensor : (64, 7) float32
        Concatenation of [distance_norm | hit_type_onehot] along the
        feature axis.  Shape: (NUM_RAYS, 1 + NUM_HIT_TYPES) = (64, 7).
        This is the primary input for a 1-D CNN over the ray dimension.

    wall_mask : (64,) bool
        True where hit_type == RAY_WALL.

    door_mask : (64,) bool
        True where hit_type is RAY_DOOR or RAY_LOCKED (any door).

    enemy_mask : (64,) bool
        True where hit_type == RAY_ENEMY.

    item_mask : (64,) bool
        True where hit_type == RAY_ITEM.

    open_mask : (64,) bool
        True where hit_type == RAY_NONE (no obstacle in range).

    forward_clearance : float
        Fraction of the 7 forward-facing rays that are open (hit_type=0
        or distance >= 0.9).  0.0 = fully blocked, 1.0 = fully clear.

    summary : _RaySummaryTuple
        Aggregate counts and min/mean distance statistics.
    """

    distance_norm:      np.ndarray   # (64,)    float32
    hit_type_raw:       np.ndarray   # (64,)    int64
    hit_type_onehot:    np.ndarray   # (64, 6)  float32
    combined_tensor:    np.ndarray   # (64, 7)  float32
    wall_mask:          np.ndarray   # (64,)    bool
    door_mask:          np.ndarray   # (64,)    bool
    enemy_mask:         np.ndarray   # (64,)    bool
    item_mask:          np.ndarray   # (64,)    bool
    open_mask:          np.ndarray   # (64,)    bool
    forward_clearance:  float
    summary:            _RaySummaryTuple


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class RaycastProcessor:
    """Stateless processor for the 64 ego-centric raycasts.

    Usage
    -----
    processor = RaycastProcessor()          # instantiate once
    features  = processor.process(obs)     # call every step

    Parameters
    ----------
    open_threshold : float
        A ray is considered 'open' (passable) if its normalised distance
        exceeds this value OR its hit_type is RAY_NONE (0).
        Default 0.9 ≈ 921 map units of clear space.
    """

    def __init__(self, open_threshold: float = 0.9) -> None:
        self._open_threshold = float(open_threshold)

    # ------------------------------------------------------------------
    def process(self, obs: dict) -> RayFeatures:
        """Process obs["rays"] into a fully-validated RayFeatures object.

        Parameters
        ----------
        obs : dict
            Raw observation dict from DoomEnv.step() or env.reset().
            Must contain obs["rays"]["distance"] and obs["rays"]["hit_type"].

        Returns
        -------
        RayFeatures
            Immutable dataclass containing all derived ray features.
            If the "rays" key is missing or malformed, safe fallback values
            are used (distance=1.0, hit_type=RAY_NONE) so callers never crash.
        """
        distance, hit_type = self._extract_safe(obs)

        # --- Boolean masks (vectorised, no loops) -------------------------
        wall_mask  = hit_type == RAY_WALL
        door_mask  = (hit_type == RAY_DOOR) | (hit_type == RAY_LOCKED)
        enemy_mask = hit_type == RAY_ENEMY
        item_mask  = hit_type == RAY_ITEM
        open_mask  = hit_type == RAY_NONE

        # --- One-hot encoding --------------------------------------------
        # _ONEHOT_MATRIX[hit_type] is equivalent to np.eye(6)[hit_type]
        # but avoids fancy indexing overhead for small arrays.
        hit_type_onehot = _ONEHOT_MATRIX[hit_type]  # (64, 6) float32

        # --- Combined tensor: [distance | one-hot] -----------------------
        # Shape: (64, 7) — the primary CNN input channel set
        combined_tensor = np.concatenate(
            [distance[:, np.newaxis], hit_type_onehot], axis=1
        )  # (64, 7)

        # --- Forward clearance -------------------------------------------
        fwd_dist = distance[_FORWARD_RAY_IDX]
        fwd_open = open_mask[_FORWARD_RAY_IDX]
        is_clear = fwd_open | (fwd_dist >= self._open_threshold)
        forward_clearance = float(is_clear.mean())

        # --- Aggregate summary -------------------------------------------
        n_wall   = int(wall_mask.sum())
        n_door   = int((hit_type == RAY_DOOR).sum())
        n_locked = int((hit_type == RAY_LOCKED).sum())
        n_enemy  = int(enemy_mask.sum())
        n_item   = int(item_mask.sum())
        n_open   = int(open_mask.sum())

        mean_d  = float(distance.mean())
        min_idx = int(np.argmin(distance))
        min_d   = float(distance[min_idx])
        min_ang = float(_RAY_ANGLES_DEG[min_idx])

        summary = _RaySummaryTuple(
            n_wall=n_wall,  n_door=n_door,  n_locked=n_locked,
            n_enemy=n_enemy, n_item=n_item, n_open=n_open,
            mean_dist_norm=mean_d, min_dist_norm=min_d,
            min_dist_angle=min_ang,
        )

        return RayFeatures(
            distance_norm=distance,
            hit_type_raw=hit_type,
            hit_type_onehot=hit_type_onehot,
            combined_tensor=combined_tensor,
            wall_mask=wall_mask,
            door_mask=door_mask,
            enemy_mask=enemy_mask,
            item_mask=item_mask,
            open_mask=open_mask,
            forward_clearance=forward_clearance,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_safe(
        self, obs: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract and sanitise distance and hit_type from obs.

        Returns
        -------
        distance : (64,) float32   values in [0.0, 1.0]; NaN→1.0, out-of-range clamped
        hit_type : (64,) int64     values in [0, 5]; invalid→0 (RAY_NONE)
        """
        rays = obs.get("rays", {})

        # --- distance ----------------------------------------------------
        raw_dist = rays.get("distance", None)
        if (
            raw_dist is None
            or not isinstance(raw_dist, np.ndarray)
            or raw_dist.shape != (NUM_RAYS,)
        ):
            # Fallback: all rays fully open
            distance = np.ones(NUM_RAYS, dtype=np.float32)
        else:
            distance = raw_dist.astype(np.float32, copy=True)

        # Replace NaN / Inf with 1.0 (clear — no obstacle)
        bad = ~np.isfinite(distance)
        if bad.any():
            distance[bad] = 1.0

        # Clamp to [0, 1]
        np.clip(distance, 0.0, 1.0, out=distance)

        # --- hit_type ----------------------------------------------------
        raw_hit = rays.get("hit_type", None)
        if (
            raw_hit is None
            or not isinstance(raw_hit, np.ndarray)
            or raw_hit.shape != (NUM_RAYS,)
        ):
            hit_type = np.zeros(NUM_RAYS, dtype=np.int64)
        else:
            hit_type = raw_hit.astype(np.int64, copy=True)

        # Clamp invalid codes to RAY_NONE (0)
        invalid = (hit_type < 0) | (hit_type >= NUM_HIT_TYPES)
        if invalid.any():
            hit_type[invalid] = RAY_NONE

        return distance, hit_type
