"""EntityProcessor — visible entity perception.

Converts obs["visible_entities"] into a typed EntityFeatures dataclass
containing normalised features, one-hot type encodings, and filtered
sub-views for enemies and pickups.

Observation contract (env/doom_env/spaces.py + docs/CONTRACT.md §2.3)
----------------------------------------------------------------------
obs["visible_entities"]["features"]    (32, 5)  float32
  col 0  distance        raw value from env = edist / RAY_RANGE = edist / 1024.
                         Max entity observation range is ENTITY_MAX_DISTANCE = 4096
                         map units, so the stored value is in [0.0, 4.0].
                         We further normalise by /4.0 → [0.0, 1.0] for the network.
  col 1  sin_rel_angle   in [-1.0, 1.0]  (already well-scaled)
  col 2  cos_rel_angle   in [-1.0, 1.0]  (already well-scaled)
  col 3  rel_vel_forward map units/tic, unbounded (projectiles ~50 u/tic)
  col 4  rel_vel_right   map units/tic, unbounded

obs["visible_entities"]["entity_type"] (32,)   int64   {0..11}
obs["visible_entities"]["mask"]        (32,)   int8    {0=pad, 1=real}

Entity type codes (from doom_env.spaces.ENTITY_TYPES):
  0  pad            7  ammo
  1  enemy_melee    8  weapon
  2  enemy_hitscan  9  key
  3  enemy_projectile  10 barrel
  4  projectile     11 other
  5  health
  6  armor

NOTE: entity distance col-0 raw max is 4.0 (= ENTITY_MAX_DISTANCE / RAY_RANGE
= 4096 / 1024). The network's EnemyAttentionModule receives features as-is
from network_input, so we normalise col-0 by /4.0 before inserting into
network_input. This is the Audit Finding 3 fix.

Design rules
------------
- Pure NumPy; zero torch dependency.
- Never modifies the caller's obs dict.
- Invalid entity_type values are clamped to 0 (pad) before one-hot.
- Extreme velocities are hard-clipped (np.clip) at ±20 u/tic then
  scaled to [-1, 1]; this bounds the network's first linear layer.
- Must complete in well under 1 ms on a single CPU thread.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants (from doom_env.spaces + audit report)
# ---------------------------------------------------------------------------

MAX_ENTITIES        = 32
ENTITY_FEATURES     = 5           # (dist, sin, cos, vel_fwd, vel_right)
NUM_ENTITY_TYPES    = 12          # {0=pad, 1..11}

# Distance normalisation: entity col-0 ∈ [0, 4.0]  → /4.0 → [0, 1]
# This is DIFFERENT from ray distance (which is already /1024).
ENTITY_DIST_MAX_NORM = 4.0        # = 4096 / 1024

# Velocity soft-clip: values beyond ±VEL_CLIP are scaled to [−1, 1]
# Fast projectiles can reach ~50 u/tic; we clip at 20 for smooth training.
_VEL_CLIP  = 20.0
_VEL_SCALE = 1.0 / _VEL_CLIP

# Combined token width: features_norm (5) + type_onehot (12) = 17
COMBINED_TOKEN_DIM = ENTITY_FEATURES + NUM_ENTITY_TYPES   # 17

# Entity type ID sets (used for sub-view filtering)
_ENEMY_TYPE_IDS   = frozenset({1, 2, 3})       # melee, hitscan, proj enemy
_PICKUP_TYPE_IDS  = frozenset({5, 6, 7, 8, 9}) # health, armor, ammo, weapon, key
_KEY_TYPE_ID      = 9
_PROJECTILE_ID    = 4

# One-hot matrix computed at import time (12×12 identity)
_TYPE_ONEHOT_MATRIX = np.eye(NUM_ENTITY_TYPES, dtype=np.float32)  # (12,12)

# Human-readable names (index = type_id)
ENTITY_TYPE_NAMES = (
    "pad", "enemy_melee", "enemy_hitscan", "enemy_projectile",
    "projectile", "health", "armor", "ammo", "weapon", "key",
    "barrel", "other",
)


# ---------------------------------------------------------------------------
# Sub-view named tuples (filtered slices of the entity table)
# ---------------------------------------------------------------------------

class EnemySlice(NamedTuple):
    """Filtered view of enemy entity slots."""
    features_norm: np.ndarray   # (K, 5)  float32  — normalised
    type_ids:      np.ndarray   # (K,)    int64
    count:         int


class PickupSlice(NamedTuple):
    """Filtered view of pickup + key entity slots."""
    features_norm: np.ndarray   # (K, 5)  float32  — normalised
    type_ids:      np.ndarray   # (K,)    int64
    count:         int


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EntityFeatures:
    """All derived entity features, ready for neural network input.

    Fields
    ------
    features_raw : (32, 5) float32
        Exact copy of obs["visible_entities"]["features"].
        col 0 = distance in [0, 4.0] (NOT normalised).

    features_norm : (32, 5) float32
        Normalised copy:
          col 0  distance  / 4.0        → [0.0, 1.0]
          col 1  sin_angle  unchanged   → [−1.0, 1.0]
          col 2  cos_angle  unchanged   → [−1.0, 1.0]
          col 3  vel_fwd  clip±20 / 20  → [−1.0, 1.0]
          col 4  vel_right clip±20 / 20 → [−1.0, 1.0]
        Padding slots (mask=0) contain zeros — no information leakage.

    entity_type : (32,) int64
        Original entity_type array, clamped to [0, 11].

    mask : (32,) bool
        True = real entity slot. False = padding (should be ignored).

    type_onehot : (32, 12) float32
        One-hot encoding of entity_type. Each row sums to 1.0.
        Padding slots (entity_type=0) have a 1 in position 0.

    combined_input : (32, 17) float32
        Concatenation of [features_norm | type_onehot] along the
        feature axis.  Shape: (MAX_ENTITIES, COMBINED_TOKEN_DIM) = (32, 17).
        This is the extended token input for EnemyAttentionModule
        (requires Member 3 to update token_encoder from 5→17 dims).

    n_visible : int
        Number of real entity slots (mask.sum()).

    enemies : EnemySlice
        Filtered sub-view of enemy slots (type_ids ∈ {1,2,3}).

    pickups : PickupSlice
        Filtered sub-view of pickup + key slots (type_ids ∈ {5..9}).

    has_incoming_projectile : bool
        True if any visible entity has type_id == 4 (projectile).
    """

    features_raw:            np.ndarray    # (32,  5)   float32
    features_norm:           np.ndarray    # (32,  5)   float32
    entity_type:             np.ndarray    # (32,)      int64
    mask:                    np.ndarray    # (32,)      bool
    type_onehot:             np.ndarray    # (32, 12)   float32
    combined_input:          np.ndarray    # (32, 17)   float32
    n_visible:               int
    enemies:                 EnemySlice
    pickups:                 PickupSlice
    has_incoming_projectile: bool


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class EntityProcessor:
    """Stateless processor for the visible entity table.

    Usage
    -----
    processor = EntityProcessor()
    features  = processor.process(obs)
    """

    # ------------------------------------------------------------------
    def process(self, obs: dict) -> EntityFeatures:
        """Process obs["visible_entities"] into EntityFeatures.

        Parameters
        ----------
        obs : dict
            Raw observation dict from DoomEnv.

        Returns
        -------
        EntityFeatures
            Immutable dataclass. Falls back to all-zero / empty tables
            if the "visible_entities" key is missing or malformed.
        """
        raw_feats, entity_type, mask = self._extract_safe(obs)

        # --- Normalise features (Audit Finding 3 fix) -------------------
        features_norm = self._normalise(raw_feats)

        # --- Mask: zero-out padding slots in features_norm --------------
        # (the env should already zero-pad, but we enforce it defensively)
        pad_slots = ~mask
        if pad_slots.any():
            features_norm[pad_slots, :] = 0.0

        # --- One-hot type encoding -------------------------------------
        type_onehot = _TYPE_ONEHOT_MATRIX[entity_type]   # (32, 12)

        # --- Combined token tensor -------------------------------------
        combined_input = np.concatenate(
            [features_norm, type_onehot], axis=1
        )  # (32, 17)

        # --- Count & sub-views ----------------------------------------
        n_visible = int(mask.sum())

        is_enemy   = np.isin(entity_type, list(_ENEMY_TYPE_IDS))
        is_pickup  = np.isin(entity_type, list(_PICKUP_TYPE_IDS))
        active     = mask  # bool (32,)

        enemy_idx   = np.flatnonzero(active & is_enemy)
        pickup_idx  = np.flatnonzero(active & is_pickup)
        proj_active = active & (entity_type == _PROJECTILE_ID)

        enemies = EnemySlice(
            features_norm=features_norm[enemy_idx],
            type_ids=entity_type[enemy_idx],
            count=len(enemy_idx),
        )
        pickups = PickupSlice(
            features_norm=features_norm[pickup_idx],
            type_ids=entity_type[pickup_idx],
            count=len(pickup_idx),
        )

        return EntityFeatures(
            features_raw=raw_feats,
            features_norm=features_norm,
            entity_type=entity_type,
            mask=mask,
            type_onehot=type_onehot,
            combined_input=combined_input,
            n_visible=n_visible,
            enemies=enemies,
            pickups=pickups,
            has_incoming_projectile=bool(proj_active.any()),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_safe(
        obs: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract and sanitise features, entity_type, mask from obs.

        Returns
        -------
        features   : (32, 5)  float32   raw, NaN→0.0
        entity_type: (32,)    int64     clamped to [0, 11]
        mask       : (32,)    bool      True = real slot
        """
        ve = obs.get("visible_entities", {})

        # --- features ---------------------------------------------------
        raw_f = ve.get("features", None)
        if (
            raw_f is None
            or not isinstance(raw_f, np.ndarray)
            or raw_f.shape != (MAX_ENTITIES, ENTITY_FEATURES)
        ):
            features = np.zeros((MAX_ENTITIES, ENTITY_FEATURES), dtype=np.float32)
        else:
            features = raw_f.astype(np.float32, copy=True)
            # Replace NaN / Inf with 0.0 (safe neutral value)
            bad = ~np.isfinite(features)
            if bad.any():
                features[bad] = 0.0

        # --- entity_type ------------------------------------------------
        raw_et = ve.get("entity_type", None)
        if (
            raw_et is None
            or not isinstance(raw_et, np.ndarray)
            or raw_et.shape != (MAX_ENTITIES,)
        ):
            entity_type = np.zeros(MAX_ENTITIES, dtype=np.int64)
        else:
            entity_type = raw_et.astype(np.int64, copy=True)
            # Clamp invalid codes to 0 (pad)
            invalid = (entity_type < 0) | (entity_type >= NUM_ENTITY_TYPES)
            if invalid.any():
                entity_type[invalid] = 0

        # --- mask -------------------------------------------------------
        raw_mask = ve.get("mask", None)
        if (
            raw_mask is None
            or not isinstance(raw_mask, np.ndarray)
            or raw_mask.shape != (MAX_ENTITIES,)
        ):
            mask = np.zeros(MAX_ENTITIES, dtype=bool)
        else:
            mask = (raw_mask == 1).astype(bool)   # tolerates int8 / int64

        return features, entity_type, mask

    @staticmethod
    def _normalise(features: np.ndarray) -> np.ndarray:
        """Return normalised copy of entity features.

        Normalisation applied per column:
          col 0  distance:   raw [0, 4.0] → /4.0 → [0, 1.0]
          col 1  sin_angle:  already [-1, 1] — unchanged
          col 2  cos_angle:  already [-1, 1] — unchanged
          col 3  vel_fwd:    hard-clip ±20, then /20 → [-1, 1]
          col 4  vel_right:  hard-clip ±20, then /20 → [-1, 1]

        Uses ellipsis indexing (`[..., k]`) so the function is safe for
        both 2-D inputs (32, 5) and batched 3-D inputs (..., 32, 5).
        """
        f = features.copy()
        # Distance column: [0, 4.0] → [0, 1.0]
        f[..., 0] = f[..., 0] / ENTITY_DIST_MAX_NORM
        # sin / cos are already in [-1, 1] — no scaling needed
        # Velocity columns: hard-clip at ±20 u/tic, then scale to [-1, 1]
        f[..., 3:5] = np.clip(f[..., 3:5], -_VEL_CLIP, _VEL_CLIP) * _VEL_SCALE
        return f
