"""VitalsProcessor — health, armor, ammunition and velocity perception.

Converts obs["vitals"] (17-element float32 vector) into a typed
VitalsFeatures dataclass with per-field accessors, normalised values,
and semantic flags useful for decision-making and reward shaping.

Observation contract (env/doom_env/spaces.py — VITALS_FIELDS)
--------------------------------------------------------------
Index  Field                 Notes
  0    health                raw value; 0=dead, typical max 100-200
  1    armor                 raw value; 0–200
  2    selected_weapon       integer 0–7 (slot number)
  3    selected_weapon_ammo  ammo for currently held weapon
  4    ammo_bullets          total bullets
  5    ammo_shells           total shells
  6    ammo_rockets          total rockets
  7    ammo_cells            total energy cells
  8    key_blue_card         {0.0, 1.0} binary flag
  9    key_yellow_card       {0.0, 1.0}
 10    key_red_card          {0.0, 1.0}
 11    key_blue_skull        {0.0, 1.0}
 12    key_yellow_skull      {0.0, 1.0}
 13    key_red_skull         {0.0, 1.0}
 14    vel_forward           map units/tic, ego frame
 15    vel_right             map units/tic, ego frame
 16    vel_up                map units/tic, ego frame

Normalisation (mirrors network.encoder.VitalsEncoder.normalize_vitals)
------------------------------------------------------------------------
[0:2]   health, armor     / 100.0           → typical [0, 2]
[2]     selected_weapon   / 7.0             → [0, 1]
[3:8]   ammo (5 fields)   log1p / 4.0       → ≈ [0, 1.5] for max ammo
[8:14]  key flags         unchanged         → {0, 1}
[14:17] velocity (3)      / 20.0            → running ≈ 0.6

Design rules
------------
- Pure NumPy; zero torch dependency.
- Never modifies the caller's obs dict.
- NaN / Inf vitals are replaced with 0.0 before all computations.
- Must complete in well under 1 ms on a single CPU thread.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_VITALS = 17

# Thresholds for semantic health flags
_LOW_HEALTH_THRESHOLD      = 25.0   # start prioritising health pickups
_CRITICAL_HEALTH_THRESHOLD = 10.0   # flee / panic mode

# Normalisation scales (identical to VitalsEncoder.normalize_vitals)
_HEALTH_ARMOR_SCALE = 1.0 / 100.0
_WEAPON_SCALE       = 1.0 / 7.0
_AMMO_LOG_SCALE     = 1.0 / 4.0
_VELOCITY_SCALE     = 1.0 / 20.0

# Vitals index slices (from doom_env.spaces.VITALS_FIELDS)
_IDX_HEALTH   = 0
_IDX_ARMOR    = 1
_IDX_WEAPON   = 2
_IDX_SEL_AMMO = 3
_IDX_AMMO     = slice(3, 8)   # sel_weapon_ammo + bullets + shells + rockets + cells
_IDX_AMMO_BULLETS  = 4
_IDX_AMMO_SHELLS   = 5
_IDX_AMMO_ROCKETS  = 6
_IDX_AMMO_CELLS    = 7
_IDX_KEYS     = slice(8, 14)
_IDX_VEL      = slice(14, 17)
_IDX_VEL_FWD  = 14
_IDX_VEL_RGT  = 15
_IDX_VEL_UP   = 16

_KEY_NAMES = (
    "blue_card", "yellow_card", "red_card",
    "blue_skull", "yellow_skull", "red_skull",
)
_AMMO_NAMES = ("sel_weapon_ammo", "bullets", "shells", "rockets", "cells")


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VitalsFeatures:
    """All derived vitals features, ready for neural network input.

    Fields
    ------
    raw : (17,) float32
        Original vitals vector (NaN/Inf replaced with 0.0).

    normalized : (17,) float32
        Normalised copy, mirrors VitalsEncoder.normalize_vitals():
          [0:2]   / 100.0
          [2]     / 7.0
          [3:8]   log1p / 4.0
          [8:14]  unchanged (key flags)
          [14:17] / 20.0

    health : float
        Raw health value.
    health_norm : float
        health / 100.0

    armor : float
        Raw armor value.
    armor_norm : float
        armor / 100.0

    is_low_health : bool
        True when health < 25.0. Agent should seek health pickups.
    is_critical : bool
        True when health < 10.0. Agent should flee / retreat.

    selected_weapon : int
        Currently equipped weapon slot (0–7).
    sel_weapon_ammo : float
        Ammo count for the currently selected weapon.

    ammo_counts : dict[str, float]
        Named raw ammo counts:
        {"sel_weapon_ammo", "bullets", "shells", "rockets", "cells"}
    ammo_norm : (5,) float32
        log1p(ammo_counts) / 4.0 for all 5 ammo slots (tensor-ready).

    keys_held : dict[str, bool]
        {blue_card, yellow_card, red_card, blue_skull, yellow_skull, red_skull}
    n_keys : int
        Total number of keys held (0–6).
    keys_vector : (6,) float32
        vitals[8:14] as float32 (same as normalized[8:14]).

    velocity : (3,) float32
        [vel_forward, vel_right, vel_up] in map units/tic (raw).
    velocity_norm : (3,) float32
        velocity / 20.0  → running ≈ [0.6, 0, 0].
    speed : float
        Euclidean XY speed = sqrt(vel_forward² + vel_right²) in map units/tic.
        Useful for detecting stuck-agent states.
    is_moving : bool
        True when speed > 1.0 map units/tic.
    """

    raw:              np.ndarray    # (17,) float32
    normalized:       np.ndarray    # (17,) float32

    health:           float
    health_norm:      float
    armor:            float
    armor_norm:       float
    is_low_health:    bool
    is_critical:      bool

    selected_weapon:  int
    sel_weapon_ammo:  float
    ammo_counts:      dict          # str → float
    ammo_norm:        np.ndarray    # (5,) float32
    keys_held:        dict          # str → bool
    n_keys:           int
    keys_vector:      np.ndarray    # (6,) float32

    velocity:         np.ndarray    # (3,) float32
    velocity_norm:    np.ndarray    # (3,) float32
    speed:            float
    is_moving:        bool


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class VitalsProcessor:
    """Stateless processor for the (17,) vitals vector.

    Usage
    -----
    processor = VitalsProcessor()
    features  = processor.process(obs)

    Parameters
    ----------
    low_health_threshold : float
        Health below this value sets is_low_health=True. Default 25.0.
    critical_health_threshold : float
        Health below this value sets is_critical=True. Default 10.0.
    moving_speed_threshold : float
        Speed (map units/tic) above which is_moving=True. Default 1.0.
    """

    def __init__(
        self,
        low_health_threshold:      float = _LOW_HEALTH_THRESHOLD,
        critical_health_threshold: float = _CRITICAL_HEALTH_THRESHOLD,
        moving_speed_threshold:    float = 1.0,
    ) -> None:
        self._low_health_thr  = float(low_health_threshold)
        self._critical_thr    = float(critical_health_threshold)
        self._moving_speed_thr = float(moving_speed_threshold)

    # ------------------------------------------------------------------
    def process(self, obs: dict) -> VitalsFeatures:
        """Process obs["vitals"] into a fully-validated VitalsFeatures object.

        Parameters
        ----------
        obs : dict
            Raw observation dict from DoomEnv.

        Returns
        -------
        VitalsFeatures
            Immutable dataclass. Falls back to all-zero vitals if the
            "vitals" key is missing or malformed.
        """
        raw = self._extract_safe(obs)

        # --- Normalised copy (mirrors VitalsEncoder.normalize_vitals) ---
        normalized = self._normalise(raw)

        # --- Scalar accessors -------------------------------------------
        health  = float(raw[_IDX_HEALTH])
        armor   = float(raw[_IDX_ARMOR])

        # --- Ammo counts ------------------------------------------------
        ammo_raw = raw[_IDX_AMMO]   # (5,) float32
        ammo_counts = {
            name: float(ammo_raw[i]) for i, name in enumerate(_AMMO_NAMES)
        }
        ammo_norm = normalized[_IDX_AMMO].copy()   # (5,) float32

        # --- Key flags --------------------------------------------------
        keys_raw = raw[_IDX_KEYS]   # (6,) float32
        keys_held = {
            name: bool(keys_raw[i] > 0.5) for i, name in enumerate(_KEY_NAMES)
        }
        n_keys = int(sum(keys_held.values()))
        keys_vector = keys_raw.astype(np.float32).copy()   # (6,)

        # --- Velocity ---------------------------------------------------
        velocity      = raw[_IDX_VEL].copy()          # (3,)
        velocity_norm = normalized[_IDX_VEL].copy()   # (3,)
        vf = float(raw[_IDX_VEL_FWD])
        vr = float(raw[_IDX_VEL_RGT])
        speed = float(math.sqrt(vf * vf + vr * vr))

        return VitalsFeatures(
            raw=raw,
            normalized=normalized,
            health=health,
            health_norm=health * _HEALTH_ARMOR_SCALE,
            armor=armor,
            armor_norm=armor * _HEALTH_ARMOR_SCALE,
            is_low_health=(health < self._low_health_thr),
            is_critical=(health < self._critical_thr),
            selected_weapon=int(raw[_IDX_WEAPON]),
            sel_weapon_ammo=float(raw[_IDX_SEL_AMMO]),
            ammo_counts=ammo_counts,
            ammo_norm=ammo_norm,
            keys_held=keys_held,
            n_keys=n_keys,
            keys_vector=keys_vector,
            velocity=velocity,
            velocity_norm=velocity_norm,
            speed=speed,
            is_moving=(speed > self._moving_speed_thr),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_safe(obs: dict) -> np.ndarray:
        """Return a sanitised (NUM_VITALS,) float32 vitals array.

        Handles missing / wrong-shape / NaN-containing input gracefully.
        """
        raw = obs.get("vitals", None)
        if (
            raw is None
            or not isinstance(raw, np.ndarray)
            or raw.shape != (NUM_VITALS,)
        ):
            return np.zeros(NUM_VITALS, dtype=np.float32)

        v = raw.astype(np.float32, copy=True)
        # Replace NaN / Inf with 0 (safe neutral — health=0 means dead)
        bad = ~np.isfinite(v)
        if bad.any():
            v[bad] = 0.0
        return v

    @staticmethod
    def _normalise(v: np.ndarray) -> np.ndarray:
        """Normalise vitals, mirroring VitalsEncoder.normalize_vitals().

        Parameters
        ----------
        v : (17,) float32  — sanitised (no NaN / Inf)

        Returns
        -------
        (17,) float32  — normalised copy, same layout as v.
        """
        out = v.copy()
        out[0:2]   *= _HEALTH_ARMOR_SCALE               # health, armor
        out[2]     *= _WEAPON_SCALE                      # weapon slot
        ammo = np.maximum(out[3:8], 0.0)                 # clip negative
        out[3:8]    = np.log1p(ammo) * _AMMO_LOG_SCALE  # log-normalise ammo
        # [8:14] key flags: unchanged
        out[14:17] *= _VELOCITY_SCALE                    # velocity
        return out
