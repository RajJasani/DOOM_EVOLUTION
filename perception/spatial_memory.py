"""EgocentricSpatialMemory — episode-scoped structured spatial memory.

Maintains interpretable, structured information about the game world across
consecutive steps within a single episode. Operates in pure ego-centric
polar coordinates — no global map, no dead-reckoning.

Architecture overview
---------------------
Two-tier memory design (complementary, not overlapping):

  Tier 1 — RecurrentMemory (network/memory.py)
    - Neural, implicit, trained end-to-end via backpropagation.
    - Encodes ALL temporal patterns the network learns.
    - Lives inside DoomNetwork; outputs a hidden state vector.

  Tier 2 — EgocentricSpatialMemory (THIS MODULE)
    - Explicit, structured, human-readable.
    - Rule-updated each step from ProcessedObservation.
    - Provides a compact 64-dim feature vector as an extra network input.
    - Can be inspected, logged, and queried at runtime.

Sub-components
--------------
  ObjectMemorySlot         Polar-frame record per remembered object.
  DirectionExplorationTracker   64-bin ray-direction history.
  KeyCardTracker           Episode-level keycard possession state.
  EgocentricSpatialMemory  Top-level manager; call update() / to_feature_vector().

Integration
-----------
    from perception.spatial_memory import EgocentricSpatialMemory, MEMORY_FEATURE_DIM

    memory = EgocentricSpatialMemory()   # create once

    # DoomAgent.reset():
    memory.reset()

    # DoomAgent.act():
    processed = pipeline.process(obs)
    memory.update(processed)
    obs["memory_features"] = memory.to_feature_vector()   # (64,) float32

Design rules
------------
- Pure Python + NumPy; zero PyTorch dependency.
- NO global coordinates, NO dead-reckoning from velocity.
- Stateful within an episode; fully cleared by reset().
- O(N) per step (N ≤ 64 slots); well under 0.1 ms on a single CPU core.
- Thread-safe if called from a single thread (standard RL use case).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Public size constant — coordinate with Member 3 before changing
# ---------------------------------------------------------------------------

MEMORY_FEATURE_DIM: int = 64
"""Dimension of the vector returned by EgocentricSpatialMemory.to_feature_vector().

If Member 3 needs a different size, change this constant here and update
the network's Linear(MEMORY_FEATURE_DIM, memory_embed_dim) accordingly.
"""

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

# Maximum slots per category (caps memory footprint)
_MAX_ENEMY_SLOTS  = 32
_MAX_DOOR_SLOTS   = 16
_MAX_PICKUP_SLOTS = 16

# -------------------------------------------------------------------------
# Confidence lifecycle — per object category
# -------------------------------------------------------------------------
# Doors are static; they should stay in memory much longer than moving enemies.
# Projectile enemies (type 3) move fast and need aggressive decay.
# Pickups are static but can be collected; medium decay is appropriate.
#
# Half-life reference at _EVICTION_THRESHOLD = 0.10:
#   decay 0.985 → ~150 steps (enemies, general)
#   decay 0.996 → ~570 steps (doors — static, stay in memory across a level section)
#   decay 0.970 → ~74  steps (projectiles — fast-moving, expire quickly)
#   decay 0.990 → ~228 steps (pickups — static but collectible)

_CONFIDENCE_DECAY_ENEMY       = 0.985   # per step; general enemies
_CONFIDENCE_DECAY_PROJECTILE  = 0.970   # faster decay for fast-moving projectiles
_CONFIDENCE_DECAY_DOOR        = 0.996   # slow decay; doors are static fixtures
_CONFIDENCE_DECAY_PICKUP      = 0.990   # medium; pickups are static but collectible

# Legacy single-value used as default for custom EgocentricSpatialMemory instances
_CONFIDENCE_DECAY       = _CONFIDENCE_DECAY_ENEMY

_CONFIDENCE_REINFORCE   = 0.35    # added when observation re-confirms an object
                                  # Reduced from 0.4: prevent over-confident slots
                                  # for objects seen intermittently
_EVICTION_THRESHOLD     = 0.10    # slots below this are recycled
_INITIAL_CONFIDENCE     = 1.0

# -------------------------------------------------------------------------
# Object matching thresholds — now dual-criterion (angle + distance)
# -------------------------------------------------------------------------
# Objects merge only when BOTH angle AND distance are within tolerance.
# This prevents merging two enemies at the same bearing but different depths.
#
# Angular merge: 30° default (tightened from 45° to reduce false merges in
#   dense scenes; still loose enough to handle observation jitter of a few rays).
# Distance merge: objects within 25% of the max entity range (1.0 on a 0–4 scale)
#   are considered the same object.

_ANGULAR_MERGE_DEG     = 30.0    # degrees; reduced from 45° to avoid false merges
_DIST_MERGE_TOL        = 1.0     # on the [0, 4] entity dist_norm scale (= 1024 map units)

# Entity distance is normalised by 1024 at the env level.
# The env allows objects up to 4096 u away → max dist_norm = 4.0.
# We re-normalise to [0, 1] for the output feature vector.
_MAX_ENTITY_DIST_NORM = 4.0  # = 4096 / 1024

# Key names, in the same order as vitals[8:14]
_KEY_NAMES = (
    "blue_card",
    "yellow_card",
    "red_card",
    "blue_skull",
    "yellow_skull",
    "red_skull",
)

# Ray hit_type codes (must match env/doom_env/spaces.py)
_RAY_NONE   = 0
_RAY_WALL   = 1
_RAY_DOOR   = 2
_RAY_LOCKED = 3
_RAY_ENEMY  = 4
_RAY_ITEM   = 5

# Entity type codes (must match env/doom_env/spaces.py)
_ENT_ENEMY_MELEE      = 1
_ENT_ENEMY_HITSCAN    = 2
_ENT_ENEMY_PROJECTILE = 3
_ENT_PROJECTILE       = 4
_ENT_HEALTH           = 5
_ENT_ARMOR            = 6
_ENT_AMMO             = 7
_ENT_WEAPON           = 8
_ENT_KEY              = 9
_ENT_BARREL           = 10
_ENT_OTHER            = 11

_ENTITY_ENEMY_TYPES  = frozenset({_ENT_ENEMY_MELEE, _ENT_ENEMY_HITSCAN, _ENT_ENEMY_PROJECTILE})
_ENTITY_PICKUP_TYPES = frozenset({_ENT_HEALTH, _ENT_ARMOR, _ENT_AMMO, _ENT_WEAPON, _ENT_KEY})

_ENTITY_NAME: Dict[int, str] = {
    _ENT_ENEMY_MELEE:      "enemy_melee",
    _ENT_ENEMY_HITSCAN:    "enemy_hitscan",
    _ENT_ENEMY_PROJECTILE: "enemy_projectile",
    _ENT_PROJECTILE:       "projectile",
    _ENT_HEALTH:           "health",
    _ENT_ARMOR:            "armor",
    _ENT_AMMO:             "ammo",
    _ENT_WEAPON:           "weapon",
    _ENT_KEY:              "key",
    _ENT_BARREL:           "barrel",
    _ENT_OTHER:            "other",
}

# Number of ray bins used for the direction tracker feature buckets
_NUM_RAYS = 64

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ObjectMemorySlot:
    """A single remembered object stored in ego-relative polar coordinates.

    COORDINATE CONVENTION — read carefully
    ---------------------------------------
    All angles are stored in the EGO-CENTRIC FRAME AT THE TIME OF LAST OBSERVATION.
    That is: angle_deg=0 means the object was directly ahead when last seen.

    **What happens when the player turns:**
    After the player turns right by 30°, an object that was at 0° (ahead) is now
    at 30° (to the left in the new frame). BUT this module does NOT update stored
    angles when the player rotates — it has no access to player heading changes
    (no global coordinates, no IMU). The stored angle becomes stale.

    **Why this is acceptable:**
    - Confidence decays each step, so stale observations lose influence over time.
    - The GRU RecurrentMemory (network/memory.py) learns residual temporal patterns
      on top of these features and can implicitly compensate for rotation-induced error.
    - The feature vector uses sin/cos encoding, so a stale angle still provides a
      soft directional signal rather than a hard discrete direction.

    **What this module does NOT claim:**
    It does NOT claim that `angle_deg` accurately represents the current
    world-frame bearing to the object after the player has moved or turned.
    It records where the object WAS relative to the player's last facing direction.

    Coordinate convention (at moment of observation)
    -------------------------------------------------
    angle_deg =   0° → straight ahead (Ray 0)
    angle_deg =  90° → left           (Ray 16)   [CCW = positive]
    angle_deg = 180° → behind         (Ray 32)
    angle_deg = 270° → right          (Ray 48)

    dist_norm scale
    ---------------
    Entity distances: divided by 1024 at env level, max 4096/1024 = 4.0.
    Ray distances:    already in [0, 1] (divided by 1024 at env level, max range 1024).
    Both are stored raw here and re-normalised to [0, 1] only in feature extraction.
    """

    type_id:    int    # entity_type or hit_type code
    type_name:  str    # human-readable label
    dist_norm:  float  # last observed dist ∈ [0, 4.0] for entities, [0, 1] for rays
    angle_deg:  float  # ego-relative angle in degrees [0, 360) at time of observation
    confidence: float  # [0.0, 1.0]; 1.0 = just seen, decays each step
    step_first: int    # step when first observed
    step_last:  int    # step when last confirmed
    n_observed: int = 1  # cumulative confirmation count


@dataclass
class KeyCardTracker:
    """Episode-level keycard possession state for a single card type."""

    name:        str    # e.g. "blue_card"
    held:        bool   # True if vitals flag is currently set
    ever_seen:   bool   # True if entity_type=key was ever in FOV
    ever_held:   bool   # True once held (latches permanently)
    step_picked: int    # step when first acquired (-1 if never)


class MemorySnapshot(NamedTuple):
    """Compact read-only summary of current memory state.

    Returned by snapshot().  All distances in [0, 1] (re-normalised from [0, 4]).
    """

    n_active_enemies:   int
    n_active_doors:     int
    n_locked_doors:     int
    n_keys_held:        int
    nearest_enemy_dist:  float  # 1.0 if none remembered
    nearest_door_dist:   float  # 1.0 if none remembered
    nearest_locked_dist: float  # 1.0 if none remembered
    steps_since_enemy:   int    # steps since any enemy was confirmed (-1 if never)
    steps_since_door:    int    # steps since any door was confirmed (-1 if never)
    total_steps:         int

    @property
    def n_remembered_enemies(self) -> int:
        """Backwards compatibility alias for n_active_enemies."""
        return self.n_active_enemies

    @property
    def n_remembered_doors(self) -> int:
        """Backwards compatibility alias for n_active_doors."""
        return self.n_active_doors

    @property
    def n_held_keys(self) -> int:
        """Backwards compatibility alias for n_keys_held."""
        return self.n_keys_held

    @property
    def nearest_remembered_enemy_dist(self) -> float:
        """Backwards compatibility alias for nearest_enemy_dist."""
        return self.nearest_enemy_dist


# ---------------------------------------------------------------------------
# Sub-component: DirectionExplorationTracker
# ---------------------------------------------------------------------------

class DirectionExplorationTracker:
    """Tracks which of the 64 ray directions have been meaningfully observed.

    Updated every step from RayFeatures.  Provides compact bucket summaries
    for the memory feature vector.

    Attributes (read-only after construction)
    -----------------------------------------
    explored_mask : (64,) bool
        True if this ray direction ever returned any hit (hit_type > 0).
    open_mask : (64,) bool
        True if this ray direction ever returned hit_type == 0 (clear).
    smoothed_dist : (64,) float32
        Exponential moving average of ray distance in each direction.
    n_steps_seen : int
        Total number of step updates so far.
    """

    _EMA_ALPHA = 0.2  # EMA smoothing for distance (0 = no update, 1 = latest)

    def __init__(self) -> None:
        self.explored_mask: np.ndarray = np.zeros(_NUM_RAYS, dtype=bool)
        self.open_mask:     np.ndarray = np.zeros(_NUM_RAYS, dtype=bool)
        self.smoothed_dist: np.ndarray = np.ones(_NUM_RAYS, dtype=np.float32)
        self.n_steps_seen:  int = 0

    def reset(self) -> None:
        """Clear all tracking state for a new episode."""
        self.explored_mask[:] = False
        self.open_mask[:] = False
        self.smoothed_dist[:] = 1.0
        self.n_steps_seen = 0

    def update(self, distance_norm: np.ndarray, hit_type_raw: np.ndarray) -> None:
        """Update direction tracking from the current step's ray features.

        Parameters
        ----------
        distance_norm : (64,) float32
            Normalised ray distances in [0, 1].
        hit_type_raw : (64,) int64
            Raw hit_type integers in {0..5}.
        """
        # Any ray that returns a hit marks that direction as explored
        hit_any = (hit_type_raw > 0)
        no_hit  = (hit_type_raw == _RAY_NONE)

        self.explored_mask |= hit_any
        self.open_mask     |= no_hit

        # EMA smoothed distance
        self.smoothed_dist = (
            (1.0 - self._EMA_ALPHA) * self.smoothed_dist
            + self._EMA_ALPHA * distance_norm.astype(np.float32)
        )
        self.n_steps_seen += 1

    def explored_fractions(self, n_buckets: int = 16) -> np.ndarray:
        """Fraction of explored rays within each of `n_buckets` sectors.

        Returns
        -------
        (n_buckets,) float32, values in [0, 1].
        """
        rays_per_bucket = _NUM_RAYS // n_buckets
        fractions = np.zeros(n_buckets, dtype=np.float32)
        for b in range(n_buckets):
            lo = b * rays_per_bucket
            hi = lo + rays_per_bucket
            fractions[b] = self.explored_mask[lo:hi].mean()
        return fractions

    def open_fractions(self, n_buckets: int = 8) -> np.ndarray:
        """Fraction of open (no-hit) rays within each of `n_buckets` sectors.

        Returns
        -------
        (n_buckets,) float32, values in [0, 1].
        """
        rays_per_bucket = _NUM_RAYS // n_buckets
        fractions = np.zeros(n_buckets, dtype=np.float32)
        for b in range(n_buckets):
            lo = b * rays_per_bucket
            hi = lo + rays_per_bucket
            fractions[b] = self.open_mask[lo:hi].mean()
        return fractions


# ---------------------------------------------------------------------------
# Sub-component helpers
# ---------------------------------------------------------------------------

def _angular_diff_deg(a: float, b: float) -> float:
    """Minimum angular difference between two angles in degrees, in [0, 180]."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _ray_index_to_angle_deg(ray_idx: int) -> float:
    """Convert a ray index [0, 63] to ego-relative angle in degrees.

    Ray 0 → 0° (forward), Ray 16 → 90° (left), Ray 32 → 180° (behind),
    Ray 48 → 270° (right).  Counter-clockwise positive.
    """
    return float(ray_idx) * 360.0 / _NUM_RAYS


def _entity_sincos_to_angle_deg(sin_val: float, cos_val: float) -> float:
    """Convert entity features col-1 (sin) and col-2 (cos) to ego-relative degrees.

    VERIFIED ENCODING (from env/doom_env/sensors.py line 295):
    The env stores entity relative angle as:
        col-1 = sin(relative_angle)   in [-1, 1]
        col-2 = cos(relative_angle)   in [-1, 1]

    where relative_angle is the angle from the player's heading to the entity,
    with CCW positive (matching ViZDoom's convention):
        0° = straight ahead
        +90° = left
        -90° (= 270°) = right

    We recover the angle using atan2(sin, cos) and convert to [0, 360) degrees.

    Parameters
    ----------
    sin_val : float
        Entity features column 1 (sin of relative angle).
    cos_val : float
        Entity features column 2 (cos of relative angle).

    Returns
    -------
    float
        Angle in degrees in [0, 360).
    """
    angle_rad = math.atan2(float(sin_val), float(cos_val))
    return angle_rad * 180.0 / math.pi % 360.0


# ---------------------------------------------------------------------------
# Main memory class
# ---------------------------------------------------------------------------

class EgocentricSpatialMemory:
    """Episode-scoped structured spatial memory.

    Call ``reset()`` at the start of every episode (before env.reset() returns).
    Call ``update(processed_obs)`` every step after SensoryPipeline.process().
    Call ``to_feature_vector()`` to get the 64-dim numpy array for the network.

    Example
    -------
    ::

        from perception.processors import SensoryPipeline
        from perception.spatial_memory import EgocentricSpatialMemory, MEMORY_FEATURE_DIM

        pipeline = SensoryPipeline()
        memory   = EgocentricSpatialMemory()

        # In DoomAgent.reset():
        memory.reset()

        # In DoomAgent.act():
        processed = pipeline.process(obs)
        memory.update(processed)
        obs["memory_features"] = memory.to_feature_vector()   # (64,) float32

    Parameters
    ----------
    confidence_decay : float
        Per-step multiplicative decay applied to enemy slots.
        Default 0.985 ≈ 150 steps to reach eviction threshold.
        Doors use _CONFIDENCE_DECAY_DOOR (0.996); pickups use _CONFIDENCE_DECAY_PICKUP (0.990).
        These per-category values can be overridden by subclassing.
    angular_merge_deg : float
        Maximum angular separation (degrees) to merge a new observation into
        an existing slot of the same type.  Default 30° (tightened from 45°
        to reduce false merges in dense scenes while handling observation jitter).
    dist_merge_tol : float
        Maximum distance difference (on the [0, 4] entity dist_norm scale) to
        allow angular merging.  Objects at the same bearing but very different
        distances are distinct objects.  Default 1.0 (= 1024 map units).
        Set to a large value (e.g. 999) to revert to angle-only matching.
    """

    def __init__(
        self,
        confidence_decay:   float = _CONFIDENCE_DECAY_ENEMY,
        angular_merge_deg:  float = _ANGULAR_MERGE_DEG,
        dist_merge_tol:     float = _DIST_MERGE_TOL,
    ) -> None:
        self._decay_enemy      = float(confidence_decay)
        self._decay_door       = _CONFIDENCE_DECAY_DOOR
        self._decay_pickup     = _CONFIDENCE_DECAY_PICKUP
        self._decay_projectile = _CONFIDENCE_DECAY_PROJECTILE
        self._merge_thresh     = float(angular_merge_deg)
        self._dist_merge_tol   = float(dist_merge_tol)

        # Object memory lists (one per category)
        self._enemies: List[ObjectMemorySlot] = []
        self._doors:   List[ObjectMemorySlot] = []
        self._pickups: List[ObjectMemorySlot] = []

        # Direction exploration state
        self._dir_tracker = DirectionExplorationTracker()

        # Keycard possession state (6 cards)
        self._keys: Dict[str, KeyCardTracker] = {
            name: KeyCardTracker(name=name, held=False, ever_seen=False,
                                 ever_held=False, step_picked=-1)
            for name in _KEY_NAMES
        }

        # Step counter (episode-local)
        self._step: int = 0

        # Last step any enemy / door was confirmed (for recency features)
        self._last_enemy_step: int = -1
        self._last_door_step:  int = -1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all episode-specific state.

        Must be called at the start of every new episode before the first
        ``update()`` call.  Resets object slots, direction tracker, key
        states, and step counter.
        """
        self._enemies.clear()
        self._doors.clear()
        self._pickups.clear()
        self._dir_tracker.reset()
        for kt in self._keys.values():
            kt.held      = False
            kt.ever_seen = False
            kt.ever_held = False
            kt.step_picked = -1
        self._step          = 0
        self._last_enemy_step = -1
        self._last_door_step  = -1

    def update(self, processed_obs: object) -> None:
        """Update memory from a single step's ProcessedObservation.

        Parameters
        ----------
        processed_obs : ProcessedObservation
            Output of ``SensoryPipeline.process(obs)``.

        Notes
        -----
        Update order:
        1. Decay all existing slot confidences.
        2. Ingest visible entities (enemies, pickups).
        3. Ingest ray door hits.
        4. Update direction tracker.
        5. Update keycard possession from vitals.
        6. Evict expired slots.
        7. Increment step counter.
        """
        rays    = processed_obs.rays      # type: ignore[attr-defined]
        entities = processed_obs.entities  # type: ignore[attr-defined]
        vitals   = processed_obs.vitals    # type: ignore[attr-defined]

        # 1. Decay
        self._decay_all()

        # 2. Visible entities
        self._ingest_entities(entities)

        # 3. Ray door hits
        self._ingest_ray_doors(rays)

        # 4. Direction tracker
        self._dir_tracker.update(rays.distance_norm, rays.hit_type_raw)

        # 5. Keycards
        self._update_keys(vitals, entities)

        # 6. Evict
        _evict(self._enemies, _MAX_ENEMY_SLOTS,  _EVICTION_THRESHOLD)
        _evict(self._doors,   _MAX_DOOR_SLOTS,   _EVICTION_THRESHOLD)
        _evict(self._pickups, _MAX_PICKUP_SLOTS, _EVICTION_THRESHOLD)

        # 7. Step
        self._step += 1

    def to_feature_vector(self) -> np.ndarray:
        """Return the MEMORY_FEATURE_DIM-dim float32 feature vector for the network.

        All values are in [0.0, 1.0] unless noted.  Structure:

        [0:8]   Enemy cluster summary (8 dims)
        [8:12]  Open door summary (4 dims)
        [12:16] Locked door summary (4 dims)
        [16:22] Key held flags (6 dims, one per key)
        [22:28] Key ever-seen flags (6 dims)
        [28:36] Pickup summary: health, armor, ammo (8 dims)
        [36:52] Direction explored fractions, 16 buckets
        [52:60] Direction open fractions, 8 buckets
        [60:62] Recency: steps_since_enemy_norm, steps_since_door_norm
        [62:64] Global: total_steps_norm, n_keys_held_norm

        Returns
        -------
        (MEMORY_FEATURE_DIM,) float32 numpy array.
        """
        vec = np.zeros(MEMORY_FEATURE_DIM, dtype=np.float32)

        active_enemies  = _active(self._enemies)
        active_doors    = _active(self._doors)
        active_pickups  = _active(self._pickups)
        locked_doors    = [d for d in active_doors if d.type_id == _RAY_LOCKED]
        open_doors      = [d for d in active_doors if d.type_id == _RAY_DOOR]

        # --- [0:8] Enemy cluster summary ---
        vec[0] = _count_norm(active_enemies,  _MAX_ENEMY_SLOTS)
        vec[1] = _nearest_dist_norm(active_enemies)
        vec[2] = _mean_dist_norm(active_enemies)
        vec[3], vec[4] = _nearest_angle_sincos(active_enemies)
        vec[5] = _mean_angle_sin(active_enemies)
        vec[6] = _max_confidence(active_enemies)
        vec[7] = _mean_confidence(active_enemies)

        # --- [8:12] Open door summary ---
        vec[8]  = _count_norm(open_doors, _MAX_DOOR_SLOTS)
        vec[9]  = _nearest_dist_norm(open_doors)
        vec[10], vec[11] = _nearest_angle_sincos(open_doors)

        # --- [12:16] Locked door summary ---
        vec[12] = _count_norm(locked_doors, _MAX_DOOR_SLOTS)
        vec[13] = _nearest_dist_norm(locked_doors)
        vec[14], vec[15] = _nearest_angle_sincos(locked_doors)

        # --- [16:22] Key held flags ---
        for i, name in enumerate(_KEY_NAMES):
            vec[16 + i] = 1.0 if self._keys[name].held else 0.0

        # --- [22:28] Key ever-seen flags ---
        for i, name in enumerate(_KEY_NAMES):
            vec[22 + i] = 1.0 if self._keys[name].ever_seen else 0.0

        # --- [28:36] Pickup summary (health / armor / ammo) ---
        health_slots = [p for p in active_pickups if p.type_id == _ENT_HEALTH]
        armor_slots  = [p for p in active_pickups if p.type_id == _ENT_ARMOR]
        ammo_slots   = [p for p in active_pickups if p.type_id in (_ENT_AMMO, _ENT_WEAPON)]
        vec[28] = _count_norm(health_slots, _MAX_PICKUP_SLOTS)
        vec[29] = _count_norm(armor_slots,  _MAX_PICKUP_SLOTS)
        vec[30] = _count_norm(ammo_slots,   _MAX_PICKUP_SLOTS)
        vec[31] = _nearest_dist_norm(health_slots)
        vec[32] = _nearest_dist_norm(armor_slots)
        vec[33] = _nearest_dist_norm(ammo_slots)
        vec[34], vec[35] = _nearest_angle_sincos(health_slots)

        # --- [36:52] Direction explored fractions (16 buckets) ---
        vec[36:52] = self._dir_tracker.explored_fractions(n_buckets=16)

        # --- [52:60] Direction open fractions (8 buckets) ---
        vec[52:60] = self._dir_tracker.open_fractions(n_buckets=8)

        # --- [60:62] Recency ---
        vec[60] = _steps_since_norm(self._last_enemy_step, self._step)
        vec[61] = _steps_since_norm(self._last_door_step,  self._step)

        # --- [62:64] Global ---
        vec[62] = min(1.0, self._step / 1000.0)                        # episode age
        vec[63] = sum(kt.held for kt in self._keys.values()) / 6.0     # keys held fraction

        return vec

    def snapshot(self) -> MemorySnapshot:
        """Return a compact, human-readable summary of current memory state."""
        active_enemies  = _active(self._enemies)
        active_doors    = _active(self._doors)
        locked_doors    = [d for d in active_doors if d.type_id == _RAY_LOCKED]
        n_keys_held     = sum(kt.held for kt in self._keys.values())

        return MemorySnapshot(
            n_active_enemies=len(active_enemies),
            n_active_doors=len(active_doors),
            n_locked_doors=len(locked_doors),
            n_keys_held=n_keys_held,
            nearest_enemy_dist=_nearest_dist_norm(active_enemies),
            nearest_door_dist=_nearest_dist_norm(active_doors),
            nearest_locked_dist=_nearest_dist_norm(locked_doors),
            steps_since_enemy=(
                self._step - self._last_enemy_step
                if self._last_enemy_step >= 0 else -1
            ),
            steps_since_door=(
                self._step - self._last_door_step
                if self._last_door_step >= 0 else -1
            ),
            total_steps=self._step,
        )

    @property
    def step(self) -> int:
        """Current episode step count."""
        return self._step

    def active_enemies(self) -> List[ObjectMemorySlot]:
        """Return a list of enemy slots above eviction threshold (live view)."""
        return _active(self._enemies)

    def active_doors(self) -> List[ObjectMemorySlot]:
        """Return a list of door slots above eviction threshold (live view)."""
        return _active(self._doors)

    def active_pickups(self) -> List[ObjectMemorySlot]:
        """Return a list of pickup slots above eviction threshold (live view)."""
        return _active(self._pickups)

    def keys_held(self) -> Dict[str, bool]:
        """Return current key-held status as a name → bool dict."""
        return {name: kt.held for name, kt in self._keys.items()}

    def __repr__(self) -> str:
        snap = self.snapshot()
        return (
            f"EgocentricSpatialMemory("
            f"step={self._step}, "
            f"enemies={snap.n_active_enemies}, "
            f"doors={snap.n_active_doors}/{snap.n_locked_doors}locked, "
            f"keys_held={snap.n_keys_held}/6)"
        )

    # ------------------------------------------------------------------
    # Private: ingest helpers
    # ------------------------------------------------------------------

    def _decay_all(self) -> None:
        """Apply per-step, per-category confidence decay to all object slots.

        Decay rates differ by category because:
        - Enemies move; their positions become stale quickly → faster decay.
        - Projectile enemies move even faster → fastest decay.
        - Doors are static fixtures; they remain valid across the whole level → slowest.
        - Pickups are static but can be collected; medium decay.
        """
        for slot in self._enemies:
            decay = (
                self._decay_projectile
                if slot.type_id == _ENT_ENEMY_PROJECTILE
                else self._decay_enemy
            )
            slot.confidence *= decay
        for slot in self._doors:
            slot.confidence *= self._decay_door
        for slot in self._pickups:
            slot.confidence *= self._decay_pickup

    def _ingest_entities(self, entities: object) -> None:
        """Update enemy and pickup memory from EntityFeatures."""
        # entities.mask is (32,) bool; entities.features_norm is (32, 5) float32
        # entities.entity_type is (32,) int64
        mask_arr     = entities.mask            # type: ignore[attr-defined]
        feat_arr     = entities.features_raw    # type: ignore[attr-defined]  # (32,5) unnorm dist
        type_arr     = entities.entity_type     # type: ignore[attr-defined]  # (32,) int64

        for i in range(len(mask_arr)):
            if not mask_arr[i]:
                continue  # padding slot

            type_id  = int(type_arr[i])
            # col-0: dist already divided by 1024 at env level (max 4.0)
            dist_norm = float(feat_arr[i, 0])
            # col-1: sin(rel_angle), col-2: cos(rel_angle) → recover angle
            angle_deg = _entity_sincos_to_angle_deg(
                float(feat_arr[i, 1]), float(feat_arr[i, 2])
            )

            if type_id in _ENTITY_ENEMY_TYPES:
                self._reinforce_or_add(
                    self._enemies, type_id,
                    _ENTITY_NAME.get(type_id, "enemy"),
                    dist_norm, angle_deg,
                )
                self._last_enemy_step = self._step

            elif type_id in _ENTITY_PICKUP_TYPES:
                self._reinforce_or_add(
                    self._pickups, type_id,
                    _ENTITY_NAME.get(type_id, "pickup"),
                    dist_norm, angle_deg,
                )
                # Mark any key sighting
                if type_id == _ENT_KEY:
                    for kt in self._keys.values():
                        kt.ever_seen = True

    def _ingest_ray_doors(self, rays: object) -> None:
        """Update door memory from RayFeatures.

        Doors observed via raycasts are stored with the ray direction as
        the angle and the ray distance as dist_norm.
        """
        distance_norm = rays.distance_norm   # type: ignore[attr-defined]  # (64,) float32
        hit_type_raw  = rays.hit_type_raw    # type: ignore[attr-defined]  # (64,) int64

        for ray_idx in range(_NUM_RAYS):
            ht = int(hit_type_raw[ray_idx])
            if ht not in (_RAY_DOOR, _RAY_LOCKED):
                continue

            # Ray distances are in [0, 1] but entity distances are in [0, 4].
            # Store on the entity scale [0, 4] so _nearest_dist_norm normalises
            # consistently (divides all by 4.0 to get [0, 1]).
            dist_norm = float(distance_norm[ray_idx]) * _MAX_ENTITY_DIST_NORM  # [0,1]→[0,4]
            angle_deg = _ray_index_to_angle_deg(ray_idx)
            type_name = "door" if ht == _RAY_DOOR else "locked_door"

            self._reinforce_or_add(
                self._doors, ht, type_name, dist_norm, angle_deg
            )
            self._last_door_step = self._step

    def _update_keys(self, vitals: object, entities: object) -> None:
        """Update KeyCardTracker from vitals[8:14] and visible entity types."""
        # vitals.keys_held is a dict[str, bool] built from vitals[8:14]
        keys_held_dict: Dict[str, bool] = vitals.keys_held  # type: ignore[attr-defined]

        for name, kt in self._keys.items():
            current_held = keys_held_dict.get(name, False)
            if current_held and not kt.ever_held:
                kt.ever_held  = True
                kt.step_picked = self._step
            kt.held = current_held

        # Cross-reference: if any entity_type == _ENT_KEY is visible → ever_seen
        mask_arr = entities.mask        # type: ignore[attr-defined]
        type_arr = entities.entity_type  # type: ignore[attr-defined]
        for i in range(len(mask_arr)):
            if mask_arr[i] and int(type_arr[i]) == _ENT_KEY:
                for kt in self._keys.values():
                    kt.ever_seen = True
                break

    def _reinforce_or_add(
        self,
        lst:       List[ObjectMemorySlot],
        type_id:   int,
        type_name: str,
        dist_norm: float,
        angle_deg: float,
    ) -> None:
        """Find the best matching slot by dual-criterion (angle + distance), then reinforce or add.

        Matching criteria (Phase 2 fix)
        --------------------------------
        A candidate slot matches the new observation ONLY when BOTH hold:
          1. ``type_id`` matches exactly (same category).
          2. Angular separation < ``self._merge_thresh`` (default 30°).
          3. Distance difference < ``self._dist_merge_tol`` (default 1.0 on [0,4] scale).

        Among all qualifying candidates, the one with the smallest angular
        separation wins (tightest directional match).

        Rationale
        ---------
        Pure angle-only matching at 45° caused two enemies at the same bearing
        but very different distances (e.g. 0.2 and 3.5) to collapse into one slot.
        Adding the distance criterion prevents this false merge while still
        allowing legitimate reinforcement of the same object seen from a slightly
        different angle (observation jitter from adjacent rays ≈ 5.6° each).

        Parameters
        ----------
        lst       : mutable list of ObjectMemorySlot for one category.
        type_id   : integer code that must match exactly.
        type_name : human-readable label for new slots.
        dist_norm : observed distance on the [0, 4] entity scale (or [0, 1] for rays).
        angle_deg : observed ego-relative angle in [0, 360).
        """
        best: Optional[ObjectMemorySlot] = None
        best_angle_diff = self._merge_thresh  # running minimum; starts at threshold

        for slot in lst:
            if slot.type_id != type_id:
                continue
            angle_diff = _angular_diff_deg(angle_deg, slot.angle_deg)
            if angle_diff >= self._merge_thresh:
                continue   # too far in angle — different object or extreme rotation
            dist_diff = abs(dist_norm - slot.dist_norm)
            if dist_diff >= self._dist_merge_tol:
                continue   # same bearing, very different depth — distinct object
            # Both criteria satisfied; prefer tightest angular match
            if angle_diff < best_angle_diff:
                best_angle_diff = angle_diff
                best = slot

        if best is not None:
            # Reinforce existing slot with updated position (exponential moving average
            # for distance to reduce jitter; hard update for angle since we want
            # the most recent bearing).
            _EMA = 0.3   # weight for new observation (0 = never update, 1 = always replace)
            best.dist_norm   = (1.0 - _EMA) * best.dist_norm + _EMA * dist_norm
            best.angle_deg   = angle_deg   # latest bearing; confidence decay handles staleness
            best.confidence  = min(1.0, best.confidence + _CONFIDENCE_REINFORCE)
            best.step_last   = self._step
            best.n_observed += 1
        else:
            # No matching slot — add a new one
            lst.append(ObjectMemorySlot(
                type_id=type_id,
                type_name=type_name,
                dist_norm=dist_norm,
                angle_deg=angle_deg,
                confidence=_INITIAL_CONFIDENCE,
                step_first=self._step,
                step_last=self._step,
            ))


# ---------------------------------------------------------------------------
# Module-level utility functions
# ---------------------------------------------------------------------------

def _active(lst: List[ObjectMemorySlot]) -> List[ObjectMemorySlot]:
    """Return slots with confidence above eviction threshold."""
    return [s for s in lst if s.confidence > _EVICTION_THRESHOLD]


def _evict(
    lst:       List[ObjectMemorySlot],
    max_size:  int,
    threshold: float,
) -> None:
    """Remove low-confidence slots in-place, then cap by max_size."""
    lst[:] = [s for s in lst if s.confidence > threshold]
    if len(lst) > max_size:
        lst.sort(key=lambda s: s.confidence, reverse=True)
        del lst[max_size:]


def _count_norm(slots: List[ObjectMemorySlot], max_count: int) -> float:
    """Fraction of maximum capacity that is currently occupied."""
    return min(1.0, len(slots) / max(1, max_count))


def _nearest_dist_norm(slots: List[ObjectMemorySlot]) -> float:
    """Re-normalised distance of the nearest slot.

    Entity distances are in [0, 4.0]; we re-normalise to [0, 1] by dividing by
    _MAX_ENTITY_DIST_NORM = 4.0.  Ray distances are already in [0, 1].
    Returns 1.0 (max = "nothing nearby") if slots is empty.
    """
    if not slots:
        return 1.0
    raw = min(s.dist_norm for s in slots)
    return float(np.clip(raw / _MAX_ENTITY_DIST_NORM, 0.0, 1.0))


def _mean_dist_norm(slots: List[ObjectMemorySlot]) -> float:
    """Average re-normalised distance across all slots.  Returns 1.0 if empty."""
    if not slots:
        return 1.0
    mean_raw = float(np.mean([s.dist_norm for s in slots]))
    return float(np.clip(mean_raw / _MAX_ENTITY_DIST_NORM, 0.0, 1.0))


def _nearest_angle_sincos(slots: List[ObjectMemorySlot]) -> tuple[float, float]:
    """Sin and cos of the nearest (by distance) slot's angle.

    Returns (0.0, 1.0) → 0° = forward, if slots is empty.
    """
    if not slots:
        return 0.0, 1.0  # 0° = forward
    nearest = min(slots, key=lambda s: s.dist_norm)
    rad = math.radians(nearest.angle_deg)
    return float(math.sin(rad)), float(math.cos(rad))


def _mean_angle_sin(slots: List[ObjectMemorySlot]) -> float:
    """Mean sine of all slot angles (approximate circular mean direction).

    Returns 0.0 if slots is empty.
    """
    if not slots:
        return 0.0
    return float(np.mean([math.sin(math.radians(s.angle_deg)) for s in slots]))


def _max_confidence(slots: List[ObjectMemorySlot]) -> float:
    """Maximum confidence across all slots.  Returns 0.0 if empty."""
    if not slots:
        return 0.0
    return float(max(s.confidence for s in slots))


def _mean_confidence(slots: List[ObjectMemorySlot]) -> float:
    """Mean confidence across all slots.  Returns 0.0 if empty."""
    if not slots:
        return 0.0
    return float(np.mean([s.confidence for s in slots]))


def _steps_since_norm(last_step: int, current_step: int, max_steps: int = 200) -> float:
    """Normalised steps since an event occurred.

    Returns 1.0 if the event never happened or happened > max_steps ago.
    Returns 0.0 if the event happened this step.
    """
    if last_step < 0:
        return 1.0
    elapsed = current_step - last_step
    return float(np.clip(elapsed / max_steps, 0.0, 1.0))
